from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_and_clean_data, prepare_pytorch_datasets
TARGETS = ['Error_X', 'Error_Y', 'Error_Z', 'Error_Clock']

def synthetic_frame(epochs: int=80, *, invalid_clock_index: int | None=10, missing_index: int | None=None) -> pd.DataFrame:
    timestamps = pd.date_range('2025-01-01', periods=epochs, freq='15min')
    index = np.arange(epochs, dtype=np.float64)
    frame = pd.DataFrame({'Timestamp': timestamps, 'Satellite_ID': 'G01', 'Constellation': 'G', 'Broadcast_X': 20000000.0 + 1000.0 * index, 'Broadcast_Y': 15000000.0 - 400.0 * index, 'Broadcast_Z': 10000000.0 + 200.0 * index, 'Broadcast_Clock': 0.0002 + index * 1e-08, 'Modelled_Clock': 0.0002 + index * 5e-09, 'Error_X': 10.0 + 0.25 * index, 'Error_Y': -20.0 + 0.5 * index, 'Error_Z': np.sin(index / 5.0) * 3.0, 'Error_Clock': index * 5e-09, '3D_Orbit_Error': 25.0 + index})
    if invalid_clock_index is not None:
        frame.loc[invalid_clock_index, 'Error_Clock'] = np.nan
    frame.loc[5, '3D_Orbit_Error'] = 100000.0
    if missing_index is not None:
        frame = frame.drop(index=missing_index).reset_index(drop=True)
    return frame

class DataPipelineContractTests(unittest.TestCase):

    def write_csv(self, frame: pd.DataFrame, directory: str) -> Path:
        path = Path(directory) / 'synthetic.csv'
        frame.to_csv(path, index=False)
        return path

    def prepare(self, path: Path):
        return prepare_pytorch_datasets(str(path), input_window=4, forecast_horizon=3, batch_size=8, train_end_date='2025-01-01 15:00:00', seed=7)

    def test_non_finite_clock_is_masked_without_removing_row(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            train, test, satellites = load_and_clean_data(str(path), train_end_date='2025-01-01 15:00:00')
        combined = pd.concat([train, test], ignore_index=True)
        invalid = combined[~combined['Error_Clock_valid']]
        self.assertEqual(len(combined), len(frame))
        self.assertEqual(satellites, ['G01'])
        self.assertEqual(len(invalid), 1)
        self.assertTrue(np.isnan(invalid.iloc[0]['Error_Clock']))
        self.assertEqual(int((combined['3D_Orbit_Error'] >= 50000).sum()), 1)

    def test_targets_are_scaled_once_and_invalid_clock_never_trains(self):
        frame = synthetic_frame()
        frame['Orbit_Class'] = 'MEO'
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)
        fit_end = pd.Timestamp(bundle['split_metadata']['scaler_fit_end_exclusive'])
        expected_clock_mean = frame.loc[frame['Timestamp'] < fit_end, 'Error_Clock'].mean()
        self.assertAlmostEqual(bundle['target_scaler'].mean_[3], expected_clock_mean)
        expected_x_mean = frame.loc[frame['Timestamp'] < fit_end, 'Error_X'].mean()
        self.assertAlmostEqual(bundle['target_scaler'].mean_[0], expected_x_mean)
        self.assertAlmostEqual(bundle['feature_scaler'].mean_[0], expected_x_mean)
        self.assertEqual(bundle['target_feature_indices'], [0, 1, 2, 3])
        self.assertIn('Broadcast_VX', bundle['feature_cols'])
        self.assertIn('Broadcast_Phase_Sin', bundle['feature_cols'])
        self.assertEqual(bundle['orbit_class_classes'], ['MEO'])
        self.assertEqual(bundle['orbit_class_by_satellite'], [0])
        json.dumps(bundle['data_quality_report'])
        json.dumps(bundle['split_metadata'])
        self.assertEqual(len(next(iter(bundle['train_loader']))), 5)
        saw_invalid_clock = False
        lookup = frame.set_index(['Satellite_ID', 'Timestamp'])
        for split in ('train', 'val', 'test'):
            restored = bundle['target_scaler'].inverse_transform(bundle[f'Y_{split}'].reshape(-1, 4)).reshape(bundle[f'Y_{split}'].shape)
            masks = bundle[f'TARGET_MASK_{split}'].astype(bool)
            self.assertEqual(masks.shape, bundle[f'Y_{split}'].shape)
            self.assertTrue(np.isfinite(bundle[f'Y_{split}']).all())
            for sample_index, satellite_id in enumerate(bundle[f'SATELLITE_IDS_{split}']):
                for horizon_index, timestamp in enumerate(bundle[f'LABEL_TIMESTAMPS_{split}'][sample_index]):
                    raw = lookup.loc[(satellite_id, pd.Timestamp(timestamp)), TARGETS].to_numpy(float)
                    valid = masks[sample_index, horizon_index]
                    np.testing.assert_allclose(restored[sample_index, horizon_index, valid], raw[valid], rtol=1e-05, atol=1e-07)
                    if not valid[3]:
                        saw_invalid_clock = True
                        self.assertEqual(bundle[f'Y_{split}'][sample_index, horizon_index, 3], 0.0)
        self.assertTrue(saw_invalid_clock)

    def test_every_emitted_window_is_contiguous_even_when_source_has_gap(self):
        frame = synthetic_frame(invalid_clock_index=None, missing_index=25)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)
        self.assertEqual(bundle['data_quality_report']['irregular_steps'], 1)
        self.assertGreater(bundle['data_quality_report']['skipped_noncontiguous_windows'], 0)
        expected_ns = pd.Timedelta(minutes=15).value
        for split in ('train', 'val', 'test'):
            combined = np.concatenate([bundle[f'INPUT_TIMESTAMPS_{split}'], bundle[f'LABEL_TIMESTAMPS_{split}']], axis=1).astype('datetime64[ns]').astype(np.int64)
            self.assertTrue(np.all(np.diff(combined, axis=1) == expected_ns))

    def test_train_validation_and_test_label_timestamps_are_disjoint(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)
        labels = {split: set(bundle[f'LABEL_TIMESTAMPS_{split}'].reshape(-1).tolist()) for split in ('train', 'val', 'test')}
        self.assertTrue(labels['train'].isdisjoint(labels['val']))
        self.assertTrue(labels['train'].isdisjoint(labels['test']))
        self.assertTrue(labels['val'].isdisjoint(labels['test']))
        self.assertGreater(bundle['split_metadata']['purged_boundary_windows'], 0)

    def test_insufficient_history_has_actionable_error(self):
        frame = synthetic_frame(epochs=10, invalid_clock_index=None)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            with self.assertRaisesRegex(ValueError, 'Insufficient history'):
                prepare_pytorch_datasets(str(path), input_window=4, forecast_horizon=3, train_end_date='2025-01-01 01:30:00')

    def test_strict_causal_test_split_prevents_input_leakage(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)
        test_start = pd.Timestamp(bundle['split_metadata']['test_start'])
        self.assertEqual(bundle['split_metadata']['evaluation_mode'], 'strict_block')
        self.assertTrue(bundle['split_metadata']['strict_test_input_boundary'])
        self.assertGreater(bundle['split_metadata']['purged_leakage_windows'], 0)
        self.assertGreater(len(bundle['X_test']), 0)

        for i in range(len(bundle['X_test'])):
            inp = bundle['INPUT_TIMESTAMPS_test'][i]
            lbl = bundle['LABEL_TIMESTAMPS_test'][i]
            max_in = pd.Timestamp(inp[-1])
            min_lbl = pd.Timestamp(lbl[0])
            self.assertLess(max_in, test_start)
            self.assertGreaterEqual(min_lbl, test_start)
            self.assertTrue(set(inp).isdisjoint(set(lbl)))
            self.assertFalse(any(pd.Timestamp(t) >= test_start for t in inp))

    def test_test_input_features_contain_no_test_period_targets_or_rolling_stats(self):
        frame = synthetic_frame()
        test_boundary = pd.Timestamp('2025-01-01 15:00:00')
        # Inject extreme, distinctive marker values into the test period
        test_mask = frame['Timestamp'] >= test_boundary
        frame.loc[test_mask, 'Error_X'] = 9999.0
        frame.loc[test_mask, 'Error_Y'] = 8888.0
        frame.loc[test_mask, 'Error_Z'] = 7777.0
        frame.loc[test_mask, 'Error_Clock'] = 6666.0

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)

        # 1. Verify scaler was fit exclusively on pre-validation training history
        self.assertLess(bundle['target_scaler'].mean_[0], 50.0)
        self.assertLess(bundle['feature_scaler'].mean_[0], 50.0)

        # 2. Recover unscaled physical features in test inputs
        unscaled_X_test = bundle['feature_scaler'].inverse_transform(
            bundle['X_test'].reshape(-1, bundle['num_features'])
        ).reshape(bundle['X_test'].shape)

        target_and_roll_cols = [
            'Error_X', 'Error_Y', 'Error_Z', 'Error_Clock',
            'Error_X_roll_mean', 'Error_Y_roll_mean', 'Error_Z_roll_mean', 'Error_Clock_roll_mean'
        ]
        for col in target_and_roll_cols:
            if col in bundle['feature_cols']:
                col_idx = bundle['feature_cols'].index(col)
                values = unscaled_X_test[:, :, col_idx]
                self.assertTrue(
                    np.all(values < 100.0),
                    f"Feature {col} leaked test-period values into test input X: max={values.max()}"
                )

        # 3. Verify actual test labels contain the test-period values
        unscaled_Y_test = bundle['target_scaler'].inverse_transform(
            bundle['Y_test'].reshape(-1, bundle['output_dim'])
        ).reshape(bundle['Y_test'].shape)
        np.testing.assert_allclose(unscaled_Y_test[..., 0], 9999.0)

    def test_validate_temporal_windows_catches_leakage_violations(self):
        from src.data import validate_temporal_windows
        val_start = pd.Timestamp('2025-01-01 12:00:00')
        test_start = pd.Timestamp('2025-01-01 15:00:00')

        # Test valid windows
        valid_parts = {
            'train': {
                'input_ts': [pd.date_range('2025-01-01 10:00:00', periods=4, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'label_ts': [pd.date_range('2025-01-01 11:00:00', periods=3, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'satellite_id': ['G01']
            },
            'val': {
                'input_ts': [pd.date_range('2025-01-01 11:00:00', periods=4, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'label_ts': [pd.date_range('2025-01-01 12:00:00', periods=3, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'satellite_id': ['G01']
            },
            'test': {
                'input_ts': [pd.date_range('2025-01-01 14:00:00', periods=4, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'label_ts': [pd.date_range('2025-01-01 15:00:00', periods=3, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'satellite_id': ['G01']
            }
        }
        report = validate_temporal_windows(valid_parts, val_start, test_start, evaluation_mode='strict_block')
        self.assertTrue(report['test']['strict_out_of_sample'])

        # Test leakage: test input containing test_start timestamp
        leaked_parts = {
            'train': valid_parts['train'],
            'val': valid_parts['val'],
            'test': {
                'input_ts': [pd.date_range('2025-01-01 14:15:00', periods=4, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'label_ts': [pd.date_range('2025-01-01 15:15:00', periods=3, freq='15min').to_numpy(dtype='datetime64[ns]')],
                'satellite_id': ['G01']
            }
        }
        with self.assertRaisesRegex(ValueError, 'Data leakage detected in test window'):
            validate_temporal_windows(leaked_parts, val_start, test_start, evaluation_mode='strict_block')

        # Test invalid evaluation_mode raises
        with self.assertRaisesRegex(ValueError, 'evaluation_mode must be'):
            validate_temporal_windows(valid_parts, val_start, test_start, evaluation_mode='invalid_mode')

    def test_rolling_evaluation_mode_is_explicit(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            # Default is strict_block
            bundle_default = prepare_pytorch_datasets(str(path), input_window=4, forecast_horizon=3, batch_size=8, train_end_date='2025-01-01 15:00:00')
            self.assertEqual(bundle_default['split_metadata']['evaluation_mode'], 'strict_block')
            self.assertTrue(bundle_default['split_metadata']['strict_test_input_boundary'])

            # Explicit rolling
            bundle_rolling = prepare_pytorch_datasets(str(path), input_window=4, forecast_horizon=3, batch_size=8, train_end_date='2025-01-01 15:00:00', evaluation_mode='rolling')
            self.assertEqual(bundle_rolling['split_metadata']['evaluation_mode'], 'rolling')
            self.assertFalse(bundle_rolling['split_metadata']['strict_test_input_boundary'])
            self.assertGreater(len(bundle_rolling['X_test']), len(bundle_default['X_test']))


if __name__ == '__main__':
    unittest.main()
