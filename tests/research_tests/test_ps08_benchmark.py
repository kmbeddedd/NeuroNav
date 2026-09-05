from pathlib import Path
import numpy as np
import pandas as pd
from scripts.benchmark.benchmark_ps08 import (
    SeriesSpec,
    TARGETS,
    compose_orbit_class_specialist,
    evaluate_predictions,
    load_series,
    save_predictions,
)

def test_prediction_exports_separate_each_input_without_changing_values(tmp_path):
    datasets = {}
    predictions = {'Persistence': {}, 'Other': {}}
    for index, name in enumerate(['GEO', 'MEO-1', 'MEO-2']):
        values = np.arange(12, dtype=float).reshape(3, 4) + index * 100
        datasets[name] = {
            'spec': SeriesSpec(name, 'GEO' if index == 0 else 'MEO',
                               f'train{index}.csv', f'test{index}.csv'),
            'test': pd.DataFrame({'utc_time': pd.date_range('2025-09-08', periods=3, freq='h'),
                                  **{target: values[:, i] for i, target in enumerate(TARGETS)}}),
        }
        predictions['Persistence'][name] = values + 1
        predictions['Other'][name] = values - 2
    output = tmp_path / 'nested' / 'day8_predictions.csv'
    save_predictions(datasets, predictions, output,
                     diagnostics={'Other': {('MEO-1', 0): {'p_gate': 0.25}}})
    combined = pd.read_csv(output)
    assert len(combined) == 18
    for index, name in enumerate(datasets):
        separate = pd.read_csv(output.parent / f'test{index}_predictions.csv')
        pd.testing.assert_frame_equal(
            separate, combined.loc[combined['series'] == name].reset_index(drop=True)
        )
        assert set(separate['model']) == {'Persistence', 'Other'}
        assert separate.loc[separate['model'] == 'Persistence', 'residual_x_error_m'].eq(1).all()

def test_load_series_normalizes_headers_and_deduplicates(tmp_path: Path):
    path = tmp_path / 'sample.csv'
    frame = pd.DataFrame({'utc_time': ['2025-09-01 01:00', '2025-09-01 00:00', '2025-09-01 00:00'], 'x_error (m)': [2.0, 1.0, 1.0], 'y_error  (m)': [2.0, 1.0, 1.0], 'z_error (m)': [2.0, 1.0, 1.0], 'satclockerror (m)': [2.0, 1.0, 1.0]})
    frame.to_csv(path, index=False)
    loaded = load_series(path)
    assert list(loaded.columns) == ['utc_time', *TARGETS]
    assert len(loaded) == 2
    assert loaded['utc_time'].is_monotonic_increasing

def test_ranking_metric_averages_targets_equally():
    times = pd.date_range('2025-09-08', periods=8, freq='h')
    actual = np.arange(32, dtype=float).reshape(8, 4)
    datasets = {'sample': {'test': pd.DataFrame({'utc_time': times, **{target: actual[:, i] for i, target in enumerate(TARGETS)}})}}
    residual = np.array([-1.2, -0.8, -0.3, -0.1, 0.0, 0.2, 0.7, 1.4])
    predictions = {'sample': actual + residual[:, None]}
    report = evaluate_predictions(datasets, predictions)
    target_scores = [report['per_series']['sample']['per_target'][target]['shapiro_w'] for target in TARGETS]
    assert report['average_shapiro_w'] == np.mean(target_scores)
    assert report['rejected_test_count'] in range(5)
    expected_vector_error = np.linalg.norm(residual[:, None] * np.ones((1, 3)), axis=1).mean()
    assert np.isclose(report['orbit_vector_mae_m'], expected_vector_error)


def test_orbit_class_specialist_routes_targets_by_orbit_class():
    datasets = {
        'GEO': {'spec': SeriesSpec('GEO', 'GEO', 'train.csv', 'test.csv')},
        'MEO-1': {'spec': SeriesSpec('MEO-1', 'MEO', 'train.csv', 'test.csv')},
    }
    all_predictions = {
        'GEO Gated MoE': {'GEO': np.full((2, 4), 1.0)},
        'Gaussian Process': {'MEO-1': np.full((2, 4), 2.0)},
        'Random Forest': {'MEO-1': np.full((2, 4), 3.0)},
    }
    selections = {
        'MEO-1': {'orbit_model': 'Gaussian Process', 'clock_model': 'Random Forest'},
    }

    predictions, routing = compose_orbit_class_specialist(
        datasets, all_predictions, selections
    )

    assert np.all(predictions['GEO'] == 1.0)
    assert np.all(predictions['MEO-1'][:, :3] == 2.0)
    assert np.all(predictions['MEO-1'][:, 3] == 3.0)
    assert routing['MEO-1'] == {
        'orbit_model': 'Gaussian Process',
        'clock_model': 'Random Forest',
    }
