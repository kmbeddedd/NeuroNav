"""Tests for NeuroNav production inference engine and deployment contracts."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.inference import NeuroNavModel, PredictionResult


def test_load_bilstm_model():
    """Verify loading BiLSTM deployment bundle and manifest."""
    model = NeuroNavModel.load('bilstm')
    assert model.model_type == 'bilstm'
    assert model.model_name == 'GNSS-BiLSTM-GRU'
    assert model.seq_len == 96
    assert model.forecast_horizon == 96
    assert len(model.feature_cols) == 21
    assert len(model.target_cols) == 4
    assert hasattr(model.feature_scaler, 'transform')
    assert hasattr(model.target_scaler, 'inverse_transform')
    assert model.uncertainty_supported is False


def test_load_transformer_model():
    """Verify loading Transformer deployment bundle and manifest."""
    model = NeuroNavModel.load('transformer')
    assert model.model_type == 'transformer'
    assert model.model_name == 'GNSS-Hybrid-Transformer'
    assert model.seq_len == 96
    assert model.forecast_horizon == 96
    assert len(model.feature_cols) == 21
    assert hasattr(model.feature_scaler, 'transform')
    assert hasattr(model.target_scaler, 'inverse_transform')
    assert model.uncertainty_supported is True


@pytest.fixture(scope="module")
def sample_data_path(tmp_path_factory):
    """Creates a temporary synthetic GNSS telemetry CSV conforming to inference contracts."""
    fn = tmp_path_factory.mktemp("sample_data") / "sample_gnss_data.csv"
    rows = []
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    for sat in ["G01", "G02"]:
        for step in range(120):
            t = start_time + pd.Timedelta(minutes=15 * step)
            phase = 2 * np.pi * step / 96.0
            rows.append({
                "Timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "Satellite_ID": sat,
                "Error_X": float(np.sin(phase) * 1.5),
                "Error_Y": float(np.cos(phase) * 1.2),
                "Error_Z": float(np.sin(2 * phase) * 0.8),
                "Error_Clock": float(0.05 * step),
                "Broadcast_X": float(26000000.0 * np.cos(phase)),
                "Broadcast_Y": float(26000000.0 * np.sin(phase)),
                "Broadcast_Z": float(1000000.0 * np.sin(phase)),
                "Broadcast_Clock": float(1e-5),
                "Broadcast_VX": float(-3000.0 * np.sin(phase)),
                "Broadcast_VY": float(3000.0 * np.cos(phase)),
                "Broadcast_VZ": float(500.0),
                "Broadcast_Clock_Drift": float(1e-11),
                "Broadcast_Radius": float(26000000.0),
                "Broadcast_Phase_Sin": float(np.sin(phase)),
                "Broadcast_Phase_Cos": float(np.cos(phase)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(fn, index=False)
    return str(fn)


def test_bilstm_predict_contract(sample_data_path):
    """Verify BiLSTM multihorizon prediction conforms to inference output contract."""
    model = NeuroNavModel.load('bilstm')
    df = model.predict(sample_data_path)

    assert isinstance(df, pd.DataFrame)
    # 2 satellites in sample dataset * 96 forecast steps = 192 rows
    assert len(df) == 192

    required_cols = [
        'forecast_step', 'forecast_time', 'Satellite_ID',
        'pred_Error_X', 'pred_Error_Y', 'pred_Error_Z', 'pred_Error_Clock',
        'pred_3D_Orbit_Error'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check for no NaNs in predictions
    assert not df['pred_Error_X'].isna().any(), "pred_Error_X contains NaNs"
    assert not df['pred_Error_Y'].isna().any(), "pred_Error_Y contains NaNs"
    assert not df['pred_Error_Z'].isna().any(), "pred_Error_Z contains NaNs"
    assert not df['pred_Error_Clock'].isna().any(), "pred_Error_Clock contains NaNs"
    assert not df['pred_3D_Orbit_Error'].isna().any(), "pred_3D_Orbit_Error contains NaNs"

    # Verify mathematical identity of derived 3D Orbit Error
    expected_3d = np.sqrt(df['pred_Error_X'] ** 2 + df['pred_Error_Y'] ** 2 + df['pred_Error_Z'] ** 2)
    assert np.allclose(df['pred_3D_Orbit_Error'], expected_3d, atol=1e-5), "Derived 3D error mismatch"


def test_transformer_predict_uncertainty(sample_data_path):
    """Verify Transformer prediction generates valid calibrated uncertainty bounds."""
    model = NeuroNavModel.load('transformer')
    df = model.predict(sample_data_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 192

    # Check uncertainty bounds
    for target in ['Error_X', 'Error_Y', 'Error_Z', 'Error_Clock']:
        low_col = f'pred_{target}_low'
        high_col = f'pred_{target}_high'
        assert low_col in df.columns
        assert high_col in df.columns
        # Low bound must be strictly less than high bound
        assert (df[low_col] < df[high_col]).all(), f"Uncertainty bound inverted for {target}"


def test_input_validation_missing_columns():
    """Verify input validation rejects DataFrames missing required columns."""
    model = NeuroNavModel.load('bilstm')

    # Missing Timestamp
    invalid_df1 = pd.DataFrame({'Satellite_ID': ['G01'], 'Error_X': [1.0], 'Error_Y': [1.0], 'Error_Z': [1.0], 'Error_Clock': [1.0]})
    with pytest.raises(ValueError, match="must contain a 'Timestamp'"):
        model.predict(invalid_df1)

    # Missing Satellite_ID
    invalid_df2 = pd.DataFrame({'Timestamp': ['2026-01-01 00:00:00'], 'Error_X': [1.0], 'Error_Y': [1.0], 'Error_Z': [1.0], 'Error_Clock': [1.0]})
    with pytest.raises(ValueError, match="must contain a 'Satellite_ID'"):
        model.predict(invalid_df2)

    # Missing error target
    invalid_df3 = pd.DataFrame({'Timestamp': ['2026-01-01 00:00:00'], 'Satellite_ID': ['G01'], 'Error_X': [1.0]})
    with pytest.raises(ValueError, match="missing required target columns"):
        model.predict(invalid_df3)


def test_input_validation_insufficient_history(sample_data_path):
    """Verify input validation rejects sequences with fewer than seq_len observations."""
    model = NeuroNavModel.load('bilstm')
    df = pd.read_csv(sample_data_path)
    short_df = df[df['Satellite_ID'] == 'G01'].iloc[:50].copy()

    with pytest.raises(ValueError, match="minimum lookback history of 96"):
        model.predict(short_df)


def test_single_satellite_selection(sample_data_path):
    """Verify specifying satellite_id filters prediction to only that satellite."""
    model = NeuroNavModel.load('bilstm')
    df = model.predict(sample_data_path, satellite_id='G01')

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 96
    assert df['Satellite_ID'].unique().tolist() == ['G01']


def test_prediction_result_dataclass(sample_data_path):
    """Verify returning raw PredictionResult dataclass objects."""
    model = NeuroNavModel.load('bilstm')
    results = model.predict(sample_data_path, satellite_id='G02', return_dataframe=False)

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, PredictionResult)
    assert res.satellite_id == 'G02'
    assert len(res.forecast_steps) == 96
    assert len(res.pred_error_x) == 96
    assert len(res.pred_3d_orbit_error) == 96

    df_converted = res.to_dataframe()
    assert isinstance(df_converted, pd.DataFrame)
    assert len(df_converted) == 96

