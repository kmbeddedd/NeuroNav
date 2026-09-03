from pathlib import Path
import numpy as np
import pandas as pd
from scripts.benchmark_ps08 import TARGETS, evaluate_predictions, load_series

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
