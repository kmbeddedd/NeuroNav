from pathlib import Path
import pandas as pd
from scripts.data.audit_data import audit_csv

def test_audit_detects_non_finite_required_value(tmp_path: Path):
    rows = []
    for step in range(3):
        rows.append({'Timestamp': f'2026-01-01 00:{step * 15:02d}:00', 'Satellite_ID': 'G01', 'Constellation': 'G', 'Broadcast_X': 1.0, 'Broadcast_Y': 2.0, 'Broadcast_Z': 3.0, 'Broadcast_Clock': 0.0, 'Modelled_X': 1.0, 'Modelled_Y': 2.0, 'Modelled_Z': 3.0, 'Modelled_Clock': 0.0, 'Error_X': 0.0, 'Error_Y': 0.0, 'Error_Z': 0.0, '3D_Orbit_Error': 0.0, 'Error_Clock': float('nan') if step == 1 else 0.0})
    csv_path = tmp_path / 'sample.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = audit_csv(str(csv_path))
    assert report['non_finite_required_values'] == 1
    assert not report['passed']
