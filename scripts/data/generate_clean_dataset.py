from __future__ import annotations
import argparse
import math
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from scripts.audit_data import audit_csv
GM = 398600441800000.0
OMEGA_E = 7.2921151467e-05
CONSTELLATION_CONFIGS = {'G': {'name': 'GPS', 'type': 'MEO', 'semi_major_axis_m': 26559700.0, 'inclination_deg': 55.0, 'eccentricity': 0.01, 'orbit_err_scale_m': 1.2, 'clock_err_scale_s': 4.5e-09, 'sat_count': 8}, 'R': {'name': 'GLONASS', 'type': 'MEO', 'semi_major_axis_m': 25510000.0, 'inclination_deg': 64.8, 'eccentricity': 0.005, 'orbit_err_scale_m': 1.8, 'clock_err_scale_s': 8e-09, 'sat_count': 6}, 'E': {'name': 'Galileo', 'type': 'MEO', 'semi_major_axis_m': 29600000.0, 'inclination_deg': 56.0, 'eccentricity': 0.002, 'orbit_err_scale_m': 0.8, 'clock_err_scale_s': 2e-09, 'sat_count': 6}, 'I': {'name': 'NavIC', 'type': 'GEO_GSO', 'semi_major_axis_m': 42164000.0, 'inclination_deg': 29.0, 'eccentricity': 0.001, 'orbit_err_scale_m': 2.2, 'clock_err_scale_s': 6e-09, 'sat_count': 7}}

def simulate_satellite_orbit_and_errors(sat_id: str, const_code: str, start_time: datetime, epochs: int, cadence_minutes: int=15, seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + hash(sat_id) % 100000)
    cfg = CONSTELLATION_CONFIGS[const_code]
    a = cfg['semi_major_axis_m']
    inc = math.radians(cfg['inclination_deg'])
    e = cfg['eccentricity']
    n = math.sqrt(GM / a ** 3)
    period_sec = 2.0 * math.pi / n
    raan_0 = rng.uniform(0.0, 2.0 * math.pi)
    arg_perigee = rng.uniform(0.0, 2.0 * math.pi)
    m_0 = rng.uniform(0.0, 2.0 * math.pi)
    srp_phase_x = rng.uniform(0.0, 2.0 * math.pi)
    srp_phase_y = rng.uniform(0.0, 2.0 * math.pi)
    srp_phase_z = rng.uniform(0.0, 2.0 * math.pi)
    clock_bias_0 = rng.normal(0.0, 5e-05)
    clock_drift_rate = rng.normal(1e-11, 2e-12)
    records = []
    cadence_sec = cadence_minutes * 60
    err_rw_x = 0.0
    err_rw_y = 0.0
    err_rw_z = 0.0
    clk_err_rw = 0.0
    for step in range(epochs):
        t_sec = step * cadence_sec
        current_time = start_time + timedelta(seconds=t_sec)
        mean_anomaly = m_0 + n * t_sec
        true_anomaly = mean_anomaly + 2.0 * e * math.sin(mean_anomaly)
        radius = a * (1.0 - e * math.cos(mean_anomaly))
        u = arg_perigee + true_anomaly
        x_orb = radius * math.cos(u)
        y_orb = radius * math.sin(u)
        raan = raan_0 - OMEGA_E * t_sec
        x_m = x_orb * math.cos(raan) - y_orb * math.cos(inc) * math.sin(raan)
        y_m = x_orb * math.sin(raan) + y_orb * math.cos(inc) * math.cos(raan)
        z_m = y_orb * math.sin(inc)
        orbit_harmonic_x = cfg['orbit_err_scale_m'] * math.sin(2.0 * math.pi * t_sec / period_sec + srp_phase_x)
        orbit_harmonic_y = cfg['orbit_err_scale_m'] * math.cos(2.0 * math.pi * t_sec / period_sec + srp_phase_y)
        orbit_harmonic_z = cfg['orbit_err_scale_m'] * math.sin(4.0 * math.pi * t_sec / period_sec + srp_phase_z)
        err_rw_x = 0.98 * err_rw_x + rng.normal(0.0, 0.05)
        err_rw_y = 0.98 * err_rw_y + rng.normal(0.0, 0.05)
        err_rw_z = 0.98 * err_rw_z + rng.normal(0.0, 0.05)
        err_x = orbit_harmonic_x + err_rw_x
        err_y = orbit_harmonic_y + err_rw_y
        err_z = orbit_harmonic_z + err_rw_z
        mod_x = x_m
        mod_y = y_m
        mod_z = z_m
        bc_x = mod_x + err_x
        bc_y = mod_y + err_y
        bc_z = mod_z + err_z
        err_3d = math.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2)
        thermal_clock_s = cfg['clock_err_scale_s'] * 0.5 * math.sin(2.0 * math.pi * t_sec / period_sec)
        clk_err_rw = 0.995 * clk_err_rw + rng.normal(0.0, cfg['clock_err_scale_s'] * 0.08)
        err_clock = thermal_clock_s + clk_err_rw
        mod_clock = clock_bias_0 + clock_drift_rate * t_sec
        bc_clock = mod_clock + err_clock
        records.append({'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'), 'Satellite_ID': sat_id, 'Constellation': const_code, 'Broadcast_X': float(bc_x), 'Broadcast_Y': float(bc_y), 'Broadcast_Z': float(bc_z), 'Broadcast_Clock': float(bc_clock), 'Modelled_X': float(mod_x), 'Modelled_Y': float(mod_y), 'Modelled_Z': float(mod_z), 'Modelled_Clock': float(mod_clock), 'Error_X': float(err_x), 'Error_Y': float(err_y), 'Error_Z': float(err_z), '3D_Orbit_Error': float(err_3d), 'Error_Clock': float(err_clock)})
    return pd.DataFrame(records)

def generate_benchmark_dataset(output_path: Path, days: int=8, constellations: list[str] | None=None, start_date: str='2026-08-01 00:00:00') -> pd.DataFrame:
    if constellations is None:
        constellations = ['G', 'R']
    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    epochs_per_day = 24 * 4
    total_epochs = days * epochs_per_day
    all_frames = []
    for const in constellations:
        cfg = CONSTELLATION_CONFIGS[const]
        count = cfg['sat_count']
        for idx in range(1, count + 1):
            sat_id = f'{const}{idx:02d}'
            df = simulate_satellite_orbit_and_errors(sat_id=sat_id, const_code=const, start_time=start_dt, epochs=total_epochs, cadence_minutes=15, seed=42 + idx * 7)
            all_frames.append(df)
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(['Satellite_ID', 'Timestamp']).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"[OK] Created clean GNSS dataset: {output_path} ({len(combined):,} rows, {combined['Satellite_ID'].nunique()} satellites)")
    return combined

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate clean, physics-based Multi-GNSS benchmark dataset')
    parser.add_argument('--out', default='data_acquisition/CLEAN_GNSS_BENCHMARK.csv')
    parser.add_argument('--days', type=int, default=8, help='Number of 24h days (default: 8)')
    parser.add_argument('--constellations', nargs='+', default=['G', 'R'], help='Constellations: G, R, E, I')
    args = parser.parse_args()
    out_file = Path(args.out)
    generate_benchmark_dataset(out_file, days=args.days, constellations=args.constellations)
    print('\n--- Running Strict Data Contract Audit ---')
    audit_report = audit_csv(str(out_file), 'configs/data_contract.json')
    print(f"Passed Data Contract: {audit_report['passed']}")
    print(f"SP3 Missing Sentinels: {audit_report['sp3_missing_clock_sentinel_rows']}")
    print(f"1 km+ Orbit Error Fraction: {audit_report['orbit_error_at_least_1km_fraction']:.4%}")
    print(f"Max 3D Norm Mismatch: {audit_report['norm_identity_max_abs_error_m']:.2e} m")
    if not audit_report['passed']:
        print(f"Failures: {audit_report['critical_failures']}")
if __name__ == '__main__':
    main()
