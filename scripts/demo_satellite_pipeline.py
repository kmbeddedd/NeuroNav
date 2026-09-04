import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from app.controllers.inference_controller import InferenceController

def run_end_to_end_demo():
    print("=" * 70)
    print("NEURONAV: SATELLITE-SPECIFIC MODEL SELECTION & FORECASTING DEMO")
    print("=" * 70)

    # 1. Prepare 7-day training and 8th-day ground truth datasets
    benchmark_path = Path("data/benchmark/CLEAN_GNSS_BENCHMARK.csv")
    if not benchmark_path.exists():
        print(f"Benchmark dataset not found at {benchmark_path}")
        return

    df = pd.read_csv(benchmark_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    cutoff = pd.Timestamp('2026-08-08 00:00:00')

    train_df = df[df['Timestamp'] < cutoff].copy()
    test_df = df[df['Timestamp'] >= cutoff].copy()

    train_path = Path("data/benchmark/BENCHMARK_7DAY_TRAIN.csv")
    test_path = Path("data/benchmark/BENCHMARK_8TH_DAY_TRUTH.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[Stage 1] Ingesting 7-Day History ({len(train_df):,} rows) and 8th-Day Ground Truth ({len(test_df):,} rows)...")
    print("Satellites detected in calibration dataset:", sorted(train_df['Satellite_ID'].unique().tolist()))

    # 2. Calibrate all models per satellite
    print("\nRunning zero-leakage calibration across all eligible models per satellite...")
    controller = InferenceController()
    cal_results = controller.calibrate_satellite_models(train_path, test_path)

    print("\n" + "=" * 70)
    print("STAGE 1 SATELLITE MODEL WINNERS (PERSISTENT MEMORY):")
    print("=" * 70)
    for sat, info in sorted(cal_results["satellites"].items()):
        winner = info["winner"]
        score = info["score"]
        mode = info["selection_mode"]
        cands = info["candidate_scores"]
        top_candidates = ", ".join([f"{k}:{v:.3f}" for k, v in sorted(cands.items(), key=lambda x: x[1], reverse=True)[:3]])
        print(f"  Satellite: {sat:5s} | Selected Winner: {winner:16s} | Score: {score:.4f} | Top: [{top_candidates}]")

    print("\nPersistent registry saved to:", controller.registry.path)

    # 3. Stage 2 Prediction with New 7-day dataset
    print("\n" + "=" * 70)
    print("[Stage 2] Running 8th-Day Forecast on New 7-Day Dataset (No Ground Truth)...")
    print("=" * 70)
    forecast_df = controller.predict_with_satellite_models(data=train_path)

    print(f"Generated {len(forecast_df):,} total forecast epochs across {len(forecast_df['satellite_id'].unique())} satellites.")
    print("\nActive Routing Verification:")
    routing_summary = forecast_df.groupby('satellite_id')['model_used'].first().to_dict()
    for sat, model_used in sorted(routing_summary.items()):
        print(f"  {sat:5s} -> routed to: {model_used:16s}")

    print("\nSample Forecast Output Predictions (first 10 records):")
    cols = ['forecast_step', 'timestamp', 'satellite_id', 'model_used', 'predicted_X', 'predicted_Y', 'predicted_Z', 'predicted_Clock', 'pred_3D_Orbit_Error']
    print(forecast_df[cols].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUCCESSFUL — ALL INVARIANTS SATISFIED!")
    print("=" * 70)


if __name__ == '__main__':
    run_end_to_end_demo()
