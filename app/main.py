"""NeuroNav GNSS Orbit & Clock Error Forecasting Desktop Application Launcher."""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.controllers.inference_controller import InferenceController


def run_headless_demo(controller: InferenceController, model_name: str, data_path: str):
    """Run headless forecast demo displaying forecast tables in terminal."""
    print("=" * 70)
    print("NeuroNav GNSS Orbit & Clock Error Forecasting System")
    print("=" * 70)

    print(f"\n[1] Loading model: {model_name}...")
    model = controller.load_model(model_name)
    print(f"    Loaded: {model.model_name} ({model.model_type})")
    print(f"    Sequence Length: {model.seq_len} epochs (15-min cadence)")
    print(f"    Forecast Horizon: {model.forecast_horizon} epochs (24 hours)")
    print(f"    Uncertainty Supported: {model.uncertainty_supported}")

    print(f"\n[2] Loading dataset: {data_path}...")
    rows, sats = controller.load_dataset(data_path)
    print(f"    Total Records: {rows:,}")
    print(f"    Available Satellites: {sats}")

    target_sat = sats[0] if sats else None
    print(f"\n[3] Running inference for satellite: {target_sat}...")
    forecast_df = controller.run_forecast(satellite_id=target_sat)

    print(f"\n[4] Forecast Results (Showing first 10 steps of 24h horizon):")
    display_cols = [
        'forecast_step', 'forecast_time', 'Satellite_ID',
        'pred_Error_X', 'pred_Error_Y', 'pred_Error_Z', 'pred_Error_Clock',
        'pred_3D_Orbit_Error'
    ]
    avail_display = [c for c in display_cols if c in forecast_df.columns]
    print(forecast_df[avail_display].head(10).to_string(index=False))

    print(f"\n[5] Forecast Statistics:")
    print(f"    Mean 3D Orbit Error: {forecast_df['pred_3D_Orbit_Error'].mean():.4f} m")
    print(f"    Max 3D Orbit Error:  {forecast_df['pred_3D_Orbit_Error'].max():.4f} m")
    print(f"    Min 3D Orbit Error:  {forecast_df['pred_3D_Orbit_Error'].min():.4f} m")
    print("=" * 70)


def launch_gui(controller: Optional[InferenceController] = None):
    """Launch full 3-page NeuroNav Desktop GUI."""
    try:
        from gui.gui_app import NeuroNavApp
        app = NeuroNavApp()
        app.mainloop()
    except Exception as e:
        print(f"Tkinter GUI could not launch ({e}). Falling back to headless demo.")
        if controller is None:
            controller = InferenceController()
        run_headless_demo(controller, 'bilstm', 'data/sample/sample_gnss_data.csv')


def main():
    parser = argparse.ArgumentParser(description="NeuroNav Desktop Application")
    parser.add_argument("--cli", action="store_true", help="Run in terminal headless demo mode")
    parser.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    parser.add_argument("--data", default="data/sample/sample_gnss_data.csv")
    args = parser.parse_args()

    controller = InferenceController()

    if args.cli or os.environ.get("HEADLESS") == "1":
        run_headless_demo(controller, args.model, args.data)
    else:
        try:
            launch_gui(controller)
        except Exception:
            run_headless_demo(controller, args.model, args.data)


if __name__ == "__main__":
    main()
