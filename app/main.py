"""NeuroNav GNSS Orbit & Clock Error Forecasting Desktop Application Launcher."""
import argparse
import os
import sys
from pathlib import Path

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


def launch_gui(controller: InferenceController):
    """Launch Tkinter Desktop GUI."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
    except ImportError:
        print("Tkinter is not available. Falling back to headless demo.")
        run_headless_demo(controller, 'bilstm', 'data/sample/sample_gnss_data.csv')
        return

    root = tk.Tk()
    root.title("NeuroNav — GNSS Satellite Orbit & Clock Error Forecaster")
    root.geometry("900x600")

    # Header
    header_frame = ttk.Frame(root, padding="10")
    header_frame.pack(fill=tk.X)
    title_label = ttk.Label(header_frame, text="NeuroNav GNSS Forecasting", font=("Helvetica", 16, "bold"))
    title_label.pack(side=tk.LEFT)

    # Main content frame
    content = ttk.Frame(root, padding="10")
    content.pack(fill=tk.BOTH, expand=True)

    # Controls Frame
    ctrl_frame = ttk.LabelFrame(content, text="Forecast Configuration", padding="10")
    ctrl_frame.pack(fill=tk.X, pady=5)

    # Model selector
    ttk.Label(ctrl_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_var = tk.StringVar(value="bilstm")
    model_combo = ttk.Combobox(ctrl_frame, textvariable=model_var, values=["bilstm", "transformer"], state="readonly")
    model_combo.grid(row=0, column=1, padx=5, pady=5)

    # Dataset selector
    ttk.Label(ctrl_frame, text="Dataset:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
    data_var = tk.StringVar(value="data/sample/sample_gnss_data.csv")
    data_entry = ttk.Entry(ctrl_frame, textvariable=data_var, width=35)
    data_entry.grid(row=0, column=3, padx=5, pady=5)

    # Run button
    status_var = tk.StringVar(value="Ready. Select model and dataset to run forecast.")

    def on_run():
        try:
            status_var.set(f"Loading {model_var.get()} and predicting...")
            root.update()
            controller.load_model(model_var.get())
            controller.load_dataset(data_var.get())
            df = controller.run_forecast()

            # Populate table
            for row in tree.get_children():
                tree.delete(row)

            for _, r in df.head(50).iterrows():
                tree.insert("", tk.END, values=(
                    r.get('forecast_step', ''),
                    str(r.get('forecast_time', ''))[:19],
                    r.get('Satellite_ID', ''),
                    f"{r.get('pred_Error_X', 0.0):.4f}",
                    f"{r.get('pred_Error_Y', 0.0):.4f}",
                    f"{r.get('pred_Error_Z', 0.0):.4f}",
                    f"{r.get('pred_Error_Clock', 0.0):.6e}",
                    f"{r.get('pred_3D_Orbit_Error', 0.0):.4f}",
                ))
            status_var.set(f"Forecast complete: {len(df)} predictions generated.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            status_var.set(f"Error: {e}")

    run_btn = ttk.Button(ctrl_frame, text="Run Forecast", command=on_run)
    run_btn.grid(row=0, column=4, padx=10, pady=5)

    # Table View
    table_frame = ttk.LabelFrame(content, text="Forecast Output Table", padding="5")
    table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    cols = ("Step", "UTC Time", "Satellite", "Pred X (m)", "Pred Y (m)", "Pred Z (m)", "Pred Clock (s)", "3D Error (m)")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=15)
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor=tk.CENTER)
    tree.column("UTC Time", width=140)

    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Status bar
    status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, padding="4")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


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
