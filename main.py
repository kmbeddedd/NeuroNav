"""NeuroNav CLI and System Dispatcher."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="NeuroNav — Production GNSS Satellite Orbit & Clock Error Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  gui         Launch the desktop application (Tkinter GUI or headless demo)
  predict     Run production model inference via NeuroNavModel
  train       Train neural forecasting models (bilstm, transformer, orbitiq)
  evaluate    Evaluate models and baseline forecasters
  audit       Audit telemetry dataset against data contract
  benchmark   Run official PS-08 competition benchmark
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational command to execute")

    # Command: calibrate
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate models and select best per satellite")
    cal_parser.add_argument("--train", required=True, help="Path to 7-day historical dataset")
    cal_parser.add_argument("--test", required=True, help="Path to 8th-day ground truth dataset")
    cal_parser.add_argument("--satellite", default=None, help="Target satellite name / identifier (e.g. GEO-01)")
    cal_parser.add_argument("--report", default=None, help="Optional output dir for audit report")

    # Command: gui
    gui_parser = subparsers.add_parser("gui", help="Launch the Desktop GUI or headless demo")
    gui_parser.add_argument("--cli", action="store_true", help="Run in terminal headless demo mode")
    gui_parser.add_argument("--model", default="auto", help="Model choice: 'auto' (per-satellite memory) or specific model")
    gui_parser.add_argument("--data", default="data/sample/sample_gnss_data.csv")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Run multihorizon forecast inference")
    pred_parser.add_argument("--model", default="auto", help="Model choice: 'auto' (per-satellite memory) or specific model")
    pred_parser.add_argument("--data", default="data/sample/sample_gnss_data.csv")
    pred_parser.add_argument("--satellite", default=None, help="Optional satellite ID (e.g. G01)")
    pred_parser.add_argument("--output", default=None, help="Path to save forecast CSV")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train a neural forecaster")
    train_parser.add_argument("model", choices=["bilstm", "transformer", "orbitiq", "tune"])
    train_parser.add_argument("--data", default="data/benchmark/CLEAN_GNSS_BENCHMARK.csv")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=32)

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models and baselines")
    eval_parser.add_argument("target", choices=["baselines", "orbitiq", "compare"])
    eval_parser.add_argument("--data", default="data/benchmark/CLEAN_GNSS_BENCHMARK.csv")

    # Command: audit
    audit_parser = subparsers.add_parser("audit", help="Audit dataset against data contract")
    audit_parser.add_argument("--data", default="data/benchmark/CLEAN_GNSS_BENCHMARK.csv")
    audit_parser.add_argument("--report", default=None)
    audit_parser.add_argument("--strict", action="store_true")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run official PS-08 benchmark")
    bench_parser.add_argument("--data-dir", default="research/ps08/data")
    bench_parser.add_argument("--output", default="research/ps08/results")

    # If legacy syntax: python main.py --model auto
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Legacy fallback compatibility
        legacy_parser = argparse.ArgumentParser()
        legacy_parser.add_argument("--model", default="auto")
        legacy_args, rem = legacy_parser.parse_known_args()
        sys.argv = [sys.argv[0], "gui", "--cli", "--model", legacy_args.model]

    args, remaining = parser.parse_known_args()

    if args.command == "calibrate":
        from app.controllers.inference_controller import InferenceController
        controller = InferenceController()
        sat_msg = f" for '{args.satellite}'" if args.satellite else " for detected satellites"
        print(f"Calibrating all models{sat_msg} in {args.train} vs {args.test} (Zero Leakage)...")
        res = controller.calibrate_satellite_models(args.train, args.test, target_satellite_id=args.satellite)
        print("\n" + "=" * 65)
        print("SATELLITE MODEL SELECTION RESULTS")
        print("=" * 65)
        for sat, info in res.get("satellites", {}).items():
            print(f"Satellite: {sat:10s} | Best Model: {info['winner']:18s} | Score: {info['score']:.4f} | Mode: {info['selection_mode']}")
        print("=" * 65)
        if res.get("audit_reports"):
            print(f"Audit report saved to: {res['audit_reports'].get('summary_json', '')}")

    elif args.command == "gui" or args.command is None:
        from app.main import launch_gui, run_headless_demo
        from app.controllers.inference_controller import InferenceController
        controller = InferenceController()
        target_model = getattr(args, 'model', 'auto')
        target_data = getattr(args, 'data', 'data/sample/sample_gnss_data.csv')
        is_cli = getattr(args, 'cli', False) or os.environ.get("HEADLESS") == "1"
        if is_cli:
            run_headless_demo(controller, target_model, target_data)
        else:
            try:
                launch_gui(controller)
            except Exception:
                run_headless_demo(controller, target_model, target_data)

    elif args.command == "predict":
        from src.inference import NeuroNavModel
        model = NeuroNavModel.load(args.model)
        df = model.predict(args.data, satellite_id=args.satellite)
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_p, index=False)
            print(f"Forecast saved to {out_p}")
        else:
            display_cols = [c for c in ['forecast_step', 'utc_time', 'forecast_time', 'Satellite_ID', 'satellite_id', 'model_used', 'selection_mode', 'pred_Error_X', 'pred_Error_Y', 'pred_Error_Z', 'pred_Error_Clock', 'pred_3D_Orbit_Error'] if c in df.columns]
            print(df[display_cols].head(20).to_string(index=False))

    elif args.command == "train":
        if args.model == "bilstm":
            from scripts.train.bilstm import run_training
            run_training()
        elif args.model == "transformer":
            from scripts.train.transformer import run_training
            run_training()
        elif args.model == "orbitiq":
            from scripts.train.orbitiq import main as run_orbitiq
            run_orbitiq()
        elif args.model == "tune":
            from scripts.train.tune import run_tuning
            run_tuning()

    elif args.command == "evaluate":
        if args.target == "baselines":
            from scripts.evaluate.evaluate_baselines import main as run_baselines
            sys.exit(run_baselines())
        elif args.target == "orbitiq":
            from scripts.evaluate.evaluate_orbitiq import main as run_orbitiq
            run_orbitiq()
        elif args.target == "compare":
            from scripts.evaluate.compare_models import main as run_comparison
            run_comparison()

    elif args.command == "audit":
        from scripts.data.audit_data import main as run_audit
        sys.argv = [sys.argv[0]] + remaining
        run_audit()

    elif args.command == "benchmark":
        from scripts.benchmark.benchmark_ps08 import main as run_benchmark
        sys.argv = [sys.argv[0]] + remaining
        run_benchmark()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
