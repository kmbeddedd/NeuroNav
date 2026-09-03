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

    # Command: gui
    gui_parser = subparsers.add_parser("gui", help="Launch the Desktop GUI or headless demo")
    gui_parser.add_argument("--cli", action="store_true", help="Run in terminal headless demo mode")
    gui_parser.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    gui_parser.add_argument("--data", default="data/sample/sample_gnss_data.csv")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Run multihorizon forecast inference")
    pred_parser.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
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

    # If legacy syntax: python main.py --model bilstm
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Legacy fallback compatibility
        legacy_parser = argparse.ArgumentParser()
        legacy_parser.add_argument("--model", default="bilstm")
        legacy_args, rem = legacy_parser.parse_known_args()
        sys.argv = [sys.argv[0], "gui", "--cli", "--model", legacy_args.model]

    args, remaining = parser.parse_known_args()

    if args.command == "gui":
        from app.main import launch_gui, run_headless_demo
        from app.controllers.inference_controller import InferenceController
        controller = InferenceController()
        if args.cli or not sys.stdin.isatty():
            run_headless_demo(controller, args.model, args.data)
        else:
            try:
                launch_gui(controller)
            except Exception:
                run_headless_demo(controller, args.model, args.data)

    elif args.command == "predict":
        from neuronav.inference import NeuroNavModel
        model = NeuroNavModel.load(args.model)
        df = model.predict(args.data, satellite_id=args.satellite)
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_p, index=False)
            print(f"Forecast saved to {out_p}")
        else:
            print(df.head(20).to_string())

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
