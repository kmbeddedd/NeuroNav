import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='GNSS Satellite Orbit & Clock Error Forecasting CLI', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python main.py --model audit --data data_acquisition/CLEAN_GNSS_BENCHMARK.csv --report ./results/local/data_quality_report.json --strict\n  python main.py --model bilstm --data data_acquisition/CLEAN_GNSS_BENCHMARK.csv --output ./results/local/bilstm\n  python main.py --model transformer --data data_acquisition/CLEAN_GNSS_BENCHMARK.csv --output ./results/local/transformer\n  python main.py --model baselines --data data_acquisition/CLEAN_GNSS_BENCHMARK.csv --output ./results/local/baseline_metrics.json\n  python main.py --model tune --data data_acquisition/CLEAN_GNSS_BENCHMARK.csv --n-trials 15\n        ')
    parser.add_argument('--model', choices=['bilstm', 'transformer', 'tune', 'audit', 'baselines', 'orbitiq', 'eval_orbitiq', 'ps08', 'compare'], default='bilstm', help='Pipeline: bilstm, transformer, tune, audit, baselines, orbitiq, eval_orbitiq, ps08, or compare')
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.model == 'bilstm':
        from scripts.train_bilstm import run_training
        run_training()
    elif args.model == 'transformer':
        from scripts.train_transformer import run_training
        run_training()
    elif args.model == 'tune':
        from scripts.tune import run_tuning
        run_tuning()
    elif args.model == 'audit':
        from scripts.audit_data import main as run_audit
        run_audit()
    elif args.model == 'baselines':
        from scripts.evaluate_baselines import main as run_baselines
        raise SystemExit(run_baselines())
    elif args.model == 'orbitiq':
        from scripts.train_orbitiq_pipeline import main as run_orbitiq
        run_orbitiq()
    elif args.model == 'eval_orbitiq':
        from scripts.evaluate_orbitiq import main as run_eval_orbitiq
        run_eval_orbitiq()
    elif args.model == 'ps08':
        from scripts.benchmark_ps08 import main as run_ps08
        run_ps08()
    elif args.model == 'compare':
        from scripts.model_comparison_window import main as run_comparison
        run_comparison()
if __name__ == '__main__':
    main()
