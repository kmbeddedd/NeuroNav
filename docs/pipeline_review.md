# Pipeline review — 2026-09-05

Scope: repository source inventory and targeted review of benchmark exports, CLI
dispatch, public API, validation, registry/router, calibration, model prediction
interfaces, legacy training/evaluation scripts, and their regression suite. This
is not a claim that every model algorithm has been independently verified.

## Fixed

- Benchmark exports now include one CSV per held-out test filename, preserving
  model identity, actuals, predictions, residuals, and available diagnostics.
  The aggregate CSV remains compatible with existing consumers.
- Root benchmark CLI now forwards epoch and device settings. Unknown arguments
  raise an error instead of silently running with defaults.
- Legacy CLI translation retains additional arguments, including the data path.
- Router type annotations now import `Tuple`, allowing runtime type inspection.

## Findings requiring follow-up

1. **High: evaluation label leakage.** `src/forecasting/api.py:evaluate_satellite`
   calls `model.predict(test_df, test_times)`. History-dependent models can consume
   the held-out labels. Persistence, for example, anchors on the last test row.
   Add an explicit historical dataset parameter and enforce history ending before
   the test interval. Recompute affected evaluation reports. The calibration
   pipeline instead passes `sat_train` to prediction and does not share this call-site bug.
2. **Medium: stale model cache.** `PredictionRouter.get_assigned_model` caches by
   satellite, model name, and version only. Replacing an artifact or physics
   configuration without changing the version can retain an old model/provider.
   Use immutable artifact versions or invalidate caches on registry changes.
3. **Medium: validated history is discarded by the inference CLI.** It validates
   through the public reporting endpoint, then predicts from the original CSV.
   Any normalization/deduplication performed during validation is not passed as
   the validated dataset. Share the validated `SatelliteDataset` with prediction.
4. **Medium: fractional cadence is truncated.** Single-satellite routing converts
   registered cadence to `int`, whereas batch routing retains fractional minutes.
   Define one cadence policy and test fractional-minute forecasts across both APIs.
5. **Low: comparison dispatcher requires unused data.** `main.py evaluate compare`
   requires `--data` but calls the report viewer with no arguments. Give this
   command report-specific options; it does not evaluate an input CSV.
6. **Low: tuning writes to the repository root.** `scripts/train/tune.py` still
   defaults to `tuning_results.json` despite the documented reports convention.
7. **Environment:** OrbitIQ's stored scaler was built with scikit-learn 1.7.1,
   while the local environment uses 1.9.0. The test suite reports a version warning.
   Validate or rebuild that research artifact with a pinned environment.

These findings are recorded without changing model/evaluation semantics or trained
artifacts in this export-focused change. The GUI still uses the legacy backend,
as documented in the main README.

## Verification

Regression tests cover per-input export isolation and equality with the combined
file, retained diagnostics, CLI option forwarding, and rejection of unknown options.
The full pytest suite is run for integration coverage. Full neural benchmark
retraining is not needed to verify the export change and was not performed.
