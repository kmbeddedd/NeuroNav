# NeuroNav Project Context

## Context maintenance policy

- Update this file after every project-related user prompt and every repository edit.
- Record new instructions, decisions, implementation changes, commands, results, validation status, and unresolved issues.
- Keep entries concise and factual so this file remains useful as a handoff document.
- Never include secrets, credentials, private keys, or unnecessarily sensitive data.
- If a new instruction supersedes an older one, mark the older guidance as superseded instead of leaving contradictory directions.

### Latest instruction

On 2026-09-04, the user approved the complete Phase 1 structural audit and requested
execution of all approved refactoring plus the README update. The implementation:

1. Established `src.forecasting` as the production API and exported all required
   single-satellite functions from the package facade.
2. Split production responsibilities into `data/`, `features/`, `evaluation/`,
   `training/`, `registry/`, and `inference/` modules while preserving old import paths.
3. Isolated the inherited global pipeline under `src/compat/global_forecasting/`; the
   unchanged Tkinter frontend continues to use compatibility imports.
4. Consolidated the supplied CSVs under `data/ps08/` and competition references under
   `docs/reference/ps08/`; removed exact duplicates from `research/ps08/data/`.
5. Moved benchmark, OrbitIQ, ablation, operational evaluation, research tests, and
   lineage tooling into purpose-specific locations with compatibility CLI wrappers.
6. Corrected registry artifact paths from the nonexistent `artifacts_test` directory to
   `models/registry/artifacts/` without changing selected models.
7. Added explicit PS-08 input and satellite prediction contracts, Git LFS attributes for
   model binaries, and a production-oriented README.
8. Verification passed: 133 tests, public/compatibility import smoke tests, and routed
   two-step forecasts for GEO, MEO-1, and MEO-2.

### Previous instruction (N-HiTS Day-8 exports)

On 2026-09-04, the user instructed to generate predicted 8th day values in the form of CSV files for datasets MEO train (`DATA_MEO_Train.csv`), MEO Train2 (`DATA_MEO_Train2.csv`), and GEO (`DATA_GEO_Train.csv`) using `nhits`:
1. Generated production exports in root directory and `results/nhits_day8_predictions/`:
   - `predictions_GEO_nhits.csv` (69 rows matching `DATA_GEO_Test.csv` with predictions, actuals, residuals, 3D orbit error)
   - `predictions_MEO_train_nhits.csv` (11 rows matching `DATA_MEO_Test.csv` row-by-row)
   - `predictions_MEO_Train2_nhits.csv` (30 rows matching `DATA_MEO_Test2.csv` row-by-row)
   - Submission-format clean 5-column CSVs: `DATA_GEO_Test_nhits_predicted.csv`, `DATA_MEO_Test_nhits_predicted.csv`, `DATA_MEO_Train2_Test_nhits_predicted.csv`
   - Deduplicated unique-epoch CSVs: `predictions_GEO_nhits_unique_epochs.csv` (69 rows), `predictions_MEO_train_nhits_unique_epochs.csv` (6 rows), `predictions_MEO_Train2_nhits_unique_epochs.csv` (18 rows)
   - Combined multi-satellite evaluation CSV: `all_satellites_nhits_day8_predictions.csv` (110 rows)
2. Verified evaluation metrics under the Official Competition Hierarchy:
   - **GEO (`GEO_nhits.pt`):** $W_{\text{avg}} = 0.889359$, 3D MAE = 23.5272 m, Clock MAE = 7.7601 m
   - **MEO train / MEO-1 (`MEO-1_nhits.pt`):** $W_{\text{avg}} = 0.942893$, 3D MAE = 1.0734 m, Clock MAE = 0.2201 m
   - **MEO Train2 / MEO-2 (`MEO-2_nhits.pt`):** $W_{\text{avg}} = 0.839277$, 3D MAE = 0.3406 m, Clock MAE = 0.0288 m
3. Persisted trained model checkpoints in `models/registry/artifacts/`: `GEO_nhits.pt`, `MEO-1_nhits.pt`, `MEO-2_nhits.pt`.
4. Automated generator: `scripts/evaluate/export_nhits_day8_predictions.py`.
5. Automated tests: Added `tests/test_nhits_predictions.py` (2 passed). Full test suite passes 100% (133/133 passed).

### Previous instruction (2026-09-04)

The satellite forecasting architecture was fully finished and integrated for single-satellite independent datasets and physics semantics:
1. Implemented single-satellite independent upload, validation, and training workflow (`train_satellite`, `train_single_satellite`, `run_satellite_validation`) primarily supporting real-world input without orbital state vectors (UTC + error measurements only).
2. Explicit `physics_mode` semantics implemented across features, models, registry, pipeline, router, and API:
   - `"none"`: Disables all RIC and SRP physics context completely; model trains on temporal and error features only.
   - `"nominal"`: Uses nominal solar radiation pressure / orbit approximation without requiring external coordinates.
   - `"provided"`: Accepts user-supplied ephemeris / orbit-state table (`state_df`), interpolates state vectors, persists `orbital_state.csv` into `models/registry/artifacts/satellites/<sat_id>/`, records `state_artifact` in the registry, and reloads it on operational inference.
3. Feature engineering parity in `src/forecasting/features.py`: Added instantaneous unit basis projections ($\hat{r}_z, \hat{i}_z, \hat{c}_z$) and `dt_seconds` time-gap features with zero target error leakage across training and inference.
4. Cadence classification in `SamplingMetadata`: Added `mean_cadence_minutes`, `cadence_std_minutes`, `duration_hours`, and `cadence_classification` (`"regular" | "mildly_irregular" | "strongly_irregular"`). Default `resample_if_irregular=False`.
5. Model Eligibility and Capabilities: Added `ModelCapabilities` with `supports_irregular_timestamps`, `requires_regular_cadence`, `supports_nominal_physics`, and `supports_provided_state`.
6. Official Competition Hierarchy candidate ranking: Implemented `rank_candidates_hierarchically` with `functools.cmp_to_key` and tie tolerance ($\tau = 10^{-4}$), strictly respecting Priority 1 ($W_{\text{avg}}$), Priority 2 (aggregate bias and std), and Priority 3 (Q-Q Blom outliers).
7. Strict fail-closed routing in `router.py`: Verifies explicitly recorded `selection.model_artifact` directly and raises `ModelArtifactError` immediately if missing, preventing silent fallbacks to leftover sibling files.
8. Enhanced `PersistenceModel.__init__` with `**kwargs` for clean parameter propagation from model factory.
9. Added `get_satellite_summary(satellite_id)` for frontend-ready provenance and configuration.
10. Added comprehensive regression tests in `tests/test_official_evaluation.py` (10 passed) and `tests/test_satellite_upload_pipeline.py` (12 passed). Full test suite passes 100% (131/131 passed).

### Governing model-selection instruction

Model selection must be driven strictly by the **official competition evaluation hierarchy**:
1. **Priority 1 (Primary Decision):** Shapiro-Wilk $W_{\text{avg}} = (W_X + W_Y + W_Z + W_{\text{Clock}}) / 4$ with exactly equal weight (25%) per parameter at significance level $\alpha = 0.05$. Higher $W_{\text{avg}}$ wins. Retains per-parameter $W$, $p$-values, hypothesis results $H_0 \in \{0, 1\}$ ($0 = \text{fail to reject normality}, 1 = \text{reject normality}$), and sample size $N$.
2. **Priority 2 (First Tie-Breaker):** Invoked strictly when Priority 1 is tied within tolerance $\tau = 10^{-4}$. Compares equal-weighted aggregate residual bias $|\mu| = \frac{1}{4} \sum_p |\mu_p|$ and aggregate standard deviation $\sigma = \frac{1}{4} \sum_p \sigma_p$. Lower aggregate error wins.
3. **Priority 3 (Second Tie-Breaker):** Invoked strictly when Priority 1 and Priority 2 remain tied within tolerance. Compares Q-Q plot outlier counts ($|z| > 3.0$ or $|\Delta_i| > 1.0$) and maximum quantile discrepancy. Fewer outliers wins.
4. **Supplementary Metrics (Diagnostics Only):** MAE, RMSE, 3D Orbit Error, and SISRE are computed strictly as supplementary diagnostics and **never** drive the official selection decision.

The backend was also required to support a satellite-specific calibration and operational inference router with persistent atomic registry updates and **zero silent fallback to BiLSTM**. Frontend/GUI modifications were strictly forbidden.

### Instruction history

- On 2026-09-04, the user requested that `context.md` be continuously updated with every subsequent prompt or edit.
- On 2026-09-04, the user requested removal of the SP3 and RINEX/RNX ingestion layer and all related operational code, configuration, documentation, tests, and historical ingestion notes.
- On 2026-09-04, the user requested an official calibration run across GEO and MEO satellites under the official evaluation hierarchy.
- On 2026-09-04, the user requested a production-readiness evaluation of the calibrated satellite-specific pipeline.
- On 2026-09-04, the user requested genuine physics layer integration (SRP features, OrbitalStateProvider, corrected GEO/MEO physical semantics, and E5 ablation fix).

## Objective

This repository forecasts GNSS satellite orbit and clock errors from the supplied PS-08 datasets. The core architectural invariant is:
> **The best forecasting model is a property of an individual satellite, not a global property of the dataset.**

The backend consists of:
- **Phase A (Calibration):** Evaluates candidate models out-of-sample on Day 8, selects satellite-specific winners under the official competition hierarchy, serializes model weights to `models/registry/artifacts/`, and atomically records provenance in `models/registry/satellite_model_registry.json`.
- **Phase B (Operational Forecast Router):** Routes live multi-satellite telemetry (without ground truth) to winning models, rotates spatial errors into Radial-Along-Cross (RIC) frame, and returns standardized forecasts with full provenance.
- **Fail-Closed Router Policy:** Unregistered satellites or corrupted artifacts trigger explicit, structured exceptions (`NoModelSelectionError`, `ModelArtifactError`) rather than falling back to default estimators.

## Non-negotiable data constraint

- Use only the CSV datasets supplied in `data/`.
- Do not generate artificial, interpolated, repeated, or physics-simulated training observations.
- Exact duplicate timestamps may be removed deterministically before training and evaluation.
- Day-8 test observations must never be used as model inputs during training or calibration.

The supplied series are:
- `DATA_GEO_Train.csv` (142 rows) and `DATA_GEO_Test.csv` (69 rows)
- `DATA_MEO_Train.csv` (90 rows) and `DATA_MEO_Test.csv` (11 rows)
- `DATA_MEO_Train2.csv` (244 rows) and `DATA_MEO_Test2.csv` (30 rows)

## Removed acquisition and ingestion surface

The repository no longer supports downloading or processing SP3, RINEX/RNX, CLK, BRDC, or IGS/MGEX products. The following components were removed:
- `scripts/data/fetch_igs_data.py`
- `scripts/data/process_gnss_errors.py`
- `scripts/data/generate_clean_dataset.py`
- `docs/data_acquisition.md`
- SP3-specific validity masking, audit metrics, configuration fields, and tests

The remaining `scripts/data/audit_data.py` utility performs source-agnostic CSV validation. Do not reintroduce an external-product acquisition layer unless the user explicitly reverses this instruction.

## Current training and evaluation workflow

The primary entry points are:
- **Python API:** `src/forecasting/api.py` (`calibrate_models`, `predict_with_satellite_models`, `get_calibration_summary`, `get_model_comparison`, `get_detailed_statistical_results`, `get_qq_data`)
- **Calibration Engine:** `src/forecasting/pipeline.py` (`CalibrationPipeline`, `evaluate_residuals_official_hierarchy`, `compare_models_hierarchical`)
- **Inference Router:** `src/forecasting/router.py` (`PredictionRouter`)
- **Benchmark Script (Historical PS-08):** `scripts/benchmark/benchmark_ps08.py`

Run official calibration with:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -c "from src.forecasting.api import calibrate_models; calibrate_models('data', 'data', run_id='official_competition_run')"
```

Run multi-satellite inference with:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -c "from src.forecasting.api import predict_with_satellite_models; df = predict_with_satellite_models('data'); print(df[['satellite_id', 'model_used']].drop_duplicates())"
```

## Official Competition Calibration Results (`run_id="official_competition_run"`)

Winning models selected strictly by Priority 1 (Shapiro-Wilk $W_{\text{avg}}$ across $X, Y, Z, \text{Clock}$):

| Satellite | Winning Model | Selection Metric | Average $W$ ($W_{\text{avg}}$) | Priority 1 Status | Aggregate Bias $|\mu|$ (m) | Aggregate Std $\sigma$ (m) | Q-Q Outliers | Supplementary 3D MAE (m) | Supplementary SISRE Mean (m) |
|---|---|---|---|---|---|---|---|---|---|
| **GEO** | **`nhits`** | `shapiro_w_avg` | **0.889359** | Won on P1 | 1.2513 | 16.7739 | 10 | 23.5272 | 14.7248 |
| **MEO-1** | **`geo_moe`** | `shapiro_w_avg` | **0.917693** | Won on P1 | 0.3124 | 0.2966 | 0 | 0.9653 | 0.3564 |
| **MEO-2** | **`random_forest`** | `shapiro_w_avg` | **0.867010** | Won on P1 | 0.0420 | 0.1704 | 1 | 0.3266 | 0.1733 |

### Granular Priority 1 Breakdown

- **GEO (`nhits`):**
  - $W$: $X = 0.9637$, $Y = 0.9496$, $Z = 0.9651$, $\text{Clock} = 0.6790$ ($W_{\text{avg}} = 0.889359$)
  - $p$-values: $p_X = 0.0425$, $p_Y = 0.0075$, $p_Z = 0.0509$, $p_{\text{Clock}} = 5.33 \times 10^{-11}$
  - $H_0$ ($\alpha=0.05$): $X=1$, $Y=1$, **$Z=0$ (fail to reject normality)**, $\text{Clock}=1$ (total rejected: 3)
  - Sample size: $N = 69$
- **MEO-1 (`geo_moe`):**
  - $W$: $X = 0.9327$, $Y = 0.9387$, $Z = 0.8792$, $\text{Clock} = 0.9202$ ($W_{\text{avg}} = 0.917693$)
  - $p$-values: $p_X = 0.6010$, $p_Y = 0.6485$, $p_Z = 0.2655$, $p_{\text{Clock}} = 0.5067$
  - $H_0$ ($\alpha=0.05$): **$X=0$, $Y=0$, $Z=0$, $\text{Clock}=0$ (All 4 components fail to reject normality! Total rejected: 0)**
  - Effective sample size after duplicate-epoch removal: $N = 6$, Q-Q outliers: 0
- **MEO-2 (`random_forest`):**
  - $W$: $X = 0.9263$, $Y = 0.9749$, $Z = 0.9006$, $\text{Clock} = 0.6662$ ($W_{\text{avg}} = 0.867010$)
  - $p$-values: $p_X = 0.1674$, $p_Y = 0.8824$, $p_Z = 0.0589$, $p_{\text{Clock}} = 3.32 \times 10^{-5}$
  - $H_0$ ($\alpha=0.05$): **$X=0$, $Y=0$, $Z=0$ (fail to reject normality)**, $\text{Clock}=1$ (total rejected: 1)
  - Effective sample size after duplicate-epoch removal: $N = 18$, Q-Q outliers: 1

### Metric Duality: Accuracy vs. Normality
While Gaussian Process models achieve the lowest Euclidean MAE (16.43 m on GEO, 0.31 m on MEO-1, 0.23 m on MEO-2), their residuals deviate significantly from Gaussian distributions ($W_{\text{avg}} \approx 0.78-0.88$). Under the official competition rules, Shapiro-Wilk $W_{\text{avg}}$ strictly governs selection, promoting `nhits` on GEO, `geo_moe` on MEO-1, and `random_forest` on MEO-2.

## Active Artifacts & Manifests

- **Model Registry:**
  `models/registry/satellite_model_registry.json`
- **Trained Model Checkpoints:**
  - `models/registry/artifacts/GEO_nhits.pt`
  - `models/registry/artifacts/MEO-1_geo_moe.pt`
  - `models/registry/artifacts/MEO-2_random_forest.joblib`
- **Calibration Reports (`reports/calibration/official_competition_run/`):**
  - `summary.json`: High-level run metrics and winners
  - `model_comparison.csv`: All candidate models ranked per satellite with all hierarchy metrics
  - `detailed_statistical_results.csv`: Granular per-target breakdown ($W, p, H_0, \mu, \sigma, N$)
  - `qq_data/<sat>_<model>_qq.json`: Complete Blom quantiles, discrepancies, and outliers
  - `eligibility.json` & `configuration.json`

## Latest evaluation request and result (2026-09-04)

The user requested a thorough production-readiness evaluation covering leakage, metrics, baselines, slices, errors, robustness, generalization, and reproducibility. The reproducible audit is at `reports/evaluation/registered_models_20260904/EVALUATION_REPORT.md`, with machine-readable results in `evaluation.json` and the evaluator in `tools/evaluate_current_pipeline.py`.

Verdict: **Resolved and Modularized**. The previously identified architectural gaps have been addressed:
- Stale dependencies on artificial files (e.g. `data/sample/sample_gnss_data.csv`) were replaced with clean in-memory test fixtures in `tests/test_inference.py`.
- Single-satellite independent dataset upload and validation pipeline added in `src/forecasting/validation.py`.
- Cadence irregularity detection, sampling metadata, and strict causal regularization implemented.
- Authoritative feature engineering pipeline in `src/forecasting/features.py` prevents target leakage across train/inference.
- Physics layer enhanced with `ProvidedStateProvider`, dynamic RIC frame conversion, and SRP solar geometry features.
- Model registry organized with nested per-satellite directories (`models/registry/artifacts/satellites/<sat_id>/`).
- Public API expanded and `get_calibration_report()` made fully functional.

## Backend Verification Status

- Automated test suite: **All 131 tests pass cleanly in ~30 seconds**.
  - `tests/test_satellite_upload_pipeline.py` (12 tests): Covers single-satellite CSV upload, header normalization, filename fallback, cadence validation, `ProvidedStateProvider` interpolation, RIC/SRP feature dimension parity, official hierarchy winner selection, end-to-end public API workflow, and `physics_mode` (`"none"`, `"nominal"`, `"provided"`, error guards).
  - `tests/test_official_evaluation.py` (10 tests): Covers equal 25% target weighting, Priority 1 dominance, Priority 2/3 tie-breakers, Q-Q Blom quantiles, Section 11 tie tolerance ($\tau = 10^{-4}$), input validation ($N \ge 3$), and fail-closed zero-fallback routing.
  - `tests/test_physics_integration.py` (12 tests): Covers dynamic cadence calculation, RIC coordinate conversion, SRP shadow factor, causal window construction, and model training with physics features.
  - `tests/test_satellite_backend.py` (16 tests): Covers RIC rotation, SISRE calculation, SRP shadow factor, atomic registry persistence, manual overrides, corrupt registry recovery, decoupled clock, and N-HiTS.
  - `tests/test_geo_regime_aware.py` (20 tests): Covers causal temporal history, residual reconstruction, Gated MoE gradient flow, and backtest invariance.
  - `tests/test_evaluate.py` (14 tests): Covers official evaluation metrics and comparisons.
  - `tests/test_model_integrity.py` (11 tests): Covers parameter serialization and reproducibility.
  - `tests/test_data_pipeline.py` (9 tests): Covers normalization, dataset schemas, and splits.
  - `tests/test_inference.py` (8 tests): Covers neural and linear inference contracts, uncertainty quantification, and input validation.
  - `tests/test_baselines.py` (7 tests): Covers persistence, harmonic ridge, random forest baselines.
  - `tests/test_orbitiq_eval.py` (4 tests): Covers OrbitIQ regression benchmarking.
  - `tests/test_physics.py` (3 tests): Covers nominal orbit state provider, coordinate transformations, and solar radiation geometry.
  - `tests/test_ps08_benchmark.py` (3 tests): Covers dataset normalization, deduplication, and orbit specialist routing.
  - `tests/test_calibration.py` (1 test): Covers calibration pipeline lifecycle.
  - `tests/test_data_audit.py` (1 test): Covers source-agnostic CSV validation.

Run the test suite with:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m pytest -v
```

## Single-Satellite Dataset & Physics Architecture (2026-09-04)

1. **Independent Upload & Validation (`src/forecasting/validation.py`)**:
   - Accepts individual CSV/DataFrame sources per satellite without requiring a `satellite_id` column.
   - Infers `satellite_id` and `orbit_type` from explicit parameters or filename stems (e.g. `DATA_MEO_Train.csv` -> `MEO-1`, `DATA_GEO_Train.csv` -> `GEO`).
   - Analyzes time intervals, computes `SamplingMetadata` (mean cadence, std, is_regular, min/max gap), and regularizes irregular cadences causally without future information leakage.
   - Encapsulates validated data in a typed `SatelliteDataset` dataclass.

2. **Causal Feature Engineering (`src/forecasting/features.py`)**:
   - Implements `FeatureManifest` tracking exact feature columns, order, normalization stats, and physics configuration (`use_ric`, `use_srp`).
   - Strictly enforces identical feature schemas between calibration and inference routing.
   - Implements `build_training_features()` and `build_inference_features()`, injecting time harmonics, lag windows, rolling stats, RIC orbital residuals, and SRP shadow factors.

3. **Physics Layer Integration (`src/physics.py`)**:
   - `ProvidedStateProvider(OrbitalStateProvider)`: Enables ephemeris interpolation from user-provided broadcast/nominal states, bypassing synthetic circular approximations when ground truth is supplied.
   - `build_ric_features()`: Transforms Cartesian ECEF error vectors into Radial, In-Track, and Cross-Track frame residuals.
   - `extract_solar_features()`: Calculates Sun-spacecraft vector, beta angle, and cylindrical/conical shadow factors.

4. **Dedicated Per-Satellite Artifact Layout**:
   - Directory: `models/registry/artifacts/satellites/<satellite_id>/`
   - Files:
     - `model.<ext>`: Model weights (`.pt` for neural / PyTorch models, `.joblib` for scikit-learn models).
     - `metadata.json`: Complete model metadata (training timestamp, features, orbit type, hyperparams).
     - `feature_manifest.json`: Ordered feature schema and physics flags.
     - `evaluation.json`: Official three-tier competition evaluation report.

5. **Clean Public API Layer (`src/forecasting/api.py`)**:
   - `validate_satellite_dataset(source, satellite_id, orbit_type, ...)`
   - `train_satellite(dataset, test_dataset, candidate_models, use_ric, use_srp, ...)`
   - `evaluate_satellite(satellite_id, test_dataset)`
   - `predict_satellite(satellite_id, history_data, horizon_steps, compute_ric, ...)`
   - `get_satellite_model(satellite_id)`
   - `get_satellite_metadata(satellite_id)`
   - `get_calibration_report(run_id, ...)`

## Working-tree guidance

- GUI files in `app/` are managed by a separate developer and must not be modified.
- Atomic registry updates use write-and-replace (`os.replace`) to prevent corruption.
- In manual mode (`selection_mode="manual"`), model assignments are pinned and protected from automated calibration overwrite until explicitly reset to automatic mode.
- Fail-closed router policy: if a model artifact is missing or an unknown satellite is requested, raise explicit exceptions (`NoModelSelectionError`, `ModelArtifactError`) with zero silent fallbacks.
