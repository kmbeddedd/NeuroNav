# NeuroNav Project Context

## Context maintenance policy

- Update this file after every project-related user prompt and every repository edit.
- Record new instructions, decisions, implementation changes, commands, results, validation status, and unresolved issues.
- Keep entries concise and factual so this file remains useful as a handoff document.
- Never include secrets, credentials, private keys, or unnecessarily sensitive data.
- If a new instruction supersedes an older one, mark the older guidance as superseded instead of leaving contradictory directions.

### Latest instruction

On 2026-09-04, the user instructed that model selection must be driven strictly by the **official competition evaluation hierarchy**:
1. **Priority 1 (Primary Decision):** Shapiro-Wilk $W_{\text{avg}} = (W_X + W_Y + W_Z + W_{\text{Clock}}) / 4$ with exactly equal weight (25%) per parameter at significance level $\alpha = 0.05$. Higher $W_{\text{avg}}$ wins. Retains per-parameter $W$, $p$-values, hypothesis results $H_0 \in \{0, 1\}$ ($0 = \text{fail to reject normality}, 1 = \text{reject normality}$), and sample size $N$.
2. **Priority 2 (First Tie-Breaker):** Invoked strictly when Priority 1 is tied within tolerance $\tau = 10^{-4}$. Compares equal-weighted aggregate residual bias $|\mu| = \frac{1}{4} \sum_p |\mu_p|$ and aggregate standard deviation $\sigma = \frac{1}{4} \sum_p \sigma_p$. Lower aggregate error wins.
3. **Priority 3 (Second Tie-Breaker):** Invoked strictly when Priority 1 and Priority 2 remain tied. Compares Q-Q plot outlier counts ($|z| > 3.0$ or $|\Delta_i| > 1.0$) and maximum quantile discrepancy. Fewer outliers wins.
4. **Supplementary Metrics (Diagnostics Only):** MAE, RMSE, 3D Orbit Error, and SISRE are computed strictly as supplementary diagnostics and **never** drive the official selection decision.

The backend was also required to support a satellite-specific calibration and operational inference router with persistent atomic registry updates and **zero silent fallback to BiLSTM**. Frontend/GUI modifications were strictly forbidden.

### Instruction history

- On 2026-09-04, the user requested that `context.md` be continuously updated with every subsequent prompt or edit.
- On 2026-09-04, the user requested removal of the SP3 and RINEX/RNX ingestion layer and all related operational code, configuration, documentation, tests, and historical ingestion notes.
- On 2026-09-04, the user requested an official calibration run across GEO and MEO satellites under the official evaluation hierarchy.

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
  - Sample size: $N = 11$, Q-Q outliers: 0
- **MEO-2 (`random_forest`):**
  - $W$: $X = 0.9263$, $Y = 0.9749$, $Z = 0.9006$, $\text{Clock} = 0.6662$ ($W_{\text{avg}} = 0.867010$)
  - $p$-values: $p_X = 0.1674$, $p_Y = 0.8824$, $p_Z = 0.0589$, $p_{\text{Clock}} = 3.32 \times 10^{-5}$
  - $H_0$ ($\alpha=0.05$): **$X=0$, $Y=0$, $Z=0$ (fail to reject normality)**, $\text{Clock}=1$ (total rejected: 1)
  - Sample size: $N = 30$, Q-Q outliers: 1

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

## Verification status

- Automated test suite: **All 48 tests pass in ~36 seconds**.
  - `tests/test_official_evaluation.py` (9 tests): Covers equal 25% target weighting, Priority 1 dominance, Priority 2/3 tie-breakers, Q-Q Blom quantiles, input validation ($N \ge 3$), and zero fallback policy.
  - `tests/test_satellite_backend.py` (16 tests): Covers RIC rotation, SISRE calculation, SRP shadow factor, atomic registry persistence, manual overrides, corrupt registry recovery, decoupled clock, and N-HiTS.
  - `tests/test_ps08_benchmark.py` (3 tests): Covers dataset normalization, deduplication, and orbit specialist routing.
  - `tests/test_geo_regime_aware.py` (20 tests): Covers causal temporal history, residual reconstruction, Gated MoE gradient flow, and backtest invariance.

Run the test suite with:
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m pytest tests/test_official_evaluation.py tests/test_satellite_backend.py tests/test_ps08_benchmark.py tests/test_geo_regime_aware.py -v
```

## Working-tree guidance

- Preserve unrelated user changes and existing result artifacts.
- Atomic registry updates use write-and-replace (`os.replace`) to prevent corruption.
- In manual mode (`selection_mode="manual"`), model assignments are pinned and protected from automated calibration overwrite until explicitly reset to automatic mode.
