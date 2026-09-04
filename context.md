# NeuroNav Project Context

## Context maintenance policy

- Update this file after every project-related user prompt and every repository edit.
- Record new instructions, decisions, implementation changes, commands, results, validation status, and unresolved issues.
- Keep entries concise and factual so this file remains useful as a handoff document.
- Never include secrets, credentials, private keys, or unnecessarily sensitive data.
- If a new instruction supersedes an older one, mark the older guidance as superseded instead of leaving contradictory directions.

### Latest instruction

On 2026-09-04, the user requested removal of the SP3 and RINEX/RNX ingestion layer and all related operational code, configuration, documentation, tests, and historical ingestion notes.

### Instruction history

- On 2026-09-04, the user requested that `context.md` be continuously updated with every subsequent prompt or edit.

## Objective

This repository forecasts GNSS satellite orbit and clock errors from the supplied PS-08 datasets. The immediate objective is accurate Day-8 prediction for GEO and MEO series while preserving a leakage-safe evaluation boundary.

## Non-negotiable data constraint

- Use only the CSV datasets supplied in `data/`.
- Do not generate artificial, interpolated, repeated, or physics-simulated training observations.
- Exact duplicate timestamps may be removed deterministically before training and evaluation.
- Day-8 test observations must never be used as model inputs or for model selection.

The supplied series are:

- `DATA_GEO_Train.csv` and `DATA_GEO_Test.csv`
- `DATA_MEO_Train.csv` and `DATA_MEO_Test.csv`
- `DATA_MEO_Train2.csv` and `DATA_MEO_Test2.csv`

The synthetic OrbitIQ benchmark generator and its dependent training pipeline were removed. Do not restore or recreate them.

## Removed acquisition and ingestion surface

The repository no longer supports downloading or processing SP3, RINEX/RNX, CLK, BRDC, or IGS/MGEX products. The following components were removed:

- `scripts/data/fetch_igs_data.py`
- `scripts/data/process_gnss_errors.py`
- `scripts/data/generate_clean_dataset.py`
- `docs/data_acquisition.md`
- SP3-specific validity masking, audit metrics, configuration fields, and tests
- Historical ingestion audit and model-review notes tied to that pipeline

The remaining `scripts/data/audit_data.py` utility performs source-agnostic CSV validation. Target availability is based on finite values rather than source-format-specific sentinels. Do not reintroduce an external-product acquisition layer unless the user explicitly reverses this instruction.

## Current training and evaluation workflow

The direct PS-08 workflow is implemented in:

```text
scripts/benchmark/benchmark_ps08.py
```

Run the refined benchmark with:

```powershell
.\.venv\Scripts\python.exe -u scripts\benchmark\benchmark_ps08.py `
  --data-dir data `
  --output results\ps08_refined_meo_20260904 `
  --max-epochs 180 `
  --device cuda
```

The workflow trains and evaluates persistence, harmonic ridge, random forest, Gaussian process, BiLSTM-GRU, Transformer, GEO Gated MoE, and the refined Orbit-Class Specialist.

## Refined model strategy

The Orbit-Class Specialist preserves the existing GEO path and improves MEO accuracy through target-specific routing:

| Series | Orbit predictor | Clock predictor |
|---|---|---|
| GEO | GEO Gated MoE | GEO Gated MoE |
| MEO-1 | Gaussian Process | Random Forest |
| MEO-2 | Gaussian Process | Gaussian Process |

MEO specialists are selected using rolling-origin validation on training data only. Day-8 labels are not used for selection.

## Latest MEO results

Compared with applying the GEO Gated MoE to every orbit class:

| Metric | Previous | Refined | Relative improvement |
|---|---:|---:|---:|
| Combined MEO 3D vector MAE | 0.461179 m | 0.253316 m | 45.1% |
| Combined MEO clock MAE | 0.060820 m | 0.052957 m | 12.9% |
| MEO-1 3D vector MAE | 0.894132 m | 0.309880 m | 65.3% |
| MEO-2 3D vector MAE | 0.316861 m | 0.234461 m | 26.0% |

The promotion gate passed because both combined MEO orbit-vector MAE and clock MAE improved.

Use true 3D vector error for orbit accuracy:

```text
sqrt(residual_x^2 + residual_y^2 + residual_z^2)
```

Do not substitute the absolute difference between actual and predicted vector magnitudes; that can conceal directional errors.

## Important metric distinction

The official PS-08 Priority-1 metric is macro-average Shapiro-Wilk W across the three series and four targets. It measures residual normality, not prediction accuracy. The GEO Gated MoE remains strongest by that official criterion, while the Orbit-Class Specialist is the preferred balanced MEO accuracy model.

## Refined artifacts

Primary outputs are under:

```text
results/ps08_refined_meo_20260904/
```

Important files:

- `benchmark_report.json`: detailed metrics, routing, validation evidence, and promotion result
- `BENCHMARK_REPORT.md`: readable benchmark summary
- `orbit_class_specialist_manifest.json`: routing policy and component artifact references
- `day8_predictions.csv`: predictions from every evaluated model
- `day8_actual_vs_predicted_orbit_class_specialist.csv`: complete refined comparison
- `meo_day8_actual_vs_predicted_refined.csv`: MEO-only refined comparison
- `geo_gated_moe_day8.pt`: trained GEO checkpoint
- `gaussian_process_day8.joblib`: fitted Gaussian-process models
- `random_forest_day8.joblib`: fitted random-forest models

The earlier unrefined results are retained under `results/ps08_direct_retrained_20260904/` for rollback and comparison.

## Verification status

- Focused PS-08 and GEO-regime tests pass.
- After removal of the acquisition layer, 33 data-pipeline, audit, PS-08, and GEO-regime tests pass.
- Full-suite verification after the removal completed with 76 passing tests and the same 5 inference-fixture failures caused by the absent `data/sample/sample_gnss_data.csv`.
- The refined artifact validation confirms 24 unique MEO test rows and verifies every stored 3D vector error.
- The full test suite currently has five unrelated inference-test failures because `data/sample/sample_gnss_data.csv` is absent.
- Do not fabricate a replacement sample dataset merely to make those tests pass. Either restore the original tracked fixture from an authorized source or update the tests only when requirements explicitly change.

Useful checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_ps08_benchmark.py tests\test_geo_regime_aware.py
.\.venv\Scripts\python.exe -m py_compile scripts\benchmark\benchmark_ps08.py
git diff --check -- scripts\benchmark\benchmark_ps08.py tests\test_ps08_benchmark.py
```

## Scientific limitations

- Only 24 unique MEO test observations are available: 6 for MEO-1 and 18 for MEO-2.
- Reported MEO gains are meaningful on the supplied split but require confirmation on additional independent MEO periods before production deployment.
- Preserve the supplied test split as an untouched final evaluation set in future experiments.

## Working-tree guidance

- Preserve unrelated user changes and existing result artifacts.
- Write new experiments to a new result directory instead of overwriting the refined artifacts.
- Record dataset paths, seed, validation policy, routing, component artifacts, and metrics in every promoted model manifest.
- Keep a known-good previous artifact available for rollback.
