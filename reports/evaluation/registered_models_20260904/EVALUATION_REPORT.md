# Registered Satellite Model Evaluation

Evaluation date: 2026-09-04  
Code revision: `0596e66f5a1cb12f279d5c3d816ef01bb8d51504`  
Registry: `models/registry/satellite_model_registry.json`  
Calibration report: `reports/calibration/official_competition_run/`

## Executive verdict

**Production gate: FAIL.** The registered artifacts reproduce the recorded calibration metrics and inference from a loaded artifact is deterministic. However, the available Day-8 labels were used to rank and select the registered models, so this is a calibration/validation result rather than an unbiased final test. There is no untouched holdout, cross-validation estimate, configured production threshold, or uncertainty output. Two serving-path defects and five failing repository tests add deployment risk.

The models do pass the repository's *selection procedure*: each has the highest equal-weighted Shapiro-Wilk residual-normality score for its satellite. That is not the same as passing an accuracy target. In particular, Shapiro-Wilk W measures the shape of residuals rather than their magnitude; it can prefer larger but more normally distributed errors.

## 1. Evaluation setup and leakage audit

Only the six supplied CSV files were used. No synthetic, augmented, or interpolated observations were created.

| Satellite | Training period | Raw / unique train rows | Evaluation period | Raw / unique eval rows | Duplicate epochs removed | Timestamp overlap |
|---|---|---:|---|---:|---:|---:|
| GEO | 2025-09-01 06:00–2025-09-07 23:41 | 142 / 142 | 2025-09-08 00:11–23:41 | 69 / 69 | 0 train, 0 eval | 0 |
| MEO-1 | 2025-09-01 14:00–2025-09-07 16:00 | 90 / 46 | 2025-09-08 14:01–19:00 | 11 / 6 | 44 train, 5 eval | 0 |
| MEO-2 | 2025-09-03 10:11–2025-09-09 11:41 | 244 / 143 | 2025-09-10 11:41–17:11 | 30 / 18 | 101 train, 12 eval | 0 |

- There is no timestamp or full-row overlap between normalized train and evaluation data.
- The calibration code fits on normalized training rows and only then predicts at evaluation timestamps. Evaluation targets are not passed to `fit()` or `predict()`.
- The registry's three 16-character training-target hashes exactly match the current normalized training arrays.
- Loaded checkpoint predictions reproduce the recorded W scores (differences are at floating-point rounding level), confirming checkpoint/config alignment.
- **Procedural contamination:** all nine candidates were scored on these Day-8 labels and the winners were persisted. Consequently, Day-8 is not an untouched test set and cannot support an unbiased production claim.
- There is no validation split or cross-validation in this satellite-specific calibration path.

## 2. Core metrics

The official primary metric is equal-weighted Shapiro-Wilk `W_avg` over X, Y, Z, and Clock residuals. MAE/RMSE/SISRE are diagnostics only under the configured selection policy.

| Satellite | Registered model | N | W_avg | Normality tests rejected (of 4) | 3D orbit MAE, m (95% CI) | 3D orbit RMSE, m | Clock MAE, m (95% CI) | SISRE mean, m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GEO | N-HiTS | 69 | 0.8894 | 3 | 23.527 (19.248–28.114) | 30.011 | 7.760 (4.565–11.456) | 14.725 |
| MEO-1 | GEO gated MoE | 6 | 0.9177 | 0 | 0.965 (0.807–1.121) | 0.986 | 0.144 (0.071–0.248) | 0.356 |
| MEO-2 | Random Forest | 18 | 0.8670 | 1 | 0.327 (0.232–0.433) | 0.392 | 0.026 (0.012–0.047) | 0.173 |

Confidence intervals are 5,000-resample percentile bootstraps over unique epochs (seed `20260904`). They characterize this small evaluation set; they do not correct model-selection bias.

### Per-target diagnostics

| Satellite | X MAE / RMSE | Y MAE / RMSE | Z MAE / RMSE | Clock MAE / RMSE | Largest component error |
|---|---:|---:|---:|---:|---:|
| GEO | 11.283 / 15.991 | 15.462 / 21.991 | 9.329 / 12.704 | 7.760 / 16.241 | 71.906 m (Y) |
| MEO-1 | 0.732 / 0.748 | 0.432 / 0.541 | 0.269 / 0.347 | 0.144 / 0.186 | 0.985 m (X) |
| MEO-2 | 0.196 / 0.253 | 0.105 / 0.123 | 0.194 / 0.272 | 0.026 / 0.048 | 0.685 m (Z) |

There is no classification, so precision, recall, confusion matrices, and per-class metrics do not apply. The models produce point forecasts only; calibration/coverage metrics cannot be computed because predictive distributions or intervals are absent.

## 3. Baseline and alternative-model comparison

Paired deltas below are registered winner minus persistence; negative error deltas are improvements.

| Satellite | W improvement | 3D orbit MAE delta, m (95% CI) | Clock MAE delta, m (95% CI) | Finding |
|---|---:|---:|---:|---|
| GEO | +0.1054 | **+6.600 (+4.547 to +8.633)** | **+1.772 (+1.138 to +2.385)** | Significantly worse accuracy than persistence |
| MEO-1 | +0.0303 | +0.109 (-0.153 to +0.375) | **-0.133 (-0.196 to -0.062)** | Orbit difference inconclusive; clock improves |
| MEO-2 | +0.0584 | +0.007 (-0.074 to +0.096) | **-0.009 (-0.014 to -0.004)** | Orbit difference inconclusive; small clock improvement |

The accuracy-optimal candidate differs from the official W winner for every satellite:

| Satellite | Lowest 3D MAE candidate | Lowest 3D MAE, m | Registered winner MAE, m | Accuracy penalty |
|---|---|---:|---:|---:|
| GEO | Gaussian Process | 16.432 | 23.527 | +7.095 m (+43.2%) |
| MEO-1 | Gaussian Process | 0.310 | 0.965 | +0.655 m (+211.5%) |
| MEO-2 | Gaussian Process | 0.234 | 0.327 | +0.092 m (+39.3%) |

This is not an implementation discrepancy: it follows the configured competition hierarchy. It is a serious objective-design issue if production accuracy, rather than residual normality, is the actual goal.

## 4. Slice analysis

| Satellite | Early 3D / clock MAE, m | Late 3D / clock MAE, m | Notable degradation |
|---|---:|---:|---|
| GEO | 21.819 / 4.799 | 25.187 / 10.636 | Late clock MAE +121.6%; late orbit +15.4% |
| MEO-1 | 0.944 / 0.223 | 0.986 / 0.066 | Orbit stable; early clock worse, but only 3 epochs per slice |
| MEO-2 | 0.333 / 0.013 | 0.320 / 0.039 | Late clock roughly triples; only 9 epochs per slice |

GEO is the clearly weak segment. Its orbit MAE is about 24× MEO-1 and 72× MEO-2, although the regimes have different physical scales. Within targets, GEO Y is the weakest axis. Small MEO sample sizes make subgroup conclusions fragile.

## 5. Error analysis

The largest GEO vector misses cluster around excursion events rather than ordinary low-error epochs:

| Satellite/time | 3D error, m | Clock abs error, m | Dominant pattern |
|---|---:|---:|---|
| GEO 2025-09-08 00:24 | 80.306 | 35.181 | Large Y miss (+71.906 m residual) |
| GEO 2025-09-08 00:11 | 68.053 | 29.298 | Large Y/Z excursion miss |
| GEO 2025-09-08 16:04 | 67.795 | 51.353 | X/Y/Z and clock excursion |
| GEO 2025-09-08 17:10 | 64.035 | 62.136 | Large Y and clock sign/magnitude miss |
| GEO 2025-09-08 17:25 | 59.525 | 57.178 | X/Y and clock excursion |
| MEO-1 2025-09-08 19:00 | 1.192 | 0.052 | Persistent positive X/Z bias |
| MEO-2 2025-09-10 13:42 | 0.862 | 0.047 | Isolated X/Z outlier |

N-HiTS tiles its learned sub-horizon when the requested horizon is longer, and ignores actual forecast timestamp spacing. The observed GEO excursion misses and zero response to a +1 minute timestamp perturbation are consistent with that architecture limitation. MEO-1 also shows a systematic +0.732 m X bias across all six epochs.

## 6. Robustness and train/serve skew

- Repeating inference from the same loaded artifacts is deterministic (maximum difference 0 for neural models and `2.8e-16` m for Random Forest).
- A +1 minute timestamp perturbation changes N-HiTS predictions by exactly zero, MEO-1 by at most 0.0051 m orbit, and MEO-2 by at most 0.1643 m orbit. The Random Forest discontinuity is large relative to its 0.327 m MAE.
- Calibration deduplicates timestamps, but `PredictionRouter.predict()` calls `load_telemetry_source()` directly and does not run validation/deduplication. With raw duplicate MEO-1 history, the same model's orbit prediction moves by 0.472 m on average (maximum 0.497 m), roughly half its measured MAE. Random Forest is history-insensitive and does not move materially.
- Single-file ingestion misidentifies both `DATA_MEO_Train.csv` and `DATA_MEO_Train2.csv` as satellite `MEO`, while the registry keys are `MEO-1` and `MEO-2`. Routing those files individually therefore cannot resolve the intended registered model.
- Calibration evaluates at the supplied irregular timestamps; serving always generates a fixed 15-minute grid by default. This is another evaluation/serving contract mismatch.
- `src/forecasting/api.py:get_calibration_report()` contains only a docstring and therefore always returns `None`; the newer summary/comparison accessors work, but this public endpoint is non-functional.

These are checks on supplied observations and timestamp perturbations only; no artificial dataset was created.

## 7. Generalization and overfitting

Train-versus-validation-versus-test curves are unavailable. The calibration path has only fit data and a winner-selection set, and checkpoints do not record epoch losses. A trustworthy overfitting assessment is therefore impossible.

The small effective MEO sample sizes are especially concerning: MEO-1's four non-rejected Shapiro tests use only six unique observations, where the test has very low power. Treating “failure to reject normality” as positive evidence is unsafe at this N.

The repository's causal-window tests cover a separate/general pipeline, but they do not create an untouched evaluation split for these registered satellite models.

## 8. Reproducibility and test health

- Artifact hashes, complete input-file SHA-256 hashes, package versions, code revision, predictions, slices, confidence intervals, and worst cases are recorded in `evaluation.json`.
- Clean retraining is not fully reproducible because the calibration API exposes no seed and the PyTorch models do not set one. Across two fresh fits, N-HiTS W varied by 0.00995 and 3D MAE by 3.286 m; MEO-1 MoE W varied by 0.00256 and MAE by 0.0196 m. Random Forest reproduced exactly because it has `random_state=42`.
- Full pytest result: **101 passed, 5 failed, 1,816 warnings**. All five failures come from `tests/test_inference.py` referencing the removed `data/sample/sample_gnss_data.csv`. The failures are stale-test/integration debt, not model-score failures, but a red test suite still blocks a production release.
- A separate unittest discovery run passed 30/30 tests.
- A legacy estimator artifact exercised by `tests/test_orbitiq_eval.py` was created under scikit-learn 1.7.1 but loaded under 1.9.0, producing an `InconsistentVersionWarning`. Cross-version pickle/joblib loading is not a reliable deployment contract.

## Decision and required remediation

**Overall: fail before production.** Strengths are explicit satellite routing, no direct train/evaluation timestamp overlap, deterministic loaded-artifact inference, reproducible recorded metrics, and strong absolute MEO-2 errors on this small sample. Weaknesses are severe GEO accuracy, tiny MEO evaluation sets, no probabilistic calibration, no learning curves, and model selection based on residual shape rather than prediction magnitude.

Release blockers, in recommended order:

1. Reserve a new, untouched chronological test period (or use nested rolling-origin evaluation) and never select models on it.
2. Define numerical production gates for 3D orbit MAE/RMSE, clock MAE, SISRE, worst-case error, and per-satellite degradation; keep Shapiro W diagnostic unless the competition explicitly requires it as primary.
3. Make inference call the same validation/deduplication path as calibration; fix MEO-1/MEO-2 filename identity and reconcile irregular evaluation timestamps with the serving cadence.
4. Add and persist a global seed plus training history, environment lock, input hashes, and artifact metadata.
5. Replace or update stale sample-data tests and eliminate the scikit-learn artifact-version mismatch.
6. Re-evaluate on multiple chronological folds and report fold-level variance before promoting any candidate.
