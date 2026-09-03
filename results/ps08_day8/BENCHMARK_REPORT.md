# PS-08 Day-8 Model Benchmark

Models were trained only on the supplied seven-day files and evaluated at every unique supplied test timestamp. Exact duplicate rows were removed. Test observations were never fed back as model inputs.

## Official ranking

**Winner: GEO Gated MoE** — highest average Shapiro–Wilk W across the four equally weighted residual parameters.

| Rank | Model | Avg W | 95% Bootstrap CI | Avg p-value | Rejected tests | MAE (m) | RMSE (m) | GEO W | GEO MAE (m) | GEO RMSE (m) |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GEO Gated MoE | 0.848836 | [0.6638, 0.9104] | 0.184691 | 7/12 | 6.2689 | 13.2577 | 0.802533 | 8.3935 | 15.0322 |
| 2 | Harmonic Ridge | 0.848832 | [0.6633, 0.9099] | 0.177741 | 7/12 | 6.2670 | 13.2564 | 0.802403 | 8.3850 | 15.0409 |
| 3 | Random Forest | 0.847138 | [0.6621, 0.9091] | 0.217385 | 6/12 | 6.2561 | 13.3577 | 0.782957 | 8.3774 | 15.1316 |
| 4 | BiLSTM-GRU | 0.827697 | [0.6417, 0.9078] | 0.134819 | 8/12 | 5.9497 | 13.1961 | 0.783743 | 7.9725 | 14.9500 |
| 5 | Persistence | 0.826648 | [0.6430, 0.9078] | 0.126571 | 8/12 | 6.0758 | 13.1989 | 0.783926 | 8.1226 | 14.9642 |
| 6 | Transformer | 0.825003 | [0.6423, 0.9058] | 0.128970 | 8/12 | 5.9522 | 13.1978 | 0.783770 | 7.9773 | 14.9527 |
| 7 | Gaussian Process | 0.821455 | [0.6361, 0.9047] | 0.126631 | 7/12 | 5.9445 | 13.1910 | 0.783932 | 7.9693 | 14.9483 |

The primary score is the macro-average of 12 per-series/per-target Shapiro-Wilk evaluations (3 series × 4 parameters); this avoids mixing different orbit distributions or weighting GEO by its larger row count.

The published reference benchmark is W = 0.9810, p = 0.5840, hypothesis result = 0. This is a normality benchmark, not an accuracy threshold.

## Judge criteria captured from `Data_PS-08/Note.pdf`

1. Priority 1: average Shapiro–Wilk W over X, Y, Z and clock residuals; higher is better. Report p-values and the α=0.05 decision (0 = fail to reject normality, 1 = reject).
2. Priority 2: residual mean and standard deviation break a Priority-1 tie.
3. Priority 3: Q-Q plots and their visible outliers break any remaining tie.

See `qq_*.png` for every model and `day8_predictions.csv` for row-level evidence, and `geo_diagnostics/` for GEO-specific validation and excursion diagnostics.
