# PS-08 Day-8 Model Benchmark

Models were trained only on the supplied seven-day files and evaluated at every unique supplied test timestamp. Exact duplicate rows were removed. Test observations were never fed back as model inputs.

## Official ranking

**Winner: Harmonic Ridge** — highest average Shapiro–Wilk W across the four equally weighted residual parameters.

| Rank | Model | Avg W | Avg p-value | Rejected tests | MAE (m) | RMSE (m) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Harmonic Ridge | 0.848832 | 0.177741 | 7/12 | 6.267048 | 13.256386 |
| 2 | Random Forest | 0.847138 | 0.217385 | 6/12 | 6.256051 | 13.357718 |
| 3 | BiLSTM-GRU | 0.827697 | 0.134819 | 8/12 | 5.949739 | 13.196067 |
| 4 | Persistence | 0.826648 | 0.126571 | 8/12 | 6.075791 | 13.198906 |
| 5 | Transformer | 0.825003 | 0.128970 | 8/12 | 5.952240 | 13.197836 |
| 6 | Gaussian Process | 0.821455 | 0.126631 | 7/12 | 5.944548 | 13.191014 |

The primary score is the macro-average of 12 independent tests (3 series × 4 parameters); this avoids mixing different orbit distributions or weighting GEO by its larger row count.

The published reference benchmark is W = 0.9810, p = 0.5840, hypothesis result = 0. This is a normality benchmark, not an accuracy threshold.

## Judge criteria captured from `Data_PS-08/Note.pdf`

1. Priority 1: average Shapiro–Wilk W over X, Y, Z and clock residuals; higher is better. Report p-values and the α=0.05 decision (0 = fail to reject normality, 1 = reject).
2. Priority 2: residual mean and standard deviation break a Priority-1 tie.
3. Priority 3: Q-Q plots and their visible outliers break any remaining tie.

See `qq_*.png` for every model and `day8_predictions.csv` for row-level evidence.
