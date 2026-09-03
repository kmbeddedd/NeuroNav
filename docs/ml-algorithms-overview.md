# Machine-Learning Algorithms

## Forecasting Models

### 1. Bidirectional LSTM

**Category:** recurrent time-series forecasting

Processes each 24-hour telemetry sequence in both directions to learn orbital dynamics, perturbations, and secular drift.

### 2. Gated Recurrent Unit

**Category:** recurrent temporal compression

Compresses bidirectional sequence representations into a compact context using reset and update gates.

### 3. Multi-Head Self-Attention

**Category:** Transformer sequence modeling

Models long-range relationships between lookback timesteps with scaled dot-product attention.

### 4. Satellite and Orbit-Class Embeddings

**Category:** categorical representation learning

Maps satellite PRNs—and MEO, GEO, or IGSO classes when supplied—to learned vectors that capture entity-specific behavior.

### 5. Gaussian Parameter Regression

**Category:** probabilistic forecasting

Predicts a location and scale for every horizon and target. Gaussian negative log-likelihood is the default objective; Student-t remains available as a robustness ablation.

### 6. Separate Orbit and Clock Heads

**Category:** multi-task deep learning

Uses independent projections and loss weighting for metre-scale XYZ residuals and statistically different clock residuals.

### 7. Conditional DDPM/DDIM

**Category:** generative residual modeling

Optionally samples stochastic residual trajectories conditioned on the Transformer context. DDIM provides an accelerated reverse process.

### 8. Gaussian Process Regression

**Category:** probabilistic classical baseline

Provides an optional RBF-plus-white-noise comparison against recurrent and attention models.

## Optimization and Calibration

### 9. Tree-Structured Parzen Estimator

Optuna's seeded TPE search tunes network width, dropout, learning rate, and batch size.

### 10. Scaled Split-Conformal Prediction

Chronological validation residuals calibrate distribution-free 90% and 95% prediction intervals.

### 11. Binary Cross-Entropy Event Head

The architecture supports event classification, but training is deliberately disabled unless externally sourced maneuver or clock-event labels are available.

## Statistical Evaluation

- Shapiro-Wilk normality test
- Anderson-Darling normality test
- Gaussian Q-Q plots
- Residual skewness and excess kurtosis

See the main [`README.md`](../README.md) for current results and reproducible commands.
