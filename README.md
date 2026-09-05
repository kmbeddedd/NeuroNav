# NeuroNav: Precision Satellite Orbit & Clock Error Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![GUI](https://img.shields.io/badge/GUI-Tkinter%20%7C%20Matplotlib-3776AB.svg)](https://docs.python.org/3/library/tkinter.html)
[![Git LFS](https://img.shields.io/badge/Git-LFS%20Enabled-F05032.svg)](https://git-lfs.github.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Zero--Leakage%20Fail--Closed-0052CC.svg)](src/forecasting/)

NeuroNav is a production-grade forecasting engine and mission control telemetry suite designed to predict satellite orbit and clock errors from independently supplied GNSS time series. The system validates per-satellite telemetry, evaluates eligible candidate models on strictly held-out observations, ranks models according to the authoritative competition hierarchy, records decisions in an atomic persistent registry, and routes operational predictions without silent fallbacks.

The repository evolved from the `kmbeddedd/kkkk` research foundation. The legacy global BiLSTM/Transformer pipeline remains available as an explicit compatibility layer under `compat/`; modern integrations should use the unified `src.forecasting` API and the high-density desktop mission control suite.

---

## Table of Contents

1. [System Highlights](#system-highlights)
2. [Production Architecture & Flow](#production-architecture--flow)
3. [Repository Layout](#repository-layout)
4. [Installation & Setup](#installation--setup)
5. [Data Contract & Ingestion](#data-contract--ingestion)
6. [Public Backend API](#public-backend-api)
7. [Inference Engine & Prediction Router](#inference-engine--prediction-router)
8. [Model Factory & Architectures](#model-factory--architectures)
9. [Official Selection Hierarchy (P1/P2/P3)](#official-selection-hierarchy-p1p2p3)
10. [Persistent Satellite Registry](#persistent-satellite-registry)
11. [Desktop Mission Control GUI Application](#desktop-mission-control-gui-application)
    - [Architecture & ML Boundary](#architecture--ml-boundary)
    - [Stage 1: Model Calibration & Candidate Ranking](#stage-1-model-calibration--candidate-ranking)
    - [Stage 2: 96-Step Forecast & Timestamp Inspector](#stage-2-96-step-forecast--timestamp-inspector)
    - [Stage 3: Residual Probability & Mathematical Formula Cards](#stage-3-residual-probability--mathematical-formula-cards)
    - [Memory Management & Deep Reset](#memory-management--deep-reset)
12. [CLI & Operational Commands](#cli--operational-commands)
13. [Verification & Test Suite](#verification--test-suite)

---

## System Highlights

- **Zero Data Leakage Guarantee**: Strict separation between historical training telemetry (Days 1–7) and held-out ground truth evaluation targets (Day 8). Candidate models never observe evaluation epochs during training.
- **Authoritative Competition Hierarchy**: Selection follows strict priority rules: **P1** (equal-weighted Shapiro-Wilk $W$ normality across $X, Y, Z, \text{Clock}$), **P2** (residual bias and standard deviation tie-breakers), and **P3** (Q-Q plot outlier counts). Diagnostic metrics (MAE, RMSE, SISRE) never override the hierarchy.
- **Fail-Closed Runtime Routing**: Predictions are resolved strictly from verified, version-locked model artifacts. Missing models, corrupt registries, or unregistered satellites fail with actionable exceptions rather than defaulting silently.
- **Physics-Informed Modeling**: Native support for Radial-Intrack-Crosstrack (RIC) orbital coordinate frames, nominal analytical solar radiation pressure (SRP), and user-provided state vectors.
- **Mission Control GUI**: Modern Tkinter/Matplotlib interface featuring clean tabular candidate views, arbitrary timestamp telemetry inspection, residual distribution plots, and interactive mathematical formula flash cards.

---

## Production Architecture & Flow

```text
Single-Satellite Historical CSV (Days 1-7) & Held-Out CSV (Day 8)
       │
       ▼
[ Schema & Cadence Validation ] ──► Auto-detects 15-min intervals & ECEF units
       │
       ▼
[ Orbit Classification ] ──────────► GEO (Geostationary) or MEO (Medium Earth Orbit)
       │
       ▼
[ Physics State Provider ] ────────► Nominal orbital mechanics, RIC transforms, SRP
       │
       ▼
[ Candidate Training ] ────────────► Evaluates Harmonic Ridge, RF, GP, BiLSTM,
       │                             Transformer, GEO MoE, Decoupled Clock, N-HiTS
       ▼
[ Held-Out Evaluation ] ───────────► Computes P1 (Shapiro W), P2 (Bias/Std), P3 (Q-Q)
       │
       ▼
[ Atomic Model Registry ] ─────────► Commits winning artifact & metadata to JSON store
       │
       ▼
[ Prediction Router ] ─────────────► Fail-closed routing: loads exact versioned weights
       │
       ▼
[ Desktop GUI & Python API ] ──────► Real-time telemetry, 96-step forecast, formula cards
```

Production code does not import experimental or research modules. Unknown satellites, missing selections, corrupt registries, and missing artifacts fail fast with diagnostic logging.

---

## Repository Layout

```text
app/                               Tkinter controllers and application dispatchers
  controllers/
    inference_controller.py        Asynchronous UI-to-ML boundary coordinator
app.py                             Desktop GUI application launcher (`python app.py`)
gui/                               Mission control GUI interface
  gui_app.py                       Tkinter presentation layer and Matplotlib renderers
  formula_tooltips.py              Interactive mathematical formula flash cards
configs/
  contracts/                       Versioned input and output JSON schema contracts
  models/                          Legacy deploy-bundle configurations
  promotion/                       Candidate promotion policies
data/ps08/                         Canonical supplied GEO/MEO CSV datasets
docs/
  reference/ps08/                  Competition specifications and problem statements
  research/                        Forecasting background notes
models/
  registry/                        Satellite selections and routed artifacts
    satellite_model_registry.json  Persistent registry store
    artifacts/                     Model weights (.pt, .joblib) and manifests
  deploy/                          Legacy GUI-compatible global bundles
  orbitiq_pretrained/              Research comparison artifacts
reports/                           Calibration and evaluation evidence
inference/                         Operational prediction CLI (`inference/predict.py`)
research/
  experiments/                     Ablation and experimental notebooks
  orbitiq/                         OrbitIQ baseline comparison implementation
  ps08/                            Historical benchmark implementation
scripts/
  benchmark/                       Compatibility CLI wrappers
  data/                            Data audit CLI
  evaluate/                        Evaluation and export CLIs
  ops/                             Registry and runtime verification scripts
src/
  forecasting/
    api.py                         Public backend interface facade
    data/validation.py             Upload validation and cadence metadata
    features/core.py               Versioned train/serve feature manifests
    physics.py                     Orbital state, RIC, solar geometry, SISRE
    models/                        Satellite model adapters and factory
    training/calibration.py        Candidate calibration and selection pipeline
    evaluation/official.py         Authoritative P1/P2/P3 evaluation boundary
    registry/store.py              Atomic persistent registry store
    inference/router.py            Fail-closed runtime routing
  compat/global_forecasting/       Inherited global training/inference stack
tests/                             Unit, integration, and UI verification tests
```

---

## Installation & Setup

### 1. Environment Configuration

NeuroNav requires **Python 3.10+**. Set up a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Large Model Weights (Git LFS)

Model binaries (`.pt`, `.joblib`, `.h5`, `.pkl`) are managed via **Git LFS**:

```powershell
git lfs install
git lfs pull
```

---

## Data Contract & Ingestion

The canonical data contract is defined in [`configs/contracts/ps08_satellite_data.json`](configs/contracts/ps08_satellite_data.json). The competition datasets reside under `data/ps08/`:

| Satellite ID | Orbit Type | Calibration History (Days 1–7) | Held-Out Evaluation (Day 8) |
|---|---|---|---|
| **GEO** | Geostationary (GEO) | `data/ps08/DATA_GEO_Train.csv` | `data/ps08/DATA_GEO_Test.csv` |
| **MEO-1** | Medium Earth Orbit (MEO) | `data/ps08/DATA_MEO_Train.csv` | `data/ps08/DATA_MEO_Test.csv` |
| **MEO-2** | Medium Earth Orbit (MEO) | `data/ps08/DATA_MEO_Train2.csv` | `data/ps08/DATA_MEO_Test2.csv` |

### Required Telemetry Columns

All satellite telemetry files must provide the following columns:

```text
utc_time          # ISO-8601 or UTC datetime string
x_error_m         # Satellite orbit X-axis error (metres, ECEF frame)
y_error_m         # Satellite orbit Y-axis error (metres, ECEF frame)
z_error_m         # Satellite orbit Z-axis error (metres, ECEF frame)
clock_error_m     # Satellite clock range error (metres)
```

> [!NOTE]
> In the PS-08 pipeline, all four error components—including `clock_error_m`—are expressed in **metres**. If using legacy bundles (`configs/contracts/legacy_gui_inference.json`), `Error_Clock` is expressed in seconds ($1\text{ s} \approx 299{,}792{,}458\text{ m}$). The system validates units during ingestion to prevent scale mixing.

---

## Public Backend API

Import the stable facade directly from `src.forecasting`:

```python
from src.forecasting import (
    register_satellite,
    validate_satellite_dataset,
    train_satellite,
    evaluate_satellite,
    predict_satellite,
    get_satellite_model,
    get_satellite_metadata,
)
```

### 1. Validate Telemetry

```python
from src.forecasting import validate_satellite_dataset

report = validate_satellite_dataset(
    "data/ps08/DATA_GEO_Train.csv",
    satellite_id="GEO",
    orbit_type="GEO",
)
print(f"Cadence: {report['cadence_minutes']} min | Rows: {report['row_count']}")
```

### 2. Calibrate and Register a Satellite

```python
from src.forecasting import register_satellite, train_satellite

register_satellite("GEO", "GEO", metadata={"source": "PS-08"})

result = train_satellite(
    dataset="data/ps08/DATA_GEO_Train.csv",
    test_dataset="data/ps08/DATA_GEO_Test.csv",
    satellite_id="GEO",
    orbit_type="GEO",
    physics_mode="nominal",
)
print(f"Selected Winner: {result['selected_model']}")
print(f"Mean Shapiro-Wilk W: {result['selection_summary']['priority_1']['W']['average']:.4f}")
```

`physics_mode` supports:
- `none`: Pure statistical time-series forecasting.
- `nominal`: Analytical GEO/MEO Keplerian orbital approximation with solar angles.
- `provided`: Caller supplies a timestamped orbital-state DataFrame; saved with the model artifact.

### 3. Generate Forecast with Registered Winner

```python
from src.forecasting import predict_satellite

forecast = predict_satellite(
    satellite_id="GEO",
    history_data="data/ps08/DATA_GEO_Train.csv",
    horizon_steps=96,
)
print(forecast[["utc_time", "x_error_m", "y_error_m", "z_error_m", "clock_error_m", "orbit_3d_error_m"]].head())
```

---

## Inference Engine & Prediction Router

The production inference engine operates under a **fail-closed** architecture:

```text
Historical Satellite CSV
        │
        ▼
[ Validate & Normalize ] ──► Validates timestamps, units, and regular 15-min cadence
        │
        ▼
[ Resolve Registry ] ──────► Queries active model assignment from JSON store
        │
        ▼
[ Load Exact Artifact ] ───► Verifies SHA/file existence; loads exact .pt/.joblib weights
        │
        ▼
[ Restore Physics ] ───────► Initializes nominal or persisted orbital coordinate transforms
        │
        ▼
[ Feature Generation ] ────► Computes harmonic cycles, lags, and solar radiation angles
        │
        ▼
[ Model Forward Pass ] ────► Generates multi-target forecasts (X, Y, Z, Clock)
        │
        ▼
[ Derived Telemetry ] ─────► Computes 3D orbit error ||(X, Y, Z)|| and optional RIC components
        │
        ▼
[ Canonical Output ] ──────► Returns structured DataFrame conforming to JSON contract
```

### CLI Inference

Run CLI predictions from the repository root:

```powershell
python -m inference.predict `
  --satellite GEO `
  --orbit-type GEO `
  --history data/ps08/DATA_GEO_Train.csv `
  --horizon-steps 96 `
  --output reports/inference/GEO_forecast.csv
```

---

## Model Factory & Architectures

NeuroNav includes a modular model factory (`src/forecasting/models/`) implementing multiple distinct forecasting families:

| Model Family | Key Characteristics | Feature Stack |
|---|---|---|
| **Harmonic Ridge** | Regularized Ridge regression over diurnal/orbital harmonic frequencies | Versioned unified feature manifest (v2) |
| **Random Forest** | Non-linear ensemble with bootstrap aggregation and tree depth constraints | Lagged error features + harmonics |
| **Gaussian Process** | Non-parametric Bayesian regression with RBF and Matérn covariance kernels | Normalized time indices & state vectors |
| **BiLSTM-GRU** | Bidirectional recurrent neural network with gated recurrent units | Scaled multi-step sequence windows |
| **Transformer** | Multi-head self-attention network capturing long-range orbital dependencies | Multi-head attention + positional encodings |
| **GEO MoE** | Gated Mixture of Experts specializing across diurnal regimes | Gating network with domain expert sub-models |
| **Decoupled Clock** | Independent modeling pathways for spatial orbit dynamics vs. atomic clock drift | Split orbit/clock feature pathways |
| **N-HiTS** | Neural Hierarchical Interpolation for Time Series with multi-rate sampling | Multi-scale residual blocks |
| **Persistence** | Zero-order baseline carrying forward the last known observation | Direct state replication |

---

## Official Selection Hierarchy (P1/P2/P3)

Model selection is strictly governed by `src.forecasting.evaluation.official`. Residual errors are calculated as $\text{Residual} = \hat{y} - y$ (predicted minus actual) with significance level $\alpha = 0.05$:

### 1. Priority 1 (P1) — Normality Maximization (Authoritative)
The candidate model that maximizes the equal-weighted average Shapiro-Wilk $W$ statistic across all four targets is selected:
$$\bar{W} = \frac{1}{4} \left( W_X + W_Y + W_Z + W_{\text{Clock}} \right)$$
Each target contributes exactly $25\%$ to the evaluation.

### 2. Priority 2 (P2) — Bias and Variance Tie-Breakers
If two candidate models have $\bar{W}$ within a tolerance of $1 \times 10^{-4}$:
1. Minimize aggregate absolute residual bias:
   $$\text{Bias}_{\text{agg}} = \frac{1}{4} \sum_{k \in \{X, Y, Z, \text{Clock}\}} |\bar{e}_k|$$
2. If still tied, minimize aggregate residual standard deviation:
   $$\sigma_{\text{agg}} = \frac{1}{4} \sum_{k \in \{X, Y, Z, \text{Clock}\}} \sigma_{e, k}$$

### 3. Priority 3 (P3) — Distribution Tail Robustness
If still tied under P1 and P2:
1. Minimize Q-Q outlier count ($|z| > 3$).
2. Minimize maximum quantile discrepancy against theoretical normal quantiles.

> [!IMPORTANT]
> Supplementary diagnostic metrics such as MAE, RMSE, 3D Orbit Error, and SISRE are computed for telemetry inspection only. They **never** override the official P1/P2/P3 ranking.

---

## Persistent Satellite Registry

Active model assignments are stored in `models/registry/satellite_model_registry.json`. The schema maintains complete model provenance:

```json
{
  "last_updated": "2026-09-05T07:30:00+00:00",
  "satellites": {
    "GEO": {
      "selected_model": "nhits",
      "selection_score": 0.8637,
      "selection_mode": "automatic",
      "selection_policy": "official_competition",
      "model_path": "models/registry/artifacts/satellites/GEO/model.pt",
      "cadence_minutes": 15.0,
      "orbit_type": "GEO",
      "physics_mode": "nominal"
    }
  }
}
```

---

## Desktop Mission Control GUI Application

NeuroNav provides a desktop mission control application built with Python Tkinter and Matplotlib (`gui/gui_app.py`), styled with dark slate aesthetics inspired by the Stitch design system.

```powershell
# Launch mission control
python app.py
```

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  NEURONAV MISSION CONTROL                                                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  STAGE 1: CALIBRATION         │ STAGE 2: INFERENCE           │ STAGE 3: DIAGNOSTICS    │
│  • Sat ID & Orbit Selector    │ • Satellite Dropdown         │ • Residual Histograms   │
│  • Train/Test CSV Upload      │ • 96-Step Forecast Table     │ • Q-Q Normal Plots      │
│  • Auto-Train All Models      │ • Timestamp Lookup (Badge)   │ • Normality Summary     │
│  • Clean Ranking Table        │ • 3D Error Vectors (X,Y,Z)   │ • Formula Flash Cards   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architecture & ML Boundary

The desktop application follows a strict Model-View-Controller pattern:
- **Presentation Layer (`gui/gui_app.py`)**: Renders interactive telemetry controls, tables, metrics badges, and Matplotlib diagnostic plots. The GUI never performs heuristic manipulation or modifies forecast values.
- **Controller Layer (`app/controllers/inference_controller.py`)**: Decouples UI interactions and worker threads from the forecasting backend, coordinating asynchronous model training, prediction, and registry queries.
- **Official Forecasting API (`src.forecasting.api`)**: Telemetry flows directly into the backend `CalibrationPipeline` and `PredictionRouter`, returning genuine model inference outputs.

---

### Stage 1: Model Calibration & Candidate Ranking

- **Satellite Identification & Orbit Selector**: Enter the satellite identifier (e.g., `GEO-01`, `MEO-01`) and select orbit classification via explicit radio buttons: **`GEO`** (Geostationary) or **`MEO`** (Medium Earth Orbit).
- **Zero-Leakage Ingestion**: Upload separate 7-day training history and held-out 8th-day ground truth evaluation CSVs.
- **Clean Candidate Ranking Table**: Displays the official competition hierarchy metrics without visual icon clutter:
  - Model Name
  - Average Shapiro-Wilk $W$ (P1 rank metric)
  - Exact two-sided $p$-values
  - Null hypothesis $H_0$ rejection status ($\alpha = 0.05$)
  - Aggregate residual bias and standard deviation (P2 tie-breakers)
- The winning candidate is automatically committed to persistent model memory with manual override capability.

---

### Stage 2: 96-Step Forecast & Timestamp Inspector

- **Persistent Model Selection**: Multi-satellite dropdown auto-populates all registered satellites. Selecting a satellite dynamically loads its calibrated model, version, score, and cadence.
- **96-Step Horizon Forecast**: Generates the complete 8th-day horizon forecast (96 steps at 15-minute intervals) and populates the telemetry table with $\Delta X$, $\Delta Y$, $\Delta Z$, $\Delta \text{Clock}$, and $3\text{D Orbit Error}$.
- **Arbitrary Timestamp Error Inspector**: Enter any target timestamp (e.g., `12:15:00`, `14:30:00`, or `2026-01-08 12:15:00`) to execute an instantaneous temporal lookup or point prediction. Results are displayed across high-visibility telemetry badges.

---

### Stage 3: Residual Probability & Mathematical Formula Cards

Stage 3 provides a comprehensive statistical validation suite to evaluate model residual quality against theoretical normality.

#### 1. Interactive Matplotlib Visualizations
- **Residual Histograms & KDE**: Visualizes error distributions for $X$, $Y$, $Z$, and $\text{Clock}$ with kernel density estimates overlaid on theoretical normal distribution curves.
- **Quantile-Quantile (Q-Q) Plots**: Standardized residuals plotted against theoretical normal quantiles with a $45^\circ$ reference line to inspect tail kurtosis and outlier behavior.

#### 2. Interactive Mathematical Formula Flash Cards
The Stage 3 Shapiro-Wilk and Performance diagnostics table features interactive mathematical formula inspection badges:
- **Translucent Badge Design**: A subtle, translucent `?` logo badge (`bg=(148, 163, 184, 45)`) is embedded directly in the right corner of each calculated parameter column header.
- **Pure White Flash Cards**: Taking the cursor to any parameter's `?` badge opens a floating mathematical card styled with:
  - Pure white background (`#FFFFFF`) with a subtle border stroke (`#CBD5E1`) and drop shadow.
  - High-contrast slate typography (`#0F172A`) for primary headers and `#1E293B` for descriptions.
  - 12px rounded corner edges with Windows-native window transparency.
  - High-resolution LaTeX/Mathtext-rendered mathematical formulas.
- **Hover & Pin Interactivity**:
  - **Hover**: Move cursor over the `?` icon to preview the mathematical formula card.
  - **Click to Pin**: Click the `?` badge to lock the card open for detailed study.
  - **Dismiss**: Click the badge again or press <kbd>Escape</kbd> to dismiss the pinned card.

#### 3. Mathematical Formulations Reference

The flash cards provide exact mathematical definitions and interpretation criteria for all 9 calculated diagnostics:

| Parameter Header | Mathematical Formula | Statistical Description & Decision Criteria |
|---|---|---|
| **Shapiro-Wilk $W$** | $$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$ | Tests whether residuals originate from a normal distribution. $x_{(i)}$ are ordered order statistics; $a_i$ are weights from normal covariance matrices. Values closer to $1.0$ indicate stronger normality. |
| **Two-Sided $p$-value** | $$p = P(W \le w_{\text{obs}} \mid H_0)$$ | Two-sided probability of observing a $W$ statistic as extreme as calculated under the assumption that residuals are normally distributed. |
| **Null Hypothesis $H_0$** | $$p \ge 0.05 \implies \text{Retain } H_0 \quad (p < 0.05 \implies \text{Reject } H_0)$$ | At significance level $\alpha = 0.05$, $p \ge 0.05$ indicates residuals cannot be distinguished from a normal distribution. |
| **Bias / Mean Error** | $$\text{Bias} = \bar{e} = \frac{1}{n}\sum_{i=1}^n e_i, \quad e_i = \hat{y}_i - y_i$$ | Mean signed residual. Measures systemic directional over-prediction ($\bar{e} > 0$) or under-prediction ($\bar{e} < 0$). Optimal value is $0.0\text{ m}$. |
| **Residual Std Dev ($\sigma_e$)** | $$\sigma_e = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (e_i - \bar{e})^2}$$ | Unbiased sample standard deviation of residuals. Measures residual dispersion around the mean error. |
| **Mean Absolute Error (MAE)** | $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |e_i|$$ | Average magnitude of errors without direction. Robust against heavy-tailed outliers. |
| **Root Mean Square Error (RMSE)**| $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n e_i^2}$$ | Quadratic scoring rule penalizing large residual deviations more severely than MAE. |
| **Coefficient of Determination ($R^2$)** | $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$ | Proportion of ground truth telemetry variance explained by the model forecast. $1.0$ represents perfect alignment. |
| **Maximum Absolute Error ($\text{Max AE}$)** | $$\text{Max AE} = \max_{1 \le i \le n} |e_i|$$ | Maximum single-epoch absolute error across the evaluation horizon. Identifies worst-case operational deviation. |

---

### Memory Management & Deep Reset

Clicking **`Clear Memory 🗑`** executes an atomic deep reset across:
1. **Persistent Registry**: Clears `models/registry/satellite_model_registry.json`.
2. **Controller Session State**: Flushes active model pipelines, data frames, and cached predictions.
3. **Mission Control UI**: Resets all tables, telemetry badges, and Matplotlib diagnostic plots.

This guarantees that subsequent calibration runs begin from a completely clean state without cross-dataset memory interference.

---

## CLI & Operational Commands

### Run PS-08 Benchmark Evaluation

```powershell
python main.py benchmark --data-dir data/ps08 --output results/ps08_day8
```

### Validate Active Registry & Artifacts

```powershell
python scripts/ops/evaluate_registry.py
```

### Export N-HiTS Day-8 Predictions

```powershell
python scripts/evaluate/export_nhits_day8_predictions.py
```

---

## Verification & Test Suite

Run the automated test suite to verify backend pipelines, registry operations, and GUI formula cards:

```powershell
# Run all tests
.\.venv\Scripts\pytest.exe -q

# Run GUI formula tooltip verification tests specifically
.\.venv\Scripts\pytest.exe -q tests/test_gui_tooltips.py
```

### Verification Gates
Before deployment or commits, verify that:
1. **Public API Contracts**: `src.forecasting` imports cleanly and model registry resolves all candidate families.
2. **Deterministic Routing**: Predictions fail closed when given unregistered satellite IDs.
3. **GUI Telemetry**: Stage 1 candidate table remains uncluttered, and Stage 3 displays interactive formula cards with white background and black typography.

---

## License & Attribution

Developed under the NeuroNav Project. Evolved from research in `kmbeddedd/kkkk`. For inquiries, refer to repository issue tracking and competition documentation under `docs/reference/ps08/`.
