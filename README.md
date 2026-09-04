# NeuroNav

NeuroNav forecasts satellite orbit and clock errors from independently supplied GNSS
time series. The production backend validates each satellite, evaluates eligible model
candidates on held-out observations, selects a winner with the official competition
hierarchy, records that decision in a persistent registry, and routes later forecasts to
the selected artifact without a silent fallback.

The repository evolved from the earlier `kmbeddedd/kkkk` research project. Its original
global BiLSTM/Transformer pipeline remains available as an explicit compatibility layer;
new integrations should use `src.forecasting`.

## Production flow

```text
single-satellite CSV upload
        -> schema and cadence validation
        -> explicit satellite ID and orbit type
        -> optional nominal/provided orbital-state physics
        -> per-model eligibility check
        -> candidate training on history only
        -> held-out official P1/P2/P3 evaluation
        -> winning artifact and metadata
        -> satellite model registry
        -> strict prediction router
        -> stable Python API for the Tkinter frontend
```

Production code does not import benchmark or experimental modules. Unknown satellites,
missing selections, corrupt registries, and missing artifacts fail with actionable errors.

## Repository layout

```text
app/                               Tkinter controllers and application dispatchers
app.py                             Desktop GUI application launcher (`python app.py`)
gui/                               Mission control GUI interface (`gui/gui_app.py`)
configs/
  contracts/                       versioned input and output contracts
  models/                          legacy deploy-bundle configurations
  promotion/                       general baseline/promotion policy
data/ps08/                         canonical supplied GEO/MEO CSV datasets
docs/
  reference/ps08/                  competition and dataset reference documents
  research/                        forecasting background notes
models/
  registry/                        satellite selections and routed artifacts
  deploy/                          legacy GUI-compatible global bundles
  orbitiq_pretrained/              research comparison artifacts
reports/                           calibration and evaluation evidence
inference/                         operational prediction CLI and usage notes
research/
  experiments/                     ablation code
  orbitiq/                          OrbitIQ comparison implementation
  ps08/                             historical benchmark implementation
  lineage/                          kkkk -> NeuroNav history tooling
scripts/
  benchmark/                       compatibility CLI wrappers
  data/                            data audit CLI
  evaluate/                        evaluation/export CLIs
  ops/                             registry and runtime verification
  train/                           compatibility global-model trainers
src/
  forecasting/
    api.py                          public backend interface
    data/validation.py              upload validation and cadence metadata
    features/core.py                versioned train/serve feature manifests
    physics.py                      orbital state, RIC, solar geometry, SISRE
    models/                         satellite model adapters and factory
    training/calibration.py         candidate calibration and selection
    evaluation/official.py          official evaluation boundary
    registry/store.py               atomic persistent selections
    inference/router.py             fail-closed runtime routing
  compat/global_forecasting/        inherited global training/inference stack
  *.py                              backward-compatible import shims
tests/                              production, compatibility, and research tests
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Model binaries use Git LFS:

```powershell
git lfs install
git lfs pull
```

## Data contract

The satellite-specific contract is
[`configs/contracts/ps08_satellite_data.json`](configs/contracts/ps08_satellite_data.json).
The supplied datasets are under `data/ps08/`:

| Satellite | Calibration history | Held-out observations |
|---|---|---|
| GEO | `DATA_GEO_Train.csv` | `DATA_GEO_Test.csv` |
| MEO-1 | `DATA_MEO_Train.csv` | `DATA_MEO_Test.csv` |
| MEO-2 | `DATA_MEO_Train2.csv` | `DATA_MEO_Test2.csv` |

Canonical internal fields are:

```text
utc_time
x_error_m
y_error_m
z_error_m
clock_error_m
```

In the PS-08 pipeline all four error values, including `clock_error_m`, are in metres.
The legacy GUI/deploy contract is separately retained at
`configs/contracts/legacy_gui_inference.json`; its `Error_Clock` field is in seconds and
must not be mixed with the PS-08 range-error field without an explicit conversion.

The default pipeline removes exact duplicate epochs deterministically, does not fabricate
training observations, does not resample irregular data unless explicitly requested, and
never uses Day-8 targets as training inputs.

## Public backend API

Import the stable facade:

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

### Validate an upload

```python
from src.forecasting import validate_satellite_dataset

report = validate_satellite_dataset(
    "data/ps08/DATA_GEO_Train.csv",
    satellite_id="GEO",
    orbit_type="GEO",
)
```

### Calibrate and register one satellite

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
print(result["selected_model"])
```

`physics_mode` is one of:

- `none`: no orbital-state-derived RIC or solar features.
- `nominal`: documented analytical GEO/MEO approximation.
- `provided`: caller supplies a timestamped orbital-state DataFrame; the state artifact is
  persisted with the selected model and restored by the router.

### Forecast with the registered winner

```python
from src.forecasting import predict_satellite

forecast = predict_satellite(
    satellite_id="GEO",
    history_data="data/ps08/DATA_GEO_Train.csv",
    horizon_steps=96,
)
```

The canonical output schema is defined in
[`configs/contracts/satellite_prediction.json`](configs/contracts/satellite_prediction.json).
It contains timestamps, ECEF X/Y/Z and clock range-error forecasts, derived 3D orbit error,
model provenance, and optional R/I/C components.

## Inference process

The production inference path has one implementation and two supported entry points:

- Python and GUI integrations call `src.forecasting.predict_satellite` or
  `predict_with_satellite_models`.
- Operators can use the thin CLI in `inference/predict.py`.

```text
historical satellite CSV
        -> validate and normalize schema, timestamps, units, and cadence
        -> resolve the satellite's active selection from the registry
        -> verify and load the exact versioned model artifact
        -> restore the registered physics/state provider when configured
        -> construct future timestamps from registered or requested cadence
        -> run the selected model with its persisted preprocessing/features
        -> derive 3D orbit error and optional R/I/C components
        -> return the canonical prediction schema with model provenance
```

Inference is fail-closed. An unknown satellite, missing selection, missing artifact,
invalid history, or incompatible artifact raises an actionable error; the router does
not silently substitute another model.

Run single-satellite inference from the repository root:

```powershell
python -m inference.predict `
  --satellite GEO `
  --orbit-type GEO `
  --history data/ps08/DATA_GEO_Train.csv `
  --horizon-steps 96 `
  --output reports/inference/GEO_forecast.csv
```

The CLI validates the input before delegating to the public backend API. Omit `--output`
to emit CSV to standard output, use `--step-interval-minutes` to explicitly override the
registered cadence, and use `--no-ric` to disable R/I/C derivation. Full usage is in
[`inference/README.md`](inference/README.md).

### Multi-satellite calibration and routing

```python
from src.forecasting import calibrate_models, predict_with_satellite_models

summary = calibrate_models(
    train_data="data/ps08",
    test_data="data/ps08",
    run_id="official_competition_run",
)

forecast = predict_with_satellite_models("data/ps08")
print(forecast[["satellite_id", "model_used"]].drop_duplicates())
```

## Models

The satellite model factory currently registers:

- persistence;
- Harmonic Ridge, including RIC and SRP variants;
- Random Forest, including RIC and SRP variants;
- Gaussian Process;
- BiLSTM-GRU;
- Transformer;
- GEO Gated Mixture of Experts;
- Decoupled Clock;
- N-HiTS.

Harmonic Ridge and Random Forest use the versioned unified feature manifest. Other model
families retain their established feature stacks to preserve trained behavior.

## Official model selection

The authoritative policy is exposed through `src.forecasting.evaluation.official`.
Residuals are `predicted - actual` and the significance level is `alpha = 0.05`.

1. **P1:** maximize the equal-weighted average of Shapiro-Wilk W for X, Y, Z, and
   Clock. Each target contributes exactly 25%.
2. **P2:** only when P1 is tied within `1e-4`, minimize aggregate absolute residual
   bias and then aggregate residual standard deviation.
3. **P3:** only when P1 and P2 remain tied, minimize Q-Q outlier count and then maximum
   quantile discrepancy.

MAE, RMSE, 3D orbit error, and SISRE are diagnostics only and never override P1/P2/P3.
The historical PS-08 benchmark under `research/ps08/` has its own research aggregation
and must not be used as the runtime selection implementation.

## Registry and artifacts

Active selections live in:

```text
models/registry/satellite_model_registry.json
```

The router first verifies the exact artifact recorded by the selection. It does not fall
back to a BiLSTM or a neighboring checkpoint. Per-satellite calibration also supports:

```text
models/registry/artifacts/satellites/<satellite_id>/
  model.pt or model.joblib
  metadata.json
  feature_manifest.json
  evaluation.json
  orbital_state.csv                 only for provided-state physics
```

Generated run output belongs under `reports/` or the ignored `results/` workspace, not in
the repository root.

## Desktop GUI Application (Mission Control)

NeuroNav includes a high-density, mission control desktop application built with Python Tkinter and Matplotlib (`gui/gui_app.py`), styled after the modern Stitch design system.

### Launching the Application

Launch the desktop interface from the project root:

```powershell
python app.py
```

Alternatively, launch via the unified CLI dispatcher or Python module:

```powershell
python main.py gui
# or
python -m app.main
```

### Architecture & ML Communication Boundary

The frontend strictly adheres to a decoupled Model-View-Controller architecture:
- **Presentation Layer (`gui/gui_app.py`)**: Renders interactive telemetry controls, dataset inspection tables, error output metrics badges, and Matplotlib statistical diagnostics. The GUI never performs heuristic manipulation or modifies forecast values.
- **Controller Layer (`app/controllers/inference_controller.py`)**: Decouples UI interactions and worker threads from the forecasting backend, coordinating asynchronous model training, prediction, and registry queries.
- **Official Forecasting API (`src.forecasting.api`)**: Clean entry point for all ML operations (`train_satellite`, `predict_satellite`, `predict_with_satellite_models`, `clear_registry`). Telemetry data flows directly into the backend `CalibrationPipeline` and `PredictionRouter`, and genuine model inference outputs are returned without alteration.

### Core Workflow & Features

#### 1. Stage 1: Zero-Leakage Satellite Model Calibration & Selection
- **Satellite Identification & Orbit Selector**:
  - Enter the target satellite identifier (e.g., `GEO-01`, `MEO-01`).
  - Select orbit classification via explicit radio buttons: **`GEO`** (Geostationary) or **`MEO`** (Medium Earth Orbit). This information directly informs regime-aware physics and model features (such as solar radiation pressure, orbital harmonics, and model eligibility like `geo_moe`).
- **Data Ingestion**:
  - Upload separate 7-day training history and held-out 8th-day ground truth evaluation CSVs.
  - Zero data leakage guarantee: candidate models are trained strictly on history without visibility into evaluation data.
- **Automated Candidate Model Calibration**:
  - Trains and evaluates all eligible candidate models (`Harmonic Ridge`, `Random Forest`, `Gaussian Process`, `BiLSTM-GRU`, `Transformer`, `GEO MoE`, `Decoupled Clock`, `N-HiTS`, `Persistence`).
- **Official Competition Hierarchy Ranking**:
  - Models are ranked strictly by the competition's official priority hierarchy:
    - **P1**: Maximum mean Shapiro-Wilk $W$ normality statistic across all 4 targets ($X, Y, Z, \text{Clock}$), with equal $25\%$ weighting.
    - **P2 / P3**: Absolute residual bias, residual standard deviation, and Q-Q outlier counts.
  - The Candidate Ranking Table displays the Shapiro-Wilk $W$ statistic, exact two-sided $p$-values formatted cleanly (preserving scientific notation for small values), $H_0$ normality rejection status ($\alpha = 0.05$), and residual bias.
  - The winning candidate is automatically recorded in persistent model memory (`models/registry/satellite_model_registry.json`), with manual override capability.

#### 2. Stage 2: Satellite-Specific 8th-Day Forecast & Arbitrary Timestamp Error Inspector
- **Persistent Model Selection**:
  - Multi-satellite dropdown auto-populates all satellites currently registered in model memory.
  - Switching satellites instantly loads the satellite's assigned winning model, version, calibration score, and cadence.
- **96-Step Forecast Time Series**:
  - Routes telemetry to the selected satellite's model via `PredictionRouter` without cross-satellite contamination.
  - Generates the full 8th-day horizon forecast (e.g. 96 steps at 15-min intervals).
  - Populates the Forecast Output Table with exact predictions for $\Delta X$ (m), $\Delta Y$ (m), $\Delta Z$ (m), $\Delta \text{Clock}$ (s/m), and $3\text{D Orbit Error}$ (m).
- **Arbitrary Timestamp Error Inspector**:
  - Enter an arbitrary target timestamp (e.g. `12:15:00`, `14:30:00`, `08:00:00`, or full ISO timestamp `2026-01-08 12:15:00`).
  - The backend executes instantaneous prediction or temporal lookup using the calibrated satellite model.
  - Displays the predicted orbit error vector ($\Delta X, \Delta Y, \Delta Z$), clock error, and total 3D error directly on dedicated mission control telemetry badges.

#### 3. Stage 3: Residual Probability & Error Distribution Diagnostics
- **Interactive Matplotlib Visualizations**:
  - **Residual Histograms & KDE**: Visualizes error distributions for $X$, $Y$, $Z$, and $\text{Clock}$ with kernel density estimates overlaid on theoretical normal distribution curves.
  - **Quantile-Quantile (Q-Q) Plots**: Standardized residuals plotted against normal theoretical quantiles with a $45^\circ$ reference line to visually inspect tail kurtosis and outlier behavior.
  - **Normality Telemetry Cards**: Displays component-level Shapiro-Wilk $W$, $p$-values, skewness, kurtosis, and standard deviations.

#### 4. Memory Management & Deep Reset
- Clicking **`Clear Memory 🗑`** performs a deep reset across:
  - Persistent satellite registry (`models/registry/satellite_model_registry.json`),
  - Active controller session state and cached DataFrames,
  - Stage 1 and Stage 2 UI tables, plots, and dropdown selectors.
- Ensures subsequent dataset uploads start with a completely clean environment without residual cross-dataset memory interference.

## CLI and research tools

Run the historical PS-08 benchmark:

```powershell
python main.py benchmark --data-dir data/ps08 --output results/ps08_day8
```

Evaluate the active registry and artifacts:

```powershell
python scripts/ops/evaluate_registry.py
```

Generate the N-HiTS Day-8 export report:

```powershell
python scripts/evaluate/export_nhits_day8_predictions.py
```

Research implementations live under `research/`; compatibility command modules remain in
`scripts/benchmark/` and `scripts/evaluate/` so established imports continue to work.

## Verification

```powershell
pytest -q
python -c "import src.forecasting as f; print(sorted(f.MODEL_REGISTRY))"
python -c "from src.forecasting import get_all_satellite_selections; print(get_all_satellite_selections())"
```

The deployment gate is stricter than passing unit tests: registry paths must resolve,
manifests must match artifacts, train/serve features must remain equivalent, and the
end-to-end upload, calibration, retrieval, and prediction path must pass.
