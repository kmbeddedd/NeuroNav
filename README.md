# NeuroNav: GNSS Satellite Orbit & Clock Error Forecasting System

**NeuroNav** is a production-grade machine learning forecasting system for Global Navigation Satellite System (GNSS) broadcast ephemeris errors. It provides high-precision multi-horizon forecasting of satellite spatial coordinate deviations ($X, Y, Z$) and satellite onboard clock bias drift ($\Delta t_{\text{clk}}$), deriving total 3D Euclidean orbit errors for mission operations, autonomous navigation, and satellite positioning integrity.

---

## 1. System Architecture

```text
NeuroNav/
│
├── app/                               # Desktop GUI application package (Tkinter-ready)
│   ├── main.py                        # GUI launcher & interactive display
│   ├── controllers/                   # Decoupled application controllers
│   │   └── inference_controller.py    # Orchestrates datasets & inference execution
│   └── ui/                            # View components
│
├── src/                          # Core production forecasting package
│   ├── config.py                      # Runtime constants & path configurations
│   ├── data.py                        # Telemetry ingestion, validation & feature engineering
│   ├── inference.py                   # High-level runtime inference engine (NeuroNavModel)
│   ├── artifacts.py                   # Scaler serialization & manifest helpers
│   ├── calibration.py                 # Conformal prediction & uncertainty calibration
│   ├── evaluation.py                  # Evaluation metrics & residual normality testing
│   ├── physics.py                     # Physical coordinate transforms (RIC basis)
│   ├── baselines.py                   # Standard baselines (Persistence, Harmonic Ridge)
│   ├── models/                        # Neural architectures
│   │   ├── bilstm.py                  # BiLSTM + GRU recurrent forecaster
│   │   ├── transformer.py             # Hybrid Transformer with temporal attention
│   │   ├── diffusion.py               # Conditional residual diffusion denoiser
│   │   └── losses.py                  # Composite & robust loss functions
│   └── visualization/                 # Visualization engine
│       ├── forecast.py                # Realtime forecast & residual plotting for GUI
│       └── scientific.py              # Scientific evaluation & reporting charts
│
├── models/
│   ├── deploy/                        # Production deployable model bundles
│   │   ├── bilstm/                    # BiLSTM bundle (model.pt + manifest.json)
│   │   └── transformer/               # Hybrid Transformer bundle (model.pt + manifest.json)
│   └── orbitiq_pretrained/            # Pretrained evaluation baseline weights
│
├── data/
│   ├── sample/                        # Representative sample dataset for GUI & smoke tests
│   │   └── sample_gnss_data.csv
│   └── benchmark/                     # Verified cleaned datasets for training/evaluation
│       ├── CLEAN_GNSS_BENCHMARK.csv
│       └── ORBITIQ_BENCHMARK.csv
│
├── scripts/                           # Orchestration CLI tools
│   ├── train/                         # Training workflows (bilstm, transformer, orbitiq, tune)
│   ├── evaluate/                      # Evaluation & comparison scripts
│   ├── benchmark/                     # Official PS-08 competition benchmark runner
│   └── data/                          # Data acquisition, IGS fetch, and audit scripts
│
├── configs/                           # System contracts & model configurations
│   ├── data_contract.json             # Ingestion data contract
│   ├── inference_contract.json        # High-level inference contract
│   ├── bilstm.json                    # BiLSTM hyperparameters
│   └── transformer.json               # Transformer hyperparameters
│
├── research/                          # Historical benchmark & competition material
│   └── ps08/                          # PS-08 dataset and winning GEO Gated MoE checkpoint
│
└── tests/                             # Automated test suite (72 unit and regression tests)
```

---

## 2. Supported Models

| Model | Architecture | Lookback / Horizon | Targets | Uncertainty Supported | Deployment Bundle |
|:---|:---|:---:|:---:|:---:|:---|
| **GNSS-BiLSTM-GRU** | Bidirectional LSTM + GRU with temporal attention pooling | 96 / 96 steps (24h / 24h) | $X, Y, Z, \text{Clock}$ | Deterministic | `models/deploy/bilstm/` |
| **GNSS-Hybrid-Transformer** | Multi-Head Self-Attention + GRU backbone + RevIN | 96 / 96 steps (24h / 24h) | $X, Y, Z, \text{Clock}$ | Probabilistic Gaussian (90% CI) | `models/deploy/transformer/` |
| **GEO Gated MoE** | Causal residual gated Mixture-of-Experts with GRU encoder | 24h physical window | $X, Y, Z, \text{Clock}$ | Deterministic | `research/ps08/models/` |

---

## 3. High-Level Inference API

The runtime inference engine is completely decoupled from training scripts, PyTorch internals, and preprocessing pipelines:

```python
from src.inference import NeuroNavModel

# 1. Load model by shortcut or directory path
model = NeuroNavModel.load("transformer")  # or "bilstm" or "models/deploy/bilstm"

# 2. Predict directly on CSV path or pandas DataFrame
forecast_df = model.predict("data/sample/sample_gnss_data.csv", satellite_id="G01")

# 3. Access structured forecast outputs
print(forecast_df[[
    "forecast_step", "forecast_time", "Satellite_ID",
    "pred_Error_X", "pred_Error_Y", "pred_Error_Z", "pred_Error_Clock",
    "pred_3D_Orbit_Error"
]].head())
```

### Returned Output Schema:
- `forecast_step`: Horizon step index ($1$ to $96$, representing $15\text{m}$ to $24\text{h}$)
- `forecast_time`: UTC timestamp of the predicted epoch
- `Satellite_ID`: PRN / identifier of the vehicle (e.g. `G01`, `R02`)
- `pred_Error_X`, `pred_Error_Y`, `pred_Error_Z`: Predicted orbit coordinate errors in **metres**
- `pred_Error_Clock`: Predicted onboard clock bias in **seconds**
- `pred_3D_Orbit_Error`: Derived Euclidean norm $\sqrt{X^2 + Y^2 + Z^2}$ in **metres**
- `pred_Error_*_low`, `pred_Error_*_high`: Calibrated uncertainty bounds (when using Transformer)

---

## 4. Desktop GUI & CLI Usage

### Running the Desktop GUI:
```bash
python main.py gui
```

### Running the Headless CLI Demo:
```bash
# Predict with BiLSTM
python main.py gui --cli --model bilstm --data data/sample/sample_gnss_data.csv

# Predict with Hybrid Transformer
python main.py gui --cli --model transformer --data data/sample/sample_gnss_data.csv
```

### Direct CLI Prediction:
```bash
python main.py predict --model bilstm --data data/sample/sample_gnss_data.csv --satellite G01 --output forecast_G01.csv
```

### Training Models:
```bash
python main.py train bilstm --data data/benchmark/CLEAN_GNSS_BENCHMARK.csv --epochs 30
python main.py train transformer --data data/benchmark/CLEAN_GNSS_BENCHMARK.csv --epochs 30
```

### Auditing Telemetry Datasets:
```bash
python main.py audit --data data/benchmark/CLEAN_GNSS_BENCHMARK.csv --strict
```

### Running Benchmarks:
```bash
python main.py benchmark --data-dir research/ps08/data
```

---

## 5. Input Data Contract

Input telemetry datasets must conform to `configs/inference_contract.json`:

- **Cadence**: $15\text{ minutes}$ ($96\text{ steps} = 24\text{ hours}$)
- **Minimum History**: At least $96$ contiguous observations per satellite
- **Required Columns**:
  - `Timestamp` or `utc_time`: ISO 8601 or parsable datetime
  - `Satellite_ID`: Vehicle identifier (e.g. `G01` for GPS, `R01` for GLONASS)
  - `Error_X`, `Error_Y`, `Error_Z`: Broadcast minus precise orbit errors (metres)
  - `Error_Clock`: Broadcast minus precise clock bias error (seconds)

---

## 6. Installation & Verification

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install development & testing tools
pip install -r requirements-dev.txt

# Run complete test suite (72 automated tests)
pytest -q
```
