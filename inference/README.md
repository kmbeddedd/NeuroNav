# NeuroNav inference entry points

This directory contains operational adapters for running forecasts. The production
implementation is kept under `src/forecasting/inference/`, and the stable callable API
is exposed by `src.forecasting`. Keeping the CLI thin prevents model loading, registry
routing, and feature preparation from diverging between Python callers and operators.

Run a registered single-satellite model with:

```powershell
python -m inference.predict `
  --satellite GEO `
  --orbit-type GEO `
  --history data/ps08/DATA_GEO_Train.csv `
  --horizon-steps 96 `
  --output reports/inference/GEO_forecast.csv
```

The command validates the history, confirms an active registry selection, delegates to
`src.forecasting.predict_satellite`, and writes the canonical prediction contract. It
fails explicitly for invalid input, missing selections, missing artifacts, or artifact
loading errors. Use `--no-ric` when R/I/C output is not required.
