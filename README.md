# Gaitonde

## Simple GNSS BiLSTM

This project trains one small multivariate BiLSTM for each supplied dataset pair: GEO, MEO, and MEO2. Each model predicts the next four error values from the previous 12 observations and recursively forecasts the supplied test period.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python train_bilstm.py
```

Outputs are written to `outputs/`:

- `*_bilstm.pt`: trained PyTorch checkpoints
- `*_predictions.csv`: timestamps, ground truth, and predictions
- `metrics.json`: MAE and RMSE results
