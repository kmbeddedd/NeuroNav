import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DEFAULT_DATA_PATH = str(PROJECT_ROOT / 'data' / 'benchmark' / 'CLEAN_GNSS_BENCHMARK.csv')
SAMPLE_DATA_PATH = str(PROJECT_ROOT / 'data' / 'sample' / 'sample_gnss_data.csv')
ORBITIQ_DATA_PATH = str(PROJECT_ROOT / 'data' / 'benchmark' / 'ORBITIQ_BENCHMARK.csv')
ORBITIQ_OUTPUT_DIR = str(PROJECT_ROOT / 'models' / 'orbitiq_pipeline')
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'models' / 'deploy' / 'bilstm')

# Scientific & Pipeline Constants
TRAIN_END_DATE = '2026-01-08 00:00:00'
TOTAL_TIMESTEPS_PER_SAT = 768
SEQ_LEN = 96
FORECAST_HORIZON = 96
OUTLIER_THRESHOLD_3D = 50000.0
SPIKE_THRESHOLD = 1.5
SP3_CLOCK_SENTINEL_SECONDS = 0.999999999999
EXPECTED_CADENCE_MINUTES = 15

TARGET_COLS_5 = ['Error_X', 'Error_Y', 'Error_Z', '3D_Orbit_Error', 'Error_Clock']
TARGET_COLS_4 = ['Error_X', 'Error_Y', 'Error_Z', 'Error_Clock']
FEATURE_COLS_PYTORCH = [
    'Error_X', 'Error_Y', 'Error_Z', 'Error_Clock',
    'time_sin', 'time_cos',
    'Error_X_roll_mean', 'Error_Y_roll_mean', 'Error_Z_roll_mean', 'Error_Clock_roll_mean'
]
HORIZON_MAP = {
    '15 min': 1,
    '30 min': 2,
    '1 hour': 4,
    '2 hours': 8,
    '6 hours': 24,
    '12 hours': 48,
    '24 hours': 96
}
DEFAULT_SEED = 42

TRANSFORMER_DEFAULTS = {
    'epochs': 30,
    'batch_size': 32,
    'd_model': 64,
    'embedding_dim': 8,
    'nhead': 4,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'lr_patience': 3
}

DIFFUSION_DEFAULTS = {
    'epochs': 80,
    'steps': 100,
    'beta_start': 1e-4,
    'beta_end': 0.02,
    'learning_rate': 1e-5,
    'weight_decay': 1e-5
}

def resolve_device(name: str = 'auto') -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)
