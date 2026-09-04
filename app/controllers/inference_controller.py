"""Controller for NeuroNav Desktop Application, decoupling UI from inference engine."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.inference import NeuroNavModel
from src.visualization.forecast import plot_3d_orbit_error, plot_forecast_components


class InferenceController:
    """Orchestrates model loading, dataset selection, and forecast execution for the GUI."""

    def __init__(self, models_dir: str | Path = 'models/deploy'):
        self.models_dir = Path(models_dir)
        self.current_model: Optional[NeuroNavModel] = None
        self.current_data: Optional[pd.DataFrame] = None
        self.current_dataset_path: Optional[str] = None
        self.current_forecast: Optional[pd.DataFrame] = None

    def get_available_models(self) -> List[Dict[str, str]]:
        """Return list of deployable models discovered on disk."""
        models = []
        if self.models_dir.exists():
            for d in self.models_dir.iterdir():
                if d.is_dir() and (d / 'model.pt').exists():
                    manifest_f = d / 'manifest.json'
                    name = d.name
                    desc = f"{d.name} model"
                    if manifest_f.exists():
                        import json
                        try:
                            meta = json.loads(manifest_f.read_text(encoding='utf-8'))
                            name = meta.get('model_name', d.name)
                            desc = meta.get('description', desc)
                        except Exception:
                            pass
                    models.append({
                        'id': d.name,
                        'name': name,
                        'path': str(d),
                        'description': desc,
                    })
        return models

    def load_model(self, model_identifier: str) -> NeuroNavModel:
        """Load selected model into memory."""
        self.current_model = NeuroNavModel.load(model_identifier)
        return self.current_model

    def load_dataset(self, file_path: str | Path) -> Tuple[int, List[str]]:
        """Load and validate telemetry data file, returning row count and satellite list."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        self.current_dataset_path = str(path)
        self.current_data = df

        if 'Satellite_ID' in df.columns:
            sats = sorted(df['Satellite_ID'].astype(str).unique().tolist())
        else:
            sats = []

        return len(df), sats

    def run_forecast(self, satellite_id: Optional[str] = None) -> pd.DataFrame:
        """Execute forecast inference for the current dataset and loaded model."""
        if self.current_model is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")
        if self.current_data is None:
            raise RuntimeError("No dataset is currently loaded. Call load_dataset() first.")

        forecast_df = self.current_model.predict(
            self.current_data,
            satellite_id=satellite_id,
            return_dataframe=True,
        )
        self.current_forecast = forecast_df
        return forecast_df

    def export_forecast(self, export_path: str | Path) -> Path:
        """Export latest forecast predictions to CSV."""
        if self.current_forecast is None:
            raise RuntimeError("No forecast available to export. Run forecast first.")
        out = Path(export_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.current_forecast.to_csv(out, index=False)
        return out

    def generate_plots(
        self,
        output_dir: Optional[str | Path] = None,
        satellite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate forecast component and 3D orbit error figures."""
        if self.current_forecast is None:
            raise RuntimeError("No forecast available to plot. Run forecast first.")

        sat = satellite_id
        if sat is None and 'Satellite_ID' in self.current_forecast.columns:
            sat = self.current_forecast['Satellite_ID'].iloc[0]

        comp_path = Path(output_dir) / f'forecast_components_{sat}.png' if output_dir else None
        orbit_path = Path(output_dir) / f'orbit_3d_error_{sat}.png' if output_dir else None

        fig_comp = plot_forecast_components(self.current_forecast, satellite_id=sat, output_path=comp_path)
        fig_orbit = plot_3d_orbit_error(self.current_forecast, satellite_id=sat, output_path=orbit_path)

        return {
            'forecast_components': fig_comp,
            'orbit_3d': fig_orbit,
        }
