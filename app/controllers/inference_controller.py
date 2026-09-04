"""Controller for NeuroNav Desktop Application, connecting UI to the official forecasting backend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.forecasting.api import (
    calibrate_models,
    clear_registry as api_clear_registry,
    get_all_satellite_selections,
    get_model_metadata,
    get_qq_data,
    get_satellite_selection,
    predict_satellite,
    predict_with_satellite_models,
    reset_to_automatic,
    set_satellite_model,
    train_satellite,
)
from src.forecasting.models import MODEL_REGISTRY, get_available_model_names
from src.forecasting.registry import SatelliteModelRegistry, SatelliteSelection
from src.forecasting.router import PredictionRouter
from src.forecasting.validation import (
    detect_satellite_col,
    detect_time_col,
    load_telemetry_source,
    normalize_dataframe_columns,
)
from src.inference import NeuroNavModel
from src.visualization.forecast import plot_3d_orbit_error, plot_forecast_components


class InferenceController:
    """Decoupled controller connecting Tkinter GUI & CLI to the official forecasting backend."""

    def __init__(
        self,
        models_dir: Union[str, Path] = "models/deploy",
        registry_path: Union[str, Path] = "models/registry/satellite_model_registry.json",
    ):
        self.models_dir = Path(models_dir)
        self.registry = SatelliteModelRegistry(registry_path=registry_path)
        self.router = PredictionRouter(registry=self.registry)

        self.current_model: Optional[Any] = None
        self.current_model_name: str = "bilstm"
        self.current_data: Optional[pd.DataFrame] = None
        self.current_dataset_path: Optional[str] = None
        self.current_forecast: Optional[pd.DataFrame] = None
        self.last_calibration_result: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # Available Models Discovery
    # -------------------------------------------------------------------------
    def get_available_models(self) -> List[Dict[str, str]]:
        """Return list of available models from canonical MODEL_REGISTRY and deploy dir."""
        models = []
        for name in get_available_model_names():
            models.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "path": f"models/registry/artifacts/{name}",
                "description": f"{name} forecasting model",
            })
        if self.models_dir.exists():
            for d in self.models_dir.iterdir():
                if d.is_dir() and (d / "model.pt").exists():
                    if not any(m["id"] == d.name for m in models):
                        models.append({
                            "id": d.name,
                            "name": d.name.upper(),
                            "path": str(d),
                            "description": f"Deployable neural checkpoint {d.name}",
                        })
        return models

    def load_model(self, model_identifier: str) -> Any:
        """Load selected model into memory for legacy/CLI execution."""
        norm = model_identifier.lower().strip()
        self.current_model_name = norm
        try:
            self.current_model = NeuroNavModel.load(norm)
        except Exception:
            self.current_model = norm
        return self.current_model

    def load_dataset(self, file_path: Union[str, Path]) -> Tuple[int, List[str]]:
        """Load and validate telemetry data file, returning row count and satellite list."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        self.current_dataset_path = str(path)
        self.current_data = df

        sat_col = detect_satellite_col(df)
        if sat_col:
            sats = sorted(df[sat_col].astype(str).dropna().unique().tolist())
            if "Satellite_ID" not in df.columns:
                df["Satellite_ID"] = df[sat_col]
        else:
            stem = path.stem.upper()
            if "GEO" in stem:
                sats = ["GEO"]
            elif "MEO" in stem:
                sats = ["MEO-1"]
            else:
                registered = self.registry.get_all_satellites()
                sats = registered if registered else ["SAT_GLOBAL"]
            df["Satellite_ID"] = sats[0]

        # Map column aliases for legacy NeuroNavModel compatibility
        col_map = {
            "x_error (m)": "Error_X", "x_error_m": "Error_X", "x_error": "Error_X",
            "y_error (m)": "Error_Y", "y_error_m": "Error_Y", "y_error": "Error_Y",
            "z_error (m)": "Error_Z", "z_error_m": "Error_Z", "z_error": "Error_Z",
            "satclockerror (m)": "Error_Clock", "clock_error_m": "Error_Clock", "satclockerror": "Error_Clock",
        }
        for src, dst in col_map.items():
            if dst not in df.columns and src in df.columns:
                df[dst] = df[src]

        return len(df), sats

    # -------------------------------------------------------------------------
    # Persistent Satellite Model Memory API
    # -------------------------------------------------------------------------
    def get_all_satellite_memories(self) -> Dict[str, Any]:
        """Retrieve all satellite persistent model selections formatted for GUI display."""
        self.registry.load()
        memories = {}
        for sat_id, sel in self.registry.get_all_selections().items():
            memories[sat_id] = self._format_satellite_entry(sel)
        return memories

    def get_model_for_satellite(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored selection metadata and validation metrics for a given satellite."""
        self.registry.load()
        sel = self.registry.get_selection(satellite_id)
        if sel is None:
            return None
        return self._format_satellite_entry(sel)

    def _format_satellite_entry(self, sel: SatelliteSelection) -> Dict[str, Any]:
        """Format SatelliteSelection into the rich dictionary structure expected by GUI."""
        sat_id = sel.satellite_id
        c_scores = dict(sel.candidate_scores)
        if not c_scores and sel.selected_model:
            c_scores[sel.selected_model] = sel.selection_score

        c_results = getattr(sel, "candidate_results", {}) or {}
        v_metrics: Dict[str, Any] = {}

        for m_id, score in c_scores.items():
            cand_info = c_results.get(m_id, {})
            p1 = cand_info.get("priority_1", {})
            p2 = cand_info.get("priority_2", {})
            p3 = cand_info.get("priority_3", {})
            supp = cand_info.get("supplementary", {})

            # Fallback to winning priorities if m_id is the selected winner
            if not p1 and m_id == sel.selected_model:
                p1 = getattr(sel, "winning_priority_1", {}) or {}
            if not p2 and m_id == sel.selected_model:
                p2 = getattr(sel, "winning_priority_2", {}) or {}
            if not p3 and m_id == sel.selected_model:
                p3 = getattr(sel, "winning_priority_3", {}) or {}
            if not supp and m_id == sel.selected_model:
                supp = getattr(sel, "supplementary_diagnostics", {}) or {}

            w_avg = float(p1.get("W", {}).get("average", score))
            p_avg = float(p1.get("p_value", {}).get("average", 1.0))
            h0_mean = 0 if p_avg >= 0.05 else 1
            mean_agg = float(p2.get("mean", {}).get("aggregate", 0.0))
            std_agg = float(p2.get("std", {}).get("aggregate", 0.0))
            mae_3d = float(supp.get("mae", {}).get("3D", supp.get("mae_3d", 0.0)))
            rmse_3d = float(supp.get("rmse", {}).get("3D", supp.get("rmse_3d", 0.0)))
            mae_clk = float(supp.get("mae", {}).get("Clock", supp.get("mae_clock", 0.0)))

            per_target: Dict[str, Any] = {}
            for tgt in ("X", "Y", "Z", "Clock"):
                t_w = float(p1.get("W", {}).get(tgt, w_avg))
                t_p = p1.get("p_value", {}).get(tgt, p_avg)
                t_h0 = int(p1.get("hypothesis_result", {}).get(tgt, 0 if float(t_p) >= 0.05 else 1))
                t_mean = float(p2.get("mean", {}).get(tgt, 0.0))
                t_std = float(p2.get("std", {}).get(tgt, 0.1))
                t_mae = float(supp.get("mae", {}).get(tgt, 0.0))
                t_rmse = float(supp.get("rmse", {}).get(tgt, 0.0))
                t_r2 = float(supp.get("r2", {}).get(tgt, 0.0))
                t_max_ae = float(supp.get("max_ae", {}).get(tgt, 0.0))

                # Check if Q-Q data has exact empirical sample quantiles/residuals
                residuals_sample = None
                try:
                    qq_file = Path("reports/calibration/official_competition_run/qq_data") / f"{sat_id}_{m_id}_qq.json"
                    if qq_file.exists():
                        qq_data = json.loads(qq_file.read_text(encoding="utf-8"))
                        if "qq_details" in qq_data and tgt in qq_data["qq_details"]:
                            residuals_sample = qq_data["qq_details"][tgt].get("sample_quantiles")
                except Exception:
                    pass

                per_target[tgt] = {
                    "shapiro_w": t_w,
                    "shapiro_p": t_p,
                    "h0_result": t_h0,
                    "bias": t_mean,
                    "std": t_std,
                    "mae": t_mae,
                    "rmse": t_rmse,
                    "r2": t_r2,
                    "max_ae": t_max_ae,
                    "residuals": residuals_sample,
                }

            v_metrics[m_id] = {
                "composite_score": w_avg,
                "shapiro_w_mean": w_avg,
                "shapiro_p_mean": p_avg,
                "h0_result_mean": h0_mean,
                "mean_res_mean": mean_agg,
                "std_res_mean": std_agg,
                "mae_3d": mae_3d,
                "rmse_3d": rmse_3d,
                "mae_clock": mae_clk,
                "per_target": per_target,
            }

        orbit = getattr(sel, "orbit_type", "GEO")
        if not orbit or str(orbit).upper() == "UNKNOWN":
            orbit = "GEO" if "GEO" in sat_id.upper() else "MEO"

        return {
            "satellite_id": sat_id,
            "orbit_type": orbit,
            "selected_model": sel.selected_model,
            "selection_mode": sel.selection_mode,
            "selection_policy": getattr(sel, "selection_policy", "official_competition"),
            "model_version": sel.model_version,
            "selection_score": sel.selection_score,
            "score": sel.selection_score,
            "shapiro_w": sel.selection_score,
            "candidate_models": c_scores,
            "candidate_scores": c_scores,
            "validation_metrics": v_metrics,
            "model_artifact": sel.model_artifact,
            "selected_at": sel.selected_at,
        }

    # -------------------------------------------------------------------------
    # Stage 1 Calibration & Manual Overrides
    # -------------------------------------------------------------------------
    def calibrate_satellite_models(
        self,
        train_data: Union[pd.DataFrame, str, Path],
        test_data: Union[pd.DataFrame, str, Path],
        target_satellite_id: Optional[str] = None,
        orbit_type: str = "GEO",
    ) -> Dict[str, Any]:
        """Stage 1: Train & evaluate candidate models across satellites using the Official Competition Hierarchy."""
        if target_satellite_id:
            res = train_satellite(
                dataset=train_data,
                test_dataset=test_data,
                satellite_id=target_satellite_id,
                orbit_type=orbit_type,
            )
            self.registry.load()
            sat_id = res.get("satellite_id", target_satellite_id)
            formatted_entry = self.get_model_for_satellite(sat_id) or {}
            out = {
                "satellites": {
                    sat_id: {
                        "entry": formatted_entry,
                        "metrics": formatted_entry.get("validation_metrics", {}),
                    }
                },
                "summary_path": str(Path("reports/calibration")),
            }
        else:
            summary = calibrate_models(
                train_data=train_data,
                test_data=test_data,
            )
            self.registry.load()
            sats_dict: Dict[str, Any] = {}
            for sat_id in self.registry.get_all_satellites():
                formatted_entry = self.get_model_for_satellite(sat_id) or {}
                sats_dict[sat_id] = {
                    "entry": formatted_entry,
                    "metrics": formatted_entry.get("validation_metrics", {}),
                }
            out = {
                "satellites": sats_dict,
                "summary_path": summary.get("run_id", ""),
            }

        self.last_calibration_result = out
        return out

    def select_model_for_satellite(self, satellite_id: str, selection: str) -> Dict[str, Any]:
        """Set manual model override for satellite in persistent memory."""
        res = set_satellite_model(satellite_id=satellite_id, model_name=selection)
        self.registry.load()
        return self.get_model_for_satellite(satellite_id) or res

    def reset_model_for_satellite(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Reset satellite selection back to automatic data-driven winner."""
        res = reset_to_automatic(satellite_id=satellite_id)
        self.registry.load()
        return self.get_model_for_satellite(satellite_id)

    def clear_registry(self) -> None:
        """Clear all satellite model entries and reset state completely."""
        self.registry.clear()
        try:
            api_clear_registry()
        except Exception:
            pass
        self.current_data = None
        self.current_dataset_path = None
        self.current_model = None
        self.current_model_name = "bilstm"
        self.last_calibration_result = None
        self.current_forecast = None

    # -------------------------------------------------------------------------
    # Stage 2 Forecasting & Arbitrary Timestamp Lookup
    # -------------------------------------------------------------------------
    def predict_with_satellite_models(
        self,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        satellite_id: Optional[str] = None,
        explicit_model: Optional[str] = None,
        target_times: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Run heterogeneous satellite forecasting routing each satellite to its stored memory."""
        target_data = data if data is not None else self.current_data
        if target_data is None:
            raise RuntimeError("No dataset available for prediction. Load dataset first.")

        self.registry.load()

        # If explicit model override was supplied for single satellite
        if explicit_model and satellite_id:
            set_satellite_model(satellite_id, explicit_model, reason="GUI explicit forecast request")
            self.registry.load()

        if satellite_id:
            df_norm = load_telemetry_source(target_data)
            sat_df = df_norm.get(satellite_id) if isinstance(df_norm, dict) else None
            if sat_df is None:
                if isinstance(target_data, pd.DataFrame):
                    sat_df = normalize_dataframe_columns(target_data.copy())
                else:
                    sat_df = normalize_dataframe_columns(pd.read_csv(target_data))

            forecast_df = predict_satellite(
                satellite_id=satellite_id,
                history_data=sat_df,
                compute_ric=True,
                target_times=target_times,
            )
        else:
            forecast_df = predict_with_satellite_models(
                data=target_data,
                compute_ric=True,
                target_times=target_times,
            )

        # Standardize column naming for GUI compatibility
        if "predicted_X" in forecast_df.columns:
            forecast_df["pred_Error_X"] = forecast_df["predicted_X"]
            forecast_df["pred_Error_Y"] = forecast_df["predicted_Y"]
            forecast_df["pred_Error_Z"] = forecast_df["predicted_Z"]
            forecast_df["pred_Error_Clock"] = forecast_df["predicted_Clock"]
        if "pred_3D_Orbit_Error" in forecast_df.columns and "predicted_3D_Error" not in forecast_df.columns:
            forecast_df["predicted_3D_Error"] = forecast_df["pred_3D_Orbit_Error"]
        elif "predicted_3D_Error" in forecast_df.columns and "pred_3D_Orbit_Error" not in forecast_df.columns:
            forecast_df["pred_3D_Orbit_Error"] = forecast_df["predicted_3D_Error"]
        if "Satellite_ID" not in forecast_df.columns and "satellite_id" in forecast_df.columns:
            forecast_df["Satellite_ID"] = forecast_df["satellite_id"]
        if "forecast_time" not in forecast_df.columns and "timestamp" in forecast_df.columns:
            forecast_df["forecast_time"] = forecast_df["timestamp"]

        self.current_forecast = forecast_df
        return forecast_df

    def run_forecast(self, satellite_id: Optional[str] = None) -> pd.DataFrame:
        """Execute forecast inference for current dataset and loaded model mode."""
        if self.current_data is None:
            raise RuntimeError("No dataset is currently loaded. Call load_dataset() first.")

        # Check if current_model has predict
        if hasattr(self.current_model, "predict"):
            res = self.current_model.predict(self.current_data, satellite_id=satellite_id, return_dataframe=True)
            self.current_forecast = res
            return res

        return self.predict_with_satellite_models(data=self.current_data, satellite_id=satellite_id)

    def export_forecast(self, export_path: Union[str, Path]) -> Path:
        """Export latest forecast predictions to CSV."""
        if self.current_forecast is None:
            raise RuntimeError("No forecast available to export. Run forecast first.")
        out = Path(export_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.current_forecast.to_csv(out, index=False)
        return out

    def generate_plots(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        satellite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate forecast component and 3D orbit error figures."""
        if self.current_forecast is None:
            raise RuntimeError("No forecast available to plot. Run forecast first.")

        sat = satellite_id
        if sat is None:
            for cand in ("Satellite_ID", "satellite_id"):
                if cand in self.current_forecast.columns:
                    sat = str(self.current_forecast[cand].iloc[0])
                    break

        comp_path = Path(output_dir) / f"forecast_components_{sat}.png" if output_dir else None
        orbit_path = Path(output_dir) / f"orbit_3d_error_{sat}.png" if output_dir else None

        fig_comp = plot_forecast_components(self.current_forecast, satellite_id=sat, output_path=comp_path)
        fig_orbit = plot_3d_orbit_error(self.current_forecast, satellite_id=sat, output_path=orbit_path)

        return {
            "forecast_components": fig_comp,
            "orbit_3d": fig_orbit,
        }
