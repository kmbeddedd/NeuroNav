"""Satellite Model Registry and Persistent Memory Store for NeuroNav."""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.adapters import MODEL_ADAPTER_CLASSES, get_adapter_by_id

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "models" / "registry" / "satellite_model_registry.json"


class SatelliteModelRegistry:
    """Persistent storage and management of satellite-specific model selections."""

    def __init__(self, registry_path: Optional[str | Path] = None):
        self.path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._data: Dict[str, Any] = {
            "version": "1.0.0",
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "satellites": {},
        }
        self.load()

    def load(self) -> None:
        """Load registry from disk or initialize empty if not found or corrupted."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, dict) and "satellites" in content:
                        self._data = content
                    else:
                        self._data["satellites"] = content if isinstance(content, dict) else {}
            except Exception:
                # If corrupted, preserve default structure
                pass
        else:
            self._ensure_dir()
            self.save()

    def _ensure_dir(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Save registry atomically and safely to disk."""
        self._ensure_dir()
        self._data["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp_path = self.path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            try:
                os.replace(temp_path, self.path)
            except (PermissionError, OSError):
                # Fallback to direct write if Windows has a temporary lock
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
        except Exception:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)

    def clear(self) -> None:
        """Clear all registered satellite memories and persist empty registry."""
        self._data["satellites"] = {}
        self.save()

    def get_all_satellites(self) -> List[str]:
        """Return list of all registered satellite IDs."""
        return sorted(list(self._data.get("satellites", {}).keys()))

    def list_registered_satellites(self) -> List[str]:
        """Alias for get_all_satellites."""
        return self.get_all_satellites()

    def get_satellite_entry(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Return full metadata dictionary for a satellite if it exists."""
        return self._data.get("satellites", {}).get(satellite_id)

    def get_selected_model(self, satellite_id: str) -> Optional[str]:
        """Return the preferred model_id for the satellite, or None if not calibrated."""
        entry = self.get_satellite_entry(satellite_id)
        if entry:
            return entry.get("selected_model")
        return None

    def save_calibration_result(
        self,
        satellite_id: str,
        best_model: str,
        score: float,
        candidate_models: Dict[str, float],
        validation_metrics: Dict[str, Any],
        model_version: str = "1.0.0",
        model_artifact: Optional[str] = None,
        dataset_hash: str = "",
        force: bool = False,
    ) -> Dict[str, Any]:
        """Record calibration results for a satellite.
        
        If an existing satellite has selection_mode == 'manual' and force is False,
        the manual selection is preserved, while candidates and validation metrics are updated.
        """
        existing = self.get_satellite_entry(satellite_id)
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if existing and existing.get("selection_mode") == "manual" and not force:
            # Preserve manual override, record recommendation
            existing["candidate_models"] = candidate_models
            existing["validation_metrics"] = validation_metrics
            existing["recommended_automatic_model"] = best_model
            existing["recommended_automatic_score"] = float(score)
            existing["dataset_hash"] = dataset_hash
            existing["updated_at"] = now_str
            self.save()
            return existing

        entry = {
            "satellite_id": satellite_id,
            "selected_model": best_model,
            "selection_mode": "automatic",
            "score": float(score),
            "candidate_models": candidate_models,
            "validation_metrics": validation_metrics,
            "model_version": model_version,
            "model_artifact": model_artifact or f"models/deploy/{best_model}/",
            "dataset_hash": dataset_hash,
            "updated_at": now_str,
        }
        self._data.setdefault("satellites", {})[satellite_id] = entry
        self.save()
        return entry

    def set_manual_override(
        self,
        satellite_id: str,
        model_id: str,
        model_version: Optional[str] = None,
        model_artifact: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually override the selected model for a satellite."""
        existing = self.get_satellite_entry(satellite_id)
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if existing:
            existing["selected_model"] = model_id
            existing["selection_mode"] = "manual"
            if model_version:
                existing["model_version"] = model_version
            if model_artifact:
                existing["model_artifact"] = model_artifact
            else:
                existing["model_artifact"] = f"models/deploy/{model_id}/"
            existing["updated_at"] = now_str
            entry = existing
        else:
            entry = {
                "satellite_id": satellite_id,
                "selected_model": model_id,
                "selection_mode": "manual",
                "score": 0.0,
                "candidate_models": {model_id: 0.0},
                "validation_metrics": {},
                "model_version": model_version or "1.0.0",
                "model_artifact": model_artifact or f"models/deploy/{model_id}/",
                "dataset_hash": "",
                "updated_at": now_str,
            }
            self._data.setdefault("satellites", {})[satellite_id] = entry

        self.save()
        return entry

    def reset_to_automatic(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Reset satellite selection to automatic, restoring the highest scoring candidate."""
        entry = self.get_satellite_entry(satellite_id)
        if not entry:
            return None

        candidates = entry.get("candidate_models", {})
        rec_model = entry.get("recommended_automatic_model")

        if rec_model and rec_model in candidates:
            best_model = rec_model
            best_score = float(candidates.get(rec_model, 0.0))
        elif candidates:
            # Pick highest score
            best_model = max(candidates.keys(), key=lambda k: float(candidates[k]))
            best_score = float(candidates[best_model])
        else:
            best_model = entry.get("selected_model", "persistence")
            best_score = float(entry.get("score", 0.0))

        entry["selected_model"] = best_model
        entry["selection_mode"] = "automatic"
        entry["score"] = best_score
        entry["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.save()
        return entry

    def verify_model_availability(self, satellite_id: str) -> Tuple[bool, str]:
        """Verify that the chosen model for satellite is valid and deployable."""
        entry = self.get_satellite_entry(satellite_id)
        if not entry:
            return False, f"Satellite '{satellite_id}' has NO_SELECTION in model memory."

        selected_model = entry.get("selected_model")
        if not selected_model:
            return False, f"Satellite '{satellite_id}' has no model selected."

        try:
            adapter = get_adapter_by_id(selected_model)
        except ValueError as e:
            return False, str(e)

        # Check model artifact requirements if applicable
        if selected_model in ("bilstm_gru", "transformer"):
            deploy_dir = Path("models/deploy") / ("bilstm" if selected_model == "bilstm_gru" else "transformer")
            model_pt = deploy_dir / "model.pt"
            if not model_pt.exists():
                return False, f"Model artifact '{model_pt}' not found for {selected_model}."

        elif selected_model == "geo_gated_moe":
            moe_path = Path("research/ps08/models/geo_gated_moe_day8.pt")
            if not moe_path.exists():
                return False, f"MoE checkpoint '{moe_path}' not found."

        return True, f"Model '{selected_model}' ({adapter.version}) ready."

    def generate_audit_report(self, output_dir: Optional[str | Path] = None) -> Dict[str, str]:
        """Generate machine-readable audit report in reports/calibration/<timestamp>/."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(output_dir) if output_dir else Path("reports/calibration") / timestamp
        target_dir.mkdir(parents=True, exist_ok=True)

        summary_file = target_dir / "summary.json"
        comparison_csv = target_dir / "satellite_model_comparison.csv"
        metrics_csv = target_dir / "detailed_metrics.csv"

        # 1. Summary JSON
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

        # 2. Comparison CSV
        comparison_rows = []
        for sat_id, entry in self._data.get("satellites", {}).items():
            candidates = entry.get("candidate_models", {})
            row: Dict[str, Any] = {
                "satellite_id": sat_id,
                "selected_model": entry.get("selected_model"),
                "selection_mode": entry.get("selection_mode"),
                "score": entry.get("score"),
                "updated_at": entry.get("updated_at"),
            }
            for model_id in MODEL_ADAPTER_CLASSES:
                row[f"score_{model_id}"] = candidates.get(model_id, "")
            comparison_rows.append(row)

        if comparison_rows:
            fieldnames = list(comparison_rows[0].keys())
            with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparison_rows)

        # 3. Detailed Metrics CSV
        metrics_rows = []
        for sat_id, entry in self._data.get("satellites", {}).items():
            v_metrics = entry.get("validation_metrics", {})
            for model_id, m_dict in v_metrics.items():
                if not isinstance(m_dict, dict):
                    continue
                row = {
                    "satellite_id": sat_id,
                    "model_id": model_id,
                    "composite_score": m_dict.get("composite_score", ""),
                    "mae_3d": m_dict.get("mae_3d", ""),
                    "rmse_3d": m_dict.get("rmse_3d", ""),
                    "mae_clock": m_dict.get("mae_clock", ""),
                    "shapiro_w_mean": m_dict.get("shapiro_w_mean", ""),
                    "shapiro_p_mean": m_dict.get("shapiro_p_mean", ""),
                }
                targets_dict = m_dict.get("per_target", {})
                for tgt, tgt_m in targets_dict.items():
                    if isinstance(tgt_m, dict):
                        row[f"{tgt}_mae"] = tgt_m.get("mae", "")
                        row[f"{tgt}_rmse"] = tgt_m.get("rmse", "")
                        row[f"{tgt}_bias"] = tgt_m.get("bias", "")
                        row[f"{tgt}_std"] = tgt_m.get("std", "")
                        row[f"{tgt}_r2"] = tgt_m.get("r2", "")
                        row[f"{tgt}_max_ae"] = tgt_m.get("max_ae", "")
                        row[f"{tgt}_w"] = tgt_m.get("shapiro_w", "")
                        row[f"{tgt}_p"] = tgt_m.get("shapiro_p", "")
                metrics_rows.append(row)

        if metrics_rows:
            fieldnames = list(metrics_rows[0].keys())
            with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_rows)

        return {
            "summary_json": str(summary_file),
            "comparison_csv": str(comparison_csv),
            "metrics_csv": str(metrics_csv),
        }


def compute_dataset_hash(df: Any) -> str:
    """Compute deterministic SHA-256 fingerprint of dataset."""
    if hasattr(df, "to_csv"):
        data_bytes = df.to_csv(index=False).encode("utf-8")
    elif isinstance(df, (str, Path)):
        with open(df, "rb") as f:
            data_bytes = f.read()
    else:
        data_bytes = str(df).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()[:16]
