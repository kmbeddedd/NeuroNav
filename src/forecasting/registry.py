"""Persistent Satellite Model Registry.

Manages satellite-to-model selection mappings, manual overrides, audit trails,
and candidate evaluation scores.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("models/registry/satellite_model_registry.json")


class RegistryError(Exception):
    """Base exception for model registry errors."""
    pass


class CorruptedRegistryError(RegistryError):
    """Raised when the registry file contains invalid JSON or schema violations."""
    pass


@dataclass
class SatelliteSelection:
    """Represents a persisted model selection for a single satellite."""
    satellite_id: str
    selected_model: str
    selection_mode: str = "automatic"  # "automatic" or "manual"
    selection_policy: str = "official_competition"
    model_version: str = "1.0.0"
    selection_score: float = 0.0
    primary_metric: str = "shapiro_w_avg"
    training_dataset_hash: str = ""
    selected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_artifact: Optional[str] = None
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    winning_priority_1: Dict[str, Any] = field(default_factory=dict)
    winning_priority_2: Dict[str, Any] = field(default_factory=dict)
    winning_priority_3: Dict[str, Any] = field(default_factory=dict)
    candidate_results: Dict[str, Any] = field(default_factory=dict)
    supplementary_diagnostics: Dict[str, Any] = field(default_factory=dict)
    feature_config: Dict[str, Any] = field(default_factory=dict)
    lookback_config: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SatelliteSelection":
        return cls(
            satellite_id=data.get("satellite_id", ""),
            selected_model=data.get("selected_model", ""),
            selection_mode=data.get("selection_mode", "automatic"),
            selection_policy=data.get("selection_policy", "official_competition"),
            model_version=data.get("model_version", "1.0.0"),
            selection_score=float(data.get("selection_score", 0.0)),
            primary_metric=data.get("primary_metric", "shapiro_w_avg"),
            training_dataset_hash=data.get("training_dataset_hash", ""),
            selected_at=data.get("selected_at", datetime.now(timezone.utc).isoformat()),
            model_artifact=data.get("model_artifact"),
            candidate_scores=data.get("candidate_scores", {}),
            winning_priority_1=data.get("winning_priority_1", {}),
            winning_priority_2=data.get("winning_priority_2", {}),
            winning_priority_3=data.get("winning_priority_3", {}),
            candidate_results=data.get("candidate_results", {}),
            supplementary_diagnostics=data.get("supplementary_diagnostics", {}),
            feature_config=data.get("feature_config", {}),
            lookback_config=data.get("lookback_config", {}),
            history=data.get("history", []),
        )


class SatelliteModelRegistry:
    """Thread-safe and persistent registry storing satellite-specific model assignments."""

    def __init__(self, registry_path: Union[str, Path] = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.satellites: Dict[str, SatelliteSelection] = {}
        self.schema_version = "1.0"
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.load()

    def load(self) -> None:
        """Loads registry from JSON file. If file is absent, initializes empty registry."""
        if not self.registry_path.exists():
            self.satellites = {}
            return

        try:
            content = self.registry_path.read_text(encoding="utf-8")
            if not content.strip():
                self.satellites = {}
                return
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise CorruptedRegistryError(
                f"Registry file at {self.registry_path} is corrupted: {exc}"
            ) from exc
        except Exception as exc:
            raise RegistryError(f"Failed to read registry: {exc}") from exc

        self.schema_version = data.get("schema_version", "1.0")
        self.last_updated = data.get("last_updated", "")
        self.satellites = {}

        raw_satellites = data.get("satellites", {})
        for sat_id, sat_data in raw_satellites.items():
            if "satellite_id" not in sat_data:
                sat_data["satellite_id"] = sat_id
            self.satellites[sat_id] = SatelliteSelection.from_dict(sat_data)

    def save(self) -> None:
        """Saves current registry state to disk atomically."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(timezone.utc).isoformat()

        payload = {
            "schema_version": self.schema_version,
            "last_updated": self.last_updated,
            "satellites": {
                sat_id: selection.to_dict()
                for sat_id, selection in self.satellites.items()
            },
        }

        temp_path = self.registry_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.registry_path)

    def get_selection(self, satellite_id: str) -> Optional[SatelliteSelection]:
        """Returns the current selection for a satellite, or None if unselected."""
        return self.satellites.get(satellite_id)

    def get_all_selections(self) -> Dict[str, SatelliteSelection]:
        """Returns all registered satellite selections."""
        return dict(self.satellites)

    def register_calibration_winner(
        self,
        satellite_id: str,
        winner_model: str,
        score: float,
        candidate_scores: Dict[str, float],
        training_dataset_hash: str = "",
        model_version: str = "1.0.0",
        model_artifact: Optional[str] = None,
        primary_metric: str = "shapiro_w_avg",
        selection_policy: str = "official_competition",
        winning_priority_1: Optional[Dict[str, Any]] = None,
        winning_priority_2: Optional[Dict[str, Any]] = None,
        winning_priority_3: Optional[Dict[str, Any]] = None,
        candidate_results: Optional[Dict[str, Any]] = None,
        supplementary_diagnostics: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        lookback_config: Optional[Dict[str, Any]] = None,
    ) -> SatelliteSelection:
        """Registers the winner of an automatic calibration run.
        
        INVARIANT: If the satellite currently has selection_mode == 'manual',
        the manual selection MUST NOT be overwritten! The calibration event is
        simply recorded in history for auditability.
        """
        existing = self.satellites.get(satellite_id)
        now = datetime.now(timezone.utc).isoformat()

        if existing is not None and existing.selection_mode == "manual":
            # Manual selection is preserved
            logger.info(
                f"Satellite {satellite_id} has a MANUAL model assignment ({existing.selected_model}); "
                f"retaining manual selection and recording automatic candidate winner ({winner_model})."
            )
            audit_entry = {
                "event": "automatic_calibration_skipped_due_to_manual_override",
                "timestamp": now,
                "manual_model": existing.selected_model,
                "suggested_winner": winner_model,
                "suggested_score": score,
                "candidate_scores": candidate_scores,
                "selection_policy": selection_policy,
                "training_dataset_hash": training_dataset_hash,
            }
            existing.history.append(audit_entry)
            self.save()
            return existing

        # Automatic assignment / replacement
        history = list(existing.history) if existing else []
        if existing:
            history.append({
                "event": "model_replaced",
                "timestamp": now,
                "previous_model": existing.selected_model,
                "previous_score": existing.selection_score,
                "previous_mode": existing.selection_mode,
                "previous_policy": getattr(existing, "selection_policy", "official_competition"),
            })

        new_selection = SatelliteSelection(
            satellite_id=satellite_id,
            selected_model=winner_model,
            selection_mode="automatic",
            selection_policy=selection_policy,
            model_version=model_version,
            selection_score=score,
            primary_metric=primary_metric,
            training_dataset_hash=training_dataset_hash,
            selected_at=now,
            model_artifact=model_artifact,
            candidate_scores=candidate_scores,
            winning_priority_1=winning_priority_1 or {},
            winning_priority_2=winning_priority_2 or {},
            winning_priority_3=winning_priority_3 or {},
            candidate_results=candidate_results or {},
            supplementary_diagnostics=supplementary_diagnostics or {},
            feature_config=feature_config or {},
            lookback_config=lookback_config or {},
            history=history,
        )
        self.satellites[satellite_id] = new_selection
        self.save()
        return new_selection

    def set_manual_selection(
        self,
        satellite_id: str,
        model_name: str,
        model_version: str = "1.0.0",
        model_artifact: Optional[str] = None,
        reason: str = "Manual operator override",
    ) -> SatelliteSelection:
        """Explicitly assigns a model to a satellite with selection_mode='manual'."""
        existing = self.satellites.get(satellite_id)
        now = datetime.now(timezone.utc).isoformat()
        history = list(existing.history) if existing else []

        history.append({
            "event": "manual_selection_set",
            "timestamp": now,
            "previous_model": existing.selected_model if existing else None,
            "new_model": model_name,
            "reason": reason,
        })

        selection = SatelliteSelection(
            satellite_id=satellite_id,
            selected_model=model_name,
            selection_mode="manual",
            model_version=model_version,
            selection_score=existing.selection_score if existing else 0.0,
            primary_metric=existing.primary_metric if existing else "manual",
            training_dataset_hash=existing.training_dataset_hash if existing else "",
            selected_at=now,
            model_artifact=model_artifact or (existing.model_artifact if existing else None),
            candidate_scores=existing.candidate_scores if existing else {},
            feature_config=existing.feature_config if existing else {},
            lookback_config=existing.lookback_config if existing else {},
            history=history,
        )
        self.satellites[satellite_id] = selection
        self.save()
        return selection

    def reset_to_automatic(self, satellite_id: str) -> bool:
        """Resets a satellite's selection_mode to 'automatic' so future calibrations can update it."""
        existing = self.satellites.get(satellite_id)
        if existing is None:
            return False

        if existing.selection_mode == "manual":
            existing.selection_mode = "automatic"
            existing.history.append({
                "event": "reset_to_automatic",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self.save()
            return True
        return False
