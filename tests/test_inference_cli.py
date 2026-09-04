from pathlib import Path

import pandas as pd

from inference import predict


def test_inference_cli_validates_routes_and_writes_csv(monkeypatch, tmp_path):
    calls = {}

    def fake_validate(history, satellite_id, orbit_type):
        calls["validation"] = (history, satellite_id, orbit_type)
        return {"satellite_id": satellite_id, "row_count": 16}

    def fake_get_model(satellite_id):
        return {"satellite_id": satellite_id, "selected_model": "persistence"}

    def fake_predict(**kwargs):
        calls["prediction"] = kwargs
        return pd.DataFrame(
            {
                "forecast_step": [1],
                "timestamp": ["2025-01-01T00:15:00"],
                "satellite_id": [kwargs["satellite_id"]],
                "predicted_X": [1.0],
                "predicted_Y": [2.0],
                "predicted_Z": [3.0],
                "predicted_Clock": [4.0],
                "pred_3D_Orbit_Error": [3.741657],
                "model_used": ["persistence"],
                "model_version": ["1.0.0"],
                "selection_mode": ["automatic"],
            }
        )

    monkeypatch.setattr(predict, "validate_satellite_dataset", fake_validate)
    monkeypatch.setattr(predict, "get_satellite_model", fake_get_model)
    monkeypatch.setattr(predict, "predict_satellite", fake_predict)

    output = tmp_path / "forecast.csv"
    result = predict.run(
        [
            "--satellite",
            "GEO",
            "--orbit-type",
            "GEO",
            "--history",
            "history.csv",
            "--horizon-steps",
            "1",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert calls["validation"] == ("history.csv", "GEO", "GEO")
    assert calls["prediction"]["satellite_id"] == "GEO"
    assert calls["prediction"]["compute_ric"] is True
    assert Path(output).exists()
    assert pd.read_csv(output)["model_used"].tolist() == ["persistence"]


def test_inference_cli_rejects_missing_selection(monkeypatch):
    monkeypatch.setattr(
        predict,
        "validate_satellite_dataset",
        lambda *args, **kwargs: {"satellite_id": "UNKNOWN"},
    )
    monkeypatch.setattr(predict, "get_satellite_model", lambda satellite_id: None)

    try:
        predict.run(["--satellite", "UNKNOWN", "--history", "history.csv"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing registry selection should stop inference")
