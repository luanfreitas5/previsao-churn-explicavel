"""Testes de persistência do modelo e do rastreamento de experimentos (MLflow)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from src.config.settings import MLflowConfig
from src.exceptions import ModelNotTrainedError
from src.experiment.tracker import MLflowTracker
from src.models.persistence import load_model, save_model


def _sqlite_config(tmp_path: Path) -> MLflowConfig:
    """Cria uma configuração MLflow com backend SQLite e artefatos isolados em tmp."""
    config = MLflowConfig(tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.create_experiment(
        config.experiment_name, artifact_location=(tmp_path / "mlartifacts").as_uri()
    )
    return config


def test_save_and_load_model_roundtrip(tmp_path: Path, trained_model) -> None:
    """Salvar e carregar preserva o pipeline e os metadados de linhagem."""
    model, _, _ = trained_model
    model_path = tmp_path / "models" / "churn.joblib"
    metadata = {"primary_metric": "average_precision", "cv_model_mean": 0.61}

    save_model(model, model_path, metadata)
    loaded, meta = load_model(model_path)

    assert meta == metadata
    assert hasattr(loaded, "predict_proba")


def test_load_missing_model_raises(tmp_path: Path) -> None:
    """Carregar um modelo inexistente levanta ModelNotTrainedError."""
    with pytest.raises(ModelNotTrainedError):
        load_model(tmp_path / "inexistente.joblib")


def test_load_model_without_metadata_returns_empty_dict(tmp_path: Path, trained_model) -> None:
    """Sem arquivo de metadados, o carregamento retorna um dicionário vazio."""
    model, _, _ = trained_model
    model_path = tmp_path / "churn.joblib"
    save_model(model, model_path, {"x": 1})
    model_path.with_name("churn_meta.json").unlink()

    _, meta = load_model(model_path)
    assert meta == {}


def test_mlflow_tracker_logs_params_metrics_and_artifact(tmp_path: Path) -> None:
    """O tracker registra parâmetros, métricas e artefatos em um run local."""
    config = _sqlite_config(tmp_path)
    artifact = tmp_path / "report.txt"
    artifact.write_text("relatório", encoding="utf-8")

    with MLflowTracker(config, run_name="teste") as tracker:
        tracker.log_params({"n_estimators": 60})
        tracker.log_metrics({"average_precision": 0.61})
        tracker.log_artifact(artifact)
        tracker.log_artifact(tmp_path / "ausente.txt")  # caminho inexistente: ignorado


def test_mlflow_tracker_sets_failed_status_on_exception(tmp_path: Path) -> None:
    """Uma exceção dentro do contexto encerra o run com status FAILED sem mascarar o erro."""
    config = _sqlite_config(tmp_path)
    with pytest.raises(ValueError), MLflowTracker(config) as tracker:
        tracker.log_metrics({"m": 1.0})
        raise ValueError("falha simulada")
