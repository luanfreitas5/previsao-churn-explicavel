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


def test_mlflow_tracker_logs_one_child_run_per_fold(tmp_path: Path) -> None:
    """Cada fold vira uma run filha com as tags de linhagem; o pai guarda média e desvio."""
    config = _sqlite_config(tmp_path)
    tags = {"git_sha": "abc", "data_hash": "def", "seed": "42"}

    with MLflowTracker(config, run_name="modelo", tags=tags) as tracker:
        tracker.log_cv_folds("average_precision", [0.6, 0.62, 0.64])

    runs = mlflow.search_runs(experiment_names=[config.experiment_name])
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert len(children) == 3
    assert set(children["tags.git_sha"]) == {"abc"}
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert parent["metrics.cv_average_precision_mean"] == pytest.approx(0.62)
    assert parent["metrics.cv_average_precision_std"] == pytest.approx(0.02)


def test_mlflow_tracker_logs_model_with_signature(tmp_path: Path, trained_model) -> None:
    """O modelo final é registrado com assinatura inferida da amostra de entrada."""
    model, x_test, _ = trained_model
    config = _sqlite_config(tmp_path)

    with MLflowTracker(config, run_name="modelo") as tracker:
        tracker.log_model(model, x_test)

    runs = mlflow.search_runs(experiment_names=[config.experiment_name])
    assert len(runs) == 1


def test_compute_git_sha_refuses_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com alterações não commitadas, o SHA só é aceito se a árvore limpa não for exigida."""
    import subprocess

    from src.experiment.tracker import compute_git_sha

    def fake_run(cmd, **_kwargs):
        stdout = " M arquivo.py\n" if "status" in cmd else "a1b2c3\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Árvore Git suja"):
        compute_git_sha(require_clean_tree=True)
    assert compute_git_sha(require_clean_tree=False) == "a1b2c3"
