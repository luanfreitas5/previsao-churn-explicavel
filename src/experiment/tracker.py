"""Gerenciador de contexto para rastreamento de experimentos com MLflow.

Registra parâmetros, métricas e artefatos por run, marcados com o hash dos
dados. Encapsula o MLflow para manter as pipelines desacopladas da ferramenta.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any

import mlflow

from src.config.settings import MLflowConfig

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Encapsula um run do MLflow como gerenciador de contexto.

    Parameters
    ----------
    config : MLflowConfig
        Configuração do experimento (URI, nome, modelo registrado).
    run_name : str or None, optional
        Nome do run, by default ``None``.

    Examples
    --------
    >>> with MLflowTracker(settings.mlflow, run_name="lgbm") as tracker:  # doctest: +SKIP
    ...     tracker.log_params({"n_estimators": 500})
    ...     tracker.log_metrics({"average_precision": 0.61})
    """

    def __init__(self, config: MLflowConfig, run_name: str | None = None) -> None:
        self.config = config
        self.run_name = run_name
        self._active = False

    def __enter__(self) -> MLflowTracker:
        """Inicia o experimento e o run do MLflow."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._active = True
        logger.info("Run MLflow iniciado no experimento '%s'", self.config.experiment_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Encerra o run do MLflow, registrando o status."""
        status = "FAILED" if exc_type is not None else "FINISHED"
        mlflow.end_run(status=status)
        self._active = False

    def log_params(self, params: dict[str, Any]) -> None:
        """Loga um dicionário de parâmetros no run atual.

        Parameters
        ----------
        params : dict
            Parâmetros a registrar.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Loga um dicionário de métricas no run atual.

        Parameters
        ----------
        metrics : dict
            Métricas numéricas a registrar.
        """
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        """Loga um arquivo como artefato do run.

        Parameters
        ----------
        path : pathlib.Path
            Caminho do artefato a registrar.
        """
        if Path(path).exists():
            mlflow.log_artifact(str(path))
