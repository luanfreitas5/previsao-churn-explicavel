"""Rastreamento de experimentos com MLflow: cada run é um registro reprodutível.

Contrato de um run: tags de linhagem (``git_sha``, ``data_hash``, ``seed``) no run
pai e nos filhos, parâmetros, uma run filha aninhada por fold, métricas com
incerteza (média e desvio) e o modelo registrado com assinatura de entrada/saída.
Encapsula o MLflow para manter as pipelines desacopladas da ferramenta.
"""

from __future__ import annotations

import logging
import subprocess  # nosec B404
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.config.environment import hash_file
from src.config.settings import MLflowConfig

logger = logging.getLogger(__name__)


def compute_git_sha(require_clean_tree: bool = True) -> str:
    """Retorna o SHA do ``HEAD``; por padrão recusa uma árvore Git suja.

    Parameters
    ----------
    require_clean_tree : bool, optional
        Se ``True``, levanta erro quando há alterações não commitadas (o SHA de
        uma árvore suja não descreve o código que rodou), by default True.

    Returns
    -------
    str
        SHA de 40 caracteres do commit atual.

    Raises
    ------
    RuntimeError
        Se a árvore estiver suja e ``require_clean_tree`` for ``True``, ou se o
        Git não estiver disponível.

    Examples
    --------
    >>> compute_git_sha(require_clean_tree=False)  # doctest: +SKIP
    '13cc32a...'
    """
    try:
        status = subprocess.run(  # nosec B603
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        sha = subprocess.run(  # nosec B603
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Não foi possível obter o SHA do Git para o experimento.") from exc
    if status and require_clean_tree:
        raise RuntimeError("Árvore Git suja: faça commit antes de registrar o experimento.")
    return sha


def build_lineage_tags(
    data_path: Path, seed: int, require_clean_tree: bool = True
) -> dict[str, str]:
    """Monta as tags de linhagem do run (código, dados e semente).

    Parameters
    ----------
    data_path : pathlib.Path
        Arquivo de dados de treino cujo SHA-256 identifica a versão dos dados.
    seed : int
        Semente global do experimento.
    require_clean_tree : bool, optional
        Recusa árvore Git suja, by default True.

    Returns
    -------
    dict of str to str
        Tags ``git_sha``, ``data_hash`` e ``seed``.

    Examples
    --------
    >>> build_lineage_tags(Path("data/gold/train.parquet"), 42)  # doctest: +SKIP
    """
    return {
        "git_sha": compute_git_sha(require_clean_tree),
        "data_hash": hash_file(Path(data_path)),
        "seed": str(seed),
    }


class MLflowTracker:
    """Encapsula um run pai do MLflow como gerenciador de contexto.

    Parameters
    ----------
    config : MLflowConfig
        Configuração do experimento (URI, nome, modelo registrado).
    run_name : str or None, optional
        Nome do run pai, by default ``None``.
    tags : dict of str to str or None, optional
        Tags de linhagem aplicadas ao run pai e aos filhos, by default ``None``.

    Examples
    --------
    >>> tags = build_lineage_tags(paths.train_file, seed=42)  # doctest: +SKIP
    >>> with MLflowTracker(settings.mlflow, "lightgbm", tags) as tracker:  # doctest: +SKIP
    ...     tracker.log_params({"n_estimators": 500})
    ...     tracker.log_cv_folds("average_precision", [0.61, 0.60, 0.62])
    """

    def __init__(
        self,
        config: MLflowConfig,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.run_name = run_name
        self.tags = tags or {}
        self._active = False

    def __enter__(self) -> MLflowTracker:
        """Inicia o experimento e o run pai do MLflow (com as tags de linhagem)."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self._active = True
        logger.info("Run MLflow iniciado no experimento '%s'", self.config.experiment_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Encerra o run pai do MLflow, registrando o status."""
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

    def log_cv_folds(self, metric: str, scores: Sequence[float]) -> None:
        """Registra uma run filha aninhada por fold e a média/desvio no run pai.

        Parameters
        ----------
        metric : str
            Nome da métrica de validação (ex.: ``"average_precision"``).
        scores : sequence of float
            Métrica obtida em cada fold.

        Examples
        --------
        >>> tracker.log_cv_folds("average_precision", [0.61, 0.6])  # doctest: +SKIP
        """
        for fold, score in enumerate(scores):
            fold_tags = {**self.tags, "fold": str(fold)}
            with mlflow.start_run(
                run_name=f"{self.run_name}-fold{fold}", nested=True, tags=fold_tags
            ):
                mlflow.log_metric(f"val_{metric}", float(score))
        values = np.asarray(scores, dtype=float)
        mlflow.log_metrics(
            {
                f"cv_{metric}_mean": float(values.mean()),
                f"cv_{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )

    def log_model(self, model: Pipeline, x_sample: pd.DataFrame) -> None:
        """Registra o modelo final com assinatura e exemplo de entrada.

        O modelo deve ter sido reajustado em todo o treino (e não ser o do último
        fold). Usa ``cloudpickle`` porque o artefato é treinado por nós, evitando
        a restrição de tipos não confiáveis do formato ``skops``.

        Parameters
        ----------
        model : sklearn.pipeline.Pipeline
            Pipeline ajustado no treino completo.
        x_sample : pandas.DataFrame
            Amostra de features usada para inferir a assinatura (até 100 linhas).

        Examples
        --------
        >>> tracker.log_model(model, x_train)  # doctest: +SKIP
        """
        sample = x_sample.head(100)
        signature = infer_signature(sample, model.predict_proba(sample))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=x_sample.head(5),
            serialization_format="cloudpickle",
        )

    def log_artifact(self, path: Path) -> None:
        """Loga um arquivo como artefato do run.

        Parameters
        ----------
        path : pathlib.Path
            Caminho do artefato a registrar.
        """
        if Path(path).exists():
            mlflow.log_artifact(str(path))
