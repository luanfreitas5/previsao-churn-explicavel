"""Persistência do pipeline treinado com ``joblib`` e metadados JSON.

Os metadados registram a linhagem do modelo (data, hash dos dados, métricas,
parâmetros) para reprodutibilidade — complementando o rastreamento no MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.exceptions import ModelNotTrainedError
from src.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


def save_model(pipeline: Pipeline, model_path: Path, metadata: dict[str, Any]) -> None:
    """Salva o pipeline treinado e seus metadados de linhagem.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline treinado.
    model_path : pathlib.Path
        Caminho de destino do artefato ``.joblib``.
    metadata : dict
        Metadados de linhagem (hash dos dados, métricas, parâmetros, data).

    Examples
    --------
    >>> save_model(pipe, Path("models/churn.joblib"), {"f1": 0.8})  # doctest: +SKIP
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    write_json(metadata, model_path.with_name(f"{model_path.stem}_meta.json"))
    logger.info("Modelo salvo em %s", model_path)


def load_model(model_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Carrega o pipeline treinado e seus metadados.

    Parameters
    ----------
    model_path : pathlib.Path
        Caminho do artefato ``.joblib``.

    Returns
    -------
    tuple
        Par ``(pipeline, metadata)``. Se não houver metadados, retorna ``{}``.

    Raises
    ------
    ModelNotTrainedError
        Se o arquivo de modelo não existir.

    Examples
    --------
    >>> pipe, meta = load_model(Path("models/churn.joblib"))  # doctest: +SKIP
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotTrainedError(
            f"Modelo não encontrado em {model_path}. Treine o modelo antes (make train)."
        )
    pipeline = joblib.load(model_path)
    meta_path = model_path.with_name(f"{model_path.stem}_meta.json")
    metadata = read_json(meta_path) if meta_path.exists() else {}
    logger.info("Modelo carregado de %s", model_path)
    return pipeline, metadata
