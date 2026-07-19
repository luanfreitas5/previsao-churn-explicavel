"""Validação cruzada estratificada e ajuste final do modelo.

A validação cruzada fornece uma estimativa de desempenho com incerteza (média
± IC 95%), evitando conclusões a partir de um único número.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def cross_validate_model(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "average_precision",
    seed: int = 42,
) -> dict[str, float | list[float]]:
    """Executa validação cruzada estratificada e retorna métrica com incerteza.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline a validar (clonado a cada fold).
    x : pandas.DataFrame
        Features de treino.
    y : pandas.Series
        Alvo de treino (0/1).
    n_splits : int, optional
        Número de folds, by default 5.
    scoring : str, optional
        Métrica de avaliação (nome do scorer sklearn), by default
        ``"average_precision"``.
    seed : int, optional
        Semente para reprodutibilidade, by default 42.

    Returns
    -------
    dict
        Chaves ``mean``, ``std``, ``ci95`` e ``scores`` (por fold).

    Examples
    --------
    >>> cross_validate_model(pipe, X_train, y_train)  # doctest: +SKIP
    {'mean': 0.61, 'std': 0.01, 'ci95': 0.009, 'scores': [...]}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(pipeline), x, y, cv=cv, scoring=scoring, n_jobs=-1)
    ci95 = float(1.96 * scores.std() / np.sqrt(len(scores)))
    logger.info(
        "CV %s: %.4f ± %.4f (IC 95%%) em %d folds",
        scoring,
        scores.mean(),
        ci95,
        n_splits,
    )
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "ci95": ci95,
        "scores": scores.tolist(),
    }


def fit_model(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Ajusta o pipeline no conjunto de treino completo.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline a ser treinado.
    x : pandas.DataFrame
        Features de treino.
    y : pandas.Series
        Alvo de treino (0/1).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline treinado.

    Examples
    --------
    >>> model = fit_model(pipe, X_train, y_train)  # doctest: +SKIP
    """
    logger.info("Treinando o modelo em %d exemplos", len(x))
    pipeline.fit(x, y)
    return pipeline
