"""Predição de probabilidade de churn e atribuição de faixas de risco."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from src.constants import defaults

logger = logging.getLogger(__name__)


def predict_proba(pipeline: Pipeline, x: pd.DataFrame) -> NDArray[np.float64]:
    """Retorna a probabilidade de churn (classe positiva) para cada cliente.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline treinado.
    x : pandas.DataFrame
        Features dos clientes a pontuar.

    Returns
    -------
    numpy.ndarray
        Vetor de probabilidades de churn.

    Examples
    --------
    >>> predict_proba(model, X_test)  # doctest: +SKIP
    array([0.12, 0.87, ...])
    """
    return pipeline.predict_proba(x)[:, 1]


def score_customers(
    pipeline: Pipeline,
    features: pd.DataFrame,
    risk_bands: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Pontua clientes com probabilidade de churn e faixa de risco.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline treinado.
    features : pandas.DataFrame
        Base de features contendo ao menos as colunas de ``defaults.FEATURES``
        e ``defaults.CUSTOMER_KEY``.
    risk_bands : dict of str to float or None, optional
        Limiares superiores por faixa de risco (ex.: ``{"baixo": 0.33, ...}``).
        Se ``None``, usa faixas em terços.

    Returns
    -------
    pandas.DataFrame
        ``features`` acrescido de ``churn_proba`` e ``risk_band``, ordenado por
        probabilidade decrescente.

    Examples
    --------
    >>> score_customers(model, features_df)  # doctest: +SKIP
    """
    bands = risk_bands or {"baixo": 0.33, "medio": 0.66, "alto": 1.01}
    # Indexação por lista de colunas devolve sempre um DataFrame.
    x = cast("pd.DataFrame", features[defaults.FEATURES])
    proba = predict_proba(pipeline, x)

    scored = features.copy()
    scored["churn_proba"] = proba

    edges = sorted(bands.items(), key=lambda item: item[1])
    labels = [name for name, _ in edges]
    thresholds = [0.0, *[value for _, value in edges]]
    scored["risk_band"] = pd.cut(
        scored["churn_proba"], bins=thresholds, labels=labels, include_lowest=True
    )
    logger.info("Clientes pontuados: %d", len(scored))
    return scored.sort_values("churn_proba", ascending=False, ignore_index=True)
