"""Auditoria de justiça: disparidade de desempenho por subgrupo (UF do cliente).

Métricas agregadas escondem falhas em subgrupos importantes. Aqui medimos
recall e taxa de seleção por UF para expor disparidades no Model Card.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from numpy.typing import NDArray
from sklearn.metrics import recall_score

logger = logging.getLogger(__name__)


def audit_fairness(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    sensitive_feature: pd.Series,
) -> pd.DataFrame:
    """Calcula recall e taxa de seleção por subgrupo do atributo sensível.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros (0/1).
    y_pred : numpy.ndarray
        Rótulos previstos (0/1).
    sensitive_feature : pandas.Series
        Valores do atributo sensível (ex.: UF do cliente), alinhados a ``y_true``.

    Returns
    -------
    pandas.DataFrame
        Uma linha por subgrupo com as colunas ``recall`` e ``selection_rate``.

    Examples
    --------
    >>> audit_fairness(y_true, y_pred, X_test["customer_state"])  # doctest: +SKIP
    """
    frame = MetricFrame(
        metrics={
            "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )
    by_group = frame.by_group
    disparity = float(by_group["recall"].max() - by_group["recall"].min())
    logger.info("Disparidade de recall entre UFs = %.3f", disparity)
    return by_group
