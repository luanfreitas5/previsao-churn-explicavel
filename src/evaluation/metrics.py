"""Métricas de classificação e intervalo de confiança via bootstrap.

Nunca reportamos uma métrica pontual sem incerteza: o bootstrap fornece um
intervalo de confiança empírico sobre o conjunto de teste.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula as principais métricas de classificação binária.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros (0/1).
    y_proba : numpy.ndarray
        Probabilidades previstas da classe positiva (churn).
    threshold : float, optional
        Limiar de decisão para rótulos, by default 0.5.

    Returns
    -------
    dict of str to float
        Métricas: ``roc_auc``, ``average_precision``, ``f1``, ``precision``,
        ``recall``, ``brier``.

    Examples
    --------
    >>> evaluate_classification(np.array([0, 1]), np.array([0.2, 0.9]))  # doctest: +SKIP
    {'roc_auc': 1.0, ...}
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def bootstrap_metric(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    metric_fn: Callable[[NDArray[np.int_], NDArray[np.float64]], float] = average_precision_score,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Estima média e intervalo de confiança de uma métrica via bootstrap.

    Reamostra o conjunto de teste com reposição para obter uma distribuição
    empírica da métrica e o intervalo de confiança percentil.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros (0/1).
    y_proba : numpy.ndarray
        Probabilidades previstas da classe positiva.
    metric_fn : callable, optional
        Função ``(y_true, y_proba) -> float``, by default ``average_precision_score``.
    n_iterations : int, optional
        Número de reamostragens bootstrap, by default 1000.
    alpha : float, optional
        Nível de significância (IC = 1 - alpha), by default 0.05.
    seed : int, optional
        Semente para reprodutibilidade, by default 42.

    Returns
    -------
    dict of str to float
        Chaves ``mean``, ``lower`` e ``upper`` (limites do IC).

    Examples
    --------
    >>> bootstrap_metric(y_true, y_proba, n_iterations=200)  # doctest: +SKIP
    {'mean': 0.61, 'lower': 0.58, 'upper': 0.64}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        # Ignora reamostras com uma única classe (métrica indefinida).
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(float(metric_fn(y_true[idx], y_proba[idx])))

    scores_arr = np.asarray(scores)
    return {
        "mean": float(scores_arr.mean()),
        "lower": float(np.percentile(scores_arr, 100 * alpha / 2)),
        "upper": float(np.percentile(scores_arr, 100 * (1 - alpha / 2))),
    }
