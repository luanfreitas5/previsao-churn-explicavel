"""Avaliação de calibração das probabilidades previstas.

Como as probabilidades de churn orientam decisões comerciais (priorização de
retenção), verificamos se elas são calibradas — ou seja, se uma probabilidade
prevista de 0,7 corresponde a ~70% de churn observado.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def compute_calibration(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, list[float] | float]:
    """Calcula o Brier score e a curva de confiabilidade.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros (0/1).
    y_proba : numpy.ndarray
        Probabilidades previstas da classe positiva (churn).
    n_bins : int, optional
        Número de faixas na curva de confiabilidade, by default 10.

    Returns
    -------
    dict
        Chaves ``brier`` (float), ``prob_true`` e ``prob_pred`` (listas com a
        curva de confiabilidade).

    Examples
    --------
    >>> compute_calibration(y_true, y_proba)  # doctest: +SKIP
    {'brier': 0.18, 'prob_true': [...], 'prob_pred': [...]}
    """
    brier = float(brier_score_loss(y_true, y_proba))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    logger.info("Brier score = %.4f", brier)
    return {
        "brier": brier,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }
