"""Divisão estratificada da base de features em treino e teste."""

from __future__ import annotations

import logging

import polars as pl
from sklearn.model_selection import train_test_split

from src.constants import defaults

logger = logging.getLogger(__name__)


def split_train_test(
    features: pl.DataFrame,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide a base de features em treino e teste de forma estratificada.

    A estratificação pelo alvo mantém a proporção de churn em ambos os
    conjuntos, importante em bases desbalanceadas.

    Parameters
    ----------
    features : polars.DataFrame
        Base de features contendo a coluna alvo ``churn``.
    test_size : float, optional
        Proporção do conjunto de teste (0 a 1), by default 0.2.
    stratify : bool, optional
        Se ``True``, estratifica pelo alvo, by default True.
    seed : int, optional
        Semente para reprodutibilidade, by default 42.

    Returns
    -------
    tuple of polars.DataFrame
        Par ``(train, test)``.

    Raises
    ------
    KeyError
        Se a coluna alvo não existir em ``features``.

    Examples
    --------
    >>> train, test = split_train_test(df, test_size=0.25)  # doctest: +SKIP
    """
    if defaults.TARGET not in features.columns:
        raise KeyError(f"Coluna alvo '{defaults.TARGET}' ausente na base de features.")

    indices = list(range(features.height))
    stratify_values = features[defaults.TARGET].to_list() if stratify else None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_values,
    )

    train = features[train_idx]
    test = features[test_idx]
    logger.info(
        "Divisão treino/teste: treino=%d (%.1f%% churn), teste=%d (%.1f%% churn)",
        train.height,
        100 * train[defaults.TARGET].mean(),
        test.height,
        100 * test[defaults.TARGET].mean(),
    )
    return train, test
