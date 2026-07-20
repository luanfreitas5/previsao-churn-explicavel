"""Utilidades compartilhadas entre os estágios da pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import polars as pl

from src.constants import defaults
from src.exceptions import DataNotFoundError


def load_frame(path: Path) -> pl.DataFrame:
    """Lê um parquet de features/treino/teste como ``polars.DataFrame``.

    Parameters
    ----------
    path : pathlib.Path
        Caminho do arquivo parquet.

    Returns
    -------
    polars.DataFrame
        Conteúdo do arquivo.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(
            f"Artefato não encontrado: {path}. Rode o estágio anterior da pipeline."
        )
    return pl.read_parquet(path)


def to_pandas_xy(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a base em features (X) e alvo (y) no formato pandas.

    O modelo (LightGBM/scikit-learn/SHAP) opera sobre pandas; a conversão ocorre
    apenas nesta fronteira de modelagem.

    Parameters
    ----------
    df : polars.DataFrame
        Base contendo as features e a coluna alvo.

    Returns
    -------
    tuple
        Par ``(X, y)`` onde ``X`` contém ``defaults.FEATURES`` e ``y`` é o alvo.
    """
    pdf = df.to_pandas()
    # Indexação por lista devolve DataFrame; por rótulo único, Series.
    x = cast("pd.DataFrame", pdf[defaults.FEATURES]).copy()
    y = cast("pd.Series", pdf[defaults.TARGET]).astype(int)
    return x, y
