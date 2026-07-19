"""Contrato de dados da base de features RFM + churn.

Valida tipos, faixas e nulabilidade na saída da engenharia de features e na
entrada do modelo, evitando corrupção silenciosa e detectando train-serve skew.
"""

from __future__ import annotations

import logging

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing.polars import Series

from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class ChurnFeaturesSchema(pa.DataFrameModel):
    """Contrato da base de features de churn (uma linha por cliente).

    As faixas refletem regras de domínio: métricas RFM não negativas, notas de
    avaliação entre 1 e 5 e alvo binário.
    """

    customer_unique_id: Series[str] = pa.Field(nullable=False)
    recency_days: Series[float] = pa.Field(ge=0)
    frequency: Series[int] = pa.Field(ge=1)
    monetary: Series[float] = pa.Field(ge=0)
    monetary_mean: Series[float] = pa.Field(ge=0)
    tenure_days: Series[float] = pa.Field(ge=0)
    avg_freight_value: Series[float] = pa.Field(ge=0, nullable=True)
    avg_review_score: Series[float] = pa.Field(ge=1, le=5, nullable=True)
    avg_installments: Series[float] = pa.Field(ge=0, nullable=True)
    distinct_products: Series[int] = pa.Field(ge=1)
    avg_delivery_days: Series[float] = pa.Field(ge=0, nullable=True)
    late_delivery_rate: Series[float] = pa.Field(ge=0, le=1)
    customer_state: Series[str] = pa.Field(nullable=False)
    churn: Series[int] = pa.Field(isin=[0, 1])

    class Config:  # type: ignore
        """Configuração do contrato: rejeita colunas inesperadas."""

        strict = True
        coerce = True


def validate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Valida a base de features contra o contrato :class:`ChurnFeaturesSchema`.

    Parameters
    ----------
    df : polars.DataFrame
        Base de features a validar.

    Returns
    -------
    polars.DataFrame
        A mesma base, validada (e com tipos coeridos).

    Raises
    ------
    DataValidationError
        Se o DataFrame violar o contrato de dados.

    Examples
    --------
    >>> validate_features(df)  # doctest: +SKIP
    """
    try:
        return ChurnFeaturesSchema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        # ``failure_cases`` (presente no modo lazy) traz coluna, regra e valor que
        # falharam; incluímos um resumo na mensagem para não depender só do log.
        failure_cases = getattr(exc, "failure_cases", None)
        detalhe = str(failure_cases) if failure_cases is not None else str(exc)
        logger.error("Falha na validação do contrato de features:\n%s", detalhe)
        raise DataValidationError(
            f"A base de features violou o contrato de dados (ChurnFeaturesSchema):\n{detalhe}"
        ) from exc
