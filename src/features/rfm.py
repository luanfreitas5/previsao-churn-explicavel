"""Monta a base de features RFM por cliente e o alvo de churn (não recompra).

Definição do alvo (ver ``configs/config.yaml``)
------------------------------------------------
Dada uma data de corte ``cutoff_date`` e um horizonte ``horizon_days``:

1. A base considera clientes com pelo menos um pedido válido **até** o corte.
2. Todas as features são calculadas **apenas** com pedidos ``<= cutoff_date``
   (garante ausência de vazamento de futuro).
3. ``churn = 1`` se o cliente **não** fizer nenhum pedido na janela
   ``(cutoff_date, cutoff_date + horizon_days]``; caso contrário ``churn = 0``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl

from src.config.settings import ChurnDefinition
from src.constants import columns as c
from src.exceptions import ChurnProjectError
from src.features import engineering as eng

logger = logging.getLogger(__name__)


def _join_customer_keys(orders: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:
    """Associa cada pedido ao ``customer_unique_id`` e à UF do cliente.

    Parameters
    ----------
    orders : polars.DataFrame
        Pedidos brutos (com ``customer_id``).
    customers : polars.DataFrame
        Clientes (mapeia ``customer_id`` para ``customer_unique_id`` e UF).

    Returns
    -------
    polars.DataFrame
        Pedidos enriquecidos com ``customer_unique_id`` e ``customer_state``.
    """
    keys = customers.select(c.CUSTOMER_ID, c.CUSTOMER_UNIQUE_ID, c.CUSTOMER_STATE)
    return orders.join(keys, on=c.CUSTOMER_ID, how="inner")


def _filter_valid_orders(orders: pl.DataFrame, valid_statuses: list[str]) -> pl.DataFrame:
    """Mantém apenas pedidos com status válido e timestamp de compra presente.

    Parameters
    ----------
    orders : polars.DataFrame
        Pedidos enriquecidos.
    valid_statuses : list of str
        Status de pedido considerados válidos.

    Returns
    -------
    polars.DataFrame
        Pedidos válidos.
    """
    return orders.filter(
        pl.col(c.ORDER_STATUS).is_in(valid_statuses)
        & pl.col(c.ORDER_PURCHASE_TIMESTAMP).is_not_null()
    )


def _build_customer_features(
    past_orders: pl.DataFrame,
    order_items: pl.DataFrame,
    cutoff: datetime,
) -> pl.DataFrame:
    """Agrega os pedidos anteriores ao corte em features por cliente.

    Parameters
    ----------
    past_orders : polars.DataFrame
        Pedidos ``<= cutoff`` já enriquecidos com métricas a nível de pedido.
    order_items : polars.DataFrame
        Itens de pedido (para contagem de produtos distintos).
    cutoff : datetime.datetime
        Data de corte do snapshot.

    Returns
    -------
    polars.DataFrame
        Uma linha por cliente com as features RFM e derivadas.
    """
    ordered = past_orders.sort(c.ORDER_PURCHASE_TIMESTAMP)

    features = ordered.group_by(c.CUSTOMER_UNIQUE_ID).agg(
        # UF do último pedido observado (estado mais recente do cliente).
        pl.col(c.CUSTOMER_STATE).last().alias(c.CUSTOMER_STATE),
        pl.col(c.ORDER_PURCHASE_TIMESTAMP).max().alias("last_purchase"),
        pl.col(c.ORDER_PURCHASE_TIMESTAMP).min().alias("first_purchase"),
        pl.col(c.ORDER_ID).n_unique().alias(c.FREQUENCY),
        pl.col("payment_value_sum").sum().alias(c.MONETARY),
        pl.col("installments_mean").mean().alias(c.AVG_INSTALLMENTS),
        pl.col("freight_sum").mean().alias(c.AVG_FREIGHT),
        pl.col("review_score_mean").mean().alias(c.AVG_REVIEW_SCORE),
        pl.col("delivery_days").mean().alias(c.AVG_DELIVERY_DAYS),
        pl.col("is_late").mean().fill_null(0.0).alias(c.LATE_DELIVERY_RATE),
    )

    # Produtos distintos por cliente (via itens dos pedidos anteriores ao corte).
    past_ids = past_orders.select(c.ORDER_ID, c.CUSTOMER_UNIQUE_ID)
    distinct_products = (
        order_items.join(past_ids, on=c.ORDER_ID, how="inner")
        .group_by(c.CUSTOMER_UNIQUE_ID)
        .agg(pl.col(c.PRODUCT_ID).n_unique().alias(c.DISTINCT_PRODUCTS))
    )

    features = features.join(distinct_products, on=c.CUSTOMER_UNIQUE_ID, how="left")

    # Deriva recência, tenure e ticket médio a partir das agregações.
    return features.with_columns(
        (pl.lit(cutoff) - pl.col("last_purchase"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias(c.RECENCY_DAYS),
        (pl.lit(cutoff) - pl.col("first_purchase"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias(c.TENURE_DAYS),
        (pl.col(c.MONETARY) / pl.col(c.FREQUENCY)).alias(c.MONETARY_MEAN),
        pl.col(c.DISTINCT_PRODUCTS).fill_null(1),
    )


def build_churn_features(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    order_items: pl.DataFrame,
    payments: pl.DataFrame,
    reviews: pl.DataFrame,
    churn: ChurnDefinition,
) -> pl.DataFrame:
    """Constrói a base de features RFM + alvo de churn a partir dos dados brutos.

    Parameters
    ----------
    orders : polars.DataFrame
        Tabela de pedidos (com timestamps já tipados).
    customers : polars.DataFrame
        Tabela de clientes.
    order_items : polars.DataFrame
        Tabela de itens de pedido.
    payments : polars.DataFrame
        Tabela de pagamentos.
    reviews : polars.DataFrame
        Tabela de avaliações.
    churn : ChurnDefinition
        Parâmetros de rotulagem (corte, horizonte, status válidos).

    Returns
    -------
    polars.DataFrame
        Base de features (uma linha por cliente) com a coluna alvo ``churn``,
        pronta para validação pelo contrato de dados.

    Raises
    ------
    ChurnProjectError
        Se, após os filtros, não restar nenhum cliente na base.

    Examples
    --------
    >>> df = build_churn_features(*tables, churn=settings.churn)  # doctest: +SKIP
    >>> "churn" in df.columns  # doctest: +SKIP
    True
    """
    cutoff = datetime.combine(churn.cutoff_date, datetime.min.time())
    horizon_end = cutoff + timedelta(days=churn.horizon_days)
    logger.info(
        "Construindo features de churn (corte=%s, horizonte=%d dias, fim=%s)",
        cutoff.date(),
        churn.horizon_days,
        horizon_end.date(),
    )

    enriched = _filter_valid_orders(
        _join_customer_keys(orders, customers), churn.valid_order_statuses
    )
    enriched = eng.add_delivery_metrics(enriched)

    # Une agregações a nível de pedido (pagamentos, itens, avaliações).
    enriched = (
        enriched.join(eng.aggregate_payments(payments), on=c.ORDER_ID, how="left")
        .join(eng.aggregate_items(order_items), on=c.ORDER_ID, how="left")
        .join(eng.aggregate_reviews(reviews), on=c.ORDER_ID, how="left")
    )

    past_orders = enriched.filter(pl.col(c.ORDER_PURCHASE_TIMESTAMP) <= cutoff)
    if past_orders.is_empty():
        raise ChurnProjectError(
            f"Nenhum pedido anterior à data de corte {cutoff.date()}. "
            "Revise 'cutoff_date' em configs/config.yaml."
        )

    features = _build_customer_features(past_orders, order_items, cutoff)

    # Clientes que compraram na janela futura => não deram churn.
    future_customers = (
        enriched.filter(
            (pl.col(c.ORDER_PURCHASE_TIMESTAMP) > cutoff)
            & (pl.col(c.ORDER_PURCHASE_TIMESTAMP) <= horizon_end)
        )
        .select(c.CUSTOMER_UNIQUE_ID)
        .unique()
        .with_columns(pl.lit(0, dtype=pl.Int64).alias(c.TARGET))
    )

    features = features.join(future_customers, on=c.CUSTOMER_UNIQUE_ID, how="left").with_columns(
        pl.col(c.TARGET).fill_null(1)
    )

    result = features.select(
        c.CUSTOMER_KEY,
        c.RECENCY_DAYS,
        c.FREQUENCY,
        c.MONETARY,
        c.MONETARY_MEAN,
        c.TENURE_DAYS,
        c.AVG_FREIGHT,
        c.AVG_REVIEW_SCORE,
        c.AVG_INSTALLMENTS,
        c.DISTINCT_PRODUCTS,
        c.AVG_DELIVERY_DAYS,
        c.LATE_DELIVERY_RATE,
        c.CUSTOMER_STATE,
        c.TARGET,
    )

    logger.info(
        "Base de features construída: %d clientes, taxa de churn = %.1f%%",
        result.height,
        100 * result[c.TARGET].mean(),
    )
    return result
