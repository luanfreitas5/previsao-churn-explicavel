"""Agregações a nível de pedido usadas pela engenharia de features RFM.

Cada função reduz uma tabela transacional (múltiplas linhas por pedido) a uma
linha por ``order_id``, prontas para serem unidas aos pedidos e agregadas por
cliente em :mod:`src.features.rfm`.
"""

from __future__ import annotations

import polars as pl

from src.constants import columns as c


def aggregate_payments(payments: pl.DataFrame) -> pl.DataFrame:
    """Agrega pagamentos por pedido (valor total e média de parcelas).

    Parameters
    ----------
    payments : polars.DataFrame
        Tabela ``olist_order_payments_dataset``.

    Returns
    -------
    polars.DataFrame
        Colunas ``order_id``, ``payment_value_sum``, ``installments_mean``.
    """
    return payments.group_by(c.ORDER_ID).agg(
        pl.col(c.PAYMENT_VALUE).sum().alias("payment_value_sum"),
        pl.col(c.PAYMENT_INSTALLMENTS).mean().alias("installments_mean"),
    )


def aggregate_items(order_items: pl.DataFrame) -> pl.DataFrame:
    """Agrega itens por pedido (frete total e preço total).

    Parameters
    ----------
    order_items : polars.DataFrame
        Tabela ``olist_order_items_dataset``.

    Returns
    -------
    polars.DataFrame
        Colunas ``order_id``, ``freight_sum``, ``price_sum``.
    """
    return order_items.group_by(c.ORDER_ID).agg(
        pl.col(c.FREIGHT_VALUE).sum().alias("freight_sum"),
        pl.col(c.PRICE).sum().alias("price_sum"),
    )


def aggregate_reviews(reviews: pl.DataFrame) -> pl.DataFrame:
    """Agrega avaliações por pedido (nota média).

    Parameters
    ----------
    reviews : polars.DataFrame
        Tabela ``olist_order_reviews_dataset``.

    Returns
    -------
    polars.DataFrame
        Colunas ``order_id``, ``review_score_mean``.
    """
    return reviews.group_by(c.ORDER_ID).agg(
        pl.col(c.REVIEW_SCORE).mean().alias("review_score_mean"),
    )


def add_delivery_metrics(orders: pl.DataFrame) -> pl.DataFrame:
    """Adiciona métricas de prazo de entrega a nível de pedido.

    Calcula o tempo de entrega (em dias) e um indicador de atraso comparando a
    data de entrega real com a estimada. Pedidos não entregues ficam nulos.

    Parameters
    ----------
    orders : polars.DataFrame
        Pedidos com ``order_purchase_timestamp``,
        ``order_delivered_customer_date`` e ``order_estimated_delivery_date``.

    Returns
    -------
    polars.DataFrame
        Pedidos com colunas ``delivery_days`` (float) e ``is_late`` (int/nulo).
    """
    return orders.with_columns(
        (
            (pl.col(c.ORDER_DELIVERED_CUSTOMER_DATE) - pl.col(c.ORDER_PURCHASE_TIMESTAMP))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("delivery_days")
        ),
        (
            (pl.col(c.ORDER_DELIVERED_CUSTOMER_DATE) > pl.col(c.ORDER_ESTIMATED_DELIVERY_DATE))
            .cast(pl.Int8)
            .alias("is_late")
        ),
    )
