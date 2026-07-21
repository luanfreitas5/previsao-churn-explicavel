"""Testes da engenharia de features RFM e da rotulagem de churn."""

from __future__ import annotations

import polars as pl

from src.config.settings import ChurnDefinition
from src.constants import columns as c
from src.features.rfm import build_churn_features


def _build(tables: dict[str, pl.DataFrame], churn: ChurnDefinition) -> pl.DataFrame:
    """Atalho para construir a base de features a partir das tabelas sintéticas."""
    return build_churn_features(
        orders=tables["orders"],
        customers=tables["customers"],
        order_items=tables["order_items"],
        payments=tables["payments"],
        reviews=tables["reviews"],
        churn=churn,
    )


def test_churn_label_matches_repurchase_window(olist_tables, churn_definition):
    """Churn = 0 apenas para quem recompra dentro da janela; 1 caso contrário."""
    features = _build(olist_tables, churn_definition)
    labels = dict(zip(features[c.CUSTOMER_KEY], features[c.TARGET], strict=True))

    assert labels["u1"] == 0  # recompra dentro da janela
    assert labels["u2"] == 1  # sem recompra
    assert labels["u3"] == 1  # recompra fora da janela => churn


def test_base_only_includes_customers_active_before_cutoff(olist_tables, churn_definition):
    """A base contém exatamente os clientes com pedido até o corte."""
    features = _build(olist_tables, churn_definition)
    assert set(features[c.CUSTOMER_KEY]) == {"u1", "u2", "u3"}


def test_no_future_leakage_in_recency(olist_tables, churn_definition):
    """Recência é medida em relação ao corte (nunca negativa: sem vazamento)."""
    features = _build(olist_tables, churn_definition)
    assert (features[c.RECENCY_DAYS] >= 0).all()
    assert (features[c.TENURE_DAYS] >= 0).all()


def test_frequency_counts_only_past_orders(olist_tables, churn_definition):
    """A frequência de u1 conta só o pedido anterior ao corte (não a recompra)."""
    features = _build(olist_tables, churn_definition)
    freq_u1 = features.filter(pl.col(c.CUSTOMER_KEY) == "u1")[c.FREQUENCY].item()
    assert freq_u1 == 1


def test_late_delivery_flag_detected(olist_tables, churn_definition):
    """u1 teve entrega atrasada no pedido passado => taxa de atraso > 0."""
    features = _build(olist_tables, churn_definition)
    late_u1 = features.filter(pl.col(c.CUSTOMER_KEY) == "u1")[c.LATE_DELIVERY_RATE].item()
    assert late_u1 == 1.0
