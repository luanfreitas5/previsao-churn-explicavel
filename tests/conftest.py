"""Fixtures compartilhadas: tabelas Olist sintéticas, base de features e modelo.

Nunca usamos dados de produção nos testes — apenas DataFrames sintéticos
pequenos e determinísticos.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from src.config.settings import ChurnDefinition, LightGBMParams
from src.constants import defaults
from src.models.pipeline import build_model

RANDOM_SEED = 42


@pytest.fixture
def churn_definition() -> ChurnDefinition:
    """Definição de churn de referência para os testes."""
    return ChurnDefinition(
        cutoff_date=date(2018, 3, 1),
        horizon_days=180,
        valid_order_statuses=["delivered", "shipped", "invoiced", "approved"],
    )


@pytest.fixture
def olist_tables() -> dict[str, pl.DataFrame]:
    """Tabelas Olist sintéticas com resultado de churn conhecido.

    Cenário (corte = 2018-03-01, horizonte = 180 dias, fim = 2018-08-28):

    - ``u1``: pedido passado + recompra dentro da janela  -> churn = 0
    - ``u2``: apenas pedido passado                        -> churn = 1
    - ``u3``: pedido passado + recompra fora da janela     -> churn = 1
    """
    orders = pl.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "customer_id": ["ca1", "ca2", "cb1", "cc1", "cc2"],
            "order_status": ["delivered"] * 5,
            "order_purchase_timestamp": [
                datetime(2018, 1, 1),
                datetime(2018, 4, 1),  # recompra de u1 (dentro da janela)
                datetime(2018, 1, 15),
                datetime(2017, 12, 1),
                datetime(2019, 1, 1),  # recompra de u3 (fora da janela)
            ],
            "order_delivered_customer_date": [
                datetime(2018, 1, 10),
                datetime(2018, 4, 10),
                datetime(2018, 1, 30),
                datetime(2017, 12, 20),
                datetime(2019, 1, 10),
            ],
            "order_estimated_delivery_date": [
                datetime(2018, 1, 8),  # atrasou
                datetime(2018, 4, 15),
                datetime(2018, 2, 5),
                datetime(2017, 12, 25),
                datetime(2019, 1, 20),
            ],
        }
    )
    customers = pl.DataFrame(
        {
            "customer_id": ["ca1", "ca2", "cb1", "cc1", "cc2"],
            "customer_unique_id": ["u1", "u1", "u2", "u3", "u3"],
            "customer_state": ["SP", "SP", "RJ", "MG", "MG"],
        }
    )
    order_items = pl.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "product_id": ["p1", "p2", "p3", "p4", "p4"],
            "price": [100.0, 150.0, 80.0, 200.0, 210.0],
            "freight_value": [10.0, 12.0, 8.0, 20.0, 21.0],
        }
    )
    payments = pl.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "payment_value": [110.0, 162.0, 88.0, 220.0, 231.0],
            "payment_installments": [1, 2, 1, 3, 3],
        }
    )
    reviews = pl.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "review_score": [5, 4, 3, 2, 5],
        }
    )
    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
    }


@pytest.fixture
def features_frame() -> pl.DataFrame:
    """Base de features sintética (~300 clientes) com sinal de churn.

    O churn depende da recência e da frequência: clientes com maior recência e
    menor frequência têm maior probabilidade de churn — permitindo testes
    comportamentais direcionais do modelo.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = 300
    recency = rng.uniform(1, 500, n)
    frequency = rng.integers(1, 6, n)

    # Probabilidade latente de churn cresce com recência e cai com frequência.
    logit = 0.006 * recency - 0.5 * frequency - 1.0
    proba = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(size=n) < proba).astype(int)

    data = {
        defaults.CUSTOMER_KEY: [f"u{i}" for i in range(n)],
        "recency_days": recency,
        "frequency": frequency.astype(int),
        "monetary": rng.uniform(50, 2000, n),
        "monetary_mean": rng.uniform(50, 800, n),
        "tenure_days": rng.uniform(1, 600, n),
        "avg_freight_value": rng.uniform(5, 60, n),
        "avg_review_score": rng.uniform(1, 5, n),
        "avg_installments": rng.uniform(1, 6, n),
        "distinct_products": rng.integers(1, 8, n).astype(int),
        "avg_delivery_days": rng.uniform(2, 30, n),
        "late_delivery_rate": rng.uniform(0, 1, n),
        "customer_state": rng.choice(["SP", "RJ", "MG", "RS", "BA"], n),
        defaults.TARGET: churn,
    }
    return pl.DataFrame(data)


@pytest.fixture
def trained_model(features_frame: pl.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Modelo LightGBM treinado em dados sintéticos, com features e alvo pandas."""
    pdf = features_frame.to_pandas()
    x = pdf[defaults.FEATURES]
    y = pdf[defaults.TARGET].astype(int)
    params = LightGBMParams(n_estimators=50, num_leaves=8)
    model = build_model(params, seed=RANDOM_SEED)
    model.fit(x, y)
    return model, x, y
