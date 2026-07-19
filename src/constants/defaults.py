"""Listas padrão de features, alvo e rótulos usados na modelagem de churn."""

from __future__ import annotations

from typing import Final

from src.constants import columns as c

# Features numéricas usadas pelo modelo (todas derivadas de dados <= corte).
NUMERIC_FEATURES: Final[list[str]] = [
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
]

# Feature categórica (UF do cliente) — também usada como atributo sensível.
CATEGORICAL_FEATURES: Final[list[str]] = [c.CUSTOMER_STATE]

# Todas as features (ordem estável para o pipeline e o SHAP).
FEATURES: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET: Final[str] = c.TARGET
CUSTOMER_KEY: Final[str] = c.CUSTOMER_KEY

# Mapeamento de rótulos do alvo.
LABELS: Final[dict[int, str]] = {0: "Recompra", 1: "Churn (não recompra)"}

# Nomes de negócio das features (para explicações SHAP legíveis).
FEATURE_LABELS_PT: Final[dict[str, str]] = {
    c.RECENCY_DAYS: "Dias desde a última compra",
    c.FREQUENCY: "Número de pedidos",
    c.MONETARY: "Valor total gasto (BRL)",
    c.MONETARY_MEAN: "Ticket médio (BRL)",
    c.TENURE_DAYS: "Tempo como cliente (dias)",
    c.AVG_FREIGHT: "Frete médio (BRL)",
    c.AVG_REVIEW_SCORE: "Nota média das avaliações",
    c.AVG_INSTALLMENTS: "Média de parcelas",
    c.DISTINCT_PRODUCTS: "Produtos distintos comprados",
    c.AVG_DELIVERY_DAYS: "Prazo médio de entrega (dias)",
    c.LATE_DELIVERY_RATE: "Taxa de entregas atrasadas",
    c.CUSTOMER_STATE: "UF do cliente",
}
