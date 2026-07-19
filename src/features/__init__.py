"""Engenharia de features RFM e rotulagem de churn (não recompra).

Módulos
-------
engineering
    Agregações a nível de pedido (pagamentos, itens, avaliações, prazos).
rfm
    Monta a base de features RFM por cliente e o alvo de churn a partir da
    data de corte, sem vazamento de futuro.
"""

from src.features.rfm import build_churn_features

__all__ = ["build_churn_features"]
