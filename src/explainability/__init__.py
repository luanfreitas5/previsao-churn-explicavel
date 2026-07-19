"""Explicabilidade do modelo de churn com SHAP.

Traduz as contribuições SHAP para linguagem de negócio, tanto de forma global
(quais fatores mais influenciam o churn na base) quanto por cliente (por que
este cliente foi classificado como risco).

Módulos
-------
shap_explainer
    Classe ``ChurnExplainer`` e utilidades de tradução para negócio.
"""

from src.explainability.shap_explainer import ChurnExplainer, InstanceContribution

__all__ = ["ChurnExplainer", "InstanceContribution"]
