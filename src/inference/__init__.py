"""Inferência e predição de churn.

Módulos
-------
predictor
    Aplica o pipeline treinado para gerar probabilidades e faixas de risco.
"""

from src.inference.predictor import predict_proba, score_customers

__all__ = ["predict_proba", "score_customers"]
