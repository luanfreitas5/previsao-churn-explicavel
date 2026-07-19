"""Treino e validação cruzada do modelo de churn.

Módulos
-------
trainer
    Validação cruzada estratificada com intervalo de confiança e ajuste final.
"""

from src.training.trainer import cross_validate_model, fit_model

__all__ = ["cross_validate_model", "fit_model"]
