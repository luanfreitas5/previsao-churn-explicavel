"""Avaliação rigorosa: métricas com incerteza, calibração e justiça.

Módulos
-------
metrics
    Métricas de classificação e intervalo de confiança via bootstrap.
calibration
    Brier score e curva de confiabilidade das probabilidades.
fairness
    Auditoria de disparidade por subgrupo (UF do cliente) com fairlearn.
"""

from src.evaluation.calibration import compute_calibration
from src.evaluation.fairness import audit_fairness
from src.evaluation.metrics import bootstrap_metric, evaluate_classification

__all__ = [
    "audit_fairness",
    "bootstrap_metric",
    "compute_calibration",
    "evaluate_classification",
]
