"""Tema e gráficos padronizados do projeto.

Módulos
-------
theme
    Paleta de cores e configuração visual compartilhada.
plots
    Gráficos de avaliação (PR, ROC, calibração, importância SHAP).
"""

from src.visualization.plots import (
    plot_calibration_curve,
    plot_precision_recall,
    plot_shap_importance,
)
from src.visualization.theme import CHURN_PALETTE, apply_theme

__all__ = [
    "CHURN_PALETTE",
    "apply_theme",
    "plot_calibration_curve",
    "plot_precision_recall",
    "plot_shap_importance",
]
