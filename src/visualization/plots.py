"""Gráficos de avaliação do modelo de churn (salvos em ``reports/figures``).

Cada figura é salva em ``.png`` (300 dpi) e ``.svg`` para uso em relatórios e
publicações.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import PrecisionRecallDisplay

from src.visualization.theme import CHURN_PALETTE, apply_theme

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    """Salva uma figura em PNG (300 dpi) e SVG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a salvar.
    figures_dir : pathlib.Path
        Diretório de destino.
    name : str
        Nome base do arquivo (sem extensão).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", figures_dir / f"{name}.png")


def plot_precision_recall(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    figures_dir: Path,
) -> None:
    """Plota e salva a curva Precision-Recall.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros (0/1).
    y_proba : numpy.ndarray
        Probabilidades previstas de churn.
    figures_dir : pathlib.Path
        Diretório de destino das figuras.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_proba, ax=ax, color=CHURN_PALETTE["churn"], name="LightGBM"
    )
    ax.set_title("Curva Precision-Recall — Churn")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    _save(fig, figures_dir, "precision_recall")


def plot_calibration_curve(
    prob_true: list[float],
    prob_pred: list[float],
    figures_dir: Path,
) -> None:
    """Plota e salva a curva de confiabilidade (calibração).

    Parameters
    ----------
    prob_true : list of float
        Frequência observada de churn por faixa.
    prob_pred : list of float
        Probabilidade média prevista por faixa.
    figures_dir : pathlib.Path
        Diretório de destino das figuras.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
    ax.plot(prob_pred, prob_true, "o-", color=CHURN_PALETTE["neutro"], label="Modelo")
    ax.set_title("Curva de Confiabilidade — Calibração")
    ax.set_xlabel("Probabilidade média prevista")
    ax.set_ylabel("Fração observada de churn")
    ax.legend()
    _save(fig, figures_dir, "calibration_curve")


def plot_shap_importance(importances: pd.DataFrame, figures_dir: Path) -> None:
    """Plota e salva a importância global das features (SHAP médio absoluto).

    Parameters
    ----------
    importances : pandas.DataFrame
        Colunas ``feature`` e ``mean_abs_shap`` (já ordenadas).
    figures_dir : pathlib.Path
        Diretório de destino das figuras.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        importances["feature"][::-1],
        importances["mean_abs_shap"][::-1],
        color=CHURN_PALETTE["neutro"],
    )
    ax.set_title("Importância Global das Features (SHAP)")
    ax.set_xlabel("Impacto médio absoluto na previsão de churn")
    _save(fig, figures_dir, "shap_importance")
