"""Testes das figuras de avaliação e do tema visual compartilhado."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend não interativo para testes

from src.visualization.plots import (
    plot_calibration_curve,
    plot_precision_recall,
    plot_shap_importance,
)
from src.visualization.theme import CHURN_PALETTE, apply_theme


def test_palette_has_expected_keys() -> None:
    """A paleta define as cores de churn, retenção, neutro e destaque."""
    assert set(CHURN_PALETTE) == {"churn", "retencao", "neutro", "destaque"}
    assert all(color.startswith("#") for color in CHURN_PALETTE.values())


def test_apply_theme_sets_dpi() -> None:
    """Aplicar o tema define o DPI de figura e de salvamento."""

    apply_theme()
    assert plt.rcParams["figure.dpi"] == 120
    assert plt.rcParams["savefig.dpi"] == 300


def _assert_png_and_svg(figures_dir: Path, name: str) -> None:
    """Verifica que a figura foi salva em PNG e SVG."""
    assert (figures_dir / f"{name}.png").exists()
    assert (figures_dir / f"{name}.svg").exists()


def test_plot_precision_recall_saves_files(tmp_path: Path) -> None:
    """A curva Precision-Recall é salva em PNG e SVG."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    plot_precision_recall(y_true, y_proba, tmp_path)
    _assert_png_and_svg(tmp_path, "precision_recall")


def test_plot_calibration_curve_saves_files(tmp_path: Path) -> None:
    """A curva de confiabilidade é salva em PNG e SVG."""
    plot_calibration_curve([0.1, 0.5, 0.9], [0.15, 0.45, 0.85], tmp_path)
    _assert_png_and_svg(tmp_path, "calibration_curve")


def test_plot_shap_importance_saves_files(tmp_path: Path) -> None:
    """A importância global SHAP é salva em PNG e SVG."""
    importances = pd.DataFrame(
        {"feature": ["recency_days", "frequency"], "mean_abs_shap": [0.4, 0.2]}
    )
    plot_shap_importance(importances, tmp_path)
    _assert_png_and_svg(tmp_path, "shap_importance")
