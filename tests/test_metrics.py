"""Testes das métricas de avaliação e do bootstrap."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import bootstrap_metric, evaluate_classification


def test_perfect_separation_gives_max_scores():
    """Separação perfeita produz ROC-AUC e average precision iguais a 1."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_classification(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_bootstrap_ci_brackets_mean():
    """O intervalo de confiança do bootstrap contém a própria média."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_proba = rng.uniform(size=200)
    ci = bootstrap_metric(y_true, y_proba, n_iterations=200, seed=0)
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


def test_metrics_are_bounded():
    """Todas as métricas de probabilidade permanecem no intervalo [0, 1]."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(size=100)
    metrics = evaluate_classification(y_true, y_proba)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, name
