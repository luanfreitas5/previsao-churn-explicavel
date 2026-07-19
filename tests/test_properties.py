"""Testes baseados em propriedades (hypothesis) para funções de avaliação."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from src.evaluation.metrics import evaluate_classification


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=20, max_value=200),
)
def test_classification_metrics_always_bounded(seed: int, n: int):
    """Invariante: todas as métricas permanecem em [0, 1] para qualquer entrada.

    Garante que ambos os rótulos estejam presentes para evitar métricas
    indefinidas.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    # Força a presença das duas classes.
    y_true[0], y_true[1] = 0, 1
    y_proba = rng.uniform(size=n)

    metrics = evaluate_classification(y_true, y_proba)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


@settings(max_examples=50, deadline=None)
@given(threshold=st.floats(min_value=0.0, max_value=1.0))
def test_threshold_does_not_change_probability_metrics(threshold: float):
    """Invariante: ROC-AUC e Brier não dependem do limiar de decisão."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, 100)
    y_true[0], y_true[1] = 0, 1
    y_proba = rng.uniform(size=100)

    at_default = evaluate_classification(y_true, y_proba, threshold=0.5)
    at_threshold = evaluate_classification(y_true, y_proba, threshold=threshold)
    assert at_default["roc_auc"] == at_threshold["roc_auc"]
    assert at_default["brier"] == at_threshold["brier"]
