"""Testes da validação cruzada e do ajuste final do modelo."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.config.settings import LightGBMParams
from src.models.pipeline import build_model
from src.training.trainer import cross_validate_model, fit_model


@pytest.fixture
def untrained_pipeline() -> Pipeline:
    """Pipeline LightGBM leve e não treinado para os testes de treino."""
    return build_model(LightGBMParams(n_estimators=40, num_leaves=8), seed=42)


def test_cross_validate_returns_metric_with_uncertainty(untrained_pipeline, trained_model) -> None:
    """A CV retorna média, desvio, IC 95% e um score por fold."""
    _, x, y = trained_model
    result = cross_validate_model(untrained_pipeline, x, y, n_splits=3, seed=42)
    assert set(result) == {"mean", "std", "ci95", "scores"}
    assert len(result["scores"]) == 3
    assert result["ci95"] >= 0.0
    assert 0.0 <= result["mean"] <= 1.0


def test_fit_model_returns_usable_pipeline(untrained_pipeline, trained_model) -> None:
    """O ajuste retorna um pipeline capaz de prever probabilidades."""
    _, x, y = trained_model
    fitted = fit_model(untrained_pipeline, x, y)
    proba = fitted.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert isinstance(x, pd.DataFrame)
