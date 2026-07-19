"""Testes de carregamento e validação da configuração."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.config.settings import ChurnDefinition, LightGBMParams, load_settings


def test_load_settings_from_project_configs():
    """As configurações do projeto carregam e validam sem erro."""
    settings = load_settings()
    assert settings.project.random_seed == 42
    assert settings.churn.horizon_days > 0
    assert settings.model.lightgbm.n_estimators > 0


def test_invalid_horizon_is_rejected():
    """Horizonte não positivo é rejeitado na validação."""
    with pytest.raises(ValidationError):
        ChurnDefinition(
            cutoff_date="2018-03-01",
            horizon_days=0,
            valid_order_statuses=["delivered"],
        )


def test_invalid_learning_rate_is_rejected():
    """Learning rate fora de (0, 1] é rejeitado."""
    with pytest.raises(ValidationError):
        LightGBMParams(learning_rate=1.5)


@pytest.mark.smoke
def test_settings_smoke():
    """Smoke test rápido: a configuração é utilizável ponta a ponta."""
    settings = load_settings()
    assert settings.mlflow.experiment_name
    assert settings.fairness.sensitive_feature == "customer_state"
