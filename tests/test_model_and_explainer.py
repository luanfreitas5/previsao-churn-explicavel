"""Testes do pipeline de modelagem, predição e explicabilidade SHAP."""

from __future__ import annotations

import pytest
from src.constants import defaults
from src.explainability.shap_explainer import ChurnExplainer
from src.inference.predictor import predict_proba, score_customers
from src.models.pipeline import get_feature_names


@pytest.mark.ml
def test_predict_proba_in_unit_interval(trained_model):
    """As probabilidades previstas ficam em [0, 1]."""
    model, x, _ = trained_model
    proba = predict_proba(model, x)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert len(proba) == len(x)


@pytest.mark.ml
def test_higher_recency_increases_churn_risk(trained_model):
    """Comportamental: aumentar a recência não deve reduzir o risco de churn.

    Compara um cliente base com uma cópia de recência muito maior; o modelo
    aprende a associação recência alta -> churn nos dados sintéticos.
    """
    model, x, _ = trained_model
    base = x.iloc[[0]].copy()
    high_recency = base.copy()
    high_recency["recency_days"] = x["recency_days"].max() + 100

    proba_base = predict_proba(model, base)[0]
    proba_high = predict_proba(model, high_recency)[0]
    assert proba_high >= proba_base - 1e-6


def test_score_customers_adds_proba_and_band(trained_model):
    """A pontuação adiciona probabilidade e faixa de risco, ordenando por risco."""
    model, x, _ = trained_model
    features = x.copy()
    features[defaults.CUSTOMER_KEY] = [f"u{i}" for i in range(len(x))]
    scored = score_customers(model, features)
    assert "churn_proba" in scored.columns
    assert "risk_band" in scored.columns
    # Ordenado por probabilidade decrescente.
    assert scored["churn_proba"].is_monotonic_decreasing


def test_global_importance_covers_all_base_features(trained_model):
    """A importância global agrega todas as features de origem."""
    model, x, _ = trained_model
    explainer = ChurnExplainer(model)
    importances = explainer.explain_global(x)
    # Uma linha por feature de origem (numéricas + UF agregada).
    assert len(importances) == len(defaults.NUMERIC_FEATURES) + 1
    assert (importances["mean_abs_shap"] >= 0).all()


def test_instance_explanation_returns_contributions(trained_model):
    """A explicação por cliente retorna contribuições ordenadas por magnitude."""
    model, x, _ = trained_model
    explainer = ChurnExplainer(model)
    contributions = explainer.explain_instance(x.iloc[[0]], top_n=5)
    assert len(contributions) <= 5
    magnitudes = [abs(item.shap_value) for item in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_get_feature_names_after_fit(trained_model):
    """Os nomes das features pós-processamento incluem colunas one-hot de UF."""
    model, _, _ = trained_model
    names = get_feature_names(model)
    assert any(name.startswith("customer_state") for name in names)
