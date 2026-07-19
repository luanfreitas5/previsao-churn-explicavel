"""Testes do contrato de dados da base de features (pandera)."""

from __future__ import annotations

import polars as pl
import pytest
from src.exceptions import DataValidationError
from src.schemas.features import validate_features


def test_valid_features_pass(features_frame):
    """Uma base bem-formada passa na validação sem erros."""
    validated = validate_features(features_frame)
    assert validated.height == features_frame.height


def test_negative_recency_is_rejected(features_frame):
    """Recência negativa (vazamento) viola o contrato e levanta erro."""
    corrupted = features_frame.with_columns(pl.lit(-5.0).alias("recency_days"))
    with pytest.raises(DataValidationError):
        validate_features(corrupted)


def test_invalid_target_value_is_rejected(features_frame):
    """Um alvo fora de {0, 1} viola o contrato."""
    corrupted = features_frame.with_columns(pl.lit(2).alias("churn"))
    with pytest.raises(DataValidationError):
        validate_features(corrupted)


def test_unexpected_column_is_rejected(features_frame):
    """Coluna inesperada é rejeitada (Config.strict)."""
    corrupted = features_frame.with_columns(pl.lit(1).alias("coluna_extra"))
    with pytest.raises(DataValidationError):
        validate_features(corrupted)
