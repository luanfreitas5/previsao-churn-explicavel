"""Testes da divisão estratificada treino/teste."""

from __future__ import annotations

import pytest

from src.constants import defaults
from src.data.splitter import split_train_test


def test_split_sizes_sum_to_total(features_frame):
    """A soma de treino e teste é igual ao total de clientes."""
    train, test = split_train_test(features_frame, test_size=0.25, seed=42)
    assert train.height + test.height == features_frame.height


def test_split_is_deterministic(features_frame):
    """Mesma semente produz a mesma divisão (reprodutibilidade)."""
    a_train, a_test = split_train_test(features_frame, seed=42)
    b_train, b_test = split_train_test(features_frame, seed=42)
    assert a_train[defaults.CUSTOMER_KEY].to_list() == b_train[defaults.CUSTOMER_KEY].to_list()
    assert a_test[defaults.CUSTOMER_KEY].to_list() == b_test[defaults.CUSTOMER_KEY].to_list()


def test_missing_target_raises(features_frame):
    """Ausência da coluna alvo levanta KeyError."""
    with pytest.raises(KeyError):
        split_train_test(features_frame.drop(defaults.TARGET))
