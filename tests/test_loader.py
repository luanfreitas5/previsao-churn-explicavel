"""Testes do carregamento de dados brutos e utilidades de IO."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.config.environment import hash_file, seed_everything
from src.data.loader import load_olist_tables
from src.exceptions import DataNotFoundError
from src.utils.io import read_json, write_json


def test_missing_raw_dir_raises(tmp_path: Path):
    """Carregar de um diretório sem os CSVs levanta DataNotFoundError."""
    with pytest.raises(DataNotFoundError):
        load_olist_tables(tmp_path / "inexistente")


def test_json_roundtrip(tmp_path: Path):
    """Escrever e ler um JSON preserva o conteúdo."""
    payload = {"metric": 0.61, "nome": "acentuação"}
    path = tmp_path / "out" / "metrics.json"
    write_json(payload, path)
    assert read_json(path) == payload


def test_hash_file_is_stable(tmp_path: Path):
    """O hash de um arquivo é determinístico para o mesmo conteúdo."""
    path = tmp_path / "data.txt"
    path.write_text("conteúdo", encoding="utf-8")
    assert hash_file(path) == hash_file(path)


def test_hash_missing_file_raises(tmp_path: Path):
    """Hashear arquivo inexistente levanta FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "nao_existe.txt")


def test_seed_everything_is_reproducible():
    """seed_everything torna a geração aleatória reprodutível."""
    import random

    seed_everything(123)
    first = [random.random() for _ in range(3)]
    seed_everything(123)
    second = [random.random() for _ in range(3)]
    assert first == second
