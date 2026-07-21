"""Testes de configuração de logging e resolução de caminhos do projeto."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.config import logging as log_config
from src.config.paths import ProjectPaths, get_paths


@pytest.fixture
def reset_logging():
    """Reseta o estado global de logging antes e depois do teste."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    log_config._STATE["configured"] = False
    root.handlers = []
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)
    log_config._STATE["configured"] = False


def test_configure_logging_adds_handlers(tmp_path: Path, reset_logging) -> None:
    """Configurar o logging cria handlers de console e arquivo e grava o log."""
    yaml_path = tmp_path / "logging.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"level": "DEBUG", "console": True, "file": True, "log_dir": "logs"}),
        encoding="utf-8",
    )
    logger = log_config.configure_logging(yaml_path)
    logger.info("mensagem de teste")

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1
    # Segunda chamada é idempotente (não duplica handlers).
    before = len(logger.handlers)
    log_config.configure_logging(yaml_path)
    assert len(logger.handlers) == before


def test_configure_logging_missing_yaml_uses_defaults(reset_logging) -> None:
    """Sem YAML, o logging usa padrões seguros e ainda configura o logger."""
    logger = log_config.configure_logging(Path("nao_existe.yaml"))
    assert logger.level == logging.INFO


def test_load_config_missing_returns_defaults(tmp_path: Path) -> None:
    """A leitura de um YAML inexistente retorna a configuração padrão."""
    cfg = log_config._load_config(tmp_path / "ausente.yaml")
    assert cfg["level"] == "INFO"
    assert cfg["console"] is True


def test_get_paths_from_project_configs() -> None:
    """Os caminhos padrão do projeto carregam com os nomes esperados."""
    paths = get_paths()
    assert isinstance(paths, ProjectPaths)
    assert paths.data_raw.name == "raw"
    assert paths.model_file.suffix == ".joblib"


def test_get_paths_missing_yaml_raises(tmp_path: Path) -> None:
    """Um diretório de configs sem paths.yaml levanta FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_paths(tmp_path / "paths.yaml")


def test_ensure_dirs_creates_output_directories(tmp_path: Path) -> None:
    """``ensure_dirs`` cria os diretórios de saída ausentes."""
    paths = ProjectPaths(
        root=tmp_path,
        data_interim=tmp_path / "data" / "interim",
        data_processed=tmp_path / "data" / "processed",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        model_cards_dir=tmp_path / "reports" / "model_cards",
        datasheets_dir=tmp_path / "reports" / "datasheets",
        logs_dir=tmp_path / "logs",
    )
    paths.ensure_dirs()
    assert paths.data_processed.is_dir()
    assert paths.figures_dir.is_dir()
    assert paths.logs_dir.is_dir()
