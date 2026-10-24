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
    """Restaura handlers e nível do root após testes que chamam ``dictConfig``."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)
    logging.captureWarnings(False)


def _write_logging_yaml(tmp_path: Path) -> Path:
    """Escreve um YAML ``dictConfig`` mínimo (arquivo em ``tmp_path``) para os testes."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"file": {"format": "%(levelname)s | %(message)s"}},
        "handlers": {
            "console": {"()": "src.config.logging.create_console_handler"},
            "file": {
                "()": "src.config.logging.create_daily_file_handler",
                "filename": (tmp_path / "logs" / "log.log").as_posix(),
                "formatter": "file",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
    }
    path = tmp_path / "logging.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_configure_logging_is_idempotent(tmp_path: Path, reset_logging) -> None:
    """Chamar ``configure_logging`` várias vezes não duplica os handlers do root."""
    yaml_path = _write_logging_yaml(tmp_path)
    log_config.configure_logging(config_path=yaml_path)
    first = len(logging.getLogger().handlers)
    log_config.configure_logging(config_path=yaml_path)
    assert len(logging.getLogger().handlers) == first == 2


def test_configure_logging_level_override(tmp_path: Path, reset_logging) -> None:
    """O argumento ``level`` sobrescreve o nível definido no YAML."""
    yaml_path = _write_logging_yaml(tmp_path)
    log_config.configure_logging(level="debug", config_path=yaml_path)
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_missing_yaml_raises(tmp_path: Path) -> None:
    """Um YAML de logging inexistente levanta ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        log_config.configure_logging(config_path=tmp_path / "ausente.yaml")


def test_project_logging_yaml_is_valid() -> None:
    """O ``configs/logging.yaml`` do projeto segue o formato dictConfig esperado."""
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "logging.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["version"] == 1
    assert config["disable_existing_loggers"] is False
    assert set(config["root"]["handlers"]) == {"console", "file"}


def test_rename_rotated_uses_daily_pattern() -> None:
    """Arquivos rotacionados seguem o padrão ``log_YYYY-MM-DD.log``."""
    renamed = log_config._rename_rotated(str(Path("logs") / "log.log.2026-09-27"))
    assert Path(renamed).name == "log_2026-09-27.log"


def test_get_paths_from_project_configs() -> None:
    """Os caminhos padrão do projeto carregam com os nomes esperados."""
    paths = get_paths()
    assert isinstance(paths, ProjectPaths)
    assert paths.data_bronze.name == "bronze"
    assert paths.data_silver.name == "silver"
    assert paths.data_gold.name == "gold"
    assert paths.model_file.suffix == ".joblib"


def test_get_paths_missing_yaml_raises(tmp_path: Path) -> None:
    """Um diretório de configs sem paths.yaml levanta FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_paths(tmp_path / "paths.yaml")


def test_ensure_dirs_creates_output_directories(tmp_path: Path) -> None:
    """``ensure_dirs`` cria os diretórios de saída ausentes."""
    paths = ProjectPaths(
        root=tmp_path,
        data_silver=tmp_path / "data" / "silver",
        data_gold=tmp_path / "data" / "gold",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        model_cards_dir=tmp_path / "reports" / "model_cards",
        datasheets_dir=tmp_path / "reports" / "datasheets",
        logs_dir=tmp_path / "logs",
    )
    paths.ensure_dirs()
    assert paths.data_silver.is_dir()
    assert paths.data_gold.is_dir()
    assert paths.figures_dir.is_dir()
    assert paths.logs_dir.is_dir()
