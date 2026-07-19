"""Gestão de configuração, paths, logging e reprodutibilidade.

Módulos
-------
paths
    Centraliza todos os caminhos do projeto com ``pathlib.Path``.
settings
    Carrega e valida as configurações YAML + ``.env`` com Pydantic.
logging
    Configura logging com ``RichHandler`` e arquivo rotativo diário.
environment
    Fixa sementes de aleatoriedade e calcula hashes de arquivos.
"""

from src.config.environment import hash_file, seed_everything
from src.config.logging import configure_logging
from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings

__all__ = [
    "ProjectPaths",
    "Settings",
    "configure_logging",
    "get_paths",
    "hash_file",
    "load_settings",
    "seed_everything",
]
