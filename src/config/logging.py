"""Configuração única de logging: console ``rich`` + arquivo com rotação diária.

A configuração vem de ``configs/logging.yaml`` (formato ``dictConfig``) e é
aplicada **uma única vez**, no ponto de entrada (``main``). Os demais módulos
apenas fazem ``logger = logging.getLogger(__name__)``. Todas as mensagens de log
são emitidas em pt-BR e nunca contêm linhas de dados (apenas metadados).
"""

from __future__ import annotations

import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

from src.config.paths import CONFIGS_DIR, ROOT

# Console compartilhado por RichHandler e rich.progress: logs saem acima da barra.
CONSOLE = Console(stderr=True)


def create_console_handler() -> RichHandler:
    """Cria o handler de console ligado ao console compartilhado.

    Returns
    -------
    RichHandler
        Handler que escreve no mesmo ``Console`` usado pelas barras de progresso.
    """
    return RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=True, markup=False)


def _rename_rotated(default_name: str) -> str:
    """Converte ``logs/log.log.2026-09-27`` em ``logs/log_2026-09-27.log``.

    Parameters
    ----------
    default_name : str
        Nome padrão gerado pelo ``TimedRotatingFileHandler``.

    Returns
    -------
    str
        Nome final do arquivo rotacionado.
    """
    path = Path(default_name)
    stem, date_str = path.name.split(".log.")
    return str(path.with_name(f"{stem}_{date_str}.log"))


def create_daily_file_handler(filename: str, backup_count: int = 30) -> TimedRotatingFileHandler:
    """Cria o handler de arquivo com rotação diária à meia-noite.

    Parameters
    ----------
    filename : str
        Caminho do arquivo ativo, relativo à raiz do projeto.
    backup_count : int, optional
        Quantidade de arquivos diários mantidos, by default 30.

    Returns
    -------
    TimedRotatingFileHandler
        Handler que rotaciona para ``log_YYYY-MM-DD.log``.

    Examples
    --------
    >>> handler = create_daily_file_handler("logs/log.log")  # doctest: +SKIP
    """
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # dictConfig falha se o diretório não existe
    handler = TimedRotatingFileHandler(
        path, when="midnight", backupCount=backup_count, encoding="utf-8"
    )
    handler.namer = _rename_rotated
    return handler


def configure_logging(level: str | None = None, config_path: Path | None = None) -> None:
    """Aplica a configuração de logging do YAML. Chamar apenas no ponto de entrada.

    Parameters
    ----------
    level : str or None, optional
        Sobrescreve o nível do root (ex.: ``--log-level DEBUG``), by default None.
    config_path : pathlib.Path or None, optional
        YAML no formato ``dictConfig``. Se ``None``, usa ``configs/logging.yaml``.

    Raises
    ------
    FileNotFoundError
        Se o YAML não existir.
    ValueError
        Se o YAML ou o nível forem inválidos.

    Examples
    --------
    >>> configure_logging(level="DEBUG")  # doctest: +SKIP
    """
    yaml_path = config_path or (CONFIGS_DIR / "logging.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo de logging não encontrado: {yaml_path}")

    config: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if level is not None:
        config["root"]["level"] = level.upper()
    logging.config.dictConfig(config)  # substitui os handlers do root: chamar de novo não duplica
    logging.captureWarnings(True)  # warnings.warn() passa pelos mesmos handlers
