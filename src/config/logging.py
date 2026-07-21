"""Configura o logging do projeto com ``RichHandler`` e arquivo rotativo diário.

Todas as mensagens de log são emitidas em pt-BR. A configuração é lida de
``configs/logging.yaml`` e aplicada uma única vez por processo.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import yaml
from rich.logging import RichHandler

from src.config.paths import CONFIGS_DIR, ROOT

_STATE: dict[str, bool] = {"configured": False}


def configure_logging(logging_yaml: Path | None = None) -> logging.Logger:
    """Configura os handlers de console (rich) e arquivo do projeto.

    Idempotente: chamadas subsequentes não duplicam handlers.

    Parameters
    ----------
    logging_yaml : pathlib.Path or None, optional
        Caminho do YAML de logging. Se ``None``, usa ``configs/logging.yaml``.

    Returns
    -------
    logging.Logger
        Logger raiz configurado.

    Examples
    --------
    >>> logger = configure_logging()
    >>> logger.info("Pipeline iniciada")  # doctest: +SKIP
    """
    root_logger = logging.getLogger()
    if _STATE["configured"]:
        return root_logger

    yaml_path = logging_yaml or (CONFIGS_DIR / "logging.yaml")
    cfg = _load_config(yaml_path)

    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    root_logger.setLevel(level)

    handlers: list[logging.Handler] = []

    if cfg.get("console", True):
        rich_handler = RichHandler(rich_tracebacks=True, show_path=True, markup=True)
        rich_handler.setFormatter(logging.Formatter("%(name)s \t %(message)s"))
        handlers.append(rich_handler)

    if cfg.get("file", True):
        log_dir = ROOT / cfg.get("log_dir", "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_{date.today():%Y-%m-%d}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                cfg.get("file_format", "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"),
                datefmt=cfg.get("date_format", "%Y-%m-%d %H:%M:%S"),
            )
        )
        handlers.append(file_handler)

    for handler in handlers:
        root_logger.addHandler(handler)

    _STATE["configured"] = True
    return root_logger


def _load_config(yaml_path: Path) -> dict:
    """Lê o YAML de logging, retornando padrões seguros se ausente.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Caminho do YAML de logging.

    Returns
    -------
    dict
        Configuração de logging (ou padrões se o arquivo não existir).
    """
    if not yaml_path.exists():
        return {"level": "INFO", "console": True, "file": True, "log_dir": "logs"}
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
