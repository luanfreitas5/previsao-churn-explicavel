"""Leitura e escrita de artefatos JSON com criação automática de diretórios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    """Escreve um dicionário como JSON, criando o diretório pai se necessário.

    Parameters
    ----------
    data : dict
        Conteúdo a ser serializado.
    path : pathlib.Path
        Caminho de destino do arquivo JSON.
    indent : int, optional
        Indentação do JSON, by default 2.

    Examples
    --------
    >>> write_json({"f1": 0.8}, Path("reports/metrics.json"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent, default=str)


def read_json(path: Path) -> dict[str, Any]:
    """Lê um arquivo JSON e retorna seu conteúdo como dicionário.

    Parameters
    ----------
    path : pathlib.Path
        Caminho do arquivo JSON.

    Returns
    -------
    dict
        Conteúdo desserializado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> read_json(Path("reports/metrics.json"))  # doctest: +SKIP
    {'f1': 0.8}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
