"""Reprodutibilidade: fixa sementes de aleatoriedade e calcula hashes de dados.

``random_state`` sozinho não garante reprodutibilidade — aqui semeamos todas as
fontes de aleatoriedade e disponibilizamos o hash SHA-256 dos arquivos de dados
para detectar mudanças silenciosas de versão.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Fixa todas as fontes de aleatoriedade para garantir reprodutibilidade.

    Parameters
    ----------
    seed : int, optional
        Semente a ser aplicada, by default ``RANDOM_SEED`` (42).

    Examples
    --------
    >>> seed_everything(123)
    >>> import random
    >>> random.random() == random.Random(123).random()  # doctest: +SKIP
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Semeia o RNG global legado do NumPy de propósito: garante reprodutibilidade
    # de qualquer código que use ``np.random.*`` sem passar um Generator explícito.
    np.random.seed(seed)  # noqa: NPY002


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Retorna o hash SHA-256 de um arquivo para rastrear a versão dos dados.

    Lê o arquivo em blocos para suportar arquivos maiores que a memória.

    Parameters
    ----------
    path : pathlib.Path
        Caminho do arquivo a ser hasheado.
    chunk_size : int, optional
        Tamanho do bloco de leitura em bytes, by default 1 MiB.

    Returns
    -------
    str
        Hash SHA-256 hexadecimal do arquivo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> hash_file(Path("data/raw/olist_orders_dataset.csv"))  # doctest: +SKIP
    '3a7bd3e2360a...'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para hashing: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
