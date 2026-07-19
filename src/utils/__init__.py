"""Utilitários compartilhados do projeto.

Módulos
-------
progress
    Fábrica de barras de progresso ``rich`` padronizadas.
io
    Leitura e escrita de artefatos JSON com criação de diretórios.
"""

from src.utils.io import read_json, write_json
from src.utils.progress import build_progress

__all__ = ["build_progress", "read_json", "write_json"]
