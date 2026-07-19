"""Carregamento e divisão dos dados brutos da Olist.

Módulos
-------
loader
    Lê os CSVs brutos da Olist como ``polars.DataFrame`` com tipagem de datas.
splitter
    Divide a base de features em treino e teste de forma estratificada.
"""

from src.data.loader import OlistTables, load_olist_tables
from src.data.splitter import split_train_test

__all__ = ["OlistTables", "load_olist_tables", "split_train_test"]
