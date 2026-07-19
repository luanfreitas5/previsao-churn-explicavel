"""Estágio de construção e validação da base de features de churn."""

from __future__ import annotations

import logging

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.data.loader import load_olist_tables
from src.data.splitter import split_train_test
from src.features.rfm import build_churn_features
from src.schemas.features import validate_features

logger = logging.getLogger(__name__)


def run_build_features(settings: Settings, paths: ProjectPaths) -> None:
    """Constrói, valida e persiste a base de features + divisão treino/teste.

    Lê os CSVs brutos da Olist, gera a base RFM + alvo de churn, valida contra o
    contrato de dados e escreve ``churn_features.parquet``, ``train.parquet`` e
    ``test.parquet`` em ``data/processed``.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Examples
    --------
    >>> run_build_features(settings, paths)  # doctest: +SKIP
    """
    paths.ensure_dirs()
    tables = load_olist_tables(paths.data_raw)

    features = build_churn_features(
        orders=tables.orders,
        customers=tables.customers,
        order_items=tables.order_items,
        payments=tables.payments,
        reviews=tables.reviews,
        churn=settings.churn,
    )
    features = validate_features(features)
    features.write_parquet(paths.features_file)
    logger.info("Base de features validada e salva em %s", paths.features_file)

    train, test = split_train_test(
        features,
        test_size=settings.split.test_size,
        stratify=settings.split.stratify,
        seed=settings.project.random_seed,
    )
    train.write_parquet(paths.train_file)
    test.write_parquet(paths.test_file)
    logger.info("Divisões salvas: %s e %s", paths.train_file, paths.test_file)
