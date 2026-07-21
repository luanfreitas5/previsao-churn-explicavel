"""Orquestração principal da pipeline de previsão de churn explicável.

Expõe uma CLI (via ``argparse``) para executar cada estágio isoladamente ou a
pipeline completa:

.. code-block:: bash

    python -m src.main features   # constrói e valida a base de features
    python -m src.main train      # treina o modelo (CV + MLflow)
    python -m src.main evaluate   # avalia (métricas, calibração, justiça)
    python -m src.main explain    # explicabilidade global (SHAP)
    python -m src.main all        # executa tudo em sequência
"""

from __future__ import annotations

import argparse
import logging
from warnings import filterwarnings

from src.config.environment import seed_everything
from src.config.logging import configure_logging
from src.config.paths import get_paths
from src.config.settings import load_settings
from src.pipelines.build_features import run_build_features
from src.pipelines.evaluate import run_evaluate
from src.pipelines.explain import run_explain
from src.pipelines.train import run_train

filterwarnings("ignore")
logger = logging.getLogger(__name__)

_STAGES = ("features", "train", "evaluate", "explain", "all")


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado com o argumento posicional ``stage``.
    """
    parser = argparse.ArgumentParser(
        prog="churn",
        description="Pipeline de previsão de churn explicável (Olist).",
    )
    parser.add_argument(
        "--stage",
        choices=_STAGES,
        help="Estágio da pipeline a executar.",
    )
    return parser


def run_stage(stage: str) -> None:
    """Executa o estágio solicitado da pipeline.

    Parameters
    ----------
    stage : str
        Um de ``{"features", "train", "evaluate", "explain", "all"}``.

    Raises
    ------
    ValueError
        Se o estágio for desconhecido.

    Examples
    --------
    >>> run_stage("features")  # doctest: +SKIP
    """
    configure_logging()
    settings = load_settings()
    paths = get_paths()
    seed_everything(settings.project.random_seed)

    logger.info("Executando estágio: %s", stage)

    if stage in ("features", "all"):
        run_build_features(settings, paths)
    if stage in ("train", "all"):
        run_train(settings, paths)
    if stage in ("evaluate", "all"):
        run_evaluate(settings, paths)
    if stage in ("explain", "all"):
        run_explain(paths)


def main() -> None:
    """Ponto de entrada da CLI da pipeline de churn."""
    args = build_parser().parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
