"""Estágio de treino: baseline, validação cruzada, ajuste e persistência."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config.environment import hash_file
from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.experiment.tracker import MLflowTracker, build_lineage_tags
from src.models.persistence import save_model
from src.models.pipeline import build_baseline, build_model
from src.pipelines.common import load_frame, to_pandas_xy
from src.training.trainer import cross_validate_model, fit_model

logger = logging.getLogger(__name__)


def run_train(settings: Settings, paths: ProjectPaths) -> None:
    """Treina o modelo de churn com validação cruzada e rastreamento MLflow.

    Compara o baseline com o LightGBM via validação cruzada, treina o modelo
    final na base de treino completa e persiste o artefato com metadados de
    linhagem (hash dos dados, parâmetros, métrica de CV).

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Examples
    --------
    >>> run_train(settings, paths)  # doctest: +SKIP
    """
    train_df = load_frame(paths.train_file)
    x_train, y_train = to_pandas_xy(train_df)

    scoring = settings.model.evaluation.primary_metric
    seed = settings.project.random_seed

    baseline = build_baseline(settings.model.baseline, seed=seed)
    baseline_cv = cross_validate_model(
        baseline, x_train, y_train, settings.cross_validation.n_splits, scoring, seed
    )

    model = build_model(settings.model.lightgbm, seed=seed)
    model_cv = cross_validate_model(
        model, x_train, y_train, settings.cross_validation.n_splits, scoring, seed
    )
    logger.info(
        "Ganho sobre o baseline (%s): %.4f -> %.4f",
        scoring,
        baseline_cv["mean"],
        model_cv["mean"],
    )

    tags = build_lineage_tags(
        paths.train_file, seed, require_clean_tree=settings.mlflow.require_clean_tree
    )

    # Um run pai por modelo, com um run filho aninhado por fold.
    with MLflowTracker(settings.mlflow, run_name="baseline", tags=tags) as baseline_tracker:
        baseline_tracker.log_params({**settings.model.baseline.model_dump(), "model": "baseline"})
        baseline_tracker.log_cv_folds(scoring, baseline_cv["scores"])

    # O modelo registrado é reajustado em todo o treino (não é o do último fold).
    model = fit_model(model, x_train, y_train)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": scoring,
        "cv_baseline_mean": baseline_cv["mean"],
        "cv_model_mean": model_cv["mean"],
        "cv_model_ci95": model_cv["ci95"],
        "train_data_hash": hash_file(paths.train_file),
        "lightgbm_params": settings.model.lightgbm.model_dump(),
        "churn_definition": {
            "cutoff_date": str(settings.churn.cutoff_date),
            "horizon_days": settings.churn.horizon_days,
        },
    }

    with MLflowTracker(settings.mlflow, run_name="lightgbm", tags=tags) as tracker:
        tracker.log_params(
            {
                **settings.model.lightgbm.model_dump(),
                "cv": "StratifiedKFold",
                "n_splits": settings.cross_validation.n_splits,
            }
        )
        tracker.log_cv_folds(scoring, model_cv["scores"])
        tracker.log_metrics(
            {
                f"cv_{scoring}_baseline": baseline_cv["mean"],
                f"cv_{scoring}_ci95": model_cv["ci95"],
            }
        )
        tracker.log_model(model, x_train)
        save_model(model, paths.model_file, metadata)
        tracker.log_artifact(paths.model_file)

    logger.info("Treino concluído. Modelo em %s", paths.model_file)
