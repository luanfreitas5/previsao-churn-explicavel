"""Estágio de avaliação: métricas com incerteza, calibração e justiça."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.constants import columns as c
from src.evaluation.calibration import compute_calibration
from src.evaluation.fairness import audit_fairness
from src.evaluation.metrics import bootstrap_metric, evaluate_classification
from src.inference.predictor import predict_proba
from src.models.persistence import load_model
from src.pipelines.common import load_frame, to_pandas_xy
from src.utils.io import write_json
from src.visualization.plots import plot_calibration_curve, plot_precision_recall

logger = logging.getLogger(__name__)


def run_evaluate(settings: Settings, paths: ProjectPaths) -> dict:
    """Avalia o modelo no conjunto de teste e escreve o relatório de métricas.

    Reporta métricas com intervalo de confiança (bootstrap), calibração e
    disparidade por UF (justiça), além de salvar as figuras correspondentes.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    dict
        Relatório de métricas persistido em ``reports/metrics.json``.

    Examples
    --------
    >>> report = run_evaluate(settings, paths)  # doctest: +SKIP
    """
    model, _ = load_model(paths.model_file)
    test_df = load_frame(paths.test_file)
    x_test, y_test = to_pandas_xy(test_df)

    threshold = settings.model.evaluation.decision_threshold
    y_proba = predict_proba(model, x_test)
    y_true = y_test.to_numpy()
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_classification(y_true, y_proba, threshold)
    ci = bootstrap_metric(
        y_true,
        y_proba,
        n_iterations=settings.model.evaluation.bootstrap_iterations,
        seed=settings.project.random_seed,
    )
    calibration = compute_calibration(y_true, y_proba)
    # Indexação por rótulo único devolve Series (atributo sensível: UF do cliente).
    sensitive = cast("pd.Series", x_test[c.CUSTOMER_STATE])
    fairness = audit_fairness(y_true, y_pred, sensitive)

    plot_precision_recall(y_true, y_proba, paths.figures_dir)
    plot_calibration_curve(calibration["prob_true"], calibration["prob_pred"], paths.figures_dir)

    report = {
        "metrics": metrics,
        "primary_metric_bootstrap_ci": ci,
        "calibration_brier": calibration["brier"],
        "fairness_recall_by_state": fairness["recall"].to_dict(),
        "n_test": len(y_true),
        "churn_rate_test": float(y_true.mean()),
    }
    write_json(report, paths.metrics_file)
    logger.info(
        "Avaliação concluída. %s = %.4f (IC 95%%: %.4f-%.4f)",
        settings.model.evaluation.primary_metric,
        metrics[settings.model.evaluation.primary_metric],
        ci["lower"],
        ci["upper"],
    )
    return report
