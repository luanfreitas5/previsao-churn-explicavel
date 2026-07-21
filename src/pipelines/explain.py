"""Estágio de explicabilidade: importância global das features via SHAP."""

from __future__ import annotations

import logging

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.explainability.shap_explainer import ChurnExplainer
from src.models.persistence import load_model
from src.pipelines.common import load_frame, to_pandas_xy
from src.utils.io import write_json
from src.visualization.plots import plot_shap_importance

logger = logging.getLogger(__name__)


def run_explain(settings: Settings, paths: ProjectPaths) -> None:
    """Gera a importância global SHAP e salva figura e tabela.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Examples
    --------
    >>> run_explain(settings, paths)  # doctest: +SKIP
    """
    model, _ = load_model(paths.model_file)
    test_df = load_frame(paths.test_file)
    x_test, _ = to_pandas_xy(test_df)

    importances = ChurnExplainer(model).explain_global(x_test)

    plot_shap_importance(importances, paths.figures_dir)
    write_json(
        {"global_importance": importances.to_dict(orient="records")},
        paths.reports_dir / "shap_global_importance.json",
    )
    logger.info(
        "Explicabilidade global concluída. Feature mais influente: %s",
        importances.iloc[0]["feature"],
    )
