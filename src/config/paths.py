"""Centraliza os caminhos do projeto usando ``pathlib.Path``.

Os caminhos são lidos de ``configs/paths.yaml`` e resolvidos de forma absoluta
a partir da raiz do projeto, evitando strings de caminho hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Raiz do projeto: .../previsao-churn-explicavel (dois níveis acima deste arquivo).
ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = ROOT / "configs"


def _resolve(relative: str) -> Path:
    """Resolve um caminho relativo do YAML para um ``Path`` absoluto sob a raiz.

    Parameters
    ----------
    relative : str
        Caminho relativo à raiz do projeto (ex.: ``"data/raw"``).

    Returns
    -------
    pathlib.Path
        Caminho absoluto correspondente.
    """
    return (ROOT / relative).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Coleção imutável de caminhos do projeto.

    Attributes
    ----------
    root : pathlib.Path
        Raiz do projeto.
    data_raw, data_interim, data_processed : pathlib.Path
        Diretórios de dados por estágio.
    features_file, train_file, test_file : pathlib.Path
        Artefatos de dados processados.
    models_dir, model_file, model_metadata_file : pathlib.Path
        Diretório e artefatos de modelo.
    reports_dir, figures_dir, metrics_file : pathlib.Path
        Diretório e artefatos de relatórios.
    model_cards_dir, datasheets_dir : pathlib.Path
        Diretórios de documentação de IA responsável.
    logs_dir : pathlib.Path
        Diretório de logs.
    """

    root: Path = ROOT
    data_raw: Path = field(default_factory=lambda: _resolve("data/raw"))
    data_interim: Path = field(default_factory=lambda: _resolve("data/interim"))
    data_processed: Path = field(default_factory=lambda: _resolve("data/processed"))
    features_file: Path = field(
        default_factory=lambda: _resolve("data/processed/churn_features.parquet")
    )
    train_file: Path = field(default_factory=lambda: _resolve("data/processed/train.parquet"))
    test_file: Path = field(default_factory=lambda: _resolve("data/processed/test.parquet"))
    models_dir: Path = field(default_factory=lambda: _resolve("models"))
    model_file: Path = field(default_factory=lambda: _resolve("models/churn_lightgbm.joblib"))
    model_metadata_file: Path = field(
        default_factory=lambda: _resolve("models/churn_lightgbm_meta.json")
    )
    reports_dir: Path = field(default_factory=lambda: _resolve("reports"))
    figures_dir: Path = field(default_factory=lambda: _resolve("reports/figures"))
    metrics_file: Path = field(default_factory=lambda: _resolve("reports/metrics.json"))
    model_cards_dir: Path = field(default_factory=lambda: _resolve("reports/model_cards"))
    datasheets_dir: Path = field(default_factory=lambda: _resolve("reports/datasheets"))
    logs_dir: Path = field(default_factory=lambda: _resolve("logs"))

    def ensure_dirs(self) -> None:
        """Cria os diretórios de saída caso ainda não existam."""
        for directory in (
            self.data_interim,
            self.data_processed,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.model_cards_dir,
            self.datasheets_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_paths(paths_yaml: Path | None = None) -> ProjectPaths:
    """Constrói ``ProjectPaths`` a partir de ``configs/paths.yaml``.

    Parameters
    ----------
    paths_yaml : pathlib.Path or None, optional
        Caminho para o YAML de paths. Se ``None``, usa ``configs/paths.yaml``.

    Returns
    -------
    ProjectPaths
        Estrutura de caminhos resolvidos.

    Raises
    ------
    FileNotFoundError
        Se o arquivo de configuração de paths não existir.

    Examples
    --------
    >>> paths = get_paths()
    >>> paths.data_raw.name
    'raw'
    """
    yaml_path = paths_yaml or (CONFIGS_DIR / "paths.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo de paths não encontrado: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data = cfg["data"]
    processed = cfg["processed_files"]
    models = cfg["models"]
    reports = cfg["reports"]

    return ProjectPaths(
        data_raw=_resolve(data["raw"]),
        data_interim=_resolve(data["interim"]),
        data_processed=_resolve(data["processed"]),
        features_file=_resolve(processed["features"]),
        train_file=_resolve(processed["train"]),
        test_file=_resolve(processed["test"]),
        models_dir=_resolve(models["dir"]),
        model_file=_resolve(models["model_file"]),
        model_metadata_file=_resolve(models["metadata_file"]),
        reports_dir=_resolve(reports["dir"]),
        figures_dir=_resolve(reports["figures"]),
        metrics_file=_resolve(reports["metrics"]),
        model_cards_dir=_resolve(reports["model_cards"]),
        datasheets_dir=_resolve(reports["datasheets"]),
        logs_dir=_resolve(cfg["logs"]),
    )
