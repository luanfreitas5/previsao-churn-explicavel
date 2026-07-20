"""Carrega e valida as configurações do projeto com Pydantic.

Combina ``configs/config.yaml`` e ``configs/model_params.yaml`` em um objeto
``Settings`` tipado. Uma configuração inválida falha no startup com erro claro,
nunca no meio da execução.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.paths import CONFIGS_DIR


class ChurnDefinition(BaseModel):
    """Parâmetros que definem o alvo de churn (não recompra)."""

    cutoff_date: date = Field(description="Data de corte do snapshot (features <= corte).")
    horizon_days: int = Field(gt=0, description="Janela futura (dias) para observar recompra.")
    valid_order_statuses: list[str] = Field(min_length=1)
    customer_key: str = "customer_unique_id"


class SplitConfig(BaseModel):
    """Configuração da divisão treino/teste."""

    test_size: float = Field(gt=0, lt=1, default=0.2)
    stratify: bool = True


class CrossValidationConfig(BaseModel):
    """Configuração da validação cruzada estratificada."""

    n_splits: int = Field(ge=2, default=5)
    shuffle: bool = True


class FairnessConfig(BaseModel):
    """Configuração da auditoria de justiça."""

    sensitive_feature: str = "customer_state"


class MLflowConfig(BaseModel):
    """Configuração do rastreamento de experimentos MLflow."""

    experiment_name: str = "previsao-churn-explicavel"
    tracking_uri: str = "file:./mlruns"
    registered_model_name: str = "churn-lightgbm"


class ProjectConfig(BaseModel):
    """Configurações gerais do projeto."""

    name: str = "previsao-churn-explicavel"
    random_seed: int = 42


class BaselineParams(BaseModel):
    """Hiperparâmetros do baseline (DummyClassifier)."""

    strategy: str = "prior"


class LightGBMParams(BaseModel):
    """Hiperparâmetros validados do LightGBM."""

    objective: str = "binary"
    n_estimators: int = Field(gt=0, default=500)
    learning_rate: float = Field(gt=0, le=1, default=0.03)
    num_leaves: int = Field(gt=1, default=31)
    max_depth: int = -1
    min_child_samples: int = Field(gt=0, default=40)
    subsample: float = Field(gt=0, le=1, default=0.8)
    subsample_freq: int = Field(ge=0, default=1)
    colsample_bytree: float = Field(gt=0, le=1, default=0.8)
    reg_alpha: float = Field(ge=0, default=0.1)
    reg_lambda: float = Field(ge=0, default=0.2)
    class_weight: str | None = "balanced"
    n_jobs: int = -1
    verbose: int = -1


class EvaluationParams(BaseModel):
    """Parâmetros de avaliação e limiares de decisão."""

    primary_metric: str = "average_precision"
    decision_threshold: float = Field(ge=0, le=1, default=0.5)
    bootstrap_iterations: int = Field(gt=0, default=1000)
    min_primary_metric: float = Field(ge=0, le=1, default=0.55)


class ModelParams(BaseModel):
    """Agrupa todos os hiperparâmetros de modelagem."""

    baseline: BaselineParams = BaselineParams()
    lightgbm: LightGBMParams = LightGBMParams()
    evaluation: EvaluationParams = EvaluationParams()


class Settings(BaseSettings):
    """Configuração global validada do projeto.

    Combina os YAML de ``configs/`` com variáveis de ambiente do ``.env``.

    Examples
    --------
    >>> settings = load_settings()
    >>> settings.churn.horizon_days > 0
    True
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project: ProjectConfig = ProjectConfig()
    churn: ChurnDefinition
    split: SplitConfig = SplitConfig()
    cross_validation: CrossValidationConfig = CrossValidationConfig()
    fairness: FairnessConfig = FairnessConfig()
    mlflow: MLflowConfig = MLflowConfig()
    model: ModelParams = ModelParams()


def _read_yaml(path: Path) -> dict:
    """Lê um arquivo YAML e retorna um dicionário.

    Parameters
    ----------
    path : pathlib.Path
        Caminho do arquivo YAML.

    Returns
    -------
    dict
        Conteúdo do YAML.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(configs_dir: Path | None = None) -> Settings:
    """Carrega e valida as configurações do projeto.

    Parameters
    ----------
    configs_dir : pathlib.Path or None, optional
        Diretório com os YAML de configuração. Se ``None``, usa ``configs/``.

    Returns
    -------
    Settings
        Configuração validada do projeto.

    Raises
    ------
    FileNotFoundError
        Se ``config.yaml`` ou ``model_params.yaml`` não existirem.
    pydantic.ValidationError
        Se algum parâmetro violar as restrições declaradas.

    Examples
    --------
    >>> settings = load_settings()
    >>> settings.project.random_seed
    42
    """
    directory = configs_dir or CONFIGS_DIR
    general = _read_yaml(directory / "config.yaml")
    model = _read_yaml(directory / "model_params.yaml")

    return Settings(**general, model=ModelParams(**model))
