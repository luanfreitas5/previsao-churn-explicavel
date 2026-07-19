"""Pipelines ponta a ponta do projeto de churn.

Cada estágio é uma função ``run_*`` idempotente que lê e escreve artefatos em
disco, permitindo executar a pipeline por partes ou completa.

Módulos
-------
common
    Utilidades compartilhadas (carregamento de features, conversão para pandas).
build_features
    Estágio de construção e validação da base de features.
train
    Estágio de treino, validação cruzada e persistência do modelo.
evaluate
    Estágio de avaliação (métricas, calibração, justiça, figuras).
explain
    Estágio de explicabilidade SHAP global.
"""

from src.pipelines.build_features import run_build_features
from src.pipelines.evaluate import run_evaluate
from src.pipelines.explain import run_explain
from src.pipelines.train import run_train

__all__ = ["run_build_features", "run_evaluate", "run_explain", "run_train"]
