"""Construção e persistência do pipeline de modelagem de churn.

Módulos
-------
pipeline
    Constrói o ``sklearn.Pipeline`` (pré-processamento + LightGBM) e o baseline.
persistence
    Salva e carrega o pipeline treinado com ``joblib`` + metadados JSON.
"""

from src.models.persistence import load_model, save_model
from src.models.pipeline import build_baseline, build_model, get_feature_names

__all__ = ["build_baseline", "build_model", "get_feature_names", "load_model", "save_model"]
