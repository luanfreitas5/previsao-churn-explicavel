"""Rastreamento de experimentos com MLflow.

Módulos
-------
tracker
    Gerenciador de contexto para logar parâmetros, métricas e artefatos, e
    registrar o modelo no MLflow Model Registry.
"""

from src.experiment.tracker import MLflowTracker

__all__ = ["MLflowTracker"]
