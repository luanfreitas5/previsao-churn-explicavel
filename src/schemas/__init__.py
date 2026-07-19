"""Contratos de dados (pandera) por estágio da pipeline.

Módulos
-------
features
    Contrato da base de features RFM + churn (entrada do modelo).
"""

from src.schemas.features import ChurnFeaturesSchema, validate_features

__all__ = ["ChurnFeaturesSchema", "validate_features"]
