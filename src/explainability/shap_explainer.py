"""Explicações SHAP globais e por cliente, em linguagem de negócio.

Aplica ``shap.TreeExplainer`` sobre o LightGBM ajustado, usando as features já
pré-processadas pelo pipeline, e traduz cada contribuição para um rótulo
compreensível por áreas de negócio (ex.: "Dias desde a última compra").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from src.constants import columns as c
from src.constants import defaults
from src.models.pipeline import get_feature_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceContribution:
    """Contribuição SHAP de uma feature para a previsão de um cliente.

    Attributes
    ----------
    feature : str
        Rótulo de negócio da feature.
    shap_value : float
        Contribuição SHAP (positiva empurra para churn; negativa para recompra).
    feature_value : str
        Valor observado da feature (formatado para exibição).
    """

    feature: str
    shap_value: float
    feature_value: str


def _humanize(feature_name: str, value: object) -> tuple[str, str]:
    """Traduz o nome técnico de uma feature transformada para linguagem de negócio.

    Parameters
    ----------
    feature_name : str
        Nome da feature após o pré-processamento (ex.: ``customer_state_SP``).
    value : object
        Valor da feature transformada.

    Returns
    -------
    tuple of str
        Par ``(rótulo, valor_formatado)`` para exibição.
    """
    # Features de UF viram colunas one-hot "customer_state_XX".
    if feature_name.startswith(f"{c.CUSTOMER_STATE}_"):
        uf = feature_name.split("_", 2)[-1]
        return "UF do cliente", uf
    label = defaults.FEATURE_LABELS_PT.get(feature_name, feature_name)
    formatted = f"{value:.2f}" if isinstance(value, int | float | np.floating) else str(value)
    return label, formatted


def _base_feature(feature_name: str) -> str:
    """Mapeia uma feature transformada de volta à feature de origem.

    Parameters
    ----------
    feature_name : str
        Nome da feature após o pré-processamento.

    Returns
    -------
    str
        Nome da feature de origem (agrega colunas one-hot de UF).
    """
    if feature_name.startswith(f"{c.CUSTOMER_STATE}_"):
        return c.CUSTOMER_STATE
    return feature_name


class ChurnExplainer:
    """Gera explicações SHAP a partir de um pipeline de churn treinado.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline treinado (``preprocess`` + LightGBM).

    Examples
    --------
    >>> explainer = ChurnExplainer(model)  # doctest: +SKIP
    >>> importances = explainer.explain_global(X_test)  # doctest: +SKIP
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.feature_names = get_feature_names(pipeline)
        self._explainer = shap.TreeExplainer(self.model)

    def _shap_values(self, transformed: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calcula os valores SHAP da classe positiva (churn).

        Parameters
        ----------
        transformed : numpy.ndarray
            Features já pré-processadas.

        Returns
        -------
        numpy.ndarray
            Matriz ``(n_amostras, n_features)`` de valores SHAP para churn.
        """
        values = self._explainer.shap_values(transformed)
        # LightGBM binário pode retornar lista [classe0, classe1] ou tensor 3D.
        if isinstance(values, list):
            return np.asarray(values[1])
        values = np.asarray(values)
        if values.ndim == 3:
            return values[:, :, 1]
        return values

    def explain_global(self, x: pd.DataFrame, max_features: int | None = None) -> pd.DataFrame:
        """Importância global das features (SHAP médio absoluto).

        Agrega as colunas one-hot de UF de volta à feature de origem.

        Parameters
        ----------
        x : pandas.DataFrame
            Amostra de features para calcular a importância global.
        max_features : int or None, optional
            Limita o número de features retornadas, by default ``None`` (todas).

        Returns
        -------
        pandas.DataFrame
            Colunas ``feature`` (rótulo de negócio) e ``mean_abs_shap``,
            ordenadas por importância decrescente.
        """
        transformed = self.preprocessor.transform(x)
        shap_values = self._shap_values(transformed)
        mean_abs = np.abs(shap_values).mean(axis=0)

        per_base: dict[str, float] = {}
        for name, importance in zip(self.feature_names, mean_abs, strict=True):
            base = _base_feature(name)
            per_base[base] = per_base.get(base, 0.0) + float(importance)

        rows = [
            {"feature": defaults.FEATURE_LABELS_PT.get(base, base), "mean_abs_shap": value}
            for base, value in per_base.items()
        ]
        result = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
        return result.head(max_features) if max_features else result

    def explain_instance(self, x_row: pd.DataFrame, top_n: int = 10) -> list[InstanceContribution]:
        """Explica a previsão de um único cliente em linguagem de negócio.

        Parameters
        ----------
        x_row : pandas.DataFrame
            DataFrame de uma única linha com as features do cliente.
        top_n : int, optional
            Número de contribuições mais relevantes a retornar, by default 10.

        Returns
        -------
        list of InstanceContribution
            Contribuições ordenadas por magnitude decrescente.

        Raises
        ------
        ValueError
            Se ``x_row`` não tiver exatamente uma linha.
        """
        if len(x_row) != 1:
            raise ValueError("explain_instance espera um DataFrame de uma única linha.")

        transformed = self.preprocessor.transform(x_row)
        shap_values = self._shap_values(transformed)[0]
        values = np.asarray(transformed)[0]

        contributions: list[InstanceContribution] = []
        for name, shap_val, feat_val in zip(self.feature_names, shap_values, values, strict=True):
            # Colunas one-hot inativas não contribuem com contexto útil.
            if _base_feature(name) == c.CUSTOMER_STATE and feat_val == 0:
                continue
            label, formatted = _humanize(name, feat_val)
            contributions.append(
                InstanceContribution(
                    feature=label,
                    shap_value=float(shap_val),
                    feature_value=formatted,
                )
            )

        contributions.sort(key=lambda item: abs(item.shap_value), reverse=True)
        return contributions[:top_n]
