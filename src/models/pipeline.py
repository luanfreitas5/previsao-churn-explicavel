"""Construção do pipeline de modelagem (pré-processamento + LightGBM) e baseline.

O pipeline combina imputação/codificação com o classificador em um único objeto
``sklearn.Pipeline``, garantindo consistência entre treino e serving e evitando
vazamento (o pré-processamento é ajustado apenas no treino, dentro de cada fold).
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config.settings import BaselineParams, LightGBMParams
from src.constants import defaults


def build_preprocessor() -> ColumnTransformer:
    """Cria o pré-processador: imputação numérica + one-hot da UF.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformador que imputa a mediana nas features numéricas e aplica
        one-hot na feature categórica (UF do cliente).
    """
    numeric = SimpleImputer(strategy="median")
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric, defaults.NUMERIC_FEATURES),
            ("cat", categorical, defaults.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(params: LightGBMParams, seed: int = 42) -> Pipeline:
    """Constrói o pipeline principal com pré-processamento e LightGBM.

    Parameters
    ----------
    params : LightGBMParams
        Hiperparâmetros validados do LightGBM.
    seed : int, optional
        Semente para reprodutibilidade, by default 42.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline ``preprocess -> model`` não treinado.

    Examples
    --------
    >>> pipe = build_model(settings.model.lightgbm)  # doctest: +SKIP
    >>> pipe.fit(X_train, y_train)  # doctest: +SKIP
    """
    model = LGBMClassifier(random_state=seed, **params.model_dump())
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_baseline(params: BaselineParams, seed: int = 42) -> Pipeline:
    """Constrói o baseline (``DummyClassifier``) para referência de métrica.

    Parameters
    ----------
    params : BaselineParams
        Estratégia do baseline (ex.: ``"prior"``).
    seed : int, optional
        Semente para reprodutibilidade, by default 42.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline ``preprocess -> dummy`` não treinado.
    """
    dummy = DummyClassifier(strategy=params.strategy, random_state=seed)
    return Pipeline([("preprocess", build_preprocessor()), ("model", dummy)])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Retorna os nomes das features após o pré-processamento (pós one-hot).

    Útil para alinhar valores SHAP às colunas transformadas.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline **treinado** contendo a etapa ``preprocess``.

    Returns
    -------
    list of str
        Nomes das features na ordem produzida pelo ``ColumnTransformer``.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())
