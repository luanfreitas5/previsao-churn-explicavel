"""Dashboard Streamlit de previsão de churn explicável (Olist).

Carrega o modelo treinado e a base de teste, pontua os clientes por risco de
não recompra e apresenta explicações SHAP globais e por cliente em linguagem
de negócio.

Execução
--------
.. code-block:: bash

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from src.config.paths import get_paths
from src.config.settings import load_settings
from src.constants import columns as c
from src.constants import defaults
from src.explainability.shap_explainer import ChurnExplainer
from src.inference.predictor import score_customers
from src.models.persistence import load_model
from src.visualization.theme import CHURN_PALETTE

st.set_page_config(page_title="Churn Explicável — Olist", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Carregando modelo...")
def _load_artifacts() -> tuple[object, pd.DataFrame, ChurnExplainer]:
    """Carrega o modelo, a base de teste e o explicador SHAP (com cache).

    Returns
    -------
    tuple
        ``(pipeline, test_features_pandas, explainer)``.
    """
    paths = get_paths()
    model, _ = load_model(paths.model_file)
    test = pl.read_parquet(paths.test_file).to_pandas()
    explainer = ChurnExplainer(model)
    return model, test, explainer


def _plot_global_importance(importances: pd.DataFrame) -> go.Figure:
    """Cria o gráfico de barras da importância global SHAP.

    Parameters
    ----------
    importances : pandas.DataFrame
        Colunas ``feature`` e ``mean_abs_shap``.

    Returns
    -------
    plotly.graph_objects.Figure
        Gráfico de barras horizontais.
    """
    fig = px.bar(
        importances.iloc[::-1],
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color_discrete_sequence=[CHURN_PALETTE["neutro"]],
    )
    fig.update_layout(
        title="Fatores que mais influenciam o churn",
        xaxis_title="Impacto médio absoluto (SHAP)",
        yaxis_title="",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def _plot_instance(contributions: pd.DataFrame) -> go.Figure:
    """Cria o gráfico de contribuições SHAP de um cliente.

    Parameters
    ----------
    contributions : pandas.DataFrame
        Colunas ``feature``, ``shap_value`` e ``feature_value``.

    Returns
    -------
    plotly.graph_objects.Figure
        Gráfico de barras com contribuições positivas (churn) e negativas.
    """
    colors = [
        CHURN_PALETTE["churn"] if value > 0 else CHURN_PALETTE["retencao"]
        for value in contributions["shap_value"]
    ]
    fig = go.Figure(
        go.Bar(
            x=contributions["shap_value"],
            y=contributions["feature"],
            orientation="h",
            marker_color=colors,
            customdata=contributions["feature_value"],
            hovertemplate="%{y}: %{customdata}<br>Contribuição: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Por que este cliente foi classificado assim?",
        xaxis_title="Contribuição para o risco de churn (SHAP)",
        yaxis={"autorange": "reversed"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def main() -> None:
    """Renderiza o dashboard de churn explicável."""
    settings = load_settings()
    st.title("📉 Previsão de Churn Explicável — Olist")
    st.caption(
        "Risco de não recompra por cliente, com explicação SHAP em linguagem de negócio. "
        f"Corte: {settings.churn.cutoff_date} · Horizonte: {settings.churn.horizon_days} dias."
    )

    try:
        model, test, explainer = _load_artifacts()
    except Exception as exc:
        st.error(f"Não foi possível carregar os artefatos: {exc}")
        st.info("Rode `make pipeline` para gerar features, modelo e avaliações.")
        return

    scored = score_customers(model, test, risk_bands=None)

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes avaliados", f"{len(scored):,}".replace(",", "."))
    col2.metric("Risco médio de churn", f"{scored['churn_proba'].mean():.1%}")
    col3.metric("Clientes de alto risco", int((scored["churn_proba"] >= 0.66).sum()))

    st.subheader("Visão global")
    importances = explainer.explain_global(test[defaults.FEATURES])
    st.plotly_chart(_plot_global_importance(importances), use_container_width=True)

    st.subheader("Explicação por cliente")
    top_customers = scored.head(200)
    selected = st.selectbox(
        "Selecione um cliente (ordenado por risco):",
        options=top_customers[c.CUSTOMER_KEY].tolist(),
        format_func=lambda cid: f"{cid[:8]}… — risco {_risk_of(top_customers, cid):.1%}",
    )

    row = test[test[c.CUSTOMER_KEY] == selected]
    proba = float(scored.loc[scored[c.CUSTOMER_KEY] == selected, "churn_proba"].iloc[0])
    st.metric("Probabilidade de churn", f"{proba:.1%}")

    contributions = explainer.explain_instance(row[defaults.FEATURES], top_n=settings_top_n())
    contrib_df = pd.DataFrame(
        [
            {
                "feature": item.feature,
                "shap_value": item.shap_value,
                "feature_value": item.feature_value,
            }
            for item in contributions
        ]
    )
    st.plotly_chart(_plot_instance(contrib_df), use_container_width=True)

    with st.expander("Ver tabela de clientes de maior risco"):
        st.dataframe(
            scored[[c.CUSTOMER_KEY, "churn_proba", "risk_band", *defaults.NUMERIC_FEATURES]].head(
                100
            ),
            use_container_width=True,
        )


def _risk_of(df: pd.DataFrame, customer_id: str) -> float:
    """Retorna a probabilidade de churn de um cliente na tabela pontuada.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabela pontuada com ``churn_proba``.
    customer_id : str
        Identificador do cliente.

    Returns
    -------
    float
        Probabilidade de churn do cliente.
    """
    return float(df.loc[df[c.CUSTOMER_KEY] == customer_id, "churn_proba"].iloc[0])


def settings_top_n() -> int:
    """Lê o número de features exibidas por cliente de ``configs/deploy.yaml``.

    Returns
    -------
    int
        Número de contribuições SHAP a exibir (padrão 12).
    """
    import yaml
    from src.config.paths import CONFIGS_DIR

    deploy_yaml = CONFIGS_DIR / "deploy.yaml"
    if not deploy_yaml.exists():
        return 12
    with deploy_yaml.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return int(cfg.get("app", {}).get("top_n_features", 12))


if __name__ == "__main__":
    main()
