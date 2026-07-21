"""Testes de integração dos estágios da pipeline (features -> treino -> avaliação -> explicação).

Um dataset bruto sintético da Olist (determinístico e pequeno) percorre toda a
pipeline ponta a ponta, exercitando ingestão, engenharia de features, validação
do contrato, treino com CV, avaliação (métricas, calibração, justiça) e
explicabilidade SHAP — sem nunca usar dados de produção.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.config.paths import ProjectPaths
from src.config.settings import load_settings
from src.exceptions import DataNotFoundError
from src.pipelines.build_features import run_build_features
from src.pipelines.common import load_frame, to_pandas_xy
from src.pipelines.evaluate import run_evaluate
from src.pipelines.explain import run_explain
from src.pipelines.train import run_train

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
_CUTOFF = datetime(2018, 3, 1)


def _write_raw_olist(raw_dir: Path, n_customers: int = 240, seed: int = 42) -> None:
    """Gera CSVs brutos sintéticos da Olist com sinal de churn (recência alta -> churn).

    Parameters
    ----------
    raw_dir : pathlib.Path
        Diretório de destino dos CSVs brutos.
    n_customers : int, optional
        Número de clientes sintéticos, by default 240.
    seed : int, optional
        Semente para reprodutibilidade, by default 42.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    orders, customers, items, payments, reviews = [], [], [], [], []

    def _add_order(idx: int, suffix: int, purchase: datetime) -> None:
        cid = f"c{idx}_{suffix}"
        oid = f"o{idx}_{suffix}"
        state = states[idx]
        customers.append((cid, f"u{idx}", state))
        delivered = purchase + timedelta(days=int(rng.integers(3, 20)))
        estimated = purchase + timedelta(days=int(rng.integers(5, 25)))
        orders.append((oid, cid, "delivered", purchase, delivered, estimated))
        price = round(float(rng.uniform(30, 500)), 2)
        freight = round(float(rng.uniform(5, 60)), 2)
        items.append((oid, f"p{int(rng.integers(0, 50))}", price, freight))
        payments.append((oid, round(price + freight, 2), int(rng.integers(1, 6))))
        reviews.append((oid, int(rng.integers(1, 6))))

    states = [str(s) for s in rng.choice(["SP", "RJ", "MG", "RS", "BA"], n_customers)]

    for i in range(n_customers):
        recency = int(rng.integers(10, 400))
        _add_order(i, 0, _CUTOFF - timedelta(days=recency))
        # Probabilidade de churn cresce com a recência (gera sinal aprendível).
        p_churn = 1.0 / (1.0 + np.exp(-(0.012 * recency - 2.5)))
        if rng.random() >= p_churn:  # recompra na janela futura => churn = 0
            _add_order(i, 1, _CUTOFF + timedelta(days=int(rng.integers(1, 178))))

    orders_df = pd.DataFrame(
        orders,
        columns=[
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    for col in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        orders_df[col] = orders_df[col].map(
            lambda d: d.strftime(_DT_FORMAT) if pd.notna(d) else None
        )
    orders_df.to_csv(raw_dir / "olist_orders_dataset.csv", index=False)

    pd.DataFrame(customers, columns=["customer_id", "customer_unique_id", "customer_state"]).to_csv(
        raw_dir / "olist_customers_dataset.csv", index=False
    )
    pd.DataFrame(items, columns=["order_id", "product_id", "price", "freight_value"]).to_csv(
        raw_dir / "olist_order_items_dataset.csv", index=False
    )
    pd.DataFrame(payments, columns=["order_id", "payment_value", "payment_installments"]).to_csv(
        raw_dir / "olist_order_payments_dataset.csv", index=False
    )
    pd.DataFrame(reviews, columns=["order_id", "review_score"]).to_csv(
        raw_dir / "olist_order_reviews_dataset.csv", index=False
    )


def _tmp_paths(tmp_path: Path, raw_dir: Path) -> ProjectPaths:
    """Constrói uma coleção de caminhos totalmente sob ``tmp_path`` para o teste."""
    processed = tmp_path / "data" / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    return ProjectPaths(
        root=tmp_path,
        data_raw=raw_dir,
        data_interim=tmp_path / "data" / "interim",
        data_processed=processed,
        features_file=processed / "churn_features.parquet",
        train_file=processed / "train.parquet",
        test_file=processed / "test.parquet",
        models_dir=models,
        model_file=models / "churn_lightgbm.joblib",
        model_metadata_file=models / "churn_lightgbm_meta.json",
        reports_dir=reports,
        figures_dir=reports / "figures",
        metrics_file=reports / "metrics.json",
        model_cards_dir=reports / "model_cards",
        datasheets_dir=reports / "datasheets",
        logs_dir=tmp_path / "logs",
    )


@pytest.fixture
def pipeline_settings(tmp_path: Path):
    """Configuração real do projeto, com CV e MLflow (SQLite) ajustados para o teste."""
    settings = load_settings()
    # Mesmo backend SQLite de produção; artefatos e banco isolados sob tmp_path
    # para não poluir o repositório durante os testes.
    settings.mlflow.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.create_experiment(
        settings.mlflow.experiment_name,
        artifact_location=(tmp_path / "mlartifacts").as_uri(),
    )
    settings.cross_validation.n_splits = 3
    settings.model.lightgbm.n_estimators = 60
    return settings


@pytest.mark.integration
def test_end_to_end_pipeline(tmp_path: Path, pipeline_settings) -> None:
    """A pipeline completa roda do dado bruto à explicação, gerando todos os artefatos."""
    raw_dir = tmp_path / "raw"
    _write_raw_olist(raw_dir)
    paths = _tmp_paths(tmp_path, raw_dir)

    # 1) Features + divisão treino/teste.
    run_build_features(pipeline_settings, paths)
    assert paths.features_file.exists()
    assert paths.train_file.exists()
    assert paths.test_file.exists()

    # 2) Treino com CV + persistência + rastreamento.
    run_train(pipeline_settings, paths)
    assert paths.model_file.exists()
    assert paths.model_metadata_file.exists()

    # 3) Avaliação: métricas, calibração, justiça e figuras.
    report = run_evaluate(pipeline_settings, paths)
    assert paths.metrics_file.exists()
    assert (paths.figures_dir / "precision_recall.png").exists()
    assert (paths.figures_dir / "calibration_curve.png").exists()
    for key in ("metrics", "primary_metric_bootstrap_ci", "calibration_brier"):
        assert key in report
    assert 0.0 <= report["churn_rate_test"] <= 1.0

    # 4) Explicabilidade global SHAP.
    run_explain(paths)
    assert (paths.figures_dir / "shap_importance.png").exists()
    assert (paths.reports_dir / "shap_global_importance.json").exists()


def test_load_frame_missing_raises(tmp_path: Path) -> None:
    """Carregar um parquet inexistente levanta DataNotFoundError."""
    with pytest.raises(DataNotFoundError):
        load_frame(tmp_path / "nao_existe.parquet")


def test_to_pandas_xy_splits_features_and_target(features_frame: pl.DataFrame) -> None:
    """A conversão separa X (features) e y (alvo inteiro) no formato pandas."""
    x, y = to_pandas_xy(features_frame)
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "churn" not in x.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(x) == len(y) == features_frame.height
