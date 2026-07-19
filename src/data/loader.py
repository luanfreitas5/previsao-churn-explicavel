"""Leitura dos datasets brutos da Olist como ``polars.DataFrame``.

Apenas lê os dados brutos (nunca os modifica). As colunas de data/hora são
convertidas para ``Datetime`` para permitir cálculos temporais de RFM e churn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.constants import columns as c
from src.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

# Formato de data/hora usado nos CSVs da Olist.
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Nomes de arquivo esperados em data/raw.
_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}


@dataclass(frozen=True)
class OlistTables:
    """Conjunto de tabelas brutas da Olist necessárias para a modelagem.

    Attributes
    ----------
    orders : polars.DataFrame
        Pedidos, com timestamps de compra e entrega já tipados.
    customers : polars.DataFrame
        Clientes, incluindo ``customer_unique_id`` e UF.
    order_items : polars.DataFrame
        Itens de pedido, com preço e frete.
    payments : polars.DataFrame
        Pagamentos, com tipo, parcelas e valor.
    reviews : polars.DataFrame
        Avaliações, com nota (``review_score``).
    """

    orders: pl.DataFrame
    customers: pl.DataFrame
    order_items: pl.DataFrame
    payments: pl.DataFrame
    reviews: pl.DataFrame


def _read_csv(path: Path) -> pl.DataFrame:
    """Lê um CSV bruto da Olist como ``polars.DataFrame``.

    Parameters
    ----------
    path : pathlib.Path
        Caminho do arquivo CSV.

    Returns
    -------
    polars.DataFrame
        Conteúdo do arquivo.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir em ``data/raw``.
    """
    if not path.exists():
        logger.error("Arquivo bruto não encontrado: %s", path)
        raise DataNotFoundError(
            f"Arquivo bruto não encontrado: {path}. Baixe o dataset da Olist e extraia em data/raw."
        )
    return pl.read_csv(path, infer_schema_length=10_000)


def _parse_datetime(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Aplica a conversão para ``Datetime`` nas colunas de string presentes.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de entrada.
    columns : list of str
        Colunas a converter para ``Datetime``.

    Returns
    -------
    polars.DataFrame
        DataFrame com as colunas de data convertidas.
    """
    present = [col for col in columns if col in df.columns]
    if not present:
        return df
    return df.with_columns(
        pl.col(col).str.strptime(pl.Datetime, format=_DATETIME_FORMAT, strict=False)
        for col in present
    )


def load_olist_tables(raw_dir: Path) -> OlistTables:
    """Carrega as tabelas brutas da Olist a partir de ``data/raw``.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Diretório com os CSVs brutos da Olist.

    Returns
    -------
    OlistTables
        Tabelas carregadas com colunas de data já tipadas.

    Raises
    ------
    DataNotFoundError
        Se algum arquivo esperado não existir.

    Examples
    --------
    >>> tables = load_olist_tables(Path("data/raw"))  # doctest: +SKIP
    >>> tables.orders.height > 0  # doctest: +SKIP
    True
    """
    raw_dir = Path(raw_dir)
    logger.info("Carregando tabelas brutas da Olist de %s", raw_dir)

    orders = _parse_datetime(
        _read_csv(raw_dir / _FILES["orders"]),
        [
            c.ORDER_PURCHASE_TIMESTAMP,
            c.ORDER_DELIVERED_CUSTOMER_DATE,
            c.ORDER_ESTIMATED_DELIVERY_DATE,
        ],
    )
    customers = _read_csv(raw_dir / _FILES["customers"])
    order_items = _read_csv(raw_dir / _FILES["order_items"])
    payments = _read_csv(raw_dir / _FILES["payments"])
    reviews = _read_csv(raw_dir / _FILES["reviews"])

    logger.info(
        "Tabelas carregadas: orders=%d, customers=%d, items=%d, payments=%d, reviews=%d",
        orders.height,
        customers.height,
        order_items.height,
        payments.height,
        reviews.height,
    )
    return OlistTables(
        orders=orders,
        customers=customers,
        order_items=order_items,
        payments=payments,
        reviews=reviews,
    )
