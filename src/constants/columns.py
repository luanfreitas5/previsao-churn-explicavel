"""Nomes de colunas dos datasets Olist e da base de features de churn.

Centralizar os nomes evita strings mágicas espalhadas e facilita refatoração.
"""

from __future__ import annotations

from typing import Final

# --- Dataset de pedidos (olist_orders_dataset) -----------------------------
ORDER_ID: Final = "order_id"
CUSTOMER_ID: Final = "customer_id"
ORDER_STATUS: Final = "order_status"
ORDER_PURCHASE_TIMESTAMP: Final = "order_purchase_timestamp"
ORDER_DELIVERED_CUSTOMER_DATE: Final = "order_delivered_customer_date"
ORDER_ESTIMATED_DELIVERY_DATE: Final = "order_estimated_delivery_date"

# --- Dataset de clientes (olist_customers_dataset) -------------------------
CUSTOMER_UNIQUE_ID: Final = "customer_unique_id"
CUSTOMER_STATE: Final = "customer_state"
CUSTOMER_CITY: Final = "customer_city"
CUSTOMER_ZIP_PREFIX: Final = "customer_zip_code_prefix"

# --- Dataset de itens (olist_order_items_dataset) --------------------------
ORDER_ITEM_ID: Final = "order_item_id"
PRODUCT_ID: Final = "product_id"
PRICE: Final = "price"
FREIGHT_VALUE: Final = "freight_value"

# --- Dataset de pagamentos (olist_order_payments_dataset) ------------------
PAYMENT_TYPE: Final = "payment_type"
PAYMENT_INSTALLMENTS: Final = "payment_installments"
PAYMENT_VALUE: Final = "payment_value"

# --- Dataset de avaliações (olist_order_reviews_dataset) -------------------
REVIEW_SCORE: Final = "review_score"

# --- Base de features RFM + churn (gerada) ---------------------------------
# Chave e alvo
CUSTOMER_KEY: Final = "customer_unique_id"
TARGET: Final = "churn"

# Features RFM
RECENCY_DAYS: Final = "recency_days"
FREQUENCY: Final = "frequency"
MONETARY: Final = "monetary"
MONETARY_MEAN: Final = "monetary_mean"

# Features de engenharia adicionais
TENURE_DAYS: Final = "tenure_days"
AVG_FREIGHT: Final = "avg_freight_value"
AVG_REVIEW_SCORE: Final = "avg_review_score"
AVG_INSTALLMENTS: Final = "avg_installments"
DISTINCT_PRODUCTS: Final = "distinct_products"
AVG_DELIVERY_DAYS: Final = "avg_delivery_days"
LATE_DELIVERY_RATE: Final = "late_delivery_rate"
