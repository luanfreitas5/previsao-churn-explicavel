# Arquitetura

## Definição de churn (não recompra)

O dataset da Olist é transacional e a maioria dos clientes compra uma única vez.
Definimos churn a partir de um **snapshot** com data de corte (`cutoff_date`) e um
horizonte de observação (`horizon_days`), configuráveis em `configs/config.yaml`:

1. A base considera clientes com pelo menos um pedido válido **até** o corte.
2. As features são calculadas **apenas com pedidos ≤ corte** (sem vazamento).
3. `churn = 1` se o cliente não comprar em `(corte, corte + horizonte]`.

Essa formulação transforma um problema de recompra em classificação binária com
uma janela temporal bem definida, evitando vazamento de futuro.

## Fluxo de dados

```
data/raw (CSVs)
  → load_olist_tables (polars)
  → build_churn_features (RFM + rotulagem)
  → validate_features (pandera)
  → data/processed/*.parquet
  → split treino/teste
  → LightGBM (sklearn Pipeline: imputação + one-hot + modelo)
  → avaliação (métricas, calibração, justiça) + SHAP
  → dashboard Streamlit
```

## Decisões de design

- **Polars** na ingestão/feature engineering (desempenho e API lazy); **pandas**
  apenas na fronteira de modelagem (exigido por scikit-learn/SHAP).
- **sklearn Pipeline** encapsula pré-processamento + modelo, garantindo que o
  pré-processamento seja ajustado dentro de cada fold (sem vazamento) e a mesma
  transformação em serving.
- **Contrato de dados (pandera)** valida a base de features na saída da engenharia.
- **Métrica principal PR-AUC**: a base é desbalanceada e o objetivo é ranquear
  corretamente os clientes de maior risco.
- **SHAP (TreeExplainer)** sobre o LightGBM, com tradução dos nomes técnicos para
  rótulos de negócio e agregação das colunas one-hot de UF na importância global.

## Justiça e IA responsável

- Auditoria de disparidade de recall por **UF do cliente** (`fairlearn`).
- [Model Card](https://github.com/luanfreitas5/previsao-churn-explicavel/tree/main/reports/model_cards)
  e [Datasheet](https://github.com/luanfreitas5/previsao-churn-explicavel/tree/main/reports/datasheets)
  documentam uso pretendido, limitações e considerações de privacidade (LGPD).
