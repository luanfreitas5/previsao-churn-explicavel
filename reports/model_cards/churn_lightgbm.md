# Model Card — Churn LightGBM (Olist)

> Preencha os valores entre `<...>` após executar `make pipeline`. As métricas
> ficam em `reports/metrics.json` e as figuras em `reports/figures/`.

## Detalhes do modelo

- **Nome:** `churn-lightgbm`
- **Tipo:** classificador binário (LightGBM / gradient boosting)
- **Versão:** 0.1.0
- **Autor:** Luan Freitas
- **Data:** `<AAAA-MM-DD>`
- **Entrada:** features RFM + derivadas por cliente (ver contrato de dados)
- **Saída:** probabilidade de churn (não recompra) em `[0, 1]`

## Uso pretendido

- **Uso primário:** priorizar clientes de maior risco de não recompra para ações
  de retenção (CRM, campanhas, ofertas).
- **Usuários:** times de marketing/retenção e analistas de dados.
- **Fora de escopo:** decisões automáticas que impactem indivíduos sem revisão
  humana; qualquer uso que discrimine subgrupos protegidos.

## Dados de treino

- **Fonte:** Brazilian E-Commerce Public Dataset by Olist (público).
- **Definição de churn:** não recompra na janela `(corte, corte + horizonte]`.
- **Corte / horizonte:** `<cutoff_date>` / `<horizon_days>` dias.
- **Taxa de churn (teste):** `<churn_rate_test>`.

## Avaliação

| Métrica | Valor | IC 95% |
|---|---|---|
| PR-AUC (average precision) | `<...>` | `<lower>–<upper>` |
| ROC-AUC | `<...>` | — |
| F1 | `<...>` | — |
| Brier (calibração) | `<...>` | — |

- **Baseline (DummyClassifier):** PR-AUC = `<...>`.
- **Curvas:** ver `reports/figures/precision_recall.*` e `calibration_curve.*`.

## Avaliação por subgrupo (justiça)

- **Atributo sensível:** UF do cliente (`customer_state`).
- **Disparidade de recall entre UFs:** `<...>` (ver `metrics.json`).
- Documente aqui as UFs com pior desempenho e possíveis causas.

## Explicabilidade

- **Método:** SHAP (TreeExplainer).
- **Fatores globais mais influentes:** `<ver reports/figures/shap_importance.*>`.

## Limitações conhecidas

- Base majoritariamente de compra única → alto desbalanceamento; o modelo é mais
  útil para **ranquear risco** do que para prever a classe absoluta.
- O rótulo depende do corte/horizonte escolhidos; janelas diferentes mudam a base.
- Dados de 2016–2018; pode não refletir comportamento atual (drift).

## Considerações éticas

- Sem PII nas saídas; identificadores são pseudônimos (`customer_unique_id`).
- Decisões de retenção devem passar por revisão humana.
