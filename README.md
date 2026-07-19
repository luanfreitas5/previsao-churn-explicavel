# 📉 Previsão de Churn Explicável — Olist

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](.github/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)](.github/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Identifica **clientes com risco de não recomprar** no marketplace da Olist usando
features **RFM** (recência, frequência, valor) e **gradient boosting (LightGBM)**,
com **SHAP** para explicar cada previsão em **linguagem de negócio**. O foco é um
caso realista, reprodutível e orientado à decisão comercial de retenção.

> **Domínio:** e-commerce · **Alvo:** churn/não recompra (binário) ·
> **Métrica principal:** PR-AUC (average precision) · **Explicabilidade:** SHAP ·
> **Deploy:** dashboard Streamlit.

---

## 🎯 Definição do problema (churn = não recompra)

O dataset da Olist é transacional e a maioria dos clientes compra **uma única vez**.
Definimos churn a partir de um **snapshot** com data de corte, evitando vazamento de
futuro (parametrizável em [`configs/config.yaml`](configs/config.yaml)):

1. A base considera clientes com **pelo menos um pedido até** `cutoff_date`.
2. Todas as features são calculadas **apenas com pedidos ≤ `cutoff_date`**.
3. `churn = 1` se o cliente **não** comprar na janela
   `(cutoff_date, cutoff_date + horizon_days]`; caso contrário `churn = 0`.

> ⚠️ **Limitação conhecida:** por ser majoritariamente compra única, a base é
> **desbalanceada** (alta taxa de churn). Por isso a métrica principal é o
> **PR-AUC** e o modelo usa `class_weight="balanced"`. Ver o
> [Model Card](reports/model_cards/) e a discussão de justiça por UF.

---

## 🏗️ Arquitetura

```
CSVs Olist (data/raw)
     │  load  (polars)
     ▼
build_churn_features  ──►  contrato pandera  ──►  data/processed/*.parquet
     │  RFM + rotulagem (sem vazamento)
     ▼
split treino/teste  ──►  LightGBM (sklearn Pipeline)  ──►  MLflow + models/*.joblib
     │
     ├──►  avaliação: PR-AUC + IC bootstrap, calibração, justiça por UF
     └──►  SHAP: importância global + explicação por cliente
                        │
                        ▼
             Dashboard Streamlit (app/)
```

Pipeline por estágios (cada um lê/escreve artefatos em disco):

| Estágio | Comando | Saída |
|---|---|---|
| Features | `make features` | `data/processed/{churn_features,train,test}.parquet` |
| Treino | `make train` | `models/churn_lightgbm.joblib` + run MLflow |
| Avaliação | `make evaluate` | `reports/metrics.json` + figuras |
| Explicabilidade | `make explain` | `reports/figures/shap_importance.*` |
| Tudo | `make pipeline` | executa os quatro em sequência |

---

## 🚀 Começando

### 1. Pré-requisitos

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de ambiente e dependências)

### 2. Instalação

```bash
uv sync --dev            # runtime + dev
uv sync --extra app      # inclui Streamlit + Plotly (dashboard)
make hooks               # instala os hooks de pre-commit
```

### 3. Dados

Baixe o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
e extraia os CSVs em [`data/raw/`](data/raw/) (não versionados no Git).

### 4. Executar a pipeline

```bash
make pipeline            # features -> train -> evaluate -> explain
make app                 # sobe o dashboard de explicabilidade
```

---

## 🧪 Qualidade e testes

```bash
make quality             # lint, type check, segurança, complexidade, docstrings
make test                # pytest com cobertura (meta ≥ 80%)
```

- **Contratos de dados** (`pandera`) validam a base de features na entrada/saída.
- **Testes comportamentais de ML**: ex. aumentar a recência não reduz o risco de churn.
- **Testes property-based** (`hypothesis`) sobre as métricas.
- **Reprodutibilidade**: `seed_everything`, hash dos dados nos metadados do modelo.

---

## 📊 Avaliação rigorosa

Nunca reportamos uma métrica pontual isolada. O relatório inclui:

- **Incerteza:** PR-AUC com intervalo de confiança 95% via bootstrap.
- **Baseline:** comparação explícita com `DummyClassifier`.
- **Calibração:** Brier score + curva de confiabilidade.
- **Justiça:** disparidade de recall por **UF do cliente** (`fairlearn`).
- **Explicabilidade:** SHAP global e por cliente, traduzido para negócio.

---

## 🗂️ Estrutura do projeto

```
configs/        YAML validados com Pydantic
data/           raw / interim / processed (não versionados)
src/
  config/       settings, paths, logging, seed
  data/         loader (polars), splitter
  features/     RFM + rotulagem de churn
  schemas/      contratos pandera
  models/       pipeline LightGBM + persistência
  training/     validação cruzada + fit
  evaluation/   métricas, calibração, justiça
  explainability/  SHAP em linguagem de negócio
  inference/    pontuação e faixas de risco
  visualization/   tema + gráficos
  pipelines/    estágios ponta a ponta
  main.py       CLI (argparse)
app/            dashboard Streamlit
tests/          unit, property-based, comportamentais
reports/        métricas, figuras, model cards, datasheets
docs/           documentação MkDocs
```

---

## 📚 Documentação

```bash
make docs-serve          # http://127.0.0.1:8000
```

A referência de API é gerada automaticamente a partir das docstrings (NumPy, pt-BR)
com **MkDocs Material** + **mkdocstrings**.

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
