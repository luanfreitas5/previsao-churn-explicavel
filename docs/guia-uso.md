# Guia de uso

## Instalação

```bash
uv sync --dev            # dependências de runtime + dev
uv sync --extra app      # inclui o dashboard (Streamlit + Plotly)
make hooks               # instala os hooks de pre-commit
```

## Dados

Baixe o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
e extraia os CSVs em `data/raw/`.

## Executando a pipeline

Cada estágio pode ser rodado isoladamente ou em sequência:

```bash
make features    # constrói e valida a base RFM + alvo de churn
make train       # treina o LightGBM (CV + MLflow)
make evaluate    # métricas com IC, calibração e justiça
make explain     # importância global via SHAP
make pipeline    # tudo em sequência
```

Ou via CLI diretamente:

```bash
uv run python -m src.main all
```

## Dashboard

```bash
make app         # streamlit run app/dashboard.py
```

O dashboard carrega o modelo e a base de teste, pontua os clientes por risco e
mostra explicações SHAP globais e por cliente.

## Qualidade

```bash
make quality     # lint, type check, segurança, complexidade, docstrings
make test        # pytest com cobertura (≥ 80%)
```
