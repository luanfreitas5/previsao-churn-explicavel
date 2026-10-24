# syntax=docker/dockerfile:1.7
# Servidor MLflow local (backend SQLite, uso individual): a imagem oficial roda como root.
# Mantenha a versão igual à do mlflow no uv.lock (cliente e servidor compatíveis).
# Para uso compartilhado/equipe troque o backend por Postgres (SQLite não suporta
# escrita concorrente) e acrescente `psycopg2-binary` aqui.
FROM ghcr.io/mlflow/mlflow:v3.17.2
# /mlflow é criado com o dono certo ANTES do volume: o volume nomeado herda a permissão.
RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid mlflow --no-create-home mlflow \
    && mkdir -p /mlflow \
    && chown mlflow:mlflow /mlflow
USER mlflow:mlflow
EXPOSE 5000
