# syntax=docker/dockerfile:1.7
# Imagem do dashboard Streamlit: código + dependências travadas (uv.lock).
# O modelo (models/) e a camada gold (data/gold/) NÃO entram na imagem: o compose
# os monta como volumes somente leitura, para que um novo modelo não exija novo build.
# Fixe as tags por patch; em produção acrescente @sha256:<digest>
# (obtido com `docker buildx imagetools inspect <imagem>:<tag>`). Mesma tag nos dois estágios.

# ---------- Estágio 1: binário do uv, versão fixa ----------
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

# ---------- Estágio 2: resolve o ambiente a partir do uv.lock ----------
FROM python:3.12.11-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Só os manifestos: esta camada fica em cache até o lock mudar.
# --locked falha se o uv.lock estiver desatualizado em relação ao pyproject.toml.
# --no-default-groups mantém fora dev/quality/docs; --extra app traz Streamlit + Plotly.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-default-groups --extra app --no-install-project

# ---------- Estágio 3: runtime enxuto, sem uv, sem compiladores ----------
FROM python:3.12.11-slim-bookworm AS runtime
# libgomp1: runtime OpenMP exigido pelo LightGBM; ausente na imagem slim.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
# Código e venv pertencem ao root: o usuário da aplicação só lê.
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
# PYTHONPATH=/app: os módulos importam como `src.<pacote>` (o projeto não é instalado na imagem).
# HOME=/tmp: o Streamlit grava cache/configuração em ~; /tmp é tmpfs no compose.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER app:app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
