# =============================================================================
# Dockerfile — imagem do dashboard Streamlit de churn explicável.
# Build multi-stage com uv; usuário não-root; base fixada.
# =============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Instala o uv (gerenciador de dependências).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Camada de dependências (cacheável) ------------------------------------
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra app --no-dev --frozen || uv sync --extra app --no-dev

# --- Código da aplicação ----------------------------------------------------
COPY src ./src
COPY app ./app
COPY configs ./configs

# Usuário não-root por segurança.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

CMD ["uv", "run", "streamlit", "run", "app/dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
