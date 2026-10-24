# Atalhos reprodutíveis. `make check` replica o CI localmente.
UV := uv
RUN := $(UV) run python -m src.main

.DEFAULT_GOAL := help
.PHONY: help install hooks update lint format typecheck security quality test smoke check \
	pre-commit update-hooks update-version features train evaluate explain pipeline reproduce \
	app docs docs-serve docs-deploy docker-build docker-up docker-down profile clean clean-data \
	clean-reports clean-notebooks

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente ---------------------------------------------------------------
install:  ## Instala todos os grupos de dependências + extra do app + hooks
	$(UV) sync --all-groups --extra app
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg

hooks:  ## Reinstala os hooks do pre-commit e regenera o baseline de segredos
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg
	$(UV) run detect-secrets scan > .secrets.baseline

update:  ## Atualiza o lock e sincroniza todas as dependências
	$(UV) lock --upgrade
	$(UV) sync --all-groups --extra app

# --- Qualidade --------------------------------------------------------------
lint:  ## Lint + checagem de formatação (ruff, sem alterar arquivos)
	$(UV) run ruff check src tests app
	$(UV) run ruff format --check src tests app

format:  ## Corrige o lint e formata o código com ruff
	$(UV) run ruff check --fix src tests app
	$(UV) run ruff format src tests app

typecheck:  ## Type checking estático (basedPyright)
	$(UV) run basedpyright

security:  ## Segurança do código (bandit) e das dependências (pip-audit)
	$(UV) run bandit -c pyproject.toml -r src
	$(UV) run pip-audit

quality:  ## Código morto, modernização, complexidade e docstrings
	$(UV) run vulture
	$(UV) run refurb src
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src
	$(UV) run interrogate -v src

# --- Testes -----------------------------------------------------------------
test:  ## Roda os testes com cobertura (meta >= 80%)
	$(UV) run pytest -m "not slow" --cov --cov-report=term-missing

smoke:  ## Roda apenas os smoke tests (os mesmos do pre-commit)
	$(UV) run pytest -m smoke -q --no-cov

check: lint typecheck security quality test  ## Roda a suíte completa do CI localmente

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

update-hooks:  ## Atualiza as versões dos hooks do pre-commit
	$(UV) run pre-commit autoupdate

update-version:  ## Gera o CHANGELOG e incrementa a versão (Conventional Commits + SemVer)
	$(UV) run cz bump --changelog --yes

# --- Pipeline de ML (bronze -> silver -> gold -> modelo) --------------------
features:  ## bronze -> silver (tipado) -> gold (RFM + alvo, treino/teste)
	$(RUN) --stage features

train:  ## Treina o LightGBM com CV e registra os runs no MLflow
	$(RUN) --stage train

evaluate:  ## Avalia o modelo (métricas com IC, calibração, justiça por UF)
	$(RUN) --stage evaluate

explain:  ## Gera as explicações SHAP globais
	$(RUN) --stage explain

pipeline:  ## Executa a pipeline ponta a ponta (features -> train -> evaluate -> explain)
	$(RUN) --stage all

reproduce: pipeline  ## Reproduz os resultados a partir da camada bronze

# --- Aplicação --------------------------------------------------------------
app:  ## Sobe o dashboard Streamlit de explicabilidade
	$(UV) run streamlit run app/dashboard.py

# --- Documentação -----------------------------------------------------------
docs:  ## Constrói a documentação em modo estrito
	$(UV) run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run mkdocs gh-deploy --force

# --- Containers -------------------------------------------------------------
docker-build:  ## Constrói a imagem do dashboard
	docker compose build dashboard

docker-up:  ## Sobe dashboard + MLflow (modelo e dados gold montados do host)
	docker compose up --build

docker-down:  ## Derruba os serviços
	docker compose down

# --- Utilitários ------------------------------------------------------------
profile:  ## Profiling da pipeline com scalene (ajuste o alvo)
	$(UV) run --group profiling scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml site
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

clean-data:  ## Remove as camadas derivadas (silver e gold); a bronze é imutável
	rm -f data/silver/*.parquet data/gold/*.parquet

clean-reports:  ## Remove figuras geradas (as métricas JSON são commitadas)
	rm -f reports/figures/*.png reports/figures/*.svg

clean-notebooks:  ## Remove os outputs dos notebooks (nbstripout)
	$(UV) run nbstripout notebooks/*.ipynb
