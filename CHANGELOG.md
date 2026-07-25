## v0.2.0 (2026-07-25)

### Feat

- adiciona workflows de CI para revisão de dependências e reprodutibilidade
- adiciona configuração do pre-commit para checagens automáticas
- implementa pipeline de previsão de churn
- adiciona testes para a pipeline de churn
- adiciona notebooks de EDA e modelagem com LightGBM Inclui análise exploratória e avaliação de modelo com explicabilidade SHAP.
- adiciona dashboard Streamlit para previsão de churn
- adiciona arquivos de configuração e documentação do projeto
- adiciona configuração do devcontainer e CI/CD
- adiciona Dockerfile e docker-compose.yml para orquestração do projeto
- adiciona arquivos de configuração e documentação do projeto
- adiciona arquivo pyproject.toml com configuração do projeto
- adiciona arquivos .gitignore e LICENSE

### Fix

- atualiza dependência gitpython para >=3.1.51 Adiciona restrições em dependências transitivas por segurança e atualiza a versão do gitpython para 3.1.53.
- adiciona .gitattributes para normalização de LF
- ajusta comando do pytest e adiciona limpeza de notebooks
- ajusta verificação de outputs em notebooks e formatação
- ajusta formatação e importações em diversos arquivos
- ajusta Makefile e refatora código em rfm.py e explain.py
- atualiza nome do kernel e adiciona informações de linguagem
- ajusta ignorados no refurb e corrige ordenação de bandas de risco
- ajusta tipos de retorno e conversões em várias funções
- corrige parâmetros do método __exit__ no MLflowTracker
- remove parâmetros não utilizados no método __exit__
- pin shap dependency to a version compatible with py3.10-3.12
- atualiza a versão do Plotly para 6.9.0
- remove plotly da lista de dependências
- substitui np.random.seed por rng.default_rng
