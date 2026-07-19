"""Pacote-raiz do projeto de previsão de churn explicável (Olist).

Organiza a pipeline por funcionalidade em subpacotes:

- ``config``: carregamento e validação de configurações, paths, logging e seed.
- ``constants``: nomes de colunas e valores padrão do domínio.
- ``exceptions``: exceções customizadas do projeto.
- ``data``: carregamento e divisão dos dados brutos da Olist.
- ``schemas``: contratos de dados (pandera) por estágio da pipeline.
- ``features``: engenharia de features RFM e rotulagem de churn.
- ``models``: construção e persistência do pipeline de modelagem.
- ``training``: treino, validação cruzada e rastreamento de experimentos.
- ``evaluation``: métricas, calibração e auditoria de justiça.
- ``explainability``: explicações SHAP globais e por cliente.
- ``visualization``: tema e gráficos padronizados.
- ``pipelines``: orquestração ponta a ponta.
- ``utils``: utilitários compartilhados.
"""

__version__ = "0.1.0"
