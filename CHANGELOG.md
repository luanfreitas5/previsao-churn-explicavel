# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o versionamento adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- Estrutura inicial do projeto (configs, `src/`, `app/`, `tests/`, docs).
- Engenharia de features RFM e rotulagem de churn (não recompra) sem vazamento.
- Contrato de dados `pandera` para a base de features.
- Pipeline de modelagem LightGBM com baseline, validação cruzada e MLflow.
- Avaliação rigorosa: PR-AUC com IC bootstrap, calibração e justiça por UF.
- Explicabilidade SHAP global e por cliente em linguagem de negócio.
- Dashboard Streamlit de risco de churn e explicações.
- Suíte de testes (unitários, property-based e comportamentais de ML).

## [0.1.0] - 2026-07-04

### Adicionado
- Definição do escopo do projeto e configuração de ferramentas de qualidade.

[Não lançado]: https://github.com/luanfreitas5/previsao-churn-explicavel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/luanfreitas5/previsao-churn-explicavel/releases/tag/v0.1.0
