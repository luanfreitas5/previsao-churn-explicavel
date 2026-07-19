# Previsão de Churn Explicável — Olist

Modelo de previsão de **churn (não recompra)** de clientes do marketplace da Olist,
combinando features **RFM**, **LightGBM** e **SHAP** para explicações em linguagem
de negócio.

## Por que este projeto?

- **Orientado à decisão:** prioriza clientes de maior risco para ações de retenção.
- **Explicável:** cada previsão vem com os fatores que a justificam (SHAP).
- **Rigoroso:** métricas com incerteza, calibração e auditoria de justiça por UF.
- **Reprodutível:** seeds fixas, contratos de dados e rastreamento com MLflow.

## Navegação

- [Guia de uso](guia-uso.md) — instalação e execução da pipeline.
- [Arquitetura](arquitetura.md) — decisões de design e fluxo de dados.
- [Referência da API](referencia.md) — documentação gerada das docstrings.

!!! warning "Definição de churn"
    Como a base da Olist é majoritariamente de compra única, definimos churn por
    **não recompra em uma janela futura** a partir de uma data de corte. Detalhes
    em [Arquitetura](arquitetura.md).
