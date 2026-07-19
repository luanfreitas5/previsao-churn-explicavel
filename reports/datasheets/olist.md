# Datasheet — Brazilian E-Commerce Public Dataset by Olist

## Motivação

- **Propósito:** estudar recompra/churn em marketplace brasileiro de forma
  realista e reprodutível.
- **Fonte:** [Olist / Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **Licença:** CC BY-NC-SA 4.0 (uso não comercial; verifique os termos na origem).

## Composição

- ~100 mil pedidos de 2016 a 2018 de múltiplos vendedores.
- Tabelas usadas: pedidos, clientes, itens, pagamentos, avaliações.
- **Granularidade:** um `customer_id` por pedido; `customer_unique_id` agrega o
  mesmo indivíduo ao longo do tempo.

## Processo de coleta

- Dados transacionais reais anonimizados pela Olist antes da publicação.
- Identificadores diretos foram substituídos por códigos (pseudônimos).

## Pré-processamento (neste projeto)

- Junção de pedidos com clientes, itens, pagamentos e avaliações.
- Filtragem por status válidos e por data de corte (snapshot).
- Agregação por cliente em features RFM + derivadas (ver `src/features/`).
- Validação por contrato `pandera` antes da modelagem.

## Privacidade / LGPD

- **Sem PII:** o dataset já é anonimizado; nenhum identificador direto é exposto
  em logs, figuras ou saídas.
- **Minimização:** apenas colunas necessárias à tarefa são carregadas.
- **Quasi-identificadores:** UF/cidade são usados de forma agregada; atenção a
  risco de reidentificação em cruzamentos externos.
- **Base legal:** dados públicos, anonimizados, para fins de pesquisa/portfólio.

## Usos recomendados e não recomendados

- **Recomendado:** modelagem de churn/recompra, estudos de explicabilidade.
- **Não recomendado:** qualquer tentativa de reidentificação de indivíduos.
