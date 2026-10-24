# Datasheet — Brazilian E-Commerce Public Dataset by Olist

## 1. Motivação
- Por que foi criado? Para qual tarefa? Publicado pela Olist (marketplace brasileiro) no Kaggle como dados reais anonimizados de pedidos, para estudo de e-commerce. Neste projeto serve à previsão de **churn (não recompra)** de clientes e à explicabilidade das previsões.
- Quem criou e quem financiou? Criado e publicado pela Olist; sem informação de financiamento. Fonte: [Olist / Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). `TODO(Luan): confirmar a descrição oficial da motivação na página do Kaggle.`

## 2. Composição
- O que cada instância representa? Nas tabelas de origem: um **pedido** (`orders`), um item de pedido, um pagamento, uma avaliação ou um cliente. Na base de modelagem (camada gold): **um cliente** (`customer_unique_id`) no snapshot de `2018-03-01`.
- Número de instâncias e período coberto: ~100 mil pedidos entre 2016-09 e 2018-10. Teste da base de modelagem: 11.272 clientes (20%); `TODO(Luan): tamanho total da base de features e do treino`.
- Campos, tipos e contrato: `src/schemas/features.py` (contrato `pandera` validado na saída da engenharia de features). Tabelas usadas: pedidos, clientes, itens, pagamentos e avaliações; geolocalização, produtos e vendedores ficam na bronze, sem uso no modelo.
- Rótulo: `churn = 1` se o cliente **não** faz pedido válido em `(2018-03-01, 2018-03-01 + 180 dias]`; `churn = 0` caso contrário (`configs/config.yaml`). Prevalência de churn no teste: **98,82%**.
- Dados ausentes (taxa por coluna): `TODO(Luan)`: gerar a tabela de ausência por coluna (as features de avaliação e entrega dependem de pedidos entregues/avaliados).
- Campos sensíveis / PII e tratamento: sem identificadores diretos (`customer_id`/`customer_unique_id` são códigos pseudônimos). UF (`customer_state`) e CEP/cidade são quasi-identificadores; apenas a UF é usada, como feature e como atributo de auditoria de justiça.
- É amostra de uma população maior? Como amostrado? Sim — dados de um marketplace em um período; o provedor não documenta o critério de seleção. `TODO(Luan): conferir nota do provedor sobre amostragem.`

## 3. Processo de coleta
- Fonte(s) e sistema(s) de origem: transações do marketplace Olist, anonimizadas pela empresa antes da publicação (nomes de lojas/parceiros trocados por nomes de casas de Game of Thrones, segundo a descrição do Kaggle).
- Período e frequência de coleta: 2016–2018, extração única (dataset estático).
- Consentimento / aviso aos titulares: não documentado pelo provedor. `TODO(Luan)`: registrar a base de legitimação declarada no Kaggle.

## 4. Pré-processamento
- Limpeza, filtragem e linhas removidas por regra: filtro de pedidos por status válidos (`delivered`, `shipped`, `invoiced`, `approved`) e por data de corte; features calculadas só com pedidos ≤ corte. `TODO(Luan)`: contagem de linhas removidas por regra.
- Camadas (arquitetura medalhão): **bronze** `data/bronze/` — CSVs exatamente como recebidos, imutáveis; **silver** `data/silver/*.parquet` — as 5 tabelas usadas, com colunas de data tipadas; **gold** `data/gold/` — `churn_features.parquet`, `train.parquet` e `test.parquet` (RFM + derivadas, alvo, divisão estratificada 80/20, `seed=42`).
- Anonimização / pseudonimização: já feita pelo provedor; o projeto não reidentifica nem acrescenta identificadores.
- Hash SHA-256 dos dados brutos: `TODO(Luan)`: `uv run python -c "from pathlib import Path; from src.config.environment import hash_file; print(hash_file(Path('data/bronze/olist_orders_dataset.csv')))"` por CSV. Hash do treino da camada gold usado no modelo atual: `39fdf32f7bfcbbf53825760d1ff2672f5e772accf98214100984f10834457622`.

## 5. Usos
- Usos pretendidos: modelagem de churn/recompra, estudos de explicabilidade e justiça, portfólio e pesquisa.
- Usos desaconselhados: tentativa de reidentificação de clientes; decisões automáticas sobre indivíduos; extrapolação para o comportamento atual (dados de 2016–2018); uso comercial (licença).
- Riscos de viés conhecidos: forte concentração geográfica (a maioria dos clientes em SP/Sudeste), pequenos `n` em UFs do Norte/Nordeste, base majoritariamente de compra única (prevalência de churn ~99%), dados de um único marketplace.

## 6. Distribuição
- Licença e restrições: CC BY-NC-SA 4.0 (atribuição, uso **não comercial**, compartilhamento pela mesma licença). Confirmar os termos na origem.
- Onde está armazenado (DVC remote, bucket): localmente em `data/bronze/` (fora do Git, `.gitignore`). Não há DVC configurado no projeto.

## 7. Manutenção
- Responsável: Luan Freitas (luan.mgf@gmail.com) — apenas o recorte usado no projeto; o dataset de origem é mantido pela Olist.
- Frequência de atualização: nenhuma (dataset estático de 2016–2018).
- Como atender pedidos de exclusão de titulares: o dataset é anonimizado e não permite localizar um titular; pedidos de exclusão devem ser tratados com o provedor. Os artefatos derivados (silver, gold, modelo) são regeneráveis com `make pipeline` a partir da bronze.

## 8. LGPD
- Base legal (art. 7 / art. 11): dados públicos e anonimizados, tratados para fins de pesquisa/portfólio (art. 7º, IV, ou dado anonimizado, fora do escopo da LGPD conforme art. 12 — `TODO(Luan)`: confirmar o enquadramento com o encarregado/DPO caso o projeto vire produto).
- Prazo de retenção: `TODO(Luan)`: definir; hoje os dados ficam apenas no ambiente local de desenvolvimento.
- Relatório de impacto (RIPD) necessário? Não para o uso atual (dado anonimizado, sem PII). Reavaliar se o modelo for usado em produção com dados próprios de clientes.
