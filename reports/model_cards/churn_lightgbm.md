# Model Card — churn-lightgbm v0.6.0

**Status:** DRAFT
**Bloqueios pendentes:** (a) baseline medido no teste, teste de significância e tamanho (`n`) por UF — bloqueiam qualquer afirmação de que o modelo é "melhor" que o baseline e o uso de recortes por UF em decisões; (b) rastreabilidade (Git SHA e run MLflow) — o modelo atual foi treinado antes das tags de linhagem, bloqueia a promoção no Registry; (c) validação externa/temporal — bloqueia uso fora de apoio à priorização de campanhas.

## 1. Detalhes do modelo
- **Responsável / contato:** Luan Freitas — luan.mgf@gmail.com
- **Data:** 2026-07-30 (treino do artefato `models/churn_lightgbm.joblib`)
- **Algoritmo:** LightGBM (500 árvores, `learning_rate=0.03`, `num_leaves=31`, `class_weight="balanced"`) dentro de um `sklearn.Pipeline` (imputação + one-hot de UF + modelo); parâmetros em `configs/model_params.yaml`
- **Rastreabilidade:** Git SHA `TODO(Luan): retreinar com require_clean_tree ativo para gravar o SHA` · MLflow run `TODO(Luan): id do run após o retreino` · hash SHA-256 dos dados de treino `39fdf32f7bfcbbf53825760d1ff2672f5e772accf98214100984f10834457622` (`data/gold/train.parquet`, de `models/churn_lightgbm_meta.json`)
- **Licença:** MIT (código). O dataset tem licença própria — ver [Datasheet](../datasheets/olist.md)

## 2. Uso pretendido
- **Uso primário:** ranquear clientes por risco de **não recompra em 180 dias** para priorizar ações de retenção (CRM, campanhas, ofertas).
- **Usuários pretendidos:** times de marketing/retenção e analistas de dados.
- **Humano no circuito:** sim — o score apoia a priorização; nenhuma ação sobre o cliente é automática.

## 3. Usos fora do escopo
- Decisões automáticas que afetem o cliente individualmente (negar crédito, preço, atendimento ou benefício) sem revisão humana.
- Uso fora do varejo/marketplace brasileiro de 2016–2018 ou em outra definição de churn (corte e horizonte diferentes) sem revalidar.
- Tratar a probabilidade como risco absoluto: com `class_weight="balanced"` e prevalência de ~99%, a probabilidade **não** está calibrada (ver seção 5).
- Uso comercial dos dados de origem (licença CC BY-NC-SA 4.0 do dataset).

## 4. Fatores
- **Grupos relevantes:** UF do cliente (`customer_state`) — atributo sensível auditado.
- **Ambientes:** snapshot único em `2018-03-01`, horizonte de 180 dias; marketplace Olist (vendedores múltiplos).

## 5. Métricas
- **Métrica principal:** PR-AUC (*average precision*) — o custo de negócio está em ranquear bem os clientes de maior risco. **Atenção:** com 98,8% de positivos, a PR-AUC de um classificador sem poder discriminativo já é ≈ 0,988 (a prevalência); por isso ela precisa ser lida junto com a ROC-AUC e o baseline.
- **Limiar de decisão:** 0,5 (`configs/model_params.yaml`) — valor padrão, **não** otimizado nem escolhido em validação. `TODO(Luan): escolher o limiar na validação pelo custo de FP vs FN`.
- **Calibração:** Brier 0,0720; curva em `reports/figures/calibration_curve.png`. Sem baseline de Brier para comparar.

## 6. Dados de avaliação
Conjunto de teste de 20% (`test_size=0.2`, estratificado, `seed=42`) da mesma base e do mesmo snapshot do treino: **n = 11.272**, prevalência de churn **98,82%**. É uma divisão aleatória, **não temporal** e **sem validação externa**.

## 7. Dados de treino
Brazilian E-Commerce Public Dataset by Olist, features RFM + derivadas até a data de corte (sem informação futura). Detalhes e LGPD no [Datasheet](../datasheets/olist.md).

## 8. Análise quantitativa
Fonte dos números: [`reports/metrics/metrics.json`](../metrics/metrics.json) e `models/churn_lightgbm_meta.json`.

| Recorte | n | Métrica ± IC 95% | Baseline |
|---|---|---|---|
| Geral — PR-AUC no teste (IC bootstrap 1000×) | 11.272 | 0,9919 (0,9897–0,9939) | `TODO(Luan)`: DummyClassifier no teste (esperado ≈ prevalência, 0,988 — não medido) |
| Geral — PR-AUC em CV (5 folds, treino) | — | 0,9902 ± 0,0009 | 0,9882 (DummyClassifier, média do CV) |
| Geral — ROC-AUC no teste | 11.272 | 0,6214 (sem IC) | 0,5 por construção |
| Geral — recall / precisão / F1 (limiar 0,5) | 11.272 | 0,9435 / 0,9893 / 0,9659 (sem IC) | — |
| UF com maior recall (AC, AL, AP, CE, MA, PI, RN, RR, SE) | `TODO(Luan)` | 1,000 | — |
| UF com menor recall (TO / AM / GO) | `TODO(Luan)` | 0,865 / 0,882 / 0,916 | — |
| SP (maior UF) | `TODO(Luan)` | 0,928 | — |

Leitura honesta: o ganho de PR-AUC sobre o baseline no CV é **+0,0020** (0,9902 vs 0,9882), da ordem do próprio IC do modelo; **nenhum teste de significância foi executado** (`TODO(Luan)`: teste *t* reamostrado corrigido nos folds ou McNemar no teste). A ROC-AUC de 0,62 indica poder discriminativo apenas modesto.

## 9. Considerações éticas e fairness
- **Atributos sensíveis auditados:** UF do cliente (também é feature do modelo; é o atributo com maior importância SHAP global, 0,577 de |SHAP| médio).
- **Disparidades encontradas:** recall por UF varia de 0,865 (TO) a 1,000 (várias UFs) — amplitude de **0,135**. As UFs com recall 1,000 provavelmente têm poucos clientes; sem o `n` por UF a diferença não é interpretável (`TODO(Luan)`: acrescentar `n` e IC por UF e métricas de paridade com `fairlearn`).
- **Mitigação aplicada:** nenhuma.
- **Base legal LGPD:** dados públicos e anonimizados pelo provedor, finalidade de pesquisa/portfólio; ver Datasheet. Não há PII nas saídas.

## 10. Limitações, riscos e monitoramento
- **Limitações conhecidas:** prevalência de ~99% (clientes de compra única) torna a PR-AUC pouco informativa e o modelo útil, no máximo, para **ranquear**; ROC-AUC baixa; rótulo depende do corte (`2018-03-01`) e do horizonte (180 dias); avaliação apenas em divisão aleatória do mesmo período; UF é feature e atributo sensível ao mesmo tempo.
- **Mudanças de distribuição previstas:** dados de 2016–2018 e um único snapshot; comportamento de compra, sazonalidade (Black Friday) e mix de vendedores mudam com o tempo.
- **Monitoramento:** nenhum implementado. `TODO(Luan)`: relatório semanal de *drift* (`evidently`) e PR-AUC/ROC-AUC com rótulos novos, se o modelo for implantado.
- **Critério de retreino / descontinuação:** `TODO(Luan)`: definir o piso das métricas (hoje `min_primary_metric = 0.55` em `configs/model_params.yaml`, que não protege contra a degeneração para o baseline) e o gatilho de *drift*.
