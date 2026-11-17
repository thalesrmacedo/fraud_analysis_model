# Detecção de Fraudes em Transações de Cartão de Crédito

## 1. Problema de Negócio

A fraude em transações de cartão de crédito representa um desafio para instituições financeiras, pois envolve a identificação de operações potencialmente fraudulentas em meio a um volume muito maior de transações legítimas. O objetivo deste projeto foi desenvolver modelos de Machine Learning capazes de identificar transações fraudulentas, utilizando dados históricos de operações com cartão de crédito. O problema foi estruturado como uma tarefa de classificação binária:

- **Classe 0:** transação legítima.
- **Classe 1:** transação fraudulenta.

Um dos principais desafios é o desbalanceamento das classes: aproximadamente 99,8% das observações representam transações legítimas, enquanto apenas 0,17% correspondem a fraudes. Nesse cenário, a acurácia não é uma métrica suficiente para avaliar o desempenho dos modelos. Um classificador que identificasse todas as transações como legítimas poderia apresentar uma acurácia elevada, mas seria incapaz de cumprir o objetivo principal do projeto de detectar fraudes. Por esse motivo, a avaliação prioriza as métricas da classe positiva (fraude), especialmente recall, precisão e F1-score.

## 2. Preparação dos Dados

A base de dados utilizada contém transações realizadas com cartões de crédito, disponibilizada publicamente para estudos de detecção de fraudes. [Clique aqui para fazer o download da base de dados.](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)

As variáveis estão organizadas da seguinte maneira:

- **Time:** representa o tempo decorrido entre uma transação e a primeira transação registrada na base.
- **Amount:** valor da transação.
- **V1 a V28:** variáveis numéricas transformadas por meio da técnica PCA (Principal Component Analysis), utilizada para preservar a privacidade dos dados originais.
- **Class:** variável-alvo, indicando se a transação é legítima ou fraudulenta.

As etapas de preparação dos dados incluem:

1. Carregamento e exploração inicial da base utilizando Pandas (head()).
2. Verificação da distribuição das classes e identificação do nível de desbalanceamento (value_counts()).
3. Transformação da variável Amount utilizando `np.log1p`, buscando reduzir a assimetria da distribuição dos valores.
4. Padronização das variáveis numéricas utilizando `StandardScaler`.
5. Separação dos dados em conjuntos de treinamento e teste, utilizando estratificação (`stratify`) para preservar a proporção das classes.

A padronização e as demais transformações que aprendem parâmetros a partir dos dados são ajustadas exclusivamente com o conjunto de treinamento, evitando vazamento de informações (*data leakage*).

## 3. Treinamento e Comparação dos Modelos

Foram selecionados diferentes algoritmos de classificação para comparar suas capacidades de identificar transações fraudulentas.

Os modelos inicialmente considerados foram:

- **Regressão Logística:** utilizada como modelo baseline, permitindo estabelecer uma referência inicial de desempenho.
- **Random Forest:** modelo baseado em múltiplas árvores de decisão, com utilização de pesos para compensar o desbalanceamento das classes (`class_weight='balanced'`).
- **XGBoost:** algoritmo baseado em Gradient Boosting, capaz de identificar relações não lineares e padrões complexos nos dados.

A comparação considera principalmente as métricas da classe fraudulenta:

| Modelo | Precisão | Recall | F1-score |
|---|---|---|---|
| Regressão Logística | [0.87] | [0.66] | [0.75] |
| Random Forest | [0.84] | [0.76] | [0.79] |
| XGBoost | [0.94] | [0.78] | [0.85] |

### Critérios de avaliação

- **Recall:** indica a proporção de fraudes reais que foram corretamente identificadas pelo modelo. Um recall elevado significa que menos fraudes passaram despercebidas.
- **Precisão:** indica quantas das transações classificadas como fraudulentas realmente eram fraudes. Essa métrica ajuda a controlar os falsos positivos.
- **F1-score:** representa a média harmônica entre precisão e recall, permitindo avaliar o equilíbrio entre as duas métricas.

A escolha do modelo não considera exclusivamente o maior resultado de uma métrica isolada. Também são considerados os impactos dos falsos positivos e falsos negativos no contexto de uma operação financeira.

## 4. Ajuste do Limiar de Decisão

Por padrão, os classificadores binários utilizam 0,5 como limiar de decisão (threshold). Isso significa que transações com probabilidade estimada de fraude igual ou superior a esse valor são classificadas como fraudulentas. Entretanto, em problemas altamente desbalanceados, esse limiar pode não ser o mais adequado. Neste projeto, foi testado um limiar de 0,3, com o objetivo de aumentar a sensibilidade do modelo à identificação de fraudes. A escolha do limiar deve considerar esse equilíbrio, pois um sistema financeiro não deve apenas identificar o maior número possível de fraudes, mas também evitar bloqueios desnecessários de operações legítimas. Ao utilizar um limiar de 0,3, verificou-se que a redução do limiar tende a aumentar o recall (0,71 - superior ao da regressão logística), mas também pode reduzir a precisão (0,85 - ligeiramente inferior ao da regressão logística) e elevar a quantidade de falsos positivos, ou seja, transações legítimas classificadas incorretamente como fraudulentas.

## 5. Explicabilidade do Modelo com SHAP

Além de identificar fraudes, é importante compreender quais variáveis contribuíram para as decisões do modelo. Foi utilizada a biblioteca SHAP (*SHapley Additive exPlanations*) para interpretar as contribuições das variáveis nas previsões realizadas pelo algoritmo. A análise verificou que as variáveis V4, V14 apresentaram maior influência nas previsões do modelo a nível global. Essa abordagem contribui para tornar as previsões mais interpretáveis, facilitando a análise dos padrões identificados pelo algoritmo. É importante destacar que a contribuição de uma variável para a previsão não representa necessariamente uma relação de causalidade.

## 6. Melhorias futuras

Além dos modelos apresentados nas aulas, foram consideradas possibilidades de utilizar outros algoritmos para aprofundar a análise e avaliar diferentes estratégias de tratamento do desbalanceamento. Dentre os escolhidos, podemos citar:

| Algoritmo | Categoria | Principal característica | Status |
|---|---|---|---|
| Regressão Logística | Modelo linear | Baseline para classificação binária. | ✅ Concluído |
| Random Forest | Ensemble de árvores | Combina várias árvores de decisão. | ✅ Concluído |
| XGBoost | Gradient Boosting | Aprende padrões complexos por meio de árvores sequenciais. | ✅ Concluído |
| LightGBM | Gradient Boosting | Otimizado para eficiência e velocidade. | 🔄 Em andamento |
| SVM | Classificador por margem | Encontra fronteiras de separação entre classes. | 🔄 Em andamento |
| MLP | Rede neural | Identifica relações não lineares entre variáveis. | 🔄 Em andamento |

Também espera-se discutir o SHAP a nível local, ou seja, investigar uma transação específica e compreender quais características contribuíram para que ela fosse classificada como legítima ou fraudulenta.

## 7. Conclusão

O projeto demonstra a aplicação de técnicas de Machine Learning em um problema real de classificação com classes altamente desbalanceadas. A comparação entre algoritmos, o ajuste do limiar de decisão e a utilização de ferramentas de explicabilidade permitem avaliar não apenas a capacidade preditiva dos modelos, mas também os impactos de suas decisões. Como resultado, busca-se desenvolver um modelo capaz de identificar uma parcela significativa das transações fraudulentas, mantendo um nível aceitável de precisão e oferecendo maior transparência sobre os critérios utilizados nas previsões. Vale destacar que nenhum modelo é capaz de eliminar completamente os falsos positivos e falsos negativos. Portanto, a escolha do modelo deve considerar o equilíbrio entre a identificação de fraudes e a redução de classificações incorretas, buscando minimizar os impactos operacionais e financeiros para a instituição.
