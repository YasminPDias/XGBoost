# Previsão de inadimplência de cartão de crédito

## 1. Qual problema o projeto resolve?

O projeto prevê se um cliente de cartão de crédito tem risco de inadimplir no
próximo mês. A inadimplência ocorre quando o cliente deixa de realizar os
pagamentos esperados da fatura.

Esse é um problema de **classificação binária**:

- `0`: o cliente não inadimpliu;
- `1`: o cliente inadimpliu.

A previsão pode ajudar uma instituição financeira a identificar clientes que
precisam de uma análise mais cuidadosa, oferecer orientação financeira ou
definir estratégias de crédito mais adequadas. O modelo serve como apoio à
decisão e não deve ser utilizado como único critério para negar crédito a um
cliente.

O conjunto de dados contém informações como:

- limite de crédito;
- idade;
- histórico de atraso dos últimos seis meses;
- valores das faturas;
- valores pagos nos últimos seis meses;
- indicação de inadimplência no mês seguinte.

## 2. Como funciona o algoritmo XGBoost?

O XGBoost, sigla de **Extreme Gradient Boosting**, é um algoritmo de
aprendizado supervisionado baseado em árvores de decisão.

Ele cria várias árvores de decisão em sequência. Cada nova árvore tenta corrigir
os erros cometidos pelo conjunto de árvores construído anteriormente. Ao final,
as previsões de todas as árvores são combinadas para produzir uma previsão mais
precisa.

O processo pode ser resumido da seguinte forma:

1. o algoritmo cria uma primeira árvore de decisão;
2. identifica os erros dessa árvore;
3. cria outra árvore dando mais atenção aos erros anteriores;
4. repete esse processo várias vezes;
5. combina todas as árvores para calcular a probabilidade de inadimplência.

O treinamento utiliza uma função de perda para medir a qualidade das previsões.
Neste projeto, o modelo usa `logloss`, uma métrica adequada para classificação
que avalia também a qualidade das probabilidades produzidas.

Durante o treinamento, são ajustados hiperparâmetros como:

- número de árvores (`n_estimators`);
- profundidade máxima das árvores (`max_depth`);
- taxa de aprendizagem (`learning_rate`);
- quantidade mínima de exemplos por divisão (`min_child_weight`);
- proporção de amostras utilizadas (`subsample`);
- regularização (`reg_lambda`).

Esses parâmetros são selecionados por busca aleatória utilizando validação
cruzada estratificada com cinco divisões. A métrica usada para escolher a
melhor configuração é a AUC-ROC.

## 3. Como o XGBoost resolve o problema?

Antes do treinamento, os dados são preparados e novas características são
criadas a partir do histórico financeiro do cliente:

- `utilization`: percentual utilizado do limite de crédito;
- `avg_status`: média dos status de pagamento;
- `max_status`: pior status de pagamento;
- `n_late`: quantidade de meses com atraso;
- `paid_ratio`: proporção entre os valores pagos e os valores das faturas.

Essas informações ajudam o modelo a reconhecer padrões associados à
inadimplência. Por exemplo, um cliente que apresenta vários meses de atraso,
utiliza grande parte do limite e paga uma pequena proporção das faturas pode
receber uma probabilidade maior de inadimplência.

O fluxo do projeto é:

1. carregar o conjunto de dados;
2. remover identificadores e registros incompletos;
3. criar as características financeiras;
4. separar os dados em treinamento e teste;
5. treinar e ajustar o XGBoost com validação cruzada;
6. calcular a probabilidade de inadimplência;
7. aplicar o limiar definido pelo treinamento;
8. classificar o cliente como de risco ou não;
9. disponibilizar o resultado pela API Flask.

O modelo foi treinado com um limiar de aproximadamente `0,267`. Portanto, se a
probabilidade calculada for igual ou superior a esse valor, o cliente é
classificado como potencial inadimplente.

Além da classificação, a API retorna os principais fatores que contribuíram
para a previsão. Isso torna o resultado mais compreensível e permite observar
quais características aumentaram ou reduziram o risco estimado.

## 4. Resultados do modelo

Após o treinamento e a avaliação no conjunto de teste, foram obtidos os
seguintes resultados:

| Métrica | Resultado |
|---|---:|
| AUC-ROC | 0,780 |
| Acurácia | 0,786 |
| Precisão | 0,514 |
| Recall | 0,572 |
| F1-score | 0,542 |
| AUC média na validação cruzada | 0,786 |

A AUC de `0,780` indica que o XGBoost consegue distinguir razoavelmente entre
clientes que tendem a inadimplir e clientes que não tendem a inadimplir.

O recall de `0,572` mostra que o modelo identificou aproximadamente 57,2% dos
clientes inadimplentes no conjunto de teste. Essa métrica é relevante porque
perder um cliente de risco pode gerar prejuízo para a instituição financeira.

## 5. Disponibilização do modelo

O modelo treinado é salvo em `model.json`. A aplicação Flask carrega esse
arquivo e disponibiliza o endpoint:

```text
POST /api/predict
```

Esse endpoint recebe os dados financeiros de um cliente e retorna:

- probabilidade de inadimplência;
- classificação de risco;
- limiar utilizado;
- taxa geral de inadimplência;
- principais fatores da previsão.

Dessa forma, o XGBoost não fica limitado a um experimento offline: ele é
integrado a um serviço que pode ser consumido por uma aplicação web ou mobile.
