# -*- coding: utf-8 -*-

# Árvores de Decisão (CARTs)
#Referencia:

# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

#%% Árvore de Regressão

import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

#%% Carregando o dataset

df = pd.read_csv("preco_casas.csv")

# Fonte: https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset

#%% Separando as variáveis que serão utilizadas na modelagem

df = df[['property_value','no_of_rooms', 'no_of_bathrooms', 'large_living_room', 'parking_space', 'front_garden', 'swimming_pool']]

#%% Separando as variáveis Y e X

X = df[['no_of_rooms', 'no_of_bathrooms', 'large_living_room', 'parking_space', 'front_garden', 'swimming_pool']].values
y = df['property_value'].values

#%% Coletar features

features = list(df.drop(columns=['property_value']).columns)

#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Gerando o modelo na base de dados de treino

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)

#%% Plotando a árvore

fig = plt.figure(figsize=(40,40))

_ = tree.plot_tree(tree_reg, 
                   feature_names=features,  
                   class_names=True,
                   filled=True)

#%% Salvando a árvore

fig.savefig("decision_tree_casa.png")

#%% Verificar a importância de cada variável do modelo

importancia_features = pd.DataFrame({'features':features,
                                     'importance':tree_reg.feature_importances_})

print(importancia_features.sort_values(by='importance', ascending = False))

#%% Predict da regressão

## Qual é o valor estimado para:
## 3 quartos
## 2 banheiros
## Sala ampla
## Com estacionamento
## Jardim frontal
## Sem piscina

print(tree_reg.predict([[3, 2, 1, 1, 1, 0]]))

## Resultado estimado: $ 146.053,33

#%% Acurácia da árvore de regressão

print(tree_reg.score(X_test, y_test))

#%% Coletando os predicts

y_pred = tree_reg.predict(X_test)

#%% Fazendo a mesma validação utilizando o método cross_val_score para o modelo utilizando o criterio gini

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
acuracias = cross_val_score(tree_reg, X, y, cv=cv, scoring='r2')

print("acuracias:", acuracias)
print("r2 final:", np.mean(acuracias), "+-", np.std(acuracias))

#%% Gráfico de dispersão com ajustes (fits) linear e os dados de teste

xdata = y_test
ydata_linear = y_pred

plt.figure(figsize=(10,10))
plt.plot(xdata,xdata, color='gray')
plt.scatter(xdata,ydata_linear, alpha=0.5)

plt.title('Dispersão dos dados')
plt.xlabel('Property Value')
plt.ylabel('Fitted Values')
plt.legend(['45º graus','Árvore de Regressão'])

#%% Gerando o modelo na base de dados de treino definindo profundidade = 10

tree_reg_final = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_reg_final.fit(X_train, y_train)

#%% Coletando os preditcs

y_pred = tree_reg_final.predict(X_test)

#%% Plotando a árvore

fig = plt.figure(figsize=(60,40))

_ = tree.plot_tree(tree_reg_final, 
                   feature_names=features,  
                   class_names=True,
                   filled=True)

#%% Salvando a árvore

fig.savefig("decision_tree_casa_final.png")

#%% Validação utilizando o cross_val_score no modelo profundidade = 10
# ATENÇÃO: Esta célula do código foi ajustada para "tree_reg_final"
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
acuracias = cross_val_score(tree_reg_final, X, y, cv=cv, scoring='r2')

print("acuracias:", acuracias)
print("r2 final:", np.mean(acuracias), "+-", np.std(acuracias))

#%% Gráfico de dispersão com ajustes (fits) linear e os dados de teste para comparativo

xdata = y_test
ydata_linear = y_pred

plt.figure(figsize=(10,10))
plt.plot(xdata,xdata, color='gray')
plt.scatter(xdata,ydata_linear, alpha=0.5)

plt.title('Dispersão dos dados')
plt.xlabel('Property Value')
plt.ylabel('Fitted Values')
plt.legend(['45º graus','Árvore de Regressão'])
