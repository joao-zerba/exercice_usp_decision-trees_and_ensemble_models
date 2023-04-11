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

df = pd.read_csv("plano_saude.csv")

# Fonte: https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset?select=insurance.csv

#%% Informações descritivas das variáveis

print(df[['age','bmi', 'children', 'charges']].describe())

print(df['sex'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())

#%% Transformando variáveis categóricas em dummies

df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

#%% Separando as variáveis Y e X

X = df[['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
        'region_northwest', 'region_southeast', 'region_southwest']].values

y = df['charges'].values

#%% Coletando as features

features = list(df.drop(columns=['charges']).columns)

#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Gerando o modelo na base de dados de treino

tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg.fit(X_train, y_train)

#%% Plotando a árvore

fig = plt.figure(figsize=(64,40))

_ = tree.plot_tree(tree_reg, 
                   feature_names=features,  
                   class_names=True,
                   filled=True)

#%% Salvando a árvore

fig.savefig("decision_tree_plano.png")

#%% Verificar a importância de cada variável do modelo

importancia_features = pd.DataFrame({'features':features,
                                     'importance':tree_reg.feature_importances_})

print(importancia_features.sort_values(by='importance', ascending = False))

#%% Predict da regressão

## Qual é o valor estimado para:
## 35 anos
## bmi = 30
## 2 filhos
## Sexo feminino
## Não fumante
## Região sudeste

print(tree_reg.predict([[35, 30, 2, 0, 0, 0, 1, 0]]))

## Resultado estimado: $ 7.377;69

#%% Acurácia da árvore de regressão

print(tree_reg.score(X_test, y_test))

#%% Fazendo a validação cruzada para capturar a natureza estocástica do modelo

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
acuracias = cross_val_score(tree_reg, X, y, cv=cv, scoring='r2')

print("acuracias:", acuracias)
print("r2 final:", np.mean(acuracias), "+-", np.std(acuracias))
