# -*- coding: utf-8 -*-

# Árvores de Decisão (CARTs)

#Referência
# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

#%% Árvore de classificação (variável dependente com mais de 2 categorias)
#%% Importando os pacotes

import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

#%% Importando o dataset

df = pd.read_excel("segmenta_telecom.xlsx")

## Fonte: adaptado de https://www.kaggle.com/datasets/prathamtripathi/customersegmentation?select=Telecust1.csv

#%% Estatísticas descritivas

print(df[['meses_cliente', 'idade', 'anos_emprego']].describe())

print(df['regiao'].value_counts())
print(df['casado'].value_counts())
print(df['aposentado'].value_counts())
print(df['masculino'].value_counts())
print(df['categoria'].value_counts())

#%% Transformando variáveis categóricas em dummies

df = pd.get_dummies(df, columns=['regiao'], drop_first=True)

#%% Separando as variáveis Y, X e coletando as features

X = df.drop(columns=['categoria']).values
y = df['categoria'].values

features = list(df.drop(columns=['categoria']).columns)

#%% Definindo bases de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%% Gerando a árvore na base de dados de treino

tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

#%% Plotando a árvore

fig = plt.figure(figsize=(64,40))

_ = tree.plot_tree(tree_clf, 
                   feature_names=features,  
                   class_names=tree_clf.classes_,
                   filled=True)

#%% Salvando a árvore

fig.savefig("decision_tree_segmenta.png")

#%% Verificar a importância de cada variável do modelo

importancia_features = pd.DataFrame({'features':features,
                                     'importance':tree_clf.feature_importances_})

print(importancia_features)

#%% Probabilidades estimadas pelo modelo

## Cliente há 6 meses
## Idade = 45 anos
## Casado
## 3 anos de emprego
## Não aposentado
## Sexo masculino
## Região B

print(tree_clf.predict_proba([[6, 45, 1, 3, 0, 1, 0, 1]]))

## Probabilidades preditas:
## Categoria A = 44,29%
## Categoria B = 9,39%
## Categoria C = 20,13%
## Categoria D = 26,17%

print(tree_clf.predict([[6, 45, 1, 3, 0, 1, 0, 1]]))

#%% Obtendo valores preditos na base de teste

y_pred = tree_clf.predict(X_test)

#%% Matriz de confusão

cm = multilabel_confusion_matrix(y_test, y_pred, labels=tree_clf.classes_)

for index, item in enumerate(cm):
    print("Classe: ",tree_clf.classes_[index], "======================")
    disp = ConfusionMatrixDisplay(confusion_matrix=item)
    disp.plot()

    plt.show()

#%% Resumo do modelo

print(classification_report(y_test, y_pred))

#%% Fazendo a validação cruzada para capturar a natureza estocástica do modelo

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
acuracias = cross_val_score(tree_clf, X, y, cv=cv, scoring='accuracy')

print("acuracias:", acuracias)
print("acuracia final:", np.mean(acuracias), "+-", np.std(acuracias))

#%% Gerando um dataframe predito vs. teste

resultado = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
