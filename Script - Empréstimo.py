# -*- coding: utf-8 -*-

# Árvores de Decisão (CARTs)
# Referencia:

# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

#%% Árvore de classificação (variável dependente com 2 categorias)
#%% Importando os pacotes

import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc#, plot_roc_curve

#%% Carregando o dataset

df = pd.read_excel("emprestimo_banco.xlsx")

# Fonte: adaptado de https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling

#%% Informações das variáveis

print(df[['Age','Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']].describe())

print(df['Education'].value_counts())
print(df['Securities_Account'].value_counts())
print(df['CD_Account'].value_counts())
print(df['Online'].value_counts())
print(df['CreditCard'].value_counts())
print(df['Personal_Loan'].value_counts())

#%% Transformando variáveis categóricas em dummies

df = pd.get_dummies(df, columns=['Education'], drop_first=True)

#%% Separando as variáveis Y e X

X = df.drop(columns=['ID', 'Personal_Loan']).values
y = df['Personal_Loan'].values

#%% Coletar features

features = list(df.drop(columns=['ID', 'Personal_Loan']).columns)

#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Quantidade de observações em cada amostra

print("Shape do dataset de treino: ", X_train.shape)
print("Shape do dataset de teste: ", X_test.shape)

#%% Gerando o modelo na base de dados de treino

tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

#%% Plotando a árvore (base de treino)

fig = plt.figure(figsize=(40,28))

_ = tree.plot_tree(tree_clf, 
                   feature_names=features,  
                   class_names=['Não','Sim'],
                   filled=True)

#%% Salvando a árvore na pasta do project

fig.savefig("decision_tree_emprest.png")

#%% Verificar a importância de cada variável do modelo

importancia_features = pd.DataFrame({'features':features,
                                     'importance':tree_clf.feature_importances_})

print(importancia_features.sort_values(by='importance', ascending = False))

#%% Probabilidades preditas pelo modelo

## Qual é a probabilidade estimada para uma pessoa:
## Age = 35
## Experience = 10
## Income = 100
## Family = 2
## CCAvg = 1.90
## Mortgage = 0 
## Securities_Account = 1
## CD_Account = 1
## Online = 1 
## CreditCard = 1
## Education_Advanced/Professional = 1

print(tree_clf.predict_proba([[35, 10, 100, 2, 1.9, 0, 1, 1, 1, 1, 1, 0]]))

#%% Escolhendo com base na maior probabilidade:

print(tree_clf.predict([[35, 10, 100, 2, 1.9, 0, 1, 1, 1, 1, 1, 0]]))

## Indica que "Obtém empréstimo"

#%% Fitted values do banco de dados de teste

## O objetivo é verificar os valores preditos para o dataset de teste

y_pred = tree_clf.predict(X_test)

#%% Gerando a matriz de confusão

## Compara os valores preditos pela árvore com os valores observados (teste)

cm = confusion_matrix(y_test, 
                      y_pred, 
                      labels=tree_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)

disp.plot()

#%% Estabelecendo um cutoff

cutoff = 0.75

y_pred = (tree_clf.predict_proba(X_test)[:, 1] > cutoff).astype('float')

#%% Matriz de confusão para o cutoff

cm = confusion_matrix(y_test, 
                      y_pred, 
                      labels=tree_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)

disp.plot()

#%% Matriz de classificação

print(classification_report(y_test, y_pred))

## Precision: VP / (VP + FP) "acurácia das predições positivas"
## Recall: VP / (VP + FN) "sensibilidade do modelo"
## F1-Score: VP / (VP + (FN + FP)/2)
## Support: são as contagens observadas nos dados
## Accuracy: (VP + VN) / TOTAL
## Macro avg: é a média aritmética dos indicadores
## Weighted avg: é a média ponderada dos indicadores (ponderada pela prop. support)

#%% Verificar acurácia do modelo de outra maneira

print(accuracy_score(y_test, y_pred))

#%% Plotar a curva ROC nos dados de teste

fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr, tpr)

gini = (roc_auc - 0.5)/(0.5)

#TODO update roc plot method
# #plot_roc_curve(tree_clf, X_test, y_test) 
# plt.title('Coeficiente de GINI: %g' % round(gini,4), fontsize=12)
# plt.xlabel('1 - Especificidade', fontsize=12)
# plt.ylabel('Sensitividade', fontsize=12)
# plt.show()

#%% Alterando para entropia

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42, criterion="entropy")
tree_clf.fit(X_train, y_train)

# Plotando a árvore

fig = plt.figure(figsize=(20,14))

_ = tree.plot_tree(tree_clf, 
                   feature_names=features,  
                   class_names=['Não','Sim'],
                   filled=True)
