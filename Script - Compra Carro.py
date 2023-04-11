# -*- coding: utf-8 -*-

# Árvores de Decisão (CARTs)

# Referencias
# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

#%% Árvore de classificação (variável dependente com 2 categorias)
#%% Importando os pacotes. Executar no console: pip install -r requirements.txt

import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc#, plot_roc_curve
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

#%% Carregando o dataset

df = pd.read_csv("dados_carros.csv")

# Fonte: https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset?select=car_data.csv

#%% Dropar colunas que não serão utilizadas na modelagem

df = df.drop(columns=['User ID'])

#%% Informações das variáveis

print(df[['Age','AnnualSalary']].describe())

print(df['Gender'].value_counts())
print(df['Purchased'].value_counts())

#%% Transformando variáveis categóricas em dummies

df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

#%% Separando as variáveis Y e X

X = df.drop(columns=['Purchased']).values
y = df['Purchased'].values

#%% Coletar features

features = list(df.drop(columns=['Purchased']).columns)

#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## O parâmetro test_size indica o percentual das observações na amostra "teste"
## O parâmetro random_state equivale a um "seed": obter resultados iguais

#%% Quantidade de observações em cada amostra

print("Shape do dataset de treino: ", X_train.shape)
print("Shape do dataset de teste: ", X_test.shape)

## Portanto, há 800 observações na amostra treino e 200 na amostra teste

#%% Gerando o modelo na base de dados de treino

## Neste caso, já está parametrizado para 4 níveis (max_depth)

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)

#%% Plotando a árvore (base de treino)

fig = plt.figure(figsize=(60,40))

_ = tree.plot_tree(tree_clf, 
                   feature_names=features,  
                   class_names=['Não Compra','Compra'],
                   filled=True)

#%% Salvando a árvore na pasta do project

fig.savefig("decision_tree_carro_gini.png")

#%% Sensibilidade aos detalhes no conjunto de treinamento

n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

target_names = ['não compra','compra']

for pairidx, pair in enumerate([list(item) for item in list(combinations(np.arange(0,len(features)),2))]):
    # Coletar um par de feature
    X = X_train[:, pair]
    y = y_train

    # Treinamento dos dados
    clf = DecisionTreeClassifier().fit(X, y)

    # Plotar os contornos de decisão
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=features[pair[0]],
        ylabel=features[pair[1]],
    )

    # Protar as observações no gráfico
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

#%% Cálculo do Coeficiente de Gini

# Profundidade 1 - Esquerda
print(1 - (np.square(441/543) + np.square(102/543)))

# Profundidade 1 - Direita
print(1 - (np.square(45/257) + np.square(212/257)))

#%% Probabilidades preditas pelo modelo

## Qual é a probabilidade estimada para uma pessoa:
## Com 45 anos
## Salário de $ 80.000
## Sexo masculino

print(tree_clf.predict_proba([[45, 80000, 1]]))

## 62,80%: Não compra
## 37,20%: Compra

## Escolhendo com base na maior probabilidade:

print(tree_clf.predict([[45, 80000, 1]]))

## Indica que "Não compra"

#%% Fitted values do banco de dados de teste

## O objetivo é verificar os valores preditos para o dataset de teste

y_pred = tree_clf.predict(X_test)

#%% Verificar a importância de cada variável do modelo

importancia_features = pd.DataFrame({'features':features,
                                     'importance':tree_clf.feature_importances_})

print(importancia_features)

#%% Gerando a matriz de confusão

## Compara os valores preditos pela árvore com os valores observados (teste)

cm = confusion_matrix(y_test, 
                      y_pred, 
                      labels=tree_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)

disp.plot()

## Acurácia de (106 + 73)/200 = 0,895 ou 89,50%

#%% Podemos estabelecer um cutoff

## Para cada observação do banco de dados "teste", obtemos os valores preditos
## Para que seja classificado como 1 (evento), deve ser maior que o cutoff

cutoff = 0.90

y_pred = (tree_clf.predict_proba(X_test)[:, 1] > cutoff).astype('float')

## Interpretação: classficar como evento (y=1) probabilidades maiores que 90%

#%% Matriz de confusão para o cutoff

cm = confusion_matrix(y_test, 
                      y_pred, 
                      labels=tree_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)

disp.plot()

## Para um cutoff de 90%, a acurácia do modelo diminui (75%)

#%% Matriz de classificação

## Apresenta parâmetros relativos à classificação dos dados
## Aqui já está considerando o cutoff definido acima

print(classification_report(y_test, y_pred))

## Precision: VP / (VP + FP) = 39 / (39 + 1) "acurácia das predições positivas"
## Recall: VP / (VP + FN) = 39 / (39 + 49) "sensibilidade do modelo"
## F1-Score: VP / (VP + (FN + FP)/2)
## Support: são as contagens observadas nos dados
## Accuracy: (VP + VN) / TOTAL
## Macro avg: é a média aritmética dos indicadores
## Weighted avg: é a média ponderada dos indicadores (ponderada pela prop. support)
    
#%% Verificar a acurácia do modelo de outra maneira

print(accuracy_score(y_test, y_pred))

#%% Plotar a curva ROC nos dados de teste

fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr, tpr)

gini = (roc_auc - 0.5)/(0.5)

#TODO update plot roc method
# plot_roc_curve(tree_clf, X_test, y_test) 
# plt.title('Coeficiente de GINI: %g' % round(gini,4), fontsize=12)
# plt.xlabel('1 - Especificidade', fontsize=12)
# plt.ylabel('Sensitividade', fontsize=12)
# plt.show()

#%% Alterando para entropia

tree_clf_ent = DecisionTreeClassifier(max_depth=4, random_state=42, criterion="entropy")
tree_clf_ent.fit(X_train, y_train)

#%% Plotando a árvore

fig = plt.figure(figsize=(40,40))

_ = tree.plot_tree(tree_clf_ent, 
                   feature_names=features,  
                   class_names=['Não Compra','Compra'],
                   filled=True)

#%% Salvando a árvore na pasta do project

fig.savefig("decision_tree_carro_entrop.png")

#%% Cálculo da entropia

# Profundidade 1 - Esquerda
print(- (441/543)*(np.log(441/543)/np.log(2)) - (102/543)*(np.log(102/543)/np.log(2)))

# Profundidade 1 - Direita
print(- (45/257)*(np.log(45/257)/np.log(2)) - (212/257)*(np.log(212/257)/np.log(2)))

## Em geral, tanto o Gini quanto a entropia geram árvores semelhantes
## O Gini tende a ser mais rápido, o que pode ser útil em grandes datasets

#%% Coletar os predict da árvore pela entropia

y_pred_ent = tree_clf_ent.predict(X_test)

#%% Matriz de classificação

## Apresenta parâmetros relativos à classificação dos dados

print(classification_report(y_test, y_pred_ent))

## Precision: VP / (VP + FP) "acurácia das predições positivas"
## Recall: VP / (VP + FN) "sensibilidade do modelo"
## F1-Score: VP / (VP + (FN + FP)/2) "média harmônica da precision e recall"
## Support: são as contagens observadas nos dados
## Accuracy: (VP + VN) / TOTAL
## Macro avg: é a média aritmética dos indicadores
## Weighted avg: é a média ponderada dos indicadores (ponderada pela prop. support)

#%% Validar o modelo incluindo a perspectiva estocástica dos dados

N = 10
acuracias = list()

for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    tree_clf.fit(X_train, y_train)
    ac_i = tree_clf.score(X_test, y_test)
    acuracias.append(ac_i)

print("- Acurácia:")
print(f"Media: {round(np.mean(acuracias) * 100, 2)}%")
print(f"Desvio padrão: {round(np.std(acuracias) * 100, 2)}%")

#%% Fazendo a mesma validação utilizando o método cross_val_score para o modelo utilizando o criterio gini

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
acuracias = cross_val_score(tree_clf, X, y, cv=cv, scoring='accuracy')

print("acuracias:", acuracias)
print("acuracia final:", np.mean(acuracias), "+-", np.std(acuracias))

#%% Fazendo a mesma validação utilizando o método cross_val_score para o modelo utilizando o criterio entropia

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
acuracias = cross_val_score(tree_clf_ent, X, y, cv=cv, scoring='accuracy')

print("acuracias:", acuracias)
print("acuracia final:", np.mean(acuracias), "+-", np.std(acuracias))

#‘accuracy’
#‘balanced_accuracy’
#‘roc_auc’
#‘f1’
#‘neg_mean_absolute_error’
#‘neg_root_mean_squared_error’
#‘r2’
