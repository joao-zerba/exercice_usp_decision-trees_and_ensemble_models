# -*- coding: utf-8 -*-

# Referencia:

# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

# Exercício: Variável Y categórica com 3 ou mais categorias

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#%% Carregando o dataset

df = pd.read_excel("segmenta_telecom.xlsx")

# Fonte: adaptado de https://www.kaggle.com/datasets/prathamtripathi/customersegmentation?select=Telecust1.csv

#%% Informações das variáveis

print(df[['meses_cliente', 'idade', 'anos_emprego']].describe())

print(df['regiao'].value_counts())
print(df['casado'].value_counts())
print(df['aposentado'].value_counts())
print(df['masculino'].value_counts())
print(df['categoria'].value_counts())

#%% Transformando variáveis categóricas em dummies

df = pd.get_dummies(df, columns=['regiao'])

#%% Separando as variáveis Y e X

X = df.drop(columns=['categoria']).values
y = df['categoria'].values

#%% Coletar os nomes das variáveis X

features = list(df.drop(columns=['categoria']).columns)

#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Bagging
#%% 1: Para fins de comparação, estima-se uma árvore de classificação

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)

# Predict do modelo de uma árvore
y_pred_tree = tree_clf.predict(X_test)

# Matriz de classificação para uma árvore
print(classification_report(y_test, y_pred_tree))

#%% 2: Estimando um modelo bagging com base em árvores de classificação

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=4), # modelo base: árvore de classificação
    n_estimators=500,
    max_samples=50,
    bootstrap=True, # bootstrap = True indica modelo Bagging / False = Pasting
    n_jobs=-1, # utiliza todos os núcleos do computador
    random_state=42) 

bag_clf.fit(X_train, y_train)

# Predict do modelo bagging de árvores
y_pred_bag = bag_clf.predict(X_test)

# Gerando a matriz de confusão

cm = confusion_matrix(y_test, 
                      y_pred_bag, 
                      labels=bag_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=bag_clf.classes_)

# Matriz de classificação do modelo bagging de árvores
print(classification_report(y_test, y_pred_bag))

#%% 3: Para fins de comparação, estima-se um modelo logístico

reg_log = LogisticRegression()
reg_log.fit(np.delete(X_train, -1, axis=1), y_train)

# Predict do modelo logístico
y_pred_reg_log = reg_log.predict(np.delete(X_test, -1, axis=1))

# Matriz de classificação do modelo logístico
print(classification_report(y_test, y_pred_reg_log))

#%% 4: Estimando um modelo bagging com base em uma logística

bag_log = BaggingClassifier(
    LogisticRegression(), # modelo base: logística
    n_estimators=500,
    max_samples=50,
    bootstrap=True, # bootstrap = True indica modelo Bagging / False = Pasting
    n_jobs=-1, # utiliza todos os núcleos do computador
    random_state=42) 

bag_log.fit(np.delete(X_train, -1, axis=1), y_train)

# Predict do modelo bagging de logística
y_pred_log = bag_log.predict(np.delete(X_test, -1, axis=1))

# Matriz de classificação do modelo bagging de logística
print(classification_report(y_test, y_pred_log))

#%% Avaliação out-of-bag

# As observações de treinamento que não são amostradas são "out-of-bag"
# O modelo pode ser avaliado nessas observações sem a necessidade de um conjunto de validação
# Trata-se de uma avaliação automática após o treinamento

bag_clf_oob = BaggingClassifier(
    DecisionTreeClassifier(max_depth=4), 
    n_estimators=500,
    max_samples=50,
    bootstrap=True,
    n_jobs=-1, 
    oob_score=True, # avaliação out-of-bag
    random_state=42) 

bag_clf_oob.fit(X, y)

# Acurácia do modelo
print(bag_clf_oob.oob_score_)

#%% Random Forests

# O ForestClassifier é mais otimizado para árvores de decisão

rnd_clf = RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

# Predict na base de teste
y_pred_rf = rnd_clf.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_rf))

# Importância das variáveis X
for name, score in zip(features, rnd_clf.feature_importances_):
    print(name, score)

#%% Classificadores de Votação

tree_clf_vot = DecisionTreeClassifier(max_depth=4, random_state=42)
rnd_clf_vot = RandomForestClassifier(max_depth=4, random_state=42)

#%% Parametrizando o classificador

voting_clf = VotingClassifier(
    estimators=[('tree', tree_clf_vot),
                ('rf', rnd_clf_vot)],
    voting='hard')

voting_clf.fit(X_train, y_train)

# Fazendo o predict e identificando a acurácia dos classificadores
for clf in (tree_clf_vot, rnd_clf_vot, voting_clf):
    clf.fit(X_train, y_train)
    y_pred_vot = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred_vot))

#%% Boosting
#%% AdaBoost

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4),
    n_estimators=100,
    algorithm='SAMME.R',
    learning_rate=0.1)

ada_clf.fit(X_train,y_train)

# Predict na base de teste
y_pred_ada = ada_clf.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_ada))

#%% Grid search para escolha da profundidade 

gs_ada = GridSearchCV(estimator=DecisionTreeClassifier(),
                          param_grid={
                              'max_depth': [2, 3, 4, 5, 6, 7 , 8, 9, 10, None],
                          },
                          cv=5,
                          return_train_score=False,
                          scoring='accuracy')

gs_ada.fit(X=X_train, y=y_train)

resultados_gs_ada = pd.DataFrame(gs_ada.cv_results_).set_index('rank_test_score').sort_index()

print(resultados_gs_ada)

#%% Lista com cada iteração

estimators = np.arange(1,101)

#%% Lista que vai receber cada resultado das iterações

scores_train = np.zeros(100, dtype=np.float64)

scores_test = np.zeros(100, dtype=np.float64)

#%% Coletando a acurácia de cada iteração nos dados de treino

from sklearn.metrics import accuracy_score

for i, y_pred in enumerate(ada_clf.staged_predict(X_train)):
    
    acc = accuracy_score(y_train, y_pred)
    
    scores_train[i] = acc
    
print(scores_train)
    
#%% Coletando a acurácia de cada iteração nos dados de teste

for i, y_pred in enumerate(ada_clf.staged_predict(X_test)):
    
    acc = accuracy_score(y_test, y_pred)
    
    scores_test[i] = acc
    
print(scores_train)
    
#%% Visualizando a acurácia ao longo de cada iteração

plt.figure(figsize=(12, 10))
plt.title("Acc por iteração")
plt.plot(estimators,scores_train, label='Dados de treino')
plt.plot(estimators,scores_test, label='Dados de teste')
plt.legend(loc="upper right")
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Acurácia", fontsize=16)
plt.show()

#%% Parametrizando o AdaBoost com base nas análise anteriores

ada_clf_best = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=3,
    algorithm='SAMME.R',
    learning_rate=0.1)

ada_clf_best.fit(X_train,y_train)

# Predict na base de teste
y_pred_ada_best = ada_clf_best.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_ada))

#%% Gradiente Boosting

gbc = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)
gbc.fit(X_train, y_train)

y_pred_gbc = gbc.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_gbc))

#%% XGBoost

labels = np.unique(y)

for index, item in enumerate(labels):
   y_train[y_train == item] = index 
   
for index, item in enumerate(labels):
   y_test[y_test == item] = index 

xgb_clf = xgboost.XGBClassifier(max_depth=3, n_estimators=100, random_state=42)

xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)

# Matriz de classificação
print(classification_report(y_test.astype("int"), y_pred_xgb))
