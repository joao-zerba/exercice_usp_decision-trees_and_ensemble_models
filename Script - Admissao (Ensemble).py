# -*- coding: utf-8 -*-

# Referencia:

# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

#%% Importando os pacotes necessários

# pip install -r requirements.txt

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error

#%% Carregando o dataset

df = pd.read_csv("admissao.csv").set_index('id')

# Fonte: https://www.kaggle.com/datasets/mohansacharya/graduate-admissions/download?datasetVersionNumber=2

# GRE Scores (até 340)
# TOEFL Scores (até 120)
# University Rating (até 5)
# Statement of Purpose and Letter of Recommendation Strength (até 5)
# Undergraduate GPA (até 10)
# Research Experience (0 ou 1)
# Chance of Admit (vai de 0 a 1)

# Referência do dataset:
# Mohan S Acharya, Asfia Armaan, Aneeta S Antony: 
# A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019

#%% Informações das variáveis

print(df.info())

#%% Análise univariada

print(df[['gre_score', 'toefl_score', 'uvinersity_rating', 'sop', 'lor', 'cgpa', 'chance_of_admit']].describe())

print(df['research'].value_counts())

#%% Verificando missing values

print(df.isnull().sum())

#%% Análise da correlação entre as variáveis

plt.figure(figsize=(18,10))

sns.heatmap(df[['gre_score', 'toefl_score', 'uvinersity_rating', 'sop', 'lor', 'cgpa']].corr(), annot=True)

# 1. À medida que a pontuação do GRE aumenta, a pontuação do TOEFL também aumenta e até a pontuação do CGPA aumenta. 
# Isso significa que os alunos com boas pontuações no GRE também obtêm boas pontuações no TOEFL e têm melhores CGPAs.
# 2. A pontuação GRE tem uma distribuição com uma média próxima a 320 e a maioria dos alunos pontua entre 310 e 330, o que é indicado pelos picos no gráfico.
# 3. Alunos com classificação universitária mais alta têm pontuações GRE, TOEFL e CGPA mais altas.
# 4. Os alunos têm melhor classificação LOR e SOP quando são de universidades com classificação mais alta.
# 5. Podemos ver uma tendência semelhante para a relação entre GRE Scores, TOEFL Scores, CGPA Scores, LOR Ratings e SOP Ratings. 
# Todos eles aumentam juntos, ou seja, como alunos com melhores SOPs também tendem a ter melhores LORs, melhores pontuações GRE, TOEFL e CGPA e são de
# universidades com classificação mais alta e têm maior chance de admissão.

#%% Visualizar o pairplot

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

graph = sns.pairplot(df.drop(columns='research'), diag_kind="kde")
graph.map(corrfunc)
plt.show()

#%% Separando as variáveis Y e X

X = df.drop(columns=["chance_of_admit"])
y = df["chance_of_admit"]

#%% Coletar os nomes das variáveis X

features = list(df.drop(columns=['chance_of_admit']).columns)

#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Verificando a amplitude das variáveis para uma possível padronização

dfm = X_train.melt(var_name='columns')
g = sns.FacetGrid(dfm, col='columns')
g = (g.map(sns.distplot, 'value'))

#%% Padronizando variáveis com o método z-score

scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test = pd.DataFrame(X_test_norm, columns=X_test.columns)

#%% Verificando a amplitude das variáveis para após a padronização

dfm = X_train.melt(var_name='columns')
g = sns.FacetGrid(dfm, col='columns')
g = (g.map(sns.distplot, 'value'))

# O modelo consegue fazer melhores previsões
#%% Bagging
#%% 1: Para fins de comparação, estima-se uma árvore de regressão através do grid search (para obtenção do melhor fit)

gs_comp = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                       param_grid={
                           'max_depth': [2, 3, 4, 5, 6, 7 , 8, 9, 10, None],
                       },
                       cv=5,
                       return_train_score=False,
                       scoring='r2')

gs_comp.fit(X=X_train, y=y_train)

resultados_tree_reg = pd.DataFrame(gs_comp.cv_results_).set_index('rank_test_score').sort_index()

print(resultados_tree_reg)

#%% 2: Estimando um modelo bagging como base para regressão

bag_reg = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=4),
        #n_estimators=10, # valor default
        #max_samples=1.0, # valor default
        #bootstrap=True, # valor default # bootstrap = True indica modelo Bagging / False = Pasting 
        #n_jobs=-1, # valor default=None  # utiliza todos os núcleos do computador
        random_state=42) 

bag_reg.fit(X_train, y_train)

# Predict do modelo bagging da regressão
y_pred_reg = bag_reg.predict(X_test)

n_scores = cross_val_score(bag_reg, X, y, scoring='r2', cv=5, n_jobs=-1, error_score='raise')

# score médio do modelo
print('R2: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Ajustando um modelo bagging através do grid search e comparativos de vários estimadores

estimator = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                             random_state=42)

gs_bag = GridSearchCV(estimator=estimator,
                        param_grid={
                            'n_estimators': [10, 50, 100, 200],
                            'base_estimator__max_depth':[3,4]
                        },
                        cv=5,
                        return_train_score=False,
                        scoring='r2')

gs_bag.fit(X=X_train, y=y_train)

resultados_bag_reg = pd.DataFrame(gs_bag.cv_results_).set_index('rank_test_score').sort_index()

#%% Avaliação out-of-bag

# As observações de treinamento que não são amostradas são "out-of-bag"
# O modelo pode ser avaliado nessas observações sem a necessidade de um conjunto de validação
# Trata-se de uma avaliação automática após o treinamento

bag_clf_oob = BaggingRegressor(
    DecisionTreeRegressor(max_depth=4), 
    n_estimators=100,
    bootstrap=True,
    n_jobs=-1, 
    oob_score=True, # avaliação out-of-bag
    random_state=42) 

bag_clf_oob.fit(X, y)

# Score do modelo
print(bag_clf_oob.oob_score_)

#%% Random Forests

# O ForestRegressor é mais otimizado para árvores de decisão

rnd_reg = RandomForestRegressor(
    n_estimators=100, 
    max_depth=4, 
    n_jobs=-1, 
    random_state=42)

rnd_reg.fit(X_train, y_train)

# Predict na base de teste
y_pred_rf = rnd_reg.predict(X_test)

n_scores = cross_val_score(rnd_reg, X, y, scoring='r2', cv=10, n_jobs=-1, error_score='raise')

# performance final do modelo
print('R2: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Grid Search com random forest

gs_rand = GridSearchCV(estimator=RandomForestRegressor(),
                          param_grid={
                              'n_estimators': [10, 50, 100, 200],
                              'max_depth':[3,4]
                          },
                          cv=5,
                          return_train_score=False,
                          scoring='r2')

gs_rand.fit(X=X_train, y=y_train)
resultados_rnd_reg = pd.DataFrame(gs_rand.cv_results_).set_index('rank_test_score').sort_index()

#%% Importância das variáveis X

importancia = pd.DataFrame({'features':features,
                            'importancia':rnd_reg.feature_importances_})

#%% coletando o melhor modelo

# melhores parâmetros
print(gs_rand.best_params_)      

#%% Coletando os fitted valores do melhor modelo

y_pred = gs_rand.predict(X=scaler.transform(df.drop(columns='chance_of_admit')))          

#%% Adicionando os predict no dataframe original

df['y_pred'] = y_pred

#%% Visualizando dispersão fitted vs observado

# Gráfico didático

plt.figure(figsize=(12, 12))

y = df['chance_of_admit']
yhat = df['y_pred']

plt.plot(y, y, color='gray')
plt.scatter(y , yhat, alpha=0.5, color='#9b59b6')
plt.ylabel("fitted", fontsize=16)
plt.xlabel("observado", fontsize=16)
plt.legend(['45º graus','RandomForest'])
plt.show()

#%% Vamos comparar agora com o nosso primeiro modelo base

# Coletando os fitted valores do primeiro modelo

y_pred_comp = gs_comp.predict(X=scaler.transform(df.drop(columns=['chance_of_admit','y_pred'])))    

df['y_pred_comp'] = y_pred_comp

#%% Visualizando disperssão fitted vs observado

# Gráfico didático

plt.figure(figsize=(12, 12))

y = df['chance_of_admit']
yhat = df['y_pred']
y_hat_comp = df['y_pred_comp']

plt.plot(y,y, color='gray')
plt.scatter(y , y_hat_comp, alpha=0.5, color='#f1c40f')
plt.scatter(y , yhat, alpha=0.5, color='#9b59b6')

plt.ylabel("fitted", fontsize=16)
plt.xlabel("observado", fontsize=16)
plt.legend(['45º graus','DecisionTreeRegressor','RandomForest'])
plt.show()

# R² da RandomForest 0.75
# R² da DecisionTreeRegression 0.69

#%% Coletando e verificando os resíduos

# resíduos da random forest
df['resid_rnd'] = df['chance_of_admit'] - df['y_pred']

# resíduos do comparativo
df['resid_reg'] = df['chance_of_admit'] - df['y_pred_comp']

#%% Visualizando a distribuição dos resíduos no modelo comparativo

plt.figure(figsize=(12, 12))

sns.histplot(data=df['resid_reg'], kde=True, bins=30)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.legend(['DecisionTreeRegressor'])
plt.plot()

#%% Visualizando a distribuição dos resíduos no modelo do melhor r²

plt.figure(figsize=(12, 12))

sns.histplot(data=df['resid_rnd'], kde=True, bins=30, label='Random Forest')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()

#%% Visualizando os dois no mesmo gráfico para comparativo

import numpy as np

plt.figure(figsize=(12, 12))

sns.histplot(df['resid_rnd'], color="skyblue", label='Random Forest', bins=30, kde=True)
sns.histplot(df['resid_reg'], color="orange", label='Decision Tree Reg', bins=30, kde=True)
plt.legend(loc='upper right')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()

#%% Também pode ser gerado um pipeline com todas as instruções para gerar o modelo final

pipe = make_pipeline(scaler, gs_rand)

# primeiro: será feita a padronização dos dados
# segundo: será feito o grid de procura dos melhores parâmetros para o modelo

#%% Realizando o predict utilizando o pipeline criado

predict = pd.DataFrame({'gre_score':[337,300], 
                        'toefl_score':[118,100], 
                        'uvinersity_rating':[5,4], 
                        'sop':[4.5,4.6], 
                        'lor':[4.5,4.9], 
                        'cgpa':[9.65,8.5],
                        'research':[1,0]})

print(f"Resultado: {pipe.predict(X=predict)}")

#%% Regressores de Votação

# Um regressor de votação é um meta-estimador de conjunto que se ajusta a vários regressores de base, 
# cada um em todo o conjunto de dados. Em seguida, calcula a média das previsões individuais para formar uma previsão final.

lin_reg_vot = LinearRegression()
tree_reg_vot = DecisionTreeRegressor(max_depth=4, random_state=42)
rnd_reg_vot = RandomForestRegressor(max_depth=4, n_estimators=100, random_state=42)

#%% Parametrizando o modelo

voting_ref = VotingRegressor(
    estimators=[('lr', lin_reg_vot), 
                ('tree', tree_reg_vot),
                ('rf', rnd_reg_vot)])

voting_ref.fit(X_train, y_train)

# Fazendo o predict e identificando o r² de cada modelo:
for clf in (lin_reg_vot, tree_reg_vot, rnd_reg_vot, voting_ref):
    clf.fit(X_train, y_train)
    y_pred_vot = clf.predict(X_test)
    
    n_scores = cross_val_score(clf, X, y, scoring='r2', cv=5, n_jobs=-1, error_score='raise')
    
    # report performance
    print(f'{clf.__class__.__name__} R2: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Boosting
### AdaBoost
    
#%% Função de perda para o MSE

# valores para comparação

y_ref = np.ones((100,1))
y_hat = np.linspace(0, 2, 100)

#%% Verificando os shapes

print("y: ", y_ref.shape)
print("y_hat: ", y_hat.shape)

#%% Alterando o shape do array

y_hat = y_hat.reshape((-1, 1))

print("y_hat: ", y_hat.shape)

#%% Valor observado para referência

print(y_ref)

#%% Valores preditos indo de 0 a 2 (o valor 1 estará compreendido no array, ou muito proximo de 1)

print(y_hat)

#%% Criando vetor do erro ao quadrado

mse_array = (y_ref - y_hat)**2

print(mse_array)

#%% Verificando a MSE (Mean Squared Error)

print(np.mean(mse_array))

#%% Obtendo o mesmo resultado utilizando a biblioteca so scikit-learn

print(mean_squared_error(y_ref, y_hat))

#%% Visualizando o gráfico da função de perda para entender a taxa de aprendizado

plt.figure(figsize=(16, 12))

plt.plot(y_hat, mse_array)
plt.ylabel("Função de perda", fontsize=16)
plt.xlabel("W", fontsize=16)
plt.show()

## vamos para a lousa!

#%%

ada_reg = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4, 
                          #random_state=42
                          ),
    n_estimators=200,
    learning_rate=0.1,
    #random_state=42
    )

ada_reg.fit(X_train, y_train)

# Predict na base de teste
y_pred_ada = ada_reg.predict(X_test)

n_scores = cross_val_score(ada_reg, X, y, scoring='r2', cv=5, n_jobs=-1, error_score='raise')

print('R2: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Lista com cada iteração

estimators = np.arange(1,201)

#%% Lista que vai receber cada resultado das iterações

scores_train = np.zeros(200, dtype=np.float64)

scores_test = np.zeros(200, dtype=np.float64)

#%% Coletando o MSE de cada iteração nos dados de treino

for i, y_pred in enumerate(ada_reg.staged_predict(X_train)):
    
    mse = mean_squared_error(y_train, y_pred)
    
    scores_train[i] = mse
    
print(scores_train)
    
#%% Coletando o MSE de cada iteração nos dados de teste

for i, y_pred in enumerate(ada_reg.staged_predict(X_test)):
    
    mse = mean_squared_error(y_test, y_pred)
    
    scores_test[i] = mse
    
print(scores_train)
    
#%% Visualizando o MSE ao longo de cada iteração

plt.figure(figsize=(12, 10))
plt.title("MSE por iteração")
plt.plot(estimators,scores_train, label='Dados de treino')
plt.plot(estimators,scores_test, label='Dados de teste')
plt.legend(loc="upper right")
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("MSE", fontsize=16)
plt.show()

#%% Coletando o r² de cada iteração nas bases de treino e teste

from sklearn.metrics import r2_score

# r² nos dados de treino
for i, y_pred in enumerate(ada_reg.staged_predict(X_train)):
    
    r2 = r2_score(y_train, y_pred)
    
    scores_train[i] = r2

# r² nos dados de teste
for i, y_pred in enumerate(ada_reg.staged_predict(X_test)):
    
    r2 = r2_score(y_test, y_pred)
    
    scores_test[i] = r2
    
#%% Visualizando o R2 ao longo de cada iteração

plt.figure(figsize=(12, 10))
plt.title("R2 por iteração")
plt.plot(estimators,scores_train, label='Dados de treino')
plt.plot(estimators,scores_test, label='Dados de teste')
plt.legend(loc="upper right")
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("R2", fontsize=16)
plt.show()

#%% Coletando o predict do adaptative boosting no dataframe original

y_pred_ada_reg = ada_reg.predict(scaler.transform(X))

df['y_pred_ada_reg'] = y_pred_ada_reg

#%% Gradient Boosting

# Como o algorítmo funciona

# faz uma árvore de regressão
# faz o fit do modelo
tree_reg1 = DecisionTreeRegressor(max_depth=4)
tree_reg1.fit(X_train, y_train)

# coleta o erro do modelo em comparativo com o dado observado
# armazena o novo y como o erro do modelo

y2 = y_train - tree_reg1.predict(X_train)

# erro coletado
print(y2)

#%% 2ª estimação

# faz o fit do modelo utilizando o erro do modelo anterior junto como o target
tree_reg2 = DecisionTreeRegressor(max_depth=4)
tree_reg2.fit(X_train, y2)

# novamente coleta o novo erro para ser utilizado na próxima estimação
y3 = y2 - tree_reg2.predict(X_train)

print(y3)

#%% 3º estimação

# faz o fit do modelo utilizando o erro do modelo anterior junto como o target
tree_reg3 = DecisionTreeRegressor(max_depth=4)
tree_reg3.fit(X_train, y3)

# novamente coleta o novo erro para ser utilizado na próxima estimação
y4 = y3 - tree_reg3.predict(X_train)

print(y4)

#%% Adquire o valor predito com a soma os predicts de cada modelo

y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))

#%% Verificando o r² do modelo teste

from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))

# 0.80 mais elevado que os outros modeleos quee não são ensemble

#%% Gradiente Boosting

gbc = GradientBoostingRegressor(max_depth=4, 
                                #n_estimators=100, # valor default
                                learning_rate=0.1, 
                                random_state=42)
gbc.fit(X_train, y_train)

n_scores = cross_val_score(gbc, X, y, scoring='r2', cv=10, n_jobs=-1, error_score='raise')

print('R2: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Coletando o predict do gradient boosting no dataframe original

y_pred_gbc = gbc.predict(scaler.transform(X))

df['y_pred_gbc'] = y_pred_gbc

#%% Visualizando dispersão fitted vs observado dos 3 modelos

# Gráfico didático

plt.figure(figsize=(12, 12))

y = df['chance_of_admit']

y_hat_gbc = df['y_pred_gbc']
y_pred_ada_reg = df['y_pred_ada_reg']

plt.plot(y,y, color='gray', label='45')
plt.scatter(y , y_pred_ada_reg, alpha=0.5, color='#3498db', label=f'AdaBoostingRegressor - r² {round(r2_score(y_pred_ada_reg, y),3)}')
plt.scatter(y , y_hat_gbc, alpha=0.5, color='#e74c3c', label=f'GradientBoostingRegressor - r² {round(r2_score(y_hat_gbc, y),3)}')

plt.ylabel("fitted", fontsize=16)
plt.xlabel("observado", fontsize=16)
plt.legend(loc="lower right")
plt.show()

#%% XGBoost

xgb_clf = xgboost.XGBRegressor(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1)
xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)

n_scores = cross_val_score(gbc, X, y, scoring='r2', cv=10, n_jobs=-1, error_score='raise')

print('R2: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Coletando o predict do gradient boosting no dataframe original

y_pred_XGBoost = xgb_clf.predict(scaler.transform(X))

df['y_pred_XGBoost'] = y_pred_XGBoost

#%% Visualizando disperssão fitted vs observado dos 3 modelos

# Gráfico didático

plt.figure(figsize=(12, 12))

y = df['chance_of_admit']
y_hat_comp = df['y_pred_comp']
y_hat_gbc = df['y_pred_gbc']
y_hat_XGBoost = df['y_pred_XGBoost']

plt.plot(y,y, color='gray', label='45ª')
plt.scatter(y , yhat, alpha=0.5, color='#9b59b6', label=f'RandomForest - r² {round(r2_score(yhat, y),3)}')
plt.scatter(y , y_hat_gbc, alpha=0.5, color='#e74c3c', label=f'Scikit-GradientBoost - r² {round(r2_score(y_hat_gbc, y),3)}')
plt.scatter(y , y_pred_XGBoost, alpha=0.5, color='#f1c40f', label=f'XGBoost - r² {round(r2_score(y_pred_XGBoost, y),3)}')

plt.ylabel("fitted", fontsize=16)
plt.xlabel("observado", fontsize=16)
plt.legend(loc="lower right")
plt.show()

# R² da RandomForest 0.75
# R² da DecisionTreeRegression 0.66

#%% Coletando os resíduos do XGBoost

df['resid_gbc'] =  df['chance_of_admit'] - df['y_pred_gbc']

#%% Visualizando a distribuição dos resíduos

plt.figure(figsize=(12, 12))

sns.histplot(df['resid_rnd'], color="skyblue", label='resid_rnd', bins=30, kde=True)
sns.histplot(df['resid_reg'], color="orange", label='resid_reg', bins=30, kde=True)
sns.histplot(df['resid_gbc'], color = "purple", label='resid_gbc', bins=30, kde=True)
plt.legend(loc='upper right')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()

#%% Fazendo o deploy do modelo de forma local

import dill

# salvar o modelo de utilizando a bibliteca dill
with open('model_pipe.pkl', 'wb') as f:
    dill.dump(pipe, f)

#%% Utilizar o modelo salvo na máquina
    
with open('model_pipe.pkl', 'rb') as f:
    modelo = dill.load(f)
    
#%% Fazendo o predict do modelo salvo
    
predict = pd.DataFrame({'gre_score':[337], 
                        'toefl_score':[118], 
                        'uvinersity_rating':[5], 
                        'sop':[4.5], 
                        'lor':[4.5], 
                        'cgpa':[9.65],
                        'research':[1]})
        
print(modelo.predict(predict))
