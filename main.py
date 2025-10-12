# Etapa 1: Importação das bibliotecas usadas no modelo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# import pygame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# from pygame.locals import *

# Etapa 2: Importando a base
diabetes_df = pd.read_csv("./datasets/diabetes.csv")

# Etapa 3: Visualizar os primeiros 10 registros e estatísticas do Dataset
# print('Etapa 3: Visualizar os primeiros registros e estatísticas do Dataset\n')
# print(f'Primeiros dados: \n{diabetes_df.head()}\n')
# print(f'Estatísticas: \n{diabetes_df.describe()}\n')

# Etapa 4: Verificar se existem valores duplicados
# print('Etapa 4: Verificar se existem valores duplicados\n')
# print(f'Valores duplicados: {diabetes_df.duplicated().sum()}\n')
no_duplicates_df = diabetes_df.drop_duplicates()

# Etapa 5: Verificar se existem valores nulos
# print('Etapa 5: Verificar se existem valores nulos\n')
# print(f'Valores nulos: \n{no_duplicates_df.isnull().sum()}\n')
no_nulls_df = no_duplicates_df.dropna()

# Etapa 6: Padronizar todos os dados como float
# print('Etapa 6: Padronizar todos os dados como float\n')
diabetes_df = diabetes_df.astype(float)
# print(f'Dados como tipo float: \n{diabetes_df.head()}\n')

# Etapa 7: Análise de dados de acordo gravidez e idade
# print('Etapa 7: Análise de dados de acordo gravidez e idade\n')
preg_min = diabetes_df["Pregnancies"].min()
preg_max = diabetes_df["Pregnancies"].max()
# print(f'Menor parteira: \n{diabetes_df[diabetes_df['Pregnancies'] == preg_min].iloc[0]}\n')
# print(f'Maior parteira: \n{diabetes_df[diabetes_df['Pregnancies'] == preg_max].iloc[0]}\n')

age_min = diabetes_df["Age"].min()
age_max = diabetes_df["Age"].max()
# print(f'Menor idade:\n{diabetes_df[diabetes_df['Age'] == age_min].iloc[0]}\n')
# print(f'Maior idade: \n{diabetes_df[diabetes_df['Age'] == age_max].iloc[0]}\n')

# Etapa 8: Analisando correlação dos dados
# print('Etapa 8: Analisando correlação dos dados\n')
correlation_matrix = (
    diabetes_df.select_dtypes(include=["float64", "int"]).corr().round(2)
)
fig, ax = plt.subplots(figsize=(8, 8))
sb.heatmap(data=correlation_matrix, annot=True, linewidths=0.5, ax=ax)
# plt.show()

# Etapa 9: Análise do gráfico de dispersão dos casos de diabetes de acordo com a Idade
# print('Etapa 9: Análise do gráfico de dispersão dos casos de diabetes de acordo com a Idade\n')
# plt.xlabel('Age')
# plt.ylabel('Outcome')
# plt.scatter(diabetes_df['Age'], diabetes_df['Outcome'])
# plt.show()

# Etapa 10: Análise do histograma da distribuição de idade
# print('Etapa 10: Análise do histograma da distribuição de idade\n')
# plt.xlabel('Age')
# plt.hist(diabetes_df['Age'], bins=10, edgecolor='black', alpha=0.7)
# plt.show()

# Etapa 11: Análise Boxplot do Target com as colunas
colunas = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
]

# 11.1: Criar figura e eixos
fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # 3 linhas x 3 colunas
axes = axes.flatten()  # Facilita indexar

# 11.2: Loop para criar cada boxplot
for i, coluna in enumerate(colunas):
    sb.boxplot(x="Outcome", y=coluna, data=diabetes_df, ax=axes[i])
    axes[i].set_xlabel("Outcome")
    axes[i].set_ylabel(coluna)

# 11.3: Se sobrarem subplots vazios, remover
for j in range(len(colunas), len(axes)):
    fig.delaxes(axes[j])

# Etapa 12: Treinamento do Modelo com o Dataframe
# print('Etapa 12: Treinamento do Modelo com o Dataframe\n')
y = diabetes_df["Outcome"]
x = diabetes_df.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Etapa: 13 Análise com KNN
# print('Etapa: 13 Análise com KNN\n')
knn_model = KNeighborsClassifier(n_neighbors=5)
# print(f'Shape da base de treino X: {x_train.shape}')
# print(f'Shape da base teste Y: {y_test.shape}\n')

# 13.1: Treinando o modelo
knn_model.fit(x_train, y_train)

# 13.2: Previsão do primeiro dado do dataset
predict = knn_model.predict(x_test)

# 13.3: Análise da previsão
# print(f'Acurácia KNN: {accuracy_score(y_test, predict)}')
# print(f'Matriz de confusão KNN:\n{confusion_matrix(y_test, predict)}\n')
# print(f'Classification Report: \n{classification_report(y_test, predict)}\n')

# 13.4: Calculando os erros de valores K entre 1 e 10
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# 13.5: Exibição das métricas para avaliação
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# plt.show()

# Etapa 14: Análise com Decision Tree
# print('Etapa 14: Análise com Decision Tree')
decision_tree_model = DecisionTreeClassifier(random_state=42)

# 14.1: Treinando o modelo
decision_tree_model.fit(x_train, y_train)
predict = decision_tree_model.predict(x_test)

# 14.2: Acurácia com o modelo de Decision Tree
# print(f'Acurácia Decision Tree: {accuracy_score(y_test, predict)}\n')
# print(f'Matriz de confusão Decision Tree:\n{confusion_matrix(y_test, predict)}\n')
# print(f'Classification Report Decision Tree:\n{classification_report(y_test, predict)}\n')

# Etapa 15: Análise com Random Forest
# print('Etapa 15: Análise com Random Forest')
random_forest_model = RandomForestClassifier(random_state=42)

# 15.1: Treinando o modelo
random_forest_model.fit(x_train, y_train)
predict_random_forest = random_forest_model.predict(x_test)

# 15.2: Acurácia com o modelo de Random Forest
# print(f'\nAcurácia Random Forest: {accuracy_score(y_test, predict_random_forest)}\n')
# print(f'Matriz de confusão:\n{confusion_matrix(y_test, predict_random_forest)}\n')
# print(f'Classification Report RandomForest:\n{classification_report(y_test, predict_random_forest)}')

# Etapa 16: Identificar colunas com valores inválidos
# print('Etapa 16: Identificar colunas com valores inválidos\n')
invalid_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# 16.1: Colunas com valores inválidos
# print('\nColunas com valores inválidos')
for col in invalid_columns:
    zeros = (diabetes_df[col] == 0).sum()
    total = diabetes_df.shape[0]
    # print(f'{col}: {zeros} valores inválidos ({(zeros / total) * 100:.2f}%)')

# 16.2: Colunas com valores inválidos após substituir pela média
# print('\nColunas com valores inválidos após substituir pela média')
for col in invalid_columns:
    mean_value = diabetes_df[diabetes_df[col] != 0][col].mean()
    diabetes_df[col] = diabetes_df[col].replace(0, mean_value)
    zeros = (diabetes_df[col] == 0).sum()
    total = diabetes_df.shape[0]
    # print(f'{col}: {zeros} valores inválidos ({(zeros / total) * 100:.2f}%)')

# Contagem da Target
# print(f'\nContagem da Target: \n{diabetes_df['Outcome'].value_counts()}\n')

# Etapa 17: Undersample do Dataset
# print('\nEtapa 17: Undersample do Dataset\n')
class_no_diabetes = diabetes_df[diabetes_df["Outcome"] == 0]
class_has_diabetes = diabetes_df[diabetes_df["Outcome"] == 1]

# 17.1: Undersample da classe 0 para ter o mesmo número de registros da classe 1
class_no_diabetes_under = class_no_diabetes.sample(
    len(class_has_diabetes), random_state=42
)

# 17.2: Juntar as duas classes balanceadas
diabetes_df_undersampled = pd.concat([class_no_diabetes_under, class_has_diabetes])

# 17.3: Embaralhar os dados dentro dataset
diabetes_df_undersampled = diabetes_df_undersampled.sample(
    frac=1, random_state=42
).reset_index(drop=True)

# 17.4: Análise dos dados com o tratamento
# print(f'Análise do Dataframe Undersampled: \n{diabetes_df_undersampled.describe()}')
# print(f'Contagem da Target: \n{diabetes_df_undersampled['Outcome'].value_counts()}\n')

# 17.5: Treinamento do Modelo Undersampled
y = diabetes_df_undersampled["Outcome"]
x = diabetes_df_undersampled.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Etapa: 18 Análise com KNN
# print('Etapa: 18 Análise com KNN')
knn_model = KNeighborsClassifier(n_neighbors=5)
# print(f'Shape da base de treino X: {x_train.shape}')
# print(f'Shape da base teste Y: {y_test.shape}\n')

# 18.1: Treinando o modelo
knn_model.fit(x_train, y_train)

# 18.2: Previsão do primeiro dado do dataset
predict = knn_model.predict(x_test)

# 18.3: Análise da previsão
# print(f'Acurácia KNN: {accuracy_score(y_test, predict)}\n')
# print(f'Matriz de confusão KNN:\n{confusion_matrix(y_test, predict)}\n')
# print(f'Relatório de Classificação: \n{classification_report(y_test, predict)}\n')

# 18.4: Calculando os erros de valores K entre 1 e 10
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# 18.5: Exibição das métricas para avaliação
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# plt.show()

# Etapa 19: Análise com Decision Tree
# print('Etapa 19: Análise com Decision Tree')
decision_tree_model = DecisionTreeClassifier(random_state=42)

# 19.1: Treinando o modelo
decision_tree_model.fit(x_train, y_train)
predict = decision_tree_model.predict(x_test)

# 19.2: Acurácia com o modelo de Decision Tree
# print(f'Acurácia Decision Tree: {accuracy_score(y_test, predict)}\n')
# print(f'Matriz de confusão Decision Tree:\n{confusion_matrix(y_test, predict)}\n')
# print(f'Relatório de Classificação:\n{classification_report(y_test, predict)}\n')

# Etapa 20: Análise com Random Forest
# print('Etapa 20: Análise com Random Forest')
random_forest_model = RandomForestClassifier(random_state=42)

# 20.1: Treinando o modelo
random_forest_model.fit(x_train, y_train)
predict_random_forest = random_forest_model.predict(x_test)

# 20.2: Acurácia com o modelo de Random Forest
# print(f'\nAcurácia Random Forest: {accuracy_score(y_test, predict_random_forest)}\n')
# print(f'Matriz de confusão:\n{confusion_matrix(y_test, predict_random_forest)}\n')
# print(f'Relatório de Classificação:\n{classification_report(y_test, predict_random_forest)}')

# Etapa 21: Oversample do Dataset
# print('Etapa 21: Oversample do Dataset\n')

# 21.1: Oversample da classe 1 para ter o mesmo numero de registros da classe 0
class_has_diabetes_over = class_has_diabetes.sample(
    len(class_no_diabetes), random_state=42, replace=True
)

# 21.2: Juntar as duas classes balanceadas
diabetes_df_oversampled = pd.concat(
    [class_no_diabetes, class_has_diabetes_over], axis=0
)

# 21.3: Embaralhar os dados dentro dataset
diabetes_df_oversampled = diabetes_df_oversampled.sample(
    frac=1, random_state=42
).reset_index(drop=True)

# 21.4: Análise dos dados com o tratamento
# print(f'Análise do Dataframe Oversampled: \n{diabetes_df_oversampled.describe()}')
# print(f'Contagem da Target: \n{diabetes_df_oversampled['Outcome'].value_counts()}\n')

# 21.5: Treinamento do Modelo Oversampled
y = diabetes_df_oversampled["Outcome"]
x = diabetes_df_oversampled.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Etapa 22: Análise com KNN
# print('Etapa: 22 Análise com KNN')
knn_model = KNeighborsClassifier(n_neighbors=5)
# print(f'Shape da base de treino X: {x_train.shape}')
# print(f'Shape da base teste Y: {y_test.shape}\n')

# 22.1: Treinando o modelo
knn_model.fit(x_train, y_train)

# 22.2: Previsão do primeiro dado do dataset
predict = knn_model.predict(x_test)

# 22.3: Análise da previsão
# print(f'Acurácia KNN: {accuracy_score(y_test, predict)}\n')
# print(f'Matriz de confusão KNN:\n{confusion_matrix(y_test, predict)}\n')
# print(f'Relatório de Classificação: \n{classification_report(y_test, predict)}\n')

# 22.4: Calculando os erros de valores K entre 1 e 10
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# 22.5: Exibição das métricas para avaliação
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# plt.show()

# Etapa 23: Análise com Decision Tree
# print('Etapa 23: Análise com Decision Tree')
decision_tree_model = DecisionTreeClassifier(random_state=42)

# 23.1: Treinando o modelo
decision_tree_model.fit(x_train, y_train)
predict = decision_tree_model.predict(x_test)

# 23.2: Acurácia com o modelo de Decision Tree
# print(f'Acurácia Decision Tree: {accuracy_score(y_test, predict)}\n')
# print(f'Matriz de confusão Decision Tree:\n{confusion_matrix(y_test, predict)}\n')
# print(f'Relatório de Classificação:\n{classification_report(y_test, predict)}\n')

# Etapa 24: Análise com Random Forest
# print('Etapa 24: Análise com Random Forest')
random_forest_model = RandomForestClassifier(random_state=42)

# 24.1: Treinando o modelo
random_forest_model.fit(x_train, y_train)
predict_random_forest = random_forest_model.predict(x_test)

# 24.2: Acurácia com o modelo de Random Forest
# print(f'\nAcurácia Random Forest: {accuracy_score(y_test, predict_random_forest)}\n')
# print(f'Matriz de confusão:\n{confusion_matrix(y_test, predict_random_forest)}\n')
# print(f'Relatório de Classificação:\n{classification_report(y_test, predict_random_forest)}')

# Etapa 25: Testando validação cruzada
# print('Etapa 25: Testando validação cruzada\n')
scores = cross_val_score(random_forest_model, x, y, cv=10, scoring="accuracy")

# 25.1: Scores da validação cruzada
print("Scores da validação cruzada (10 folds):")
print(scores)

# 25.2: Média da validação
print(f"Média da validação: {scores.mean()}")

# ----------- FASE 2 ---------------
# 1: Imports das libs de algoritmo genético
from hiperparameters_otimization import run_genetic_algorithm

# 2: Rodar algoritmo de otimização dos hiper-parametros
run_genetic_algorithm(
    model=random_forest_model,
    x_train=x_train,
    x_test=x_test,
    y_train=y_train,
    y_test=y_test,
    x_initial=x,
    y_initial=y,
    cv=10,
    scoring="accuracy"
)
