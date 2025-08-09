import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# Importando a base
diabetes_df = pd.read_csv("./datasets/diabetes.csv")

# Análise de dados do Dataset
print(f'Primeiros dados: \n{diabetes_df.head()}\n')
# print(f'Informações dataset: \n{diabetes_df.info()}')
# print(f'Shape dataset: {diabetes_df.shape}')
# print(f'Describe dataset: \n{diabetes_df.describe()}')
# no_duplicates_df = diabetes_df.drop_duplicates()
# print(f'No duplicates: \n{no_duplicates_df}')

# Identificar colunas com zeros inválidos
invalid_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in invalid_columns:
    mean_value = diabetes_df[diabetes_df[column] != 0][column].mean()
    diabetes_df[column] = diabetes_df[column].replace(0, mean_value)

for col in invalid_columns:
  zeros = (diabetes_df[col] == 0).sum()
  total = diabetes_df.shape[0]
  # print(f"{col}: {zeros} zeros ({(zeros / total) * 100:.2f}%)")

# Análise de dados das pessoas com mais e menos gravidez
# print(f'Maior parteira: \n{diabetes_df[diabetes_df['Pregnancies'] == 17]}')
# print(f'Menor parteira: \n{diabetes_df[diabetes_df['Pregnancies'] == 0]}')

# Criação de uma feature com base em Idade + Gestação
# diabetes_df['Age_Pregnancy'] = np.sqrt(diabetes_df['Age'] * diabetes_df['Pregnancies'])
# print(f'Dataset com a nova feature: \n{diabetes_df}')

# Criação de uma feature com base em Insulina + Espessura da Pele
# diabetes_df['Insulin_SkinThickness'] = diabetes_df['Insulin'] / diabetes_df['SkinThickness']
# print(f'Dataset com a nova feature: \n{diabetes_df}')

# Criação de uma feature com base em IMC + Espessura da Pele
# diabetes_df['BMI_SkinThickness'] = diabetes_df['BMI'] / diabetes_df['SkinThickness']
# print(f'Dataset com a nova feature: \n{diabetes_df}')

# Analisando correlação dos dados
# correlation_matrix = diabetes_df.select_dtypes(include=['float64', 'int']).corr().round(2)
# fig, ax = plt.subplots(figsize=(8,8))    
# sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
# plt.show();

# Observações sobre a corelação
# • As colunas não possuem correlações fortes entre si e nem com a target
# • Para a primeira analise vamos executar o treinamento do modelo e observar as metricas de validação

# Analisando a concentração de casos de diabetes de acordo com a Idade
# Scatter
# plt.scatter(diabetes_df['Age'], diabetes_df['Outcome'])
# plt.xlabel('Age')
# plt.ylabel('Outcome')
# plt.show()

# Hist
# plt.hist(diabetes_df['Age'], bins=10, edgecolor='black', alpha=0.7)
# plt.xlabel('Age')
# plt.show()

# Boxplot
# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='Pregnancies', data=diabetes_df)
# plt.title('Pregnancies')
# plt.xlabel('Outcome')
# plt.ylabel('Pregnancies')

# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='Glucose', data=diabetes_df)
# plt.title('Glucose')
# plt.xlabel('Outcome')
# plt.ylabel('Glucose')

# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='BloodPressure', data=diabetes_df)
# plt.title('BloodPressure')
# plt.xlabel('Outcome')
# plt.ylabel('BloodPressure')

# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='SkinThickness', data=diabetes_df)
# plt.title('SkinThickness')
# plt.xlabel('Outcome')
# plt.ylabel('SkinThickness')

# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='Insulin', data=diabetes_df)
# plt.title('Insulin')
# plt.xlabel('Outcome')
# plt.ylabel('Insulin')

# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='BMI', data=diabetes_df)
# plt.title('BMI')
# plt.xlabel('Outcome')
# plt.ylabel('BMI')

# plt.figure(figsize=(10, 6))
# sb.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=diabetes_df)
# plt.title('DiabetesPedigreeFunction')
# plt.xlabel('Outcome')
# plt.ylabel('DiabetesPedigreeFunction')

# plt.show()
# A principio achamos que quanto maior a idade maior era a chance de diabetes, porém com a ajuda da matriz de correlação e do gráfico boxplot identificamos que
# não a idade não é um fator determinante para identificarmos pacientes com casos de diabetes

# df_padronizado = diabetes_df.astype(float)

# Undersample do Dataset
class_no_diabetes = diabetes_df[diabetes_df['Outcome'] == 0]
class_has_diabetes = diabetes_df[diabetes_df['Outcome'] == 1]


# Undersample da classe 0 para ter o mesmo número de registros da classe de casos com diabetes
class_no_diabetes_under = class_no_diabetes.sample(len(class_has_diabetes), random_state=42)

#Oversample da classe 1 para ter o mesmo numero de registros da classe de casos com não diabetes
class_has_diabetes_over = class_has_diabetes.sample(len(class_no_diabetes), random_state=42, replace=True)
df_test_over = pd.concat([class_no_diabetes, class_has_diabetes_over], axis=0)
# print(f'---info da base oversampleada: \n{df_test_over['Outcome'].value_counts()}')

# Juntar as duas classes balanceadas
diabetes_df_undersampled = pd.concat([class_no_diabetes_under, class_has_diabetes])

# Embaralhar o dataset
diabetes_df_undersampled = diabetes_df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

# print(f'Visualizando df Undersample: \n{diabetes_df_undersampled.describe()}')
# print(f'Count Outcome: \n{diabetes_df_undersampled['Outcome'].value_counts()}')

# Treinamento do Modelo
y = df_test_over['Outcome'];
x = df_test_over.drop('Outcome', axis=1)
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(x_train)
X_test_standardized = scaler.transform(x_test)

#Logistic Regression
logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_standardized, y_train)
predict_logistic = logistic_regression.predict(X_test_standardized)
print('Acurácia Logistic Regression: ', accuracy_score(y_test, predict_logistic))
print('Matriz de confusão Logistic Regression: ', confusion_matrix(y_test, predict_logistic))
print(f'Classification Report Logistic Regression:\n{classification_report(y_test, predict_logistic)}')

# KNN
# knn_model = KNeighborsClassifier(n_neighbors=5);
# print(f'Shape treino X: \n{X_test_standardized.shape}')
# print(f'Shape test X: \n{y_test.shape}')
# knn_model.fit(X_train_standardized, y_train)
# x_test_first_person = x_test.iloc[[70]]
# y_test_first_person = y_test.iloc[[70]]
# predict = knn_model.predict(X_test_standardized)
# predict_one_value = knn_model.predict(x_test_first_person)
# print('Valor real: ', y_test_first_person)
# print('Valor predito: ', predict_one_value)
# error = []
# print('Acurácia KNN: ', accuracy_score(y_test, predict))
# print('Matriz de confusão KNN: ', confusion_matrix(y_test, predict))
# print(f'Classification Report KNN:\n{classification_report(y_test, predict)}')

# Calculando os erros de valores K entre 1 e 10
# for i in range(1, 10): #range de tentativas para k
#     knn = KNeighborsClassifier(n_neighbors=i) # aqui definimos  o k
#     knn.fit(X_train_standardized, y_train) # treinando o algoritmo para encontrar o erro
#     pred_i = knn.predict(X_test_standardized) # armazenando as previsões
#     error.append(np.mean(pred_i != y_test)) # armazenando o valor do erro médio na lista de erros

# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# plt.show()

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Treinando o modelo
decision_tree_model.fit(X_train_standardized, y_train)
predict = decision_tree_model.predict(X_test_standardized)

# Acurácia com o modelo de Decision Tree: 0.72
# print('Acurácia Decision Tree: ', accuracy_score(y_test, predict))
# print('Matriz de confusão Decision Tree: ', confusion_matrix(y_test, predict))
# print(f'Classification Report Decision Tree:\n{classification_report(y_test, predict)}')

# Observação: O modelo tem dificuldades em classificar casos de diabetes como verdadeiros.

# Random Forest Classifier
random_forest_model = RandomForestClassifier(random_state=42)

# Treinando o modelo
random_forest_model.fit(X_train_standardized, y_train)
predict_random_forest = random_forest_model.predict(X_test_standardized)

# Acurácia com o modelo de Random Forest: 0.77
print(f'\nAcurácia Random Forest: {accuracy_score(y_test, predict_random_forest)}\n')
print(f'Matriz de confusão:\n{confusion_matrix(y_test, predict_random_forest)}\n')
print(f'Classification Report RandomForest:\n{classification_report(y_test, predict_random_forest)}')


# Observações: 
# • Através da matriz de confusão no modelo Random Forest, foi possível concluir que o modelo tem dificuldade em classificar casos de diabetes como verdadeiros.
# • Após substituirmos os valores zerados em colunas pela média, foi possível aumentar a acurácia de 75% para 77%
# • Adicionado o class_weight='balanced' para prever melhor a classe minoritária (casos de diabetes)
# • O modelo generalizou melhor os resultados após adicionar max_depth=10
# • Com undersample, o modelo apresentou uma acurácia, taxa de f1-score e recall melhores 
# porém a base é menor e a capacidade de generalização foi reduzida

# Análise de importância das colunas com o modelo
# importances = random_forest_model.feature_importances_
# for feature, importance in zip(x.columns, importances):
#   print(f"{feature}: importance")

# y_pred = random_forest_model.predict(X_test_standardized)
# print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Observações com Classification Report:
# • O modelo tem fraco desempenho com a classe 1 (casos de diabetes)
# • Necessidade de melhorar o desempenho da classe minoritária

#Testando validação cruzada
scores = cross_val_score(random_forest_model, x, y, cv=10, scoring='accuracy')

# 5. Imprimir os resultados
print("Scores da validação cruzada (10 folds):")
print(scores)

param_grid = {
    'n_estimators': [50, 100, 150],  # Número de árvores
    'max_depth': [5, 10, None],      # Profundidade máxima das árvores (None = sem limite)
    'min_samples_split': [2, 5, 10]  # Mínimo de amostras para fazer uma divisão
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# print("Iniciando a busca em grade (Grid Search)...")
# # O .fit() fará o treinamento de todas as combinações de parâmetros
# grid_search.fit(x, y)

# # 4. Exibir os melhores resultados
# print("\nMelhores hiperparâmetros encontrados:")
# print(grid_search.best_params_)

# print("\nMelhor score de validação cruzada:")
# print(f"{grid_search.best_score_:.4f}")
