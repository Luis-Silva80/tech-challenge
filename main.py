import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Importando a base
diabetes_df = pd.read_csv("datasets\diabetes.csv")

# ToDo:
# - Analisar correçalação com target do dados
# - Preparação dos dados
# - Análise exploratória dos dados

# Análise do dataset de casos de diabetes
print(f'Primeiros dados: \n{diabetes_df.head()}')
print(f'Informações dataset: \n{diabetes_df.info()}')
print(f'Shape dataset: {diabetes_df.shape}')
print(f'Describe dataset: \n{diabetes_df.describe()}')
print(diabetes_df.duplicated())
no_duplicates_df = diabetes_df.drop_duplicates()
print(f'No duplicates: \n{no_duplicates_df}')
print(f'Maior parteira: \n{diabetes_df[diabetes_df['Pregnancies'] == 17]}')
print(f'Menor parteira: \n{diabetes_df[diabetes_df['Pregnancies'] == 0]}')

#Analisando correlação dos dados
correlation_matrix = diabetes_df.select_dtypes(include=['float64', 'int']).corr().round(2)

# fig, ax = plt.subplots(figsize=(8,8))    
# sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
# plt.show();

#Observações sobre a corelação
# - As colunas não possuem correlações fortes entre si e nem com a target
# - Para a primeira analise vamos executar o treinamento do modelo e observar as metricas de validação

#Analisando a concentração de casos de diabetes de acordo com a Idade

#Scatter
# plt.scatter(diabetes_df['Age'], diabetes_df['Outcome'])
# plt.xlabel('Age')
# plt.ylabel('Outcome')
# plt.show()

# #Hist
# plt.hist(diabetes_df['Age'], bins=10, edgecolor='black', alpha=0.7)
# plt.xlabel('Age')
# plt.show()

#Boxplot
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


#Treinarmento  

y = diabetes_df['Outcome'];
x = diabetes_df.drop('Outcome', axis=1)

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_padronizado = scaler.fit_transform(x_train)

X_train_padronizado = scaler.fit_transform(x_train)
X_test_padronizado = scaler.transform(x_test)

modelo = KNeighborsClassifier(n_neighbors=5);

print(f'Shape treino X: \n{X_test_padronizado.shape}')
print(f'Shape test X: \n{y_test.shape}')

modelo.fit(x_train, y_train)

x_test_example = x_test.iloc[[1]]
y_test_example = y_test.iloc[[1]]

predict_model = modelo.predict(X_test_padronizado)

# print('Valor real: ', y_test_example)
# print('Valor predito: ', predict_model)

error = []

# Calculating error for K values between 1 and 10
for i in range(1, 10): #range de tentativas para k
    knn = KNeighborsClassifier(n_neighbors=i)# aqui definimos  o k
    knn.fit(X_train_padronizado, y_train) #treinando o algoritmo para encontrar o erro
    pred_i = knn.predict(X_test_padronizado) #armazenando as previsões
    error.append(np.mean(pred_i != y_test)) #armazenando o valor do erro médio na lista de erros

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


