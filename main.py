# Importação de bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./datasets/breast_cancer.csv")

# .shape: número de linhas pelo número de colunas
print(df.shape)

# .colums: nome de todas as colunas
print(df.columns)

# .head(): retorno das primeiras 5 linhas
print(df.head())

# .info(): tipo das columas e quantidades de valores não nulos
print(df.info())

# .describre(): resume a contagem, média, desvio padrão, mínimo e máximo
print(df.describe())

# Checando valores duplicados
print(df.duplicated().sum())

# Histograma
print(df['diagnosis'].hist(bins=50))

# Exibição de gráficos
plt.figure(figsize=(20,15))
plt.suptitle("Cancer Breast Diagnosis Analysis", fontsize=20, fontweight='bold', alpha=0.8, y=1)
cat = ['diagnosis', 'radius_mean', 'radius_se', 'radius_worst']

for i in range(len(cat)):
  plt.subplot(2,2,i+1)
  sns.countplot(x=df[cat[i]])
  plt.xlabel(cat[i])
  plt.xticks(rotation=45)
  plt.tight_layout()
plt.show()

# Removendo o id da análise dos dados
df_cleaned = df.copy().drop([id])

# Rotulando a target como variável numérica (LabelEncoder)
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
print(df.head())

df_malignant = df[df['diagnosis'] == 0]
print(f'Malignos: ', df_malignant.head())

df_benign = df[df['diagnosis'] == 1]
print(f'Benignos: ', df_benign.head())

