import pandas as pd

diatabetes_df = pd.read_csv("diabetes.csv")

# ToDo:
# - Analisar correçalação com target do dados
# - Preparação dos dados
# - Análise exploratória dos dados

# Análise do dataset de casos de diabetes
print(diatabetes_df.head())
print(diatabetes_df.info())
print(diatabetes_df.describe())
print(diatabetes_df.duplicated())
no_duplicates_df = diatabetes_df.drop_duplicates()
print(no_duplicates_df)
