import pandas as pd

# Carregar a base fictícia
df = pd.read_csv("dados/base_ficticia.csv", encoding="utf-8")

# Verificar estrutura
print(df.shape)         # quantas linhas e colunas
print(df.dtypes)        # tipos de cada coluna
print(df.isnull().sum()) # verificar valores ausentes

# Converter data para formato datetime
df["data"] = pd.to_datetime(df["data"])

# Visualizar distribuição por categoria
print(df["categoria"].value_counts())
print(df["gravidade"].value_counts())

df.to_csv("dados/corpus_organizado.csv", index=False)