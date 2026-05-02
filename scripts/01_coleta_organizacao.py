import pandas as pd

# Carregar a base fictícia
df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\base_ficticia.csv", encoding="utf-8")

# Verificar estrutura
print(f"Base carregada: {df.shape[0]} registros, {df.shape[1]} colunas")
print(df.columns.tolist())

# Converter data para datetime
df["data"] = pd.to_datetime(df["data"])

# Distribuição por categoria e gravidade
print("\nDistribuição por categoria:")
print(df["categoria"].value_counts())

print("\nDistribuição por gravidade:")
print(df["gravidade"].value_counts())

# Salvar corpus organizado
df.to_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_organizado.csv", index=False, encoding="utf-8")
print("\nArquivo salvo: corpus_organizado.csv")