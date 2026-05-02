import os
import pdfplumber
import pandas as pd

PASTA_PDFS = r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\pdfs_sipia"
registros = []

for arquivo in os.listdir(PASTA_PDFS):
    if arquivo.endswith(".pdf"):
        caminho = os.path.join(PASTA_PDFS, arquivo)
        with pdfplumber.open(caminho) as pdf:
            texto = ""
            for pagina in pdf.pages:
                texto += pagina.extract_text() or ""
        registros.append({
            "id": arquivo.replace(".pdf", ""),
            "texto": texto.strip(),
            "fonte": "SIPIA"
        })

df = pd.DataFrame(registros)
# As demais colunas (data, conselho, categoria etc.) virão do
# preenchimento manual ou exportação estruturada do SIPIA
df.to_csv(r"...\dados\corpus_bruto.csv", index=False, encoding="utf-8")
print(f"{len(df)} documentos extraídos dos PDFs.")

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