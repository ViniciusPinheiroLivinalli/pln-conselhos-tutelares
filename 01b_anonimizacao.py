# 01b_anonimizacao.py
# Roda APENAS com dados reais, antes do pré-processamento
# Nunca rode com dados identificados diretamente no 02

import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Instalar antes:
# pip install presidio-analyzer presidio-anonymizer
# python -m spacy download pt_core_news_lg

df = pd.read_csv(r"...\dados\corpus_bruto.csv", encoding="utf-8")

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def anonimizar(texto):
    if not isinstance(texto, str) or texto.strip() == "":
        return texto
    resultados = analyzer.analyze(
        text=texto,
        language="pt",        # português
        entities=[
            "PERSON",         # nomes
            "LOCATION",       # endereços
            "PHONE_NUMBER",   # telefones
            "EMAIL_ADDRESS",  # emails
            "BR_CPF",         # CPFs brasileiros
            "DATE_TIME"       # datas que possam identificar
        ]
    )
    anonimizado = anonymizer.anonymize(text=texto, analyzer_results=resultados)
    return anonimizado.text

df["texto_anonimizado"] = df["texto"].apply(anonimizar)

# Salvar versão anonimizada — a partir daqui só trabalha com essa coluna
df = df.drop(columns=["texto"])
df = df.rename(columns={"texto_anonimizado": "texto"})
df.to_csv(r"...\dados\corpus_anonimizado.csv", index=False, encoding="utf-8")

print(f"Anonimização concluída: {len(df)} documentos processados.")
print("\nExemplo antes/depois:")
print("ANTES:", df["texto"].iloc[0][:200])