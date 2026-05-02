#pip install spacy unidecode
#pip install nltk spacy unidecode
#python -m spacy download pt_core_news_lg

import spacy
import re
from unidecode import unidecode
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import pandas as pd
df = pd.read_csv("dados/corpus_organizado.csv", encoding="utf-8")

nlp = spacy.load("pt_core_news_lg")
stop_pt = set(stopwords.words("portuguese"))

def preprocessar(texto):
    # 1. Lowercase e remove acentos
    texto = unidecode(texto.lower())
    # 2. Remove caracteres especiais
    texto = re.sub(r"[^a-z\s]", " ", texto)
    # 3. Remove espaços extras
    texto = re.sub(r"\s+", " ", texto).strip()
    # 4. spaCy: tokeniza e lematiza, remove stopwords
    doc = nlp(texto)
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha
        and token.lemma_ not in stop_pt
        and len(token.lemma_) > 2
    ]
    return tokens  # retorna lista de tokens

df["tokens"] = df["texto"].apply(preprocessar)
df["texto_limpo"] = df["tokens"].apply(lambda t: " ".join(t))

print(df[["texto", "texto_limpo"]].head(3))
df.to_csv("dados/corpus_preprocessado.csv", index=False)