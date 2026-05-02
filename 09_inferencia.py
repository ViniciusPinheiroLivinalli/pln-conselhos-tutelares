# 09_inferencia.py
# Usa os modelos já treinados para analisar documentos novos
# Não treina nada — só carrega e aplica

import joblib
import gensim
import pandas as pd
from bertopic import BERTopic
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

# ── Carregar modelos ──────────────────────────────────────────────────────────
print("Carregando modelos...")

lda        = gensim.models.LdaModel.load(r"...\modelos\lda_modelo")
dicionario = gensim.corpora.Dictionary.load(r"...\modelos\lda_dicionario")
bertopic_model = BERTopic.load(r"...\modelos\bertopic_modelo")
svm        = joblib.load(r"...\modelos\svm_gravidade.pkl")
vec        = joblib.load(r"...\modelos\tfidf_vectorizer.pkl")

nlp        = spacy.load("pt_core_news_lg")
stop_pt    = set(stopwords.words("portuguese"))
analyzer   = AnalyzerEngine()
anonymizer = AnonymizerEngine()

print("Modelos carregados.\n")

# ── Funções ───────────────────────────────────────────────────────────────────

def anonimizar(texto):
    resultados = analyzer.analyze(
        text=texto, language="pt",
        entities=["PERSON","LOCATION","PHONE_NUMBER",
                  "EMAIL_ADDRESS","BR_CPF","DATE_TIME"]
    )
    return anonymizer.anonymize(text=texto, analyzer_results=resultados).text

def preprocessar(texto):
    texto = unidecode(texto.lower())
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    doc   = nlp(texto)
    return [t.lemma_ for t in doc
            if t.is_alpha and t.lemma_ not in stop_pt and len(t.lemma_) > 2]

def analisar_documento(texto_bruto):
    # 1. Anonimizar
    texto_anon  = anonimizar(texto_bruto)

    # 2. Pré-processar
    tokens      = preprocessar(texto_anon)
    texto_limpo = " ".join(tokens)

    # 3. LDA
    bow         = dicionario.doc2bow(tokens)
    dist_topicos = lda[bow]
    topico_lda  = max(dist_topicos, key=lambda x: x[1])[0] if dist_topicos else -1
    termos_lda  = lda.show_topic(topico_lda, topn=5) if topico_lda >= 0 else []

    # 4. BERTopic
    topico_bert, _ = bertopic_model.transform([texto_bruto])

    # 5. Gravidade
    x_tfidf   = vec.transform([texto_limpo])
    gravidade = svm.predict(x_tfidf)[0]

    return {
        "texto_anonimizado" : texto_anon[:200] + "...",
        "topico_lda"        : topico_lda,
        "termos_lda"        : [t for t, _ in termos_lda],
        "topico_bertopic"   : topico_bert[0],
        "gravidade"         : gravidade,
    }

# ── Executar ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Opção A: analisar um documento avulso
    texto_teste = """
    Criança de 10 anos, João Silva, residente na Rua das Flores 123, 
    foi encontrada sozinha em casa por três dias consecutivos. 
    Mãe, Maria Silva, CPF 123.456.789-00, ausentou-se sem deixar 
    responsável ou alimentação. Sinais de desnutrição identificados.
    """
    resultado = analisar_documento(texto_teste)
    print("=== Resultado da análise ===")
    for chave, valor in resultado.items():
        print(f"{chave}: {valor}")

    # Opção B: analisar um lote de documentos de um CSV
    # df_novos = pd.read_csv(r"...\dados\novos_documentos.csv", encoding="utf-8")
    # resultados = df_novos["texto"].apply(analisar_documento)
    # df_novos = pd.concat([df_novos, pd.DataFrame(resultados.tolist())], axis=1)
    # df_novos.to_csv(r"...\dados\novos_documentos_analisados.csv", index=False)
    # print(f"{len(df_novos)} documentos analisados e salvos.")