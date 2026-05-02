import pandas as pd
from transformers import pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_preprocessado.csv", encoding="utf-8")

# ── Parte A: Análise de sentimentos ──────────────────────────────────────────
print("Carregando modelo de sentimentos...")
sentimentos = pipeline(
    "text-classification",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=1
)

def analisar_sentimento(texto):
    resultado = sentimentos(texto[:512])[0][0]
    return resultado["label"], round(resultado["score"], 3)

df[["sentimento", "confianca_sentimento"]] = df["texto"].apply(
    lambda t: pd.Series(analisar_sentimento(t))
)

print("\n--- Sentimentos ---")
print(df[["id", "gravidade", "sentimento", "confianca_sentimento"]].head(10))

# ── Parte B: Classificação de gravidade com SVM ───────────────────────────────
print("\nTreinando classificador de gravidade...")

X = df["texto_limpo"]
y = df["gravidade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vec = TfidfVectorizer(max_features=500)
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf  = vec.transform(X_test)

svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_test_tfidf)

print("\n--- Relatório de Classificação (SVM) ---")
print(classification_report(y_test, y_pred))
print("Nota: métricas baixas são esperadas com 30 registros fictícios.")

# ── Salvar modelos ────────────────────────────────────────────────────────────
os.makedirs(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos", exist_ok=True)
joblib.dump(svm, r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\svm_gravidade.pkl")
joblib.dump(vec, r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\tfidf_vectorizer.pkl")
print("\nModelos SVM salvos em modelos/")

# ── Salvar base com sentimentos ───────────────────────────────────────────────
df.to_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_com_sentimentos.csv", index=False, encoding="utf-8")
print("Arquivo salvo: corpus_com_sentimentos.csv")