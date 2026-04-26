# !pip install transformers torch scikit-learn

# --- Parte A: Análise de sentimentos com BERTimbau ---
from transformers import pipeline
import pandas as pd
df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_preprocessado.csv", encoding="utf-8")

# Modelo de sentimentos em português
sentimentos = pipeline(
    "text-classification",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=1
)

def analisar_sentimento(texto):
    resultado = sentimentos(texto[:512])[0]  # limite de tokens
    return resultado["label"], round(resultado["score"], 3)

df[["sentimento", "confianca_sentimento"]] = df["texto"].apply(
    lambda t: pd.Series(analisar_sentimento(t))
)
print(df[["id", "gravidade", "sentimento", "confianca_sentimento"]].head(10))

# --- Parte B: Classificação de gravidade com SVM (baseline) ---
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = df["texto_limpo"]
y = df["gravidade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vec = TfidfVectorizer(max_features=500)
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))