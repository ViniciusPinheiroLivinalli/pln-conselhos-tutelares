from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import pandas as pd

# Carregar apenas os documentos rotulados manualmente
df_rotulado = pd.read_csv(r"...\dados\corpus_rotulado.csv", encoding="utf-8")

X = df_rotulado["texto_limpo"]
y = df_rotulado["gravidade"]

# Divisão 80/10/10
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

# Treinar
vec = TfidfVectorizer(max_features=1000)
X_train_tfidf = vec.fit_transform(X_train)
X_val_tfidf   = vec.transform(X_val)
X_test_tfidf  = vec.transform(X_test)

svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train_tfidf, y_train)

# Avaliar no conjunto de validação (para ajustar parâmetros)
print("\n--- Validação ---")
print(classification_report(y_val, svm.predict(X_val_tfidf)))

# Avaliar no conjunto de teste (apenas uma vez, no final)
print("\n--- Teste Final ---")
print(classification_report(y_test, svm.predict(X_test_tfidf)))

# Salvar modelos
joblib.dump(svm, r"...\modelos\svm_gravidade.pkl")
joblib.dump(vec, r"...\modelos\tfidf_vectorizer.pkl")
print("Modelos salvos.")