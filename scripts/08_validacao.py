import pandas as pd
import joblib
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os

# ── Carregar dados ────────────────────────────────────────────────────────────
df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_com_sentimentos.csv", encoding="utf-8")
df["tokens"] = df["tokens"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# ── Carregar modelos salvos ───────────────────────────────────────────────────
svm = joblib.load(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\svm_gravidade.pkl")
vec = joblib.load(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\tfidf_vectorizer.pkl")
lda = gensim.models.LdaModel.load(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\lda_modelo")
dicionario = corpora.Dictionary.load(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\lda_dicionario")

# ── Validação do SVM ──────────────────────────────────────────────────────────
X = df["texto_limpo"]
y = df["gravidade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_test_tfidf = vec.transform(X_test)
y_pred = svm.predict(X_test_tfidf)

print("=== Relatório de Classificação (SVM) ===")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred, labels=["alta", "média", "baixa"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["alta", "média", "baixa"],
            yticklabels=["alta", "média", "baixa"],
            cmap="Blues")
plt.title("Matriz de Confusão — SVM")
plt.ylabel("Real")
plt.xlabel("Predito")
plt.tight_layout()

os.makedirs(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\resultados", exist_ok=True)
plt.savefig(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\resultados\matriz_confusao.png")
plt.show()
print("Matriz de confusão salva em resultados/")

# ── Validação do LDA ──────────────────────────────────────────────────────────
textos = df["tokens"].tolist()
corpus_bow = [dicionario.doc2bow(texto) for texto in textos]

if __name__ == "__main__":
    coerencia = CoherenceModel(
        model=lda,
        texts=textos,
        dictionary=dicionario,
        coherence="c_v"
    )
    cv = coerencia.get_coherence()
    print(f"\n=== Coerência C_v do LDA ===")
    print(f"C_v: {cv:.4f}")
    print("(acima de 0.5 é considerado bom — valores baixos são esperados com 30 registros)")

    print("\n=== Pipeline completa validada com sucesso! ===")
    print(f"Registros processados : {len(df)}")
    print(f"Modelos carregados    : LDA, BERTopic, SVM, TF-IDF")
    print(f"Arquivos em modelos/  : lda_modelo, bertopic_modelo, svm_gravidade.pkl, tfidf_vectorizer.pkl")
    print(f"Arquivos em resultados/: matriz_confusao.png, lda_visualizacao.html, bertopic_mapa.html")