
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel

# --- Métricas do classificador SVM ---
print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred,
      target_names=["alta", "baixa", "média"]))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["alta", "baixa", "média"],
            yticklabels=["alta", "baixa", "média"])
plt.title("Matriz de Confusão - SVM")
plt.savefig("resultados/matriz_confusao.png")
plt.show()

# --- Coerência do LDA ---
coerencia_final = CoherenceModel(
    model=lda_final, texts=textos,
    dictionary=dicionario, coherence="c_v"
)
print(f"\nCoerência C_v do LDA final: {coerencia_final.get_coherence():.4f}")
# Referência: valores acima de 0.5 são considerados aceitáveis