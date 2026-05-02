import ast
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models

df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_preprocessado.csv", encoding="utf-8")

# Reconstruir tokens como lista
df["tokens"] = df["tokens"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

textos = df["tokens"].tolist()
dicionario = corpora.Dictionary(textos)
dicionario.filter_extremes(no_below=2, no_above=0.9)
corpus_bow = [dicionario.doc2bow(texto) for texto in textos]

# ← OBRIGATÓRIO NO WINDOWS: todo código que usa multiprocessing
# precisa estar dentro deste bloco
if __name__ == '__main__':

    import os
    os.makedirs(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos", exist_ok=True)

    lda_final.save(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\lda_modelo")
    dicionario.save(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\lda_dicionario")
    print("Modelos LDA salvos.")

    # Testar diferentes números de tópicos
    for n in [3, 4, 5, 6, 7, 8, 9, 10]:
        lda = models.LdaModel(
            corpus_bow,
            num_topics=n,
            id2word=dicionario,
            passes=20,
            random_state=42
        )
        coerencia = CoherenceModel(
            model=lda,
            texts=textos,
            dictionary=dicionario,
            coherence="c_v"
        )
        print(f"Tópicos: {n} | Coerência C_v: {coerencia.get_coherence():.4f}")

    # Treinar modelo final
    lda_final = models.LdaModel(
        corpus_bow,
        num_topics=4,
        id2word=dicionario,
        passes=20,
        random_state=42
    )

    # Exibir tópicos
    for i, topico in lda_final.print_topics(num_words=8):
        print(f"\nTópico {i}: {topico}")

    # Visualização interativa
    vis = pyLDAvis.gensim_models.prepare(lda_final, corpus_bow, dicionario)
    pyLDAvis.save_html(vis, r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\resultados\lda_visualizacao.html")
    print("\nVisualização salva em resultados/lda_visualizacao.html")