import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_preprocessado.csv", encoding="utf-8")
textos_brutos = df["texto"].tolist()

if __name__ == '__main__':
    embedding_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

    bertopic_model = BERTopic(
        embedding_model=embedding_model,
        language="portuguese",
        min_topic_size=2,
        verbose=True
    )
    topicos, probabilidades = bertopic_model.fit_transform(textos_brutos)

    print(bertopic_model.get_topic_info())

    df["topico_bertopic"] = topicos
    print(df[["id", "categoria", "topico_bertopic"]].head(10))

    bertopic_model.visualize_topics().write_html(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\resultados\bertopic_mapa.html")
    bertopic_model.visualize_barchart().write_html(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\resultados\bertopic_barras.html")
    bertopic_model.save(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\modelos\bertopic_modelo")
    print("Modelo salvo.") 