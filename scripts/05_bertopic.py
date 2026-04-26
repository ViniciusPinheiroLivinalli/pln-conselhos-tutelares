# !pip install bertopic sentence-transformers

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_preprocessado.csv", encoding="utf-8")

textos_brutos = df["texto"].tolist()

# Carregar BERTimbau como modelo de embeddings
embedding_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

# Criar e treinar o modelo
bertopic_model = BERTopic(
    embedding_model=embedding_model,
    language="portuguese",
    min_topic_size=2,      # importante para base pequena
    verbose=True
)
topicos, probabilidades = bertopic_model.fit_transform(textos_brutos)

# Ver os tópicos encontrados
print(bertopic_model.get_topic_info())

# Associar tópico a cada documento
df["topico_bertopic"] = topicos
print(df[["id", "categoria", "topico_bertopic"]].head(10))

# Visualizações
bertopic_model.visualize_topics().write_html("resultados/bertopic_mapa.html")
bertopic_model.visualize_barchart().write_html("resultados/bertopic_barras.html")

# Salvar modelo
bertopic_model.save("modelos/bertopic_modelo")