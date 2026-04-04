# !pip install wordcloud matplotlib seaborn scikit-learn

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Nuvem de palavras geral
todos_tokens = " ".join(df["texto_limpo"])
wc = WordCloud(width=800, height=400, background_color="white").generate(todos_tokens)
plt.figure(figsize=(12, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Termos mais frequentes no corpus")
plt.savefig("resultados/nuvem_geral.png")
plt.show()

# Distribuição por categoria
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x="categoria", order=df["categoria"].value_counts().index)
plt.xticks(rotation=45, ha="right")
plt.title("Distribuição por categoria")
plt.tight_layout()
plt.savefig("resultados/distribuicao_categorias.png")
plt.show()

# Distribuição por gravidade e raça/cor
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="gravidade", hue="raca_cor")
plt.title("Gravidade por raça/cor")
plt.tight_layout()
plt.savefig("resultados/gravidade_raca.png")
plt.show()

# TF-IDF: termos mais relevantes por categoria
vectorizer = TfidfVectorizer(max_features=20)
for cat in df["categoria"].unique():
    subset = df[df["categoria"] == cat]["texto_limpo"]
    if len(subset) > 1:
        tfidf = vectorizer.fit_transform(subset)
        print(f"\n--- {cat} ---")
        print(vectorizer.get_feature_names_out())