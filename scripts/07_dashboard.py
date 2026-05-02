import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

df = pd.read_csv(r"c:\Users\vinil\Documents\Projeto_IC\Pipeline\dados\corpus_com_sentimentos.csv", encoding="utf-8")
df["data"] = pd.to_datetime(df["data"])

app = dash.Dash(__name__)

fig_categorias = px.bar(
    df["categoria"].value_counts().reset_index(),
    x="categoria", y="count",
    title="Atendimentos por Categoria",
    color="categoria"
)

fig_gravidade_raca = px.histogram(
    df, x="gravidade", color="raca_cor",
    barmode="group",
    title="Gravidade por Raça/Cor"
)

fig_timeline = px.histogram(
    df, x="data",
    title="Volume de Atendimentos ao Longo do Tempo"
)

fig_sentimentos = px.histogram(
    df, x="sentimento", color="gravidade",
    barmode="group",
    title="Sentimentos por Gravidade"
)

app.layout = html.Div([
    html.H1(
        "Dashboard — Conselhos Tutelares RMC",
        style={"textAlign": "center", "fontFamily": "Arial", "marginBottom": "30px"}
    ),
    html.Div([
        dcc.Graph(figure=fig_categorias),
        dcc.Graph(figure=fig_gravidade_raca),
    ], style={"display": "flex", "flexWrap": "wrap"}),
    html.Div([
        dcc.Graph(figure=fig_timeline),
        dcc.Graph(figure=fig_sentimentos),
    ], style={"display": "flex", "flexWrap": "wrap"}),
])

if __name__ == "__main__":
    app.run(debug=True)