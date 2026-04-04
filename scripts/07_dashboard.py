# !pip install dash plotly

import dash
from dash import dcc, html
import plotly.express as px

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
    df, x="data", title="Volume de Atendimentos ao Longo do Tempo"
)

app.layout = html.Div([
    html.H1("Dashboard - Conselhos Tutelares RMC",
            style={"textAlign": "center", "fontFamily": "Arial"}),
    dcc.Graph(figure=fig_categorias),
    dcc.Graph(figure=fig_gravidade_raca),
    dcc.Graph(figure=fig_timeline),
])

app.run(debug=True)