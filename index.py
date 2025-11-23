import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

app.layout = html.Div([
    # >>> STORES GLOBALES <<<
    dcc.Store(id="stored-data", storage_type="session"),
    dcc.Store(id="group-map-store", storage_type="session"),
    dcc.Store(id="flotation-config", storage_type="session"),  # 👈 NUEVO

    # Header
    html.Div(
        className="header-container",
        children=[
            html.Div(
                html.Img(src="/assets/MetsoLogo.png", className="logo"),
                className="logo-container",
            ),
            html.Div(
                html.H1(
                    "Flotation Data Analysis Assistant",
                    className="titulo-principal",
                ),
                className="titulo-container",
            ),
            html.Div(
                id="report-message",
                className="report-message-header",
            ),
        ],
    ),

    # Contenido multipage
    dcc.Loading(dash.page_container, type="circle"),

    # Footer
    html.Div(
        className="footer",
        children=html.P("Copyright © 2025 Metso"),
    ),
])

if __name__ == "__main__":
    app.run(debug=True)
