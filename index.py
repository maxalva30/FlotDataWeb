import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# Initialize Dash app with multipage support enabled
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

# Main layout with header, global stores, page container and footer
app.layout = html.Div([

    # ---------- Header ----------
    html.Div(
        className="header-container",
        children=[
            html.Div(
                className="logo-container",
                children=html.Img(
                    src="/assets/MetsoLogo.png",
                    className="logo"
                )
            ),
            html.Div(
                className="titulo-container",
                children=html.H1(
                    "Flotation Data Analysis Assistant",
                    className="titulo-principal"
                )
            ),
        ],
    ),

    # ---------- Global Stores (available for all pages) ----------
    # These stores will hold the uploaded flotation data and
    # the mapping: group name (Feed, Reagents, etc.) -> list of variables
    dcc.Store(id="stored-data", storage_type="session"),
    dcc.Store(id="group-map-store", storage_type="session"),

    # ---------- Page content (Dash Pages) ----------
    dcc.Loading(
        id="page-loading",
        type="circle",
        children=dash.page_container
    ),

    # ---------- Footer ----------
    html.Div(
        className="footer",
        children=html.P("Copyright © 2025 Metso")
    ),
])

if __name__ == "__main__":
    app.run(debug=True)
