import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# ============================================
# NAVBAR COMPONENT (inline)
# ============================================

def create_navbar():
    """Creates a persistent navbar that appears on all pages"""
    return html.Div(
        className="navbar-container",
        children=[
            dbc.Navbar(
                dbc.Container(
                    fluid=True,
                    children=[
                        # Left: Metso Logo + Brand
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            "Metso",
                                            className="navbar-brand-text"
                                        )
                                    ),
                                ],
                                align="center",
                                className="g-0",
                            ),
                            href="/",
                            style={"textDecoration": "none"}
                        ),
                        
                        # Center: Navigation Links
                        dbc.Nav(
                            className="navbar-nav-center",
                            children=[
                                dbc.NavItem(
                                    dbc.NavLink(
                                        "Flotation Analysis Tool",
                                        href="/",
                                        className="nav-link-custom"
                                    )
                                ),
                            ],
                            navbar=True,
                        ),
                        
                        # Right: Menu Button (optional - for future use)
                        html.Div(
                            className="navbar-menu-button",
                            children=[
                                html.Button(
                                    "☰",
                                    id="menu-button",
                                    className="menu-btn"
                                )
                            ]
                        )
                    ],
                ),
                color="white",
                light=True,
                className="custom-navbar",
                sticky="top",
            )
        ]
    )

# ============================================
# APP INITIALIZATION
# ============================================

# Initialize the Dash app
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Metso - Flotation Data Assistant"
)

# ============================================
# APP LAYOUT
# ============================================

app.layout = html.Div([
    # Data stores (persist across pages)
    dcc.Store(id='stored-data', storage_type='session'),
    dcc.Store(id='group-map-store', storage_type='session'),
    dcc.Store(id='flotation-config', storage_type='session'),
    
    # Navbar (appears on all pages)
    create_navbar(),
    
    # Main content container with padding
    html.Div(
        id="main-content-container",
        style={
            "padding": "2rem",
            "maxWidth": "1400px",
            "margin": "0 auto"
        },
        children=[
            # Page container (where individual pages render)
            dash.page_container
        ]
    )
])

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8050
    )