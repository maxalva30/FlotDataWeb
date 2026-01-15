
import dash
from dash import html, dcc, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import base64

dash.register_page(__name__, path="/")

# ============================================
# PREVIEW CHART FUNCTIONS
# ============================================

def create_feed_rate_chart(df_flat):
    """
    Chart 1: Feed Rate Over Time (Last 24 entries)
    Bar chart showing recent feed rate data
    """
    feed_col = "Feed__Feed, tph"
    time_col = "Time__DateTime_Measured"
    
    if feed_col not in df_flat.columns or time_col not in df_flat.columns:
        return go.Figure().add_annotation(
            text="Feed data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Get last 24 entries
    df_recent = df_flat.tail(24).copy()
    df_recent[time_col] = pd.to_datetime(df_recent[time_col])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_recent[time_col],
        y=df_recent[feed_col],
        marker=dict(
            color='#3b82f6',
            line=dict(color='#1e3a8a', width=1)
        ),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Feed Rate: %{y:.2f} tph<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Feed Rate (Last 24h)",
            font=dict(size=14, weight=600)
        ),
        height=420,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            title=""
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            title="tph"
        ),
        hovermode='x unified'
    )
    
    return fig


def create_mineral_type_chart(df_flat):
    """
    Chart 2: Mineral Type Distribution
    Donut chart showing ore type composition
    """
    supergene_col = "Mineral type__Ore Type %Supergene"
    hypogene_col = "Mineral type__Ore Type %Hypogene"
    
    if supergene_col not in df_flat.columns or hypogene_col not in df_flat.columns:
        return go.Figure().add_annotation(
            text="Mineral type data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Calculate average percentages
    supergene_avg = df_flat[supergene_col].mean()
    hypogene_avg = df_flat[hypogene_col].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=['Supergene', 'Hypogene'],
        values=[supergene_avg, hypogene_avg],
        hole=0.4,
        marker=dict(
            colors=['#3b82f6', '#fbbf24'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Mineral Type Distribution",
            font=dict(size=14, weight=600)
        ),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def create_recovery_histogram(df_flat):
    """
    Chart 3: Recovery Distribution
    Histogram showing recovery rate distribution
    """
    recovery_col = "Process Calculated__%Rec L1"
    
    if recovery_col not in df_flat.columns:
        return go.Figure().add_annotation(
            text="Recovery data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_flat[recovery_col],
        marker=dict(
            color='#10b981',
            line=dict(color='#059669', width=1)
        ),
        nbinsx=20,
        hovertemplate='Recovery: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_recovery = df_flat[recovery_col].mean()
    fig.add_vline(
        x=mean_recovery,
        line_dash="dash",
        line_color="#059669",
        annotation_text=f"Mean: {mean_recovery:.1f}%",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(
            text="Recovery Distribution (Line 1)",
            font=dict(size=14, weight=600)
        ),
        height=420,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            title="Recovery (%)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            title="Frequency"
        ),
        bargap=0.1
    )
    
    return fig


# ============================================
# LAYOUT COMPONENTS
# ============================================

def create_hero_carousel():
    """Hero section with image carousel"""
    return html.Div(
        className="hero-carousel",
        children=[
            dcc.Interval(
                id='carousel-interval',
                interval=4000,
                n_intervals=0
            ),
            html.Div(
                id="carousel-content",
                children=[
                    html.Div(
                        className="hero-slide",
                        children=[
                            html.Img(src="/assets/flotation-machines-cells.jpg"),
                            html.Div(
                                className="hero-overlay",
                                children=[
                                    html.H1("Flotation Cells", className="hero-title"),
                                    html.P(
                                        "Advanced data analysis assistant for optimizing flotation "
                                        "cell performance and mineral recovery processes.",
                                        className="hero-subtitle"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def create_guide_section():
    """Guide section with instructions"""
    return html.Div(
        className="section-card",
        children=[
            html.Div(
                className="section-header",
                children=[
                    html.Span("📋", className="section-icon"),
                    html.H2("Guide", className="section-title")
                ]
            ),
            html.Div(
                className="guide-content",
                children=[
                    html.Div(
                        className="guide-text",
                        children=html.P(
                            "Welcome to the Flotation Data Analysis Assistant. This tool allows you to "
                            "upload operational data from your flotation cells and perform comprehensive "
                            "analysis including recovery statistics, correlation studies, and process optimization. "
                            "Start by uploading your Excel file with the standardized template format."
                        )
                    ),
                    html.Div(
                        className="guide-illustration",
                        children=html.Img(
                            src="/assets/flotationcell.png",
                            style={"maxWidth": "180px"}
                        )
                    )
                ]
            )
        ]
    )


def create_data_loading_section():
    """Data upload section"""
    return html.Div(
        className="section-card",
        children=[
            html.Div(
                className="section-header",
                children=[
                    html.Span("📊", className="section-icon"),
                    html.H2("Data Loading", className="section-title")
                ]
            ),
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    className="upload-container",
                    id="upload-container-div",
                    children=[
                        html.Div(
                            children=[
                                html.Div("📁", className="upload-icon"),
                                html.Div(
                                    id="upload-text",
                                    children="Drag your Excel file here or click to select",
                                    className="upload-text"
                                ),
                                html.Div(
                                    "Supported formats: .xls, .xlsx",
                                    style={"fontSize": "0.8rem", "color": "#9ca3af", "marginTop": "0.5rem"}
                                )
                            ]
                        )
                    ]
                ),
                multiple=False,
                accept=".xls,.xlsx"
            ),
            html.Div(id="excel-preview-container", className="excel-preview-container")
        ]
    )


def create_flotation_config_section():
    """Flotation configuration section"""
    return html.Div(
        className="section-card",
        children=[
            html.Div(
                className="section-header",
                children=[
                    html.Span("⚙️", className="section-icon"),
                    html.H2("Flotation Configuration", className="section-title")
                ]
            ),
            html.Div(
                className="config-grid",
                children=[
                    html.Div(
                        className="config-item",
                        children=[
                            html.Label("Flotation Rows"),
                            dcc.Dropdown(
                                id="dropdown-flotation-rows",
                                options=[{"label": str(i), "value": i} for i in range(1, 11)],
                                value=4,  # Default value
                                placeholder="Select number of rows",
                                clearable=False
                            )
                        ]
                    ),
                    html.Div(
                        className="config-item",
                        children=[
                            html.Label("Flotation Cells"),
                            dcc.Dropdown(
                                id="dropdown-cells-per-row",
                                options=[{"label": str(i), "value": i} for i in range(1, 21)],
                                value=8,  # Default value
                                placeholder="Select cells per row",
                                clearable=False
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                className="total-cells-display",
                children=[
                    html.Div(
                        className="cell-counter",
                        children=[
                            html.Div(id="total-rows-display", children="4", className="cell-counter-value"),
                            html.Div("Total Rows", className="cell-counter-label")
                        ]
                    ),
                    html.Div(
                        className="cell-counter",
                        children=[
                            html.Div(id="total-cells-display", children="32", className="cell-counter-value"),
                            html.Div("Total Cells", className="cell-counter-label")
                        ]
                    )
                ]
            )
        ]
    )


def create_preview_graphics_section():
    """Preview graphics section with 3 charts"""
    return html.Div(
        children=[
            html.H2(
                "Preview of Data",
                style={
                    "fontSize": "1.5rem",
                    "fontWeight": "600",
                    "marginTop": "2rem",
                    "marginBottom": "1rem",
                    "color": "#1f2937"
                }
            ),
            html.Div(
                id="preview-graphics-container",
                className="preview-graphics-grid",
                children=[
                    html.Div("Upload data to see preview charts", style={"textAlign": "center", "padding": "2rem"})
                ]
            )
        ]
    )


def create_go_to_analysis_button():
    """Go to analysis button"""
    return html.Div(
        style={"textAlign": "center", "marginTop": "2rem", "marginBottom": "2rem"},
        children=[
            html.A(
                dbc.Button(
                    "Go to Analysis",
                    id="go-to-analysis-btn",
                    className="btn-primary",
                    disabled=True,
                    size="lg",
                    style={"padding": "1rem 3rem", "fontSize": "1.1rem"}
                ),
                href="/plots"
            )
        ]
    )


# ============================================
# MAIN LAYOUT
# ============================================

layout = html.Div(
    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "2rem"},
    children=[
        # Hero Carousel
        create_hero_carousel(),
        
        # Guide Section
        create_guide_section(),
        
        # Data Loading Section
        create_data_loading_section(),
        
        # Split Layout: Excel Preview + Flotation Config
        html.Div(
            style={
                "display": "grid", 
                "gridTemplateColumns": "2fr 1fr",  # ← Excel más ancho, config más angosto
                "gap": "1.5rem", 
                "marginTop": "1.5rem"
            },
            children=[
                # Left: Excel preview section
                html.Div(
                    id="excel-preview-section",
                    className="section-card",
                    style={
                        "minWidth": "0",           # ← CRÍTICO: Permite que el grid contraiga el contenido
                        "overflow": "hidden"       # ← CRÍTICO: Oculta desbordamiento
                    },
                    children=[
                        html.Div(
                            className="section-header",
                            children=[
                                html.Span("📊", className="section-icon"),
                                html.H2("Visualization of File", className="section-title")
                            ]
                        ),
                        html.Div(id="excel-preview-container-left")
                    ]
                ),
                
                # Right: Flotation Configuration
                create_flotation_config_section()
            ]
        ),
        
        # Preview Graphics
        create_preview_graphics_section(),
        
        # Go to Analysis Button
        create_go_to_analysis_button(),
        
        # Footer
        html.Div(
            style={"textAlign": "center", "marginTop": "3rem", "padding": "1rem", "color": "#9ca3af", "fontSize": "0.85rem"},
            children="Copyright © 2025 Metso - Flotation Data Analysis Assistant"
        )
    ]
)


# ============================================
# CALLBACKS
# ============================================

@callback(
    Output("carousel-content", "children"),
    Input("carousel-interval", "n_intervals")
)
def update_carousel(n):
    """Alternate between carousel images"""
    images = [
        {
            "src": "/assets/flotation-machines-cells.jpg",
            "title": "Flotation Cells",
            "subtitle": "Advanced data analysis assistant for optimizing flotation cell performance and mineral recovery processes."
        },
        {
            "src": "/assets/Metso-flag.png",
            "title": "Metso",
            "subtitle": "Leading the industry with innovative solutions for sustainable mining and mineral processing."
        }
    ]
    
    current_img = images[n % 2]
    
    return html.Div(
        className="hero-slide",
        children=[
            html.Img(src=current_img["src"], style={"objectFit": "cover"}),
            html.Div(
                className="hero-overlay",
                children=[
                    html.H1(current_img["title"], className="hero-title"),
                    html.P(current_img["subtitle"], className="hero-subtitle")
                ]
            )
        ]
    )


@callback(
    Output("stored-data", "data"),
    Output("group-map-store", "data"),
    Output("upload-text", "children"),
    Output("upload-container-div", "className"),
    Output("excel-preview-container-left", "children"),
    Output("go-to-analysis-btn", "disabled"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    """Handle Excel file upload with validation and preview"""
    if contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    
    try:
        # Read Excel file
        xls = pd.ExcelFile(io.BytesIO(decoded))
        
        # Validate sheet exists
        if "Data" not in xls.sheet_names:
            raise ValueError("Sheet 'Data' not found. Please use the standard template.")
        
        # Read with multi-level headers
        df_multi = pd.read_excel(xls, sheet_name="Data", header=[5, 6])
        df_multi.columns = df_multi.columns.set_names(["group", "variable"])
        
        # Build group mapping
        group_to_columns = {}
        for grp, var in df_multi.columns:
            if pd.isna(grp) or pd.isna(var):
                continue
            g = str(grp).strip()
            v = str(var).strip()
            group_to_columns.setdefault(g, []).append(v)
        
        # Flatten columns for storage
        df_flat = df_multi.copy()
        flat_columns = [f"{g}__{v}" for g, v in df_flat.columns]
        df_flat.columns = flat_columns
        
        # Convert time column
        if len(df_flat.columns) > 0:
            first_col = df_flat.columns[0]
            df_flat[first_col] = pd.to_datetime(df_flat[first_col], errors="coerce")
        
        df_json = df_flat.to_json(date_format="iso", orient="split")
        
        # Create preview
        preview = create_excel_preview(df_flat, group_to_columns, filename)
        
        # Success message
        upload_text = f"✓ {filename} loaded successfully"
        upload_class = "upload-container file-loaded"
        
        return df_json, group_to_columns, upload_text, upload_class, preview, False
        
    except Exception as e:
        print(f"Error reading Excel: {e}")
        
        error_preview = html.Div(
            style={"color": "#ef4444", "marginTop": "1rem", "padding": "1rem", "background": "#fee2e2", "borderRadius": "8px"},
            children=[
                html.Strong("Error loading file: "),
                html.Span(str(e))
            ]
        )
        
        return dash.no_update, dash.no_update, "⚠ Error loading file", "upload-container", error_preview, True


def create_excel_preview(df_flat, group_map, filename):
    """Create preview of uploaded Excel data"""
    
    # Statistics
    n_records = len(df_flat)
    n_groups = len(group_map)
    n_variables = len(df_flat.columns)
    
    time_col = df_flat.columns[0]
    date_range = f"{df_flat[time_col].min():%Y-%m-%d} to {df_flat[time_col].max():%Y-%m-%d}"
    
    # Select important columns for preview (max 15)
    important_cols = []
    
    # Add time
    important_cols.append(time_col)
    
    # Add some from each major group
    for group in ["Feed", "Process Calculated", "Operational Parameters"]:
        if group in group_map:
            group_cols = [col for col in df_flat.columns if col.startswith(f"{group}__")]
            important_cols.extend(group_cols[:3])  # First 3 from each group
    
    important_cols = important_cols[:15]  # Max 15 columns
    df_preview = df_flat[important_cols].head(10)
    
    # Create nice column names for display
    display_columns = []
    for col in important_cols:
        if "__" in col:
            var_name = col.split("__")[1]
            display_columns.append({"name": var_name, "id": col})
        else:
            display_columns.append({"name": col, "id": col})
    
    return html.Div([
        # Statistics cards
        html.Div(
            className="stats-grid",
            children=[
                html.Div(
                    className="stat-card",
                    children=[
                        html.P(str(n_records), className="stat-value"),
                        html.P("Records", className="stat-label")
                    ]
                ),
                html.Div(
                    className="stat-card",
                    children=[
                        html.P(str(n_groups), className="stat-value"),
                        html.P("Data Groups", className="stat-label")
                    ]
                ),
                html.Div(
                    className="stat-card",
                    children=[
                        html.P(str(n_variables), className="stat-value"),
                        html.P("Variables", className="stat-label")
                    ]
                )
            ]
        ),
        
        # Date range
        html.Div(
            style={"marginBottom": "1rem", "color": "#6b7280", "fontSize": "0.9rem"},
            children=f"📅 Date Range: {date_range}"
        ),
        
        # Data table
        html.Div(
            style={
                "maxHeight": "500px",           # ← Altura máxima ajustable
                "height": "auto",                # ← Se adapta al contenido
                "overflowY": "auto",             # ← Scroll vertical
                "overflowX": "auto",             # ← Scroll horizontal
                "border": "1px solid #e5e7eb",
                "borderRadius": "8px",
                "width": "100%",                 # ← Ocupa todo el ancho disponible
            },
            children=[
                dbc.Table.from_dataframe(
                    df_preview,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                    style={
                        "fontSize": "0.85rem", 
                        "margin": "0",
                        "width": "100%",         # ← Tabla ocupa todo el ancho
                        "tableLayout": "auto"    # ← Se adapta al contenido
                    }
                )
            ]
        )
    ])


@callback(
    Output("total-rows-display", "children"),
    Output("total-cells-display", "children"),
    Output("flotation-config", "data"),
    Input("dropdown-flotation-rows", "value"),
    Input("dropdown-cells-per-row", "value")
)
def update_flotation_config(n_rows, cells_per_row):
    """Update flotation configuration display and storage"""
    if n_rows is None or cells_per_row is None:
        raise PreventUpdate
    
    total_cells = n_rows * cells_per_row
    
    config_data = {
        "rows": int(n_rows),
        "cells_per_row": int(cells_per_row)
    }
    
    return str(n_rows), str(total_cells), config_data


@callback(
    Output("preview-graphics-container", "children"),
    Input("stored-data", "data"),
    prevent_initial_call=True
)
def update_preview_graphics(stored_json):
    """Generate preview graphics when data is loaded"""
    if stored_json is None:
        raise PreventUpdate
    
    try:
        df_flat = pd.read_json(stored_json, orient="split")
        
        # Generate the 3 preview charts
        fig1 = create_feed_rate_chart(df_flat)
        fig2 = create_mineral_type_chart(df_flat)
        fig3 = create_recovery_histogram(df_flat)
        
        return [
            html.Div(
                className="preview-chart-container",
                children=[
                    html.Div("Feed Rate Overview", className="chart-title"),
                    dcc.Graph(
                        figure=fig1,
                        config={'displayModeBar': False}  # Mini-interactive: hover only
                    )
                ]
            ),
            html.Div(
                className="preview-chart-container",
                children=[
                    html.Div("Mineral Composition", className="chart-title"),
                    dcc.Graph(
                        figure=fig2,
                        config={'displayModeBar': False}
                    )
                ]
            ),
            html.Div(
                className="preview-chart-container",
                children=[
                    html.Div("Recovery Performance", className="chart-title"),
                    dcc.Graph(
                        figure=fig3,
                        config={'displayModeBar': False}
                    )
                ]
            )
        ]
        
    except Exception as e:
        print(f"Error generating preview charts: {e}")
        return html.Div(
            "Error generating preview charts",
            style={"textAlign": "center", "padding": "2rem", "color": "#ef4444"}
        )
