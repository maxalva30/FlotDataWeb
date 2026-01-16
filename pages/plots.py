import os
import math
import tempfile
import dash
from dash import html, dcc, callback, Input, Output, State, ctx, no_update, ALL
from dash.dcc import Download, send_file
from dash.exceptions import PreventUpdate
import pandas as pd
import sweetviz as sv
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib
matplotlib.use('Agg')

dash.register_page(__name__, path="/plots")

# =======================
# HELPER FUNCTIONS
# =======================

def _get_group_df(stored_data: str, group_map: dict, group_name: str) -> pd.DataFrame:
    """
    Rebuild dataframe only with the columns belonging to one group.
    
    CRITICAL: This function renames columns to remove the group prefix,
    allowing us to search for "Airflow 1" instead of "Operational Parameters__Airflow 1"
    """
    if stored_data is None or group_map is None:
        return pd.DataFrame()
    
    if group_name not in group_map:
        return pd.DataFrame()
    
    df_flat = pd.read_json(stored_data, orient="split")
    
    # Get the list of variables for this group from group_map
    variables = group_map.get(group_name, [])
    
    # Build column names with the group prefix
    col_names = [
        f"{group_name}__{v}"
        for v in variables
        if f"{group_name}__{v}" in df_flat.columns
    ]
    
    if not col_names:
        return pd.DataFrame()
    
    df_group = df_flat[col_names].copy()
    
    # ✅ CRITICAL: Rename back to clean variable names (remove prefix)
    # This converts "Operational Parameters__Airflow 1" → "Airflow 1"
    df_group.columns = [
        v for v in variables if f"{group_name}__{v}" in df_flat.columns
    ]
    
    if df_group.empty:
        return pd.DataFrame()
    
    return df_group


def sanitize_filename(text):
    """
    Sanitize text to make it safe for use as a filename on Windows/Unix.
    Removes or replaces characters that are invalid in filenames.
    """
    # Replace common problematic characters
    replacements = {
        '/': '_',       # Forward slash
        '\\': '_',      # Backslash
        ':': '_',       # Colon
        '*': '_',       # Asterisk
        '?': '_',       # Question mark
        '"': '_',       # Double quote
        '<': '_',       # Less than
        '>': '_',       # Greater than
        '|': '_',       # Pipe
        '³': '3',       # Superscript 3
        '²': '2',       # Superscript 2
        '¹': '1',       # Superscript 1
        '°': 'deg',     # Degree symbol
        '%': 'pct',     # Percent
        ' ': '_',       # Space
    }
    
    sanitized = text
    for old_char, new_char in replacements.items():
        sanitized = sanitized.replace(old_char, new_char)
    
    # Remove any other non-alphanumeric characters except underscore, dash, and dot
    sanitized = re.sub(r'[^\w\-.]', '_', sanitized)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized


def build_param_lines(
    df_ops,
    n_rows,
    cells_per_row,
    param_col_name,
    unit_label,
    param_title,
):
    """
    Generates line plots for a parameter across all flotation cells,
    grouped by lines/rows.
    """
    total_cells = n_rows * cells_per_row
    cell_columns = []
    
    # DEBUG: Print available columns
    print(f"\n=== DEBUG build_param_lines ===")
    print(f"Looking for param: {param_col_name}")
    print(f"Total cells expected: {total_cells}")
    print(f"Available columns in df_ops:")
    for col in df_ops.columns:
        print(f"  - {col}")
    print(f"=== END DEBUG ===\n")
    
    # Find columns that START with param_col_name and extract cell number
    for col in df_ops.columns:
        if col.startswith(param_col_name):
            # Extract number at the end of column name
            match = re.search(r"(\d+)$", col)
            if match:
                cell_num = int(match.group(1))
                if cell_num <= total_cells:
                    cell_columns.append((cell_num, col))
    
    if not cell_columns:
        print(f"⚠️ WARNING: No columns found for param '{param_col_name}'")
        return go.Figure().add_annotation(
            text=f"No data for {param_title}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    cell_columns.sort(key=lambda x: x[0])
    
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=[f"Line {i+1}" for i in range(n_rows)],
        vertical_spacing=0.08,
    )
    
    colors = px.colors.qualitative.Set3[:cells_per_row]
    
    for line_idx in range(n_rows):
        start_cell = line_idx * cells_per_row + 1
        end_cell = start_cell + cells_per_row
        
        for cell_num, col_name in cell_columns:
            if start_cell <= cell_num < end_cell:
                cell_data = df_ops[col_name].dropna()
                if not cell_data.empty:
                    local_idx = cell_num - start_cell
                    color_idx = local_idx % len(colors)
                    
                    fig.add_trace(
                        go.Scatter(
                            y=cell_data,
                            mode='lines',
                            name=f'Cell {cell_num}',
                            line=dict(color=colors[color_idx], width=2),
                            showlegend=(line_idx == 0),
                        ),
                        row=line_idx + 1,
                        col=1,
                    )
    
    fig.update_layout(
        title=dict(
            text=f"{param_title} – Raw Data",
            font=dict(size=16, weight=600)
        ),
        height=300 * n_rows,
        template="simple_white",
        hovermode='x unified',
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    for i in range(n_rows):
        fig.update_yaxes(
            title_text=unit_label,
            row=i+1,
            col=1,
            showgrid=True,
            gridcolor='lightgray',
        )
        fig.update_xaxes(
            title_text="Sample Index" if i == n_rows-1 else "",
            row=i+1,
            col=1,
            showgrid=True,
            gridcolor='lightgray',
        )
    
    return fig


def _build_param_line_boxplots(df_ops, n_rows, cells_per_row,
                                 param_col_name, unit_label, param_title):
    """
    Crea una figura con:
    - 1 subplot por línea (vertical)
    - En cada subplot, 1 boxplot por celda de esa línea
    - Cada línea tiene su propio color
    
    This is the ORIGINAL function from plots_antiguo.py that works correctly.
    """
    param_cols = [c for c in df_ops.columns if c.startswith(param_col_name)]
    if not param_cols:
        return go.Figure().add_annotation(
            text=f"No data for {param_title}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    def cell_index(col):
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 0

    param_cols = sorted(param_cols, key=cell_index)

    total_cells = len(param_cols)
    expected_cells = n_rows * cells_per_row
    if expected_cells != total_cells:
        print(f"WARNING: Configuration {n_rows} x {cells_per_row} = {expected_cells} cells, "
              f"but found {total_cells} '{param_col_name}' columns.")

    # ✅ COLORS for each line (Blue, Orange, Green, Red)
    line_colors = [
        "#1f77b4",  # Blue for Line 1
        "#ff7f0e",  # Orange for Line 2
        "#2ca02c",  # Green for Line 3
        "#d62728",  # Red for Line 4
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"Line {i+1}" for i in range(n_rows)],
    )

    for line_idx in range(n_rows):
        start = line_idx * cells_per_row
        end = start + cells_per_row
        line_cols = param_cols[start:end]

        color = line_colors[line_idx % len(line_colors)]

        for col in line_cols:
            serie = df_ops[col].dropna()
            if serie.empty:
                continue

            m = re.search(r"(\d+)$", col)
            cell_label = f"Cell {m.group(1)}" if m else col

            fig.add_trace(
                go.Box(
                    y=serie,
                    name=cell_label,
                    boxmean="sd",
                    marker=dict(color=color),
                ),
                row=line_idx + 1,
                col=1,
            )

        fig.update_yaxes(
            title_text=unit_label,
            row=line_idx + 1,
            col=1,
        )

    fig.update_layout(
        title_text=f"{param_title} Distribution – Rougher Lines",  # Title like in image 1
        showlegend=True,
        height=250 * n_rows,
    )

    return fig


def _generate_param_raw_report(n_clicks, stored_data, group_map, flot_config,
                                 param_col_name, unit_label, param_title):
    """
    Generate raw data BOXPLOTS for a given parameter.
    Shows distribution of each cell organized by lines.
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_data is None:
        raise PreventUpdate
    if flot_config is None:
        raise PreventUpdate
    
    n_rows = flot_config.get("rows", 1)
    cells_per_row = flot_config.get("cells_per_row", 1)
    
    df_ops = _get_group_df(stored_data, group_map, "Operational Parameters")
    if df_ops.empty:
        raise PreventUpdate
    
    # ✅ CHANGED: Use boxplots instead of line plots for Raw Data
    fig = _build_param_line_boxplots(
        df_ops, n_rows, cells_per_row,
        param_col_name, unit_label, param_title
    )
    
    tmp_dir = tempfile.gettempdir()
    filename = f"{sanitize_filename(param_title)}_Raw_Data.html"
    path = os.path.join(tmp_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    
    return path


def _generate_param_BA_stats_report(
    n_clicks,
    selected_date,
    stored_data,
    group_map,
    flot_config,
    param_col_name,
    unit_label,
    param_title,
):
    """
    Genera boxplots Before vs After por línea y celda para un parámetro.
    This is the ORIGINAL function from plots_antiguo.py that works correctly.
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_data is None or group_map is None or flot_config is None:
        raise PreventUpdate
    if selected_date is None:
        raise PreventUpdate

    try:
        n_rows = flot_config.get("rows")
        cells_per_row = flot_config.get("cells_per_row")
        if not n_rows or not cells_per_row:
            raise ValueError(
                "Flotation configuration (rows / cells per row) is missing."
            )

        df_flat = pd.read_json(stored_data, orient="split")

        # Find time column
        time_col = None
        for col in df_flat.columns:
            if "DateTime" in col or "Time" in col:
                time_col = col
                break
        
        if time_col is None:
            raise ValueError("Time column not found in data.")

        time_series = pd.to_datetime(df_flat[time_col])

        df_ops = _get_group_df(stored_data, group_map, "Operational Parameters")

        param_cols = [c for c in df_ops.columns if c.startswith(param_col_name)]
        if not param_cols:
            raise ValueError(
                f"No '{param_col_name}' columns found in Operational Parameters."
            )

        def cell_index(col):
            m = re.search(r"(\d+)$", col)
            return int(m.group(1)) if m else 0

        param_cols = sorted(param_cols, key=cell_index)

        total_cells = len(param_cols)
        expected_cells = n_rows * cells_per_row
        if total_cells != expected_cells:
            raise ValueError(
                f"Configuration {n_rows} x {cells_per_row} = {expected_cells} cells, "
                f"but found {total_cells} '{param_col_name}' columns."
            )

        event_date = pd.to_datetime(selected_date).date()
        records = []

        for idx, ts in time_series.items():
            if pd.isna(ts):
                continue
            current_date = ts.date()
            period = "Before" if current_date < event_date else "After"

            for col in param_cols:
                val = df_ops.at[idx, col]
                if pd.isna(val):
                    continue

                m = re.search(r"(\d+)$", col)
                if not m:
                    continue
                cell_num = int(m.group(1))

                line_idx = (cell_num - 1) // cells_per_row + 1
                line_name = f"Line {line_idx}"

                records.append(
                    {
                        "Line": line_name,
                        "CellNum": cell_num,
                        "CellLabel": f"Cell {cell_num}",
                        "Period": period,
                        "Value": val,
                    }
                )

        if not records:
            raise ValueError(
                f"No valid '{param_col_name}' data found to build Before/After plot."
            )

        df_long = pd.DataFrame(records)

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=[f"Line {i+1}" for i in range(n_rows)],
        )

        first_before = True
        first_after = True

        for line_idx in range(n_rows):
            line_name = f"Line {line_idx + 1}"
            df_line = df_long[df_long["Line"] == line_name]

            unique_cells = sorted(df_line["CellNum"].unique())

            for cell_num in unique_cells:
                df_cell = df_line[df_line["CellNum"] == cell_num]
                cell_label = f"Cell {cell_num}"

                before_vals = df_cell[df_cell["Period"] == "Before"]["Value"].dropna()
                after_vals = df_cell[df_cell["Period"] == "After"]["Value"].dropna()

                if before_vals.empty and after_vals.empty:
                    continue

                show_legend_before = first_before
                fig.add_trace(
                    go.Box(
                        x=[cell_label] * len(before_vals),
                        y=before_vals,
                        name="Before",
                        legendgroup="Before",
                        showlegend=show_legend_before,
                        boxmean="sd",
                        marker=dict(color="#1f77b4"),
                        offsetgroup="Before",
                    ),
                    row=line_idx + 1,
                    col=1,
                )
                if first_before:
                    first_before = False

                show_legend_after = first_after
                fig.add_trace(
                    go.Box(
                        x=[cell_label] * len(after_vals),
                        y=after_vals,
                        name="After",
                        legendgroup="After",
                        showlegend=show_legend_after,
                        boxmean="sd",
                        marker=dict(color="#ff7f0e"),  # Orange color
                        offsetgroup="After",
                    ),
                    row=line_idx + 1,
                    col=1,
                )
                if first_after:
                    first_after = False

            fig.update_yaxes(
                title_text=unit_label,
                row=line_idx + 1,
                col=1,
            )

        fig.update_layout(
            title_text=f"{param_title} – Before vs After (per line & cell)",
            boxmode="group",
            showlegend=True,
            height=250 * n_rows,
        )

        tmp_dir = tempfile.gettempdir()
        filename = f"{sanitize_filename(param_title)}_Statistical_Analysis.html"
        path = os.path.join(tmp_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")
        
        return path

    except Exception as e:
        print(f"Error generating Statistical Analysis report: {e}")
        raise PreventUpdate


def generate_sweetviz_report(n_clicks, stored_data, group_map, report_name):
    """
    Generate Sweetviz HTML report for a specific data group.
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_data is None:
        raise PreventUpdate
    
    group_df = _get_group_df(stored_data, group_map, report_name)
    
    if group_df.empty:
        raise PreventUpdate
    
    report = sv.analyze(group_df)
    tmp_dir = tempfile.gettempdir()
    filename = f"{report_name.replace(' ', '_')}_Report.html"
    report_path = os.path.join(tmp_dir, filename)
    
    report.show_html(filepath=report_path, open_browser=False)
    
    return report_path


def create_large_modal_with_html(title, html_path, download_id, close_id):
    """
    Helper function to create a large modal with HTML content displayed in iframe.
    Uses srcDoc to avoid file:// issues.
    """
    html_content = ""
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    
    return html.Div(
        className="modal-overlay",
        children=[
            html.Div(
                className="large-modal",
                children=[
                    html.Div(
                        className="large-modal-header",
                        children=[
                            html.H3(title, className="large-modal-title"),
                            html.Div(
                                className="large-modal-actions",
                                children=[
                                    html.Button(
                                        "Download",
                                        id=download_id,
                                        className="btn-download-modal",
                                        n_clicks=0
                                    ),
                                    html.Button("✕", id=close_id, className="btn-close-modal", n_clicks=0)
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="large-modal-body",
                        children=[
                            html.Iframe(
                                srcDoc=html_content,
                                style={"width": "100%", "height": "700px", "border": "none"}
                            )
                        ]
                    )
                ]
            )
        ]
    )


# =======================
# LAYOUT - NEW DESIGN
# =======================



# =======================
# MODAL BUILDERS
# =======================

def build_config_modal(param_id, title):
    """Build configuration modal for parameter statistical analysis"""
    return html.Div(
        id=f"modal-overlay-{param_id}",
        style={"display": "none"},
        className="modal-overlay",
        children=[
            html.Div(
                className="config-modal",
                children=[
                    html.Div(
                        className="config-modal-header",
                        children=[
                            html.H3(title, className="config-modal-title")
                        ]
                    ),
                    html.Div(
                        className="config-modal-body",
                        children=[
                            html.H6("Current available time", style={"marginBottom": "1rem"}),
                            
                            dbc.Row(
                                [
                                    dbc.Col(html.Div("From"), width="auto"),
                                    dbc.Col(
                                        dbc.Input(
                                            id=f"{param_id}-stats-from",
                                            type="text",
                                            disabled=True,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(html.Div("To"), width="auto"),
                                    dbc.Col(
                                        dbc.Input(
                                            id=f"{param_id}-stats-to",
                                            type="text",
                                            disabled=True,
                                        ),
                                        width=4,
                                    ),
                                ],
                                style={"marginBottom": "1rem"},
                            ),
                            
                            html.H6("Date for analysis", style={"marginBottom": "0.5rem"}),
                            
                            dcc.DatePickerSingle(
                                id=f"{param_id}-stats-date",
                                display_format="DD/MM/YYYY",
                                style={"marginBottom": "1.5rem"},
                            ),
                            
                            html.Div(
                                style={"display": "flex", "gap": "1rem", "justifyContent": "flex-end"},
                                children=[
                                    html.Button(
                                        "Cancel",
                                        id=f"btn-cancel-{param_id}",
                                        className="btn-close-modal",
                                        n_clicks=0
                                    ),
                                    html.Button(
                                        "Generate plot",
                                        id=f"{param_id}-stats-generate",
                                        className="btn-download-modal",
                                        n_clicks=0
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def build_corr2d_modal():
    """Build 2D correlation configuration modal"""
    return html.Div(
        id="modal-overlay-corr2d",
        style={"display": "none"},
        className="modal-overlay",
        children=[
            html.Div(
                className="config-modal",
                children=[
                    html.Div(
                        className="config-modal-header",
                        children=[html.H3("2D Correlation – Scatter Plot", className="config-modal-title")]
                    ),
                    html.Div(
                        className="config-modal-body",
                        children=[
                            html.H6("Select date range", style={"marginBottom": "1rem"}),
                            dcc.DatePickerRange(
                                id="corr2d-range",
                                display_format="DD/MM/YYYY",
                                style={"marginBottom": "1.5rem"}
                            ),
                            
                            html.H6("Select X axis", style={"marginBottom": "0.5rem"}),
                            dcc.Dropdown(id="corr2d-x", placeholder="Choose X variable", style={"marginBottom": "1rem"}),
                            
                            html.H6("Select Y axis", style={"marginBottom": "0.5rem"}),
                            dcc.Dropdown(id="corr2d-y", placeholder="Choose Y variable", style={"marginBottom": "1rem"}),
                            
                            dbc.Checklist(
                                id="corr2d-enable-color",
                                options=[{"label": " Enable color mapping", "value": True}],
                                value=[],
                                style={"marginBottom": "1rem"}
                            ),
                            
                            dcc.Dropdown(id="corr2d-color", placeholder="Choose color variable", disabled=True, style={"marginBottom": "1.5rem"}),
                            
                            html.Div(
                                style={"display": "flex", "gap": "1rem", "justifyContent": "flex-end"},
                                children=[
                                    html.Button("Cancel", id="btn-cancel-corr2d", className="btn-close-modal", n_clicks=0),
                                    html.Button("Generate plot", id="corr2d-generate", className="btn-download-modal", n_clicks=0),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def build_corr3d_modal():
    """Build 3D correlation configuration modal"""
    return html.Div(
        id="modal-overlay-corr3d",
        style={"display": "none"},
        className="modal-overlay",
        children=[
            html.Div(
                className="config-modal",
                children=[
                    html.Div(
                        className="config-modal-header",
                        children=[html.H3("3D Correlation – Scatter Plot", className="config-modal-title")]
                    ),
                    html.Div(
                        className="config-modal-body",
                        children=[
                            html.H6("Select date range", style={"marginBottom": "1rem"}),
                            dcc.DatePickerRange(
                                id="corr3d-range",
                                display_format="DD/MM/YYYY",
                                style={"marginBottom": "1.5rem"}
                            ),
                            
                            html.H6("Select X axis", style={"marginBottom": "0.5rem"}),
                            dcc.Dropdown(id="corr3d-x", placeholder="Choose X variable", style={"marginBottom": "1rem"}),
                            
                            html.H6("Select Y axis", style={"marginBottom": "0.5rem"}),
                            dcc.Dropdown(id="corr3d-y", placeholder="Choose Y variable", style={"marginBottom": "1rem"}),
                            
                            html.H6("Select Z axis", style={"marginBottom": "0.5rem"}),
                            dcc.Dropdown(id="corr3d-z", placeholder="Choose Z variable", style={"marginBottom": "1rem"}),
                            
                            dbc.Checklist(
                                id="corr3d-enable-color",
                                options=[{"label": " Enable color mapping", "value": True}],
                                value=[],
                                style={"marginBottom": "1rem"}
                            ),
                            
                            dcc.Dropdown(id="corr3d-color", placeholder="Choose color variable", disabled=True, style={"marginBottom": "1.5rem"}),
                            
                            html.Div(
                                style={"display": "flex", "gap": "1rem", "justifyContent": "flex-end"},
                                children=[
                                    html.Button("Cancel", id="btn-cancel-corr3d", className="btn-close-modal", n_clicks=0),
                                    html.Button("Generate plot", id="corr3d-generate", className="btn-download-modal", n_clicks=0),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def build_timeseries_modal():
    """Build time series configuration modal"""
    return html.Div(
        id="modal-overlay-timeseries",
        style={"display": "none"},
        className="modal-overlay",
        children=[
            html.Div(
                className="config-modal",
                children=[
                    html.Div(
                        className="config-modal-header",
                        children=[html.H3("Time Series Analysis", className="config-modal-title")]
                    ),
                    html.Div(
                        className="config-modal-body",
                        children=[
                            html.H6("Select parameter to analyze", style={"marginBottom": "0.5rem"}),
                            dcc.Dropdown(
                                id="timeseries-param",
                                placeholder="Choose parameter",
                                style={"marginBottom": "1.5rem"}
                            ),
                            
                            html.Div(
                                style={"display": "flex", "gap": "1rem", "justifyContent": "flex-end"},
                                children=[
                                    html.Button("Cancel", id="btn-cancel-timeseries", className="btn-close-modal", n_clicks=0),
                                    html.Button("Generate plot", id="timeseries-generate", className="btn-download-modal", n_clicks=0),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


# =======================
# CALLBACKS - PART 1: MODAL TOGGLES
# =======================

# Toggle config modals (airflow, froth, power, recovery)


layout = html.Div([
    # Data stores
    dcc.Store(id='current-viz-data', storage_type='memory'),  # Stores current graph data
    dcc.Store(id='pdf-charts-list', storage_type='memory', data=[]),  # List of charts for PDF
    
    # Hero image section (similar to home page)
    html.Div(
        className="hero-carousel",
        style={"height": "250px", "marginBottom": "2rem"},
        children=[
            html.Div(
                className="hero-slide",
                children=[
                    html.Img(src="/assets/flotation-machines-cells.jpg"),
                    html.Div(
                        className="hero-overlay",
                        children=[
                            html.H1("Flotation Cell Analysis", className="hero-title", style={"fontSize": "2.5rem"}),
                            html.P(
                                "Comprehensive process parameter analysis and correlation studies",
                                className="hero-subtitle", style={"fontSize": "1.1rem"}
                            )
                        ]
                    )
                ]
            )
        ]
    ),
    
    # ============================================
    # SECTION 1: REPORT GENERATION BUTTONS
    # ============================================
    html.Div(
        className="section-card reports-section",
        children=[
            html.Div(
                className="section-header",
                children=[
                    html.Span("📊", className="section-icon"),
                    html.H2("Generate Reports", className="section-title")
                ]
            ),
            html.Div(
                className="reports-buttons-grid",
                children=[
                    html.Button("Feed", id="btn-feed-report", className="report-button", n_clicks=0),
                    html.Button("Reagents", id="btn-reagents-report", className="report-button", n_clicks=0),
                    html.Button("Mineral Type", id="btn-mineral-report", className="report-button", n_clicks=0),
                    html.Button("Concentrate", id="btn-concentrate-report", className="report-button", n_clicks=0),
                    html.Button("Tails", id="btn-tails-report", className="report-button", n_clicks=0),
                ]
            )
        ]
    ),
    
    # ============================================
    # SECTION 2: PROCESS PARAMETERS
    # ============================================
    html.Div(
        className="section-card",
        children=[
            html.Div(
                className="section-header",
                children=[
                    html.Span("⚙️", className="section-icon"),
                    html.H2("Process Parameters", className="section-title")
                ]
            ),
            
            # Grid layout: Controls (left) + Visualization (right)
            html.Div(
                className="process-parameters-container",
                children=[
                    # Left: Parameter controls (Bloque 1)
                    html.Div(
                        className="parameters-controls",
                        children=[
                            # Airflow
                            html.Div(
                                className="parameter-item",
                                children=[
                                    html.Label("Airflow Nm³/h", className="parameter-label"),
                                    html.Div(
                                        className="parameter-buttons",
                                        children=[
                                            html.Button("Raw Data", id="btn-airflow-raw", className="param-btn", n_clicks=0),
                                            html.Button("Statistical Analysis", id="btn-airflow-stats", className="param-btn", n_clicks=0),
                                        ]
                                    )
                                ]
                            ),
                            
                            # Froth Depth
                            html.Div(
                                className="parameter-item",
                                children=[
                                    html.Label("Froth Depth mm", className="parameter-label"),
                                    html.Div(
                                        className="parameter-buttons",
                                        children=[
                                            html.Button("Raw Data", id="btn-froth-raw", className="param-btn", n_clicks=0),
                                            html.Button("Statistical Analysis", id="btn-froth-stats", className="param-btn", n_clicks=0),
                                        ]
                                    )
                                ]
                            ),
                            
                            # Cell Power
                            html.Div(
                                className="parameter-item",
                                children=[
                                    html.Label("Cell Power kW", className="parameter-label"),
                                    html.Div(
                                        className="parameter-buttons",
                                        children=[
                                            html.Button("Raw Data", id="btn-power-raw", className="param-btn", n_clicks=0),
                                            html.Button("Statistical Analysis", id="btn-power-stats", className="param-btn", n_clicks=0),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                    
                    # Right: Visualization panel (Bloque 2)
                    html.Div(
                        id="visualization-panel",
                        className="visualization-panel",
                        children=[
                            html.Div(
                                className="viz-panel-empty",
                                children=[
                                    html.Div("📈", className="viz-panel-empty-icon"),
                                    html.Div("First, generate a graphic.", className="viz-panel-empty-text")
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    ),
    
    # ============================================
    # SECTION 3: RECOVERY STATISTICAL ANALYSIS
    # ============================================
    html.Div(
        className="section-card recovery-section",
        children=[
            html.Div(
                className="recovery-header",
                children=[
                    html.H3("Recovery Statistical Analysis", className="recovery-title"),
                    html.Div(
                        className="recovery-buttons",
                        children=[
                            html.Button("View", id="btn-recovery-view", className="btn-view", n_clicks=0),
                            html.Button("Add to PDF", id="btn-recovery-pdf", className="btn-add-pdf-recovery", n_clicks=0),
                        ]
                    )
                ]
            )
        ]
    ),
    
    # ============================================
    # SECTION 4: CORRELATION ANALYSIS CARDS
    # ============================================
    html.Div(
        className="section-card",
        children=[
            html.Div(
                className="section-header",
                children=[
                    html.Span("🔍", className="section-icon"),
                    html.H2("Correlation Analysis", className="section-title")
                ]
            ),
            html.Div(
                className="correlation-cards-grid",
                children=[
                    html.Div(
                        id="card-2d-corr",
                        className="correlation-card",
                        n_clicks=0,
                        children=[
                            html.Div("📊", className="correlation-card-icon"),
                            html.H3("2D Correlation – Scatter plot", className="correlation-card-title")
                        ]
                    ),
                    html.Div(
                        id="card-3d-corr",
                        className="correlation-card",
                        n_clicks=0,
                        children=[
                            html.Div("🎲", className="correlation-card-icon"),
                            html.H3("3D Correlation – Scatter plot", className="correlation-card-title")
                        ]
                    ),
                    html.Div(
                        id="card-timeseries",
                        className="correlation-card",
                        n_clicks=0,
                        children=[
                            html.Div("📈", className="correlation-card-icon"),
                            html.H3("TIME SERIES ANALYSIS", className="correlation-card-title")
                        ]
                    ),
                ]
            )
        ]
    ),
    
    # ============================================
    # SECTION 5: GENERATE PDF BUTTON
    # ============================================
    html.Div(
        className="pdf-generate-section",
        children=[
            html.Button(
                id="btn-generate-pdf",
                className="btn-generate-pdf",
                disabled=True,
                children=[
                    "Generate PDF Report",
                    html.Span("0", id="pdf-counter", className="pdf-counter")
                ]
            ),
            dcc.Download(id="download-pdf-report")
        ]
    ),
    
    # ============================================
    # MODALS - Configuration modals for parameters
    # ============================================
    build_config_modal("airflow", "Airflow – Before vs After statistical analysis"),
    build_config_modal("froth", "Froth Depth – Before vs After statistical analysis"),
    build_config_modal("power", "Cell Power – Before vs After statistical analysis"),
    build_config_modal("recovery", "Recovery – Before vs After statistical analysis"),
    build_corr2d_modal(),
    build_corr3d_modal(),
    build_timeseries_modal(),
    
    # ============================================
    # LARGE MODALS - For viewing reports/graphs
    # ============================================
    html.Div(id="report-modal-container"),  # Container for report modals
    html.Div(id="graph-expand-modal-container"),  # Container for expanded graph modal
])



@callback(
    Output("modal-overlay-airflow", "style"),
    Input("btn-airflow-stats", "n_clicks"),
    Input("btn-cancel-airflow", "n_clicks"),
    Input("airflow-stats-generate", "n_clicks"),
    State("modal-overlay-airflow", "style"),
    prevent_initial_call=True
)
def toggle_airflow_modal(open_clicks, cancel_clicks, generate_clicks, current_style):
    trigger = ctx.triggered_id
    if trigger == "btn-airflow-stats":
        return {"display": "flex"}
    else:  # cancel or generate
        return {"display": "none"}


@callback(
    Output("modal-overlay-froth", "style"),
    Input("btn-froth-stats", "n_clicks"),
    Input("btn-cancel-froth", "n_clicks"),
    Input("froth-stats-generate", "n_clicks"),
    State("modal-overlay-froth", "style"),
    prevent_initial_call=True
)
def toggle_froth_modal(open_clicks, cancel_clicks, generate_clicks, current_style):
    trigger = ctx.triggered_id
    if trigger == "btn-froth-stats":
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("modal-overlay-power", "style"),
    Input("btn-power-stats", "n_clicks"),
    Input("btn-cancel-power", "n_clicks"),
    Input("power-stats-generate", "n_clicks"),
    State("modal-overlay-power", "style"),
    prevent_initial_call=True
)
def toggle_power_modal(open_clicks, cancel_clicks, generate_clicks, current_style):
    trigger = ctx.triggered_id
    if trigger == "btn-power-stats":
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("modal-overlay-recovery", "style"),
    Input("btn-recovery-view", "n_clicks"),
    Input("btn-recovery-pdf", "n_clicks"),
    Input("btn-cancel-recovery", "n_clicks"),
    Input("recovery-stats-generate", "n_clicks"),
    State("modal-overlay-recovery", "style"),
    prevent_initial_call=True
)
def toggle_recovery_modal(view_clicks, pdf_clicks, cancel_clicks, generate_clicks, current_style):
    trigger = ctx.triggered_id
    if trigger in ["btn-recovery-view", "btn-recovery-pdf"]:
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("modal-overlay-corr2d", "style"),
    Input("card-2d-corr", "n_clicks"),
    Input("btn-cancel-corr2d", "n_clicks"),
    Input("corr2d-generate", "n_clicks"),
    prevent_initial_call=True
)
def toggle_corr2d_modal(card_clicks, cancel_clicks, generate_clicks):
    trigger = ctx.triggered_id
    if trigger == "card-2d-corr":
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("modal-overlay-corr3d", "style"),
    Input("card-3d-corr", "n_clicks"),
    Input("btn-cancel-corr3d", "n_clicks"),
    Input("corr3d-generate", "n_clicks"),
    prevent_initial_call=True
)
def toggle_corr3d_modal(card_clicks, cancel_clicks, generate_clicks):
    trigger = ctx.triggered_id
    if trigger == "card-3d-corr":
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("modal-overlay-timeseries", "style"),
    Input("card-timeseries", "n_clicks"),
    Input("btn-cancel-timeseries", "n_clicks"),
    Input("timeseries-generate", "n_clicks"),
    prevent_initial_call=True
)
def toggle_timeseries_modal(card_clicks, cancel_clicks, generate_clicks):
    trigger = ctx.triggered_id
    if trigger == "card-timeseries":
        return {"display": "flex"}
    else:
        return {"display": "none"}


# =======================
# CALLBACKS - PART 2: FILL MODAL DATA
# =======================

@callback(
    Output("airflow-stats-from", "value"),
    Output("airflow-stats-to", "value"),
    Output("airflow-stats-date", "min_date_allowed"),
    Output("airflow-stats-date", "max_date_allowed"),
    Output("airflow-stats-date", "initial_visible_month"),
    Input("modal-overlay-airflow", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_airflow_dates(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = df_flat.columns[0]
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    
    min_date = df_flat[time_col].min()
    max_date = df_flat[time_col].max()
    
    return (
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
        min_date.date(),
        max_date.date(),
        max_date.date()
    )


@callback(
    Output("froth-stats-from", "value"),
    Output("froth-stats-to", "value"),
    Output("froth-stats-date", "min_date_allowed"),
    Output("froth-stats-date", "max_date_allowed"),
    Output("froth-stats-date", "initial_visible_month"),
    Input("modal-overlay-froth", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_froth_dates(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = df_flat.columns[0]
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    
    min_date = df_flat[time_col].min()
    max_date = df_flat[time_col].max()
    
    return (
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
        min_date.date(),
        max_date.date(),
        max_date.date()
    )


@callback(
    Output("power-stats-from", "value"),
    Output("power-stats-to", "value"),
    Output("power-stats-date", "min_date_allowed"),
    Output("power-stats-date", "max_date_allowed"),
    Output("power-stats-date", "initial_visible_month"),
    Input("modal-overlay-power", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_power_dates(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = df_flat.columns[0]
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    
    min_date = df_flat[time_col].min()
    max_date = df_flat[time_col].max()
    
    return (
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
        min_date.date(),
        max_date.date(),
        max_date.date()
    )


@callback(
    Output("recovery-stats-from", "value"),
    Output("recovery-stats-to", "value"),
    Output("recovery-stats-date", "min_date_allowed"),
    Output("recovery-stats-date", "max_date_allowed"),
    Output("recovery-stats-date", "initial_visible_month"),
    Input("modal-overlay-recovery", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_recovery_dates(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = df_flat.columns[0]
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    
    min_date = df_flat[time_col].min()
    max_date = df_flat[time_col].max()
    
    return (
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
        min_date.date(),
        max_date.date(),
        max_date.date()
    )


@callback(
    Output("corr2d-range", "start_date"),
    Output("corr2d-range", "end_date"),
    Output("corr2d-x", "options"),
    Output("corr2d-y", "options"),
    Output("corr2d-color", "options"),
    Input("modal-overlay-corr2d", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_corr2d_data(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = df_flat.columns[0]
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    
    min_date = df_flat[time_col].min().date()
    max_date = df_flat[time_col].max().date()
    
    numeric_cols = [col for col in df_flat.columns if pd.api.types.is_numeric_dtype(df_flat[col])]
    options = [{"label": col.split("__")[1] if "__" in col else col, "value": col} for col in numeric_cols]
    
    return min_date, max_date, options, options, options


@callback(
    Output("corr2d-color", "disabled"),
    Input("corr2d-enable-color", "value"),
    prevent_initial_call=True
)
def toggle_corr2d_color(enabled):
    return len(enabled) == 0


@callback(
    Output("corr3d-range", "start_date"),
    Output("corr3d-range", "end_date"),
    Output("corr3d-x", "options"),
    Output("corr3d-y", "options"),
    Output("corr3d-z", "options"),
    Output("corr3d-color", "options"),
    Input("modal-overlay-corr3d", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_corr3d_data(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = df_flat.columns[0]
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    
    min_date = df_flat[time_col].min().date()
    max_date = df_flat[time_col].max().date()
    
    numeric_cols = [col for col in df_flat.columns if pd.api.types.is_numeric_dtype(df_flat[col])]
    options = [{"label": col.split("__")[1] if "__" in col else col, "value": col} for col in numeric_cols]
    
    return min_date, max_date, options, options, options, options


@callback(
    Output("corr3d-color", "disabled"),
    Input("corr3d-enable-color", "value"),
    prevent_initial_call=True
)
def toggle_corr3d_color(enabled):
    return len(enabled) == 0


@callback(
    Output("timeseries-param", "options"),
    Input("modal-overlay-timeseries", "style"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def fill_timeseries_data(modal_style, stored_json):
    if modal_style.get("display") == "none" or stored_json is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    numeric_cols = [col for col in df_flat.columns if pd.api.types.is_numeric_dtype(df_flat[col])]
    options = [{"label": col.split("__")[1] if "__" in col else col, "value": col} for col in numeric_cols]
    
    return options


# =======================
# CALLBACKS - PART 3: GENERATE GRAPHS & UPDATE VISUALIZATION PANEL
# =======================

@callback(
    Output("visualization-panel", "children"),
    Output("current-viz-data", "data"),
    Output("btn-airflow-raw", "className"),
    Output("btn-airflow-stats", "className"),
    Output("btn-froth-raw", "className"),
    Output("btn-froth-stats", "className"),
    Output("btn-power-raw", "className"),
    Output("btn-power-stats", "className"),
    Input("btn-airflow-raw", "n_clicks"),
    Input("btn-airflow-stats", "n_clicks"),
    Input("airflow-stats-generate", "n_clicks"),
    Input("btn-froth-raw", "n_clicks"),
    Input("btn-froth-stats", "n_clicks"),
    Input("froth-stats-generate", "n_clicks"),
    Input("btn-power-raw", "n_clicks"),
    Input("btn-power-stats", "n_clicks"),
    Input("power-stats-generate", "n_clicks"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    State("airflow-stats-date", "date"),
    State("froth-stats-date", "date"),
    State("power-stats-date", "date"),
    State("pdf-charts-list", "data"),
    prevent_initial_call=True
)
def update_visualization_panel(
    airflow_raw_clicks, airflow_stats_clicks, airflow_gen_clicks,
    froth_raw_clicks, froth_stats_clicks, froth_gen_clicks,
    power_raw_clicks, power_stats_clicks, power_gen_clicks,
    stored_data, group_map, flot_config,
    airflow_date, froth_date, power_date,
    pdf_list
):
    trigger = ctx.triggered_id
    
    if not trigger or stored_data is None or flot_config is None:
        raise PreventUpdate
    
    # Reset all button classes
    classes = ["param-btn"] * 6
    
    # Determine which parameter and type
    param_info = None
    
    if trigger == "btn-airflow-raw":
        param_info = {
            "type": "raw",
            "param": "airflow",
            "title": "Airflow Nm³/h",
            "col_name": "Airflow",
            "unit": "Nm³/h"
        }
        classes[0] = "param-btn active"
        
    elif trigger in ["btn-airflow-stats", "airflow-stats-generate"]:
        if airflow_date is None:
            raise PreventUpdate
        param_info = {
            "type": "stats",
            "param": "airflow",
            "title": "Airflow Nm³/h",
            "col_name": "Airflow",
            "unit": "Nm³/h",
            "date": airflow_date
        }
        classes[1] = "param-btn active"
        
    elif trigger == "btn-froth-raw":
        param_info = {
            "type": "raw",
            "param": "froth",
            "title": "Froth Depth mm",
            "col_name": "Froth Depth",
            "unit": "mm"
        }
        classes[2] = "param-btn active"
        
    elif trigger in ["btn-froth-stats", "froth-stats-generate"]:
        if froth_date is None:
            raise PreventUpdate
        param_info = {
            "type": "stats",
            "param": "froth",
            "title": "Froth Depth mm",
            "col_name": "Froth Depth",
            "unit": "mm",
            "date": froth_date
        }
        classes[3] = "param-btn active"
        
    elif trigger == "btn-power-raw":
        param_info = {
            "type": "raw",
            "param": "power",
            "title": "Cell Power kW",
            "col_name": "Power Motor",
            "unit": "kW"
        }
        classes[4] = "param-btn active"
        
    elif trigger in ["btn-power-stats", "power-stats-generate"]:
        if power_date is None:
            raise PreventUpdate
        param_info = {
            "type": "stats",
            "param": "power",
            "title": "Cell Power kW",
            "col_name": "Power Motor",
            "unit": "kW",
            "date": power_date
        }
        classes[5] = "param-btn active"
    
    if param_info is None:
        raise PreventUpdate
    
    # Generate the graph
    try:
        if param_info["type"] == "raw":
            html_path = _generate_param_raw_report(
                1, stored_data, group_map, flot_config,
                param_info["col_name"], param_info["unit"], param_info["title"]
            )
        else:  # stats
            html_path = _generate_param_BA_stats_report(
                1, param_info["date"], stored_data, group_map, flot_config,
                param_info["col_name"], param_info["unit"], param_info["title"]
            )
        
        # Check if this chart is in PDF list
        chart_id = f"{param_info['param']}_{param_info['type']}"
        is_in_pdf = any(c["id"] == chart_id for c in pdf_list)
        
        # Create visualization panel with graph
        viz_panel = html.Div([
            # Header with title and actions
            html.Div(
                className="viz-panel-header",
                children=[
                    html.Div(
                        f"{param_info['title']} – {'Raw Data' if param_info['type'] == 'raw' else 'Statistical Analysis'}",
                        className="viz-panel-title"
                    ),
                    html.Div(
                        className="viz-panel-actions",
                        children=[
                            html.Button("🔍 Expand", id="btn-expand-graph", className="btn-expand", n_clicks=0),
                            html.Button(
                                "Remove from PDF" if is_in_pdf else "Add to PDF",
                                id={"type": "btn-pdf-toggle", "index": chart_id},
                                className="btn-remove-pdf" if is_in_pdf else "btn-add-pdf",
                                n_clicks=0
                            ),
                        ]
                    )
                ]
            ),
            # Graph iframe
            html.Div(
                className="viz-panel-graph",
                children=[
                    html.Iframe(
                        srcDoc=open(html_path, 'r', encoding='utf-8').read() if os.path.exists(html_path) else "",
                        style={
                            "width": "100%",
                            "height": "450px",
                            "border": "none",
                            "borderRadius": "8px"
                        }
                    )
                ]
            )
        ])
        
        # Store current viz data
        viz_data = {
            "html_path": html_path,
            "chart_id": chart_id,
            "title": f"{param_info['title']} – {'Raw Data' if param_info['type'] == 'raw' else 'Statistical Analysis'}"
        }
        
        return viz_panel, viz_data, *classes
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        raise PreventUpdate


# =======================
# CALLBACKS - PART 4: ADD/REMOVE TO PDF
# =======================

@callback(
    Output("pdf-charts-list", "data"),
    Output("btn-generate-pdf", "disabled"),
    Output("pdf-counter", "children"),
    Input({"type": "btn-pdf-toggle", "index": ALL}, "n_clicks"),
    State({"type": "btn-pdf-toggle", "index": ALL}, "id"),
    State("pdf-charts-list", "data"),
    State("current-viz-data", "data"),
    prevent_initial_call=True
)
def toggle_chart_in_pdf(n_clicks_list, button_ids, pdf_list, current_viz):
    if not ctx.triggered or current_viz is None:
        raise PreventUpdate
    
    # Find which button was clicked
    trigger_id = ctx.triggered_id
    chart_id = trigger_id["index"]
    
    # Check if chart is already in list
    is_in_list = any(c["id"] == chart_id for c in pdf_list)
    
    if is_in_list:
        # Remove from list
        pdf_list = [c for c in pdf_list if c["id"] != chart_id]
    else:
        # Add to list
        pdf_list.append({
            "id": chart_id,
            "html_path": current_viz["html_path"],
            "title": current_viz["title"]
        })
    
    # Update button state
    is_disabled = len(pdf_list) == 0
    counter_text = str(len(pdf_list))
    
    return pdf_list, is_disabled, counter_text


@callback(
    Output("btn-recovery-pdf", "className"),
    Input("btn-recovery-pdf", "n_clicks"),
    State("pdf-charts-list", "data"),
    prevent_initial_call=True
)
def update_recovery_pdf_button(n_clicks, pdf_list):
    # Check if recovery is in PDF list
    is_in_pdf = any(c["id"] == "recovery_stats" for c in pdf_list)
    return "btn-remove-pdf-recovery" if is_in_pdf else "btn-add-pdf-recovery"


# =======================
# CALLBACKS - PART 5: GENERATE SWEETVIZ REPORTS
# =======================

@callback(
    Output("report-modal-container", "children"),
    Input("btn-feed-report", "n_clicks"),
    Input("btn-reagents-report", "n_clicks"),
    Input("btn-mineral-report", "n_clicks"),
    Input("btn-concentrate-report", "n_clicks"),
    Input("btn-tails-report", "n_clicks"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    prevent_initial_call=True
)
def show_report_modal(feed_clicks, reagents_clicks, mineral_clicks, concentrate_clicks, tails_clicks, stored_data, group_map):
    trigger = ctx.triggered_id
    
    if not trigger or stored_data is None:
        raise PreventUpdate
    
    # Determine which report to generate
    report_map = {
        "btn-feed-report": "Feed",
        "btn-reagents-report": "Reagents",
        "btn-mineral-report": "Mineral type",
        "btn-concentrate-report": "Concentrate",
        "btn-tails-report": "Tails"
    }
    
    report_name = report_map.get(trigger)
    if not report_name:
        raise PreventUpdate
    
    try:
        # Generate report
        html_path = generate_sweetviz_report(1, stored_data, group_map, report_name)
        
        # Read HTML content directly (no base64 encoding)
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create modal with iframe using srcDoc (better for large files like Sweetviz reports)
        modal = html.Div(
            className="modal-overlay",
            children=[
                html.Div(
                    className="large-modal",
                    children=[
                        html.Div(
                            className="large-modal-header",
                            children=[
                                html.H3(f"{report_name} Report", className="large-modal-title"),
                                html.Div(
                                    className="large-modal-actions",
                                    children=[
                                        html.Button(
                                            "Download", 
                                            id={"type": "btn-download-report", "index": report_name},
                                            className="btn-download-modal",
                                            n_clicks=0
                                        ),
                                        html.Button("✕", id="btn-close-report-modal", className="btn-close-modal", n_clicks=0),
                                        dcc.Download(id={"type": "download-report-file", "index": report_name}),
                                        # Store HTML path for download
                                        dcc.Store(id={"type": "report-path-store", "index": report_name}, data=html_path)
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className="large-modal-body",
                            children=[
                                html.Iframe(
                                    srcDoc=html_content,  # Using srcDoc instead of base64 src
                                    className="iframe-container",
                                    style={"width": "100%", "height": "700px", "border": "none"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        
        return modal
        
    except Exception as e:
        print(f"Error generating report: {e}")
        raise PreventUpdate


@callback(
    Output("report-modal-container", "children", allow_duplicate=True),
    Input("btn-close-report-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_report_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None


@callback(
    Output({"type": "download-report-file", "index": ALL}, "data"),
    Input({"type": "btn-download-report", "index": ALL}, "n_clicks"),
    State({"type": "report-path-store", "index": ALL}, "data"),
    State({"type": "btn-download-report", "index": ALL}, "id"),
    prevent_initial_call=True
)
def download_report(n_clicks_list, paths_list, button_ids):
    if not ctx.triggered:
        raise PreventUpdate
    
    # Find which button was clicked
    trigger_id = ctx.triggered_id
    if not trigger_id or not trigger_id.get("index"):
        raise PreventUpdate
    
    report_name = trigger_id["index"]
    
    # Find corresponding path
    for i, btn_id in enumerate(button_ids):
        if btn_id["index"] == report_name and n_clicks_list[i]:
            html_path = paths_list[i]
            if html_path and os.path.exists(html_path):
                return [send_file(html_path) if j == i else no_update for j in range(len(button_ids))]
    
    raise PreventUpdate


# =======================
# CALLBACKS - PART 6: EXPAND GRAPH MODAL
# =======================

@callback(
    Output("graph-expand-modal-container", "children"),
    Input("btn-expand-graph", "n_clicks"),
    State("current-viz-data", "data"),
    prevent_initial_call=True
)
def show_expanded_graph(n_clicks, current_viz):
    if not n_clicks or current_viz is None:
        raise PreventUpdate
    
    html_path = current_viz['html_path']
    html_content = open(html_path, 'r', encoding='utf-8').read() if os.path.exists(html_path) else ""
    
    modal = html.Div(
        className="modal-overlay",
        children=[
            html.Div(
                className="large-modal",
                children=[
                    html.Div(
                        className="large-modal-header",
                        children=[
                            html.H3(current_viz["title"], className="large-modal-title"),
                            html.Div(
                                className="large-modal-actions",
                                children=[
                                    html.Button(
                                        "Download",
                                        id="btn-download-expanded-graph",
                                        className="btn-download-modal",
                                        n_clicks=0
                                    ),
                                    html.Button("✕", id="btn-close-expand-modal", className="btn-close-modal", n_clicks=0),
                                    dcc.Download(id="download-expanded-graph-file"),
                                    dcc.Store(id="expanded-graph-path-store", data=html_path)
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="large-modal-body",
                        children=[
                            html.Iframe(
                                srcDoc=html_content,
                                style={"width": "100%", "height": "700px", "border": "none"}
                            )
                        ]
                    )
                ]
            )
        ]
    )
    
    return modal


@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Input("btn-close-expand-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_expanded_graph(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None


@callback(
    Output("download-expanded-graph-file", "data"),
    Input("btn-download-expanded-graph", "n_clicks"),
    State("expanded-graph-path-store", "data"),
    prevent_initial_call=True
)
def download_expanded_graph(n_clicks, html_path):
    if not n_clicks or not html_path:
        raise PreventUpdate
    if os.path.exists(html_path):
        return send_file(html_path)
    raise PreventUpdate


# =======================
# CALLBACKS - PART 7: CORRELATION GRAPHS
# =======================

@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Input("corr2d-generate", "n_clicks"),
    State("corr2d-range", "start_date"),
    State("corr2d-range", "end_date"),
    State("corr2d-x", "value"),
    State("corr2d-y", "value"),
    State("corr2d-enable-color", "value"),
    State("corr2d-color", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def generate_2d_corr_graph(n_clicks, start_date, end_date, x_col, y_col, enable_color, color_col, stored_json):
    if not n_clicks or stored_json is None or x_col is None or y_col is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    
    if time_col not in df_flat.columns:
        raise PreventUpdate
    
    time_series = pd.to_datetime(df_flat[time_col])
    
    # Filter by date range
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (time_series >= start) & (time_series <= end)
        df_sel = df_flat.loc[mask].copy()
    else:
        df_sel = df_flat.copy()
    
    if df_sel.empty:
        raise PreventUpdate
    
    # Create nice labels
    def _nice_label(col):
        return col.split("__", 1)[1] if "__" in col else col
    
    x_label = _nice_label(x_col)
    y_label = _nice_label(y_col)
    
    use_color = bool(enable_color) and color_col is not None
    
    if use_color:
        fig = px.scatter(df_sel, x=x_col, y=y_col, color=color_col, color_continuous_scale="Plasma")
    else:
        fig = px.scatter(df_sel, x=x_col, y=y_col)
    
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        title="2D CORRELATION – SCATTER PLOT",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="simple_white",
        height=700,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    
    # Save to temp file
    tmp_dir = tempfile.gettempdir()
    filename = "Correlation_2D_Scatter.html"
    path = os.path.join(tmp_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    
    # Create modal
    modal = html.Div(
        className="modal-overlay",
        children=[
            html.Div(
                className="large-modal",
                children=[
                    html.Div(
                        className="large-modal-header",
                        children=[
                            html.H3("2D Correlation – Scatter Plot", className="large-modal-title"),
                            html.Div(
                                className="large-modal-actions",
                                children=[
                                    html.A(
                                        html.Button("Download", className="btn-download-modal"),
                                        href=f"file://{path}",
                                        download="2D_Correlation.html"
                                    ),
                                    html.Button("✕", id="btn-close-corr-modal", className="btn-close-modal", n_clicks=0)
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="large-modal-body",
                        children=[
                            html.Iframe(
                                srcDoc=open(path, "r", encoding="utf-8").read() if os.path.exists(path) else "",
                                style={"width": "100%", "height": "700px", "border": "none"}
                            )
                        ]
                    )
                ]
            )
        ]
    )
    
    return modal


@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Input("corr3d-generate", "n_clicks"),
    State("corr3d-range", "start_date"),
    State("corr3d-range", "end_date"),
    State("corr3d-x", "value"),
    State("corr3d-y", "value"),
    State("corr3d-z", "value"),
    State("corr3d-enable-color", "value"),
    State("corr3d-color", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def generate_3d_corr_graph(n_clicks, start_date, end_date, x_col, y_col, z_col, enable_color, color_col, stored_json):
    if not n_clicks or stored_json is None or x_col is None or y_col is None or z_col is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    
    if time_col not in df_flat.columns:
        raise PreventUpdate
    
    time_series = pd.to_datetime(df_flat[time_col])
    
    # Filter by date range
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (time_series >= start) & (time_series <= end)
        df_sel = df_flat.loc[mask].copy()
    else:
        df_sel = df_flat.copy()
    
    if df_sel.empty:
        raise PreventUpdate
    
    def _nice_label(col):
        return col.split("__", 1)[1] if "__" in col else col
    
    x_label = _nice_label(x_col)
    y_label = _nice_label(y_col)
    z_label = _nice_label(z_col)
    
    use_color = bool(enable_color) and color_col is not None
    
    if use_color:
        fig = px.scatter_3d(df_sel, x=x_col, y=y_col, z=z_col, color=color_col, color_continuous_scale="Plasma")
    else:
        fig = px.scatter_3d(df_sel, x=x_col, y=y_col, z=z_col)
    
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        title="3D CORRELATION – SCATTER PLOT",
        scene=dict(
            xaxis=dict(title=x_label, showbackground=True, backgroundcolor="rgba(240, 240, 240, 1)"),
            yaxis=dict(title=y_label, showbackground=True, backgroundcolor="rgba(240, 240, 240, 1)"),
            zaxis=dict(title=z_label, showbackground=True, backgroundcolor="rgba(240, 240, 240, 1)"),
            aspectmode="cube",
        ),
        template="simple_white",
        height=800,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    
    tmp_dir = tempfile.gettempdir()
    filename = "Correlation_3D_Scatter.html"
    path = os.path.join(tmp_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    
    modal = html.Div(
        className="modal-overlay",
        children=[
            html.Div(
                className="large-modal",
                children=[
                    html.Div(
                        className="large-modal-header",
                        children=[
                            html.H3("3D Correlation – Scatter Plot", className="large-modal-title"),
                            html.Div(
                                className="large-modal-actions",
                                children=[
                                    html.A(
                                        html.Button("Download", className="btn-download-modal"),
                                        href=f"file://{path}",
                                        download="3D_Correlation.html"
                                    ),
                                    html.Button("✕", id="btn-close-corr-modal", className="btn-close-modal", n_clicks=0)
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="large-modal-body",
                        children=[
                            html.Iframe(
                                srcDoc=open(path, "r", encoding="utf-8").read() if os.path.exists(path) else "",
                                style={"width": "100%", "height": "750px", "border": "none"}
                            )
                        ]
                    )
                ]
            )
        ]
    )
    
    return modal


@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Input("timeseries-generate", "n_clicks"),
    State("timeseries-param", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def generate_timeseries_graph(n_clicks, param_col, stored_json):
    if not n_clicks or stored_json is None or param_col is None:
        raise PreventUpdate
    
    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    
    if time_col not in df_flat.columns:
        raise PreventUpdate
    
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    df_sel = df_flat[[time_col, param_col]].dropna()
    df_sel = df_sel.sort_values(time_col)
    
    if df_sel.empty:
        raise PreventUpdate
    
    def _nice_label(col):
        return col.split("__", 1)[1] if "__" in col else col
    
    y_label = _nice_label(param_col)
    
    fig = px.line(df_sel, x=time_col, y=param_col)
    fig.update_layout(
        title=f"Time Series – {y_label}",
        xaxis_title="Date / Time",
        yaxis_title=y_label,
        template="simple_white",
        height=600,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    
    tmp_dir = tempfile.gettempdir()
    filename = "TimeSeries_Analysis.html"
    path = os.path.join(tmp_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    
    modal = html.Div(
        className="modal-overlay",
        children=[
            html.Div(
                className="large-modal",
                children=[
                    html.Div(
                        className="large-modal-header",
                        children=[
                            html.H3("Time Series Analysis", className="large-modal-title"),
                            html.Div(
                                className="large-modal-actions",
                                children=[
                                    html.A(
                                        html.Button("Download", className="btn-download-modal"),
                                        href=f"file://{path}",
                                        download="TimeSeries_Analysis.html"
                                    ),
                                    html.Button("✕", id="btn-close-corr-modal", className="btn-close-modal", n_clicks=0)
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="large-modal-body",
                        children=[
                            html.Iframe(
                                srcDoc=open(path, "r", encoding="utf-8").read() if os.path.exists(path) else "",
                                style={"width": "100%", "height": "650px", "border": "none"}
                            )
                        ]
                    )
                ]
            )
        ]
    )
    
    return modal


@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Input("btn-close-corr-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_corr_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None


# =======================
# CALLBACKS - PART 8: RECOVERY VIEW & ADD TO PDF
# =======================

@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Output("pdf-charts-list", "data", allow_duplicate=True),
    Input("btn-recovery-view", "n_clicks"),
    Input("btn-recovery-pdf", "n_clicks"),
    Input("recovery-stats-generate", "n_clicks"),
    State("recovery-stats-date", "date"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    State("pdf-charts-list", "data"),
    prevent_initial_call=True
)
def handle_recovery_actions(view_clicks, pdf_clicks, generate_clicks, recovery_date, stored_data, group_map, flot_config, pdf_list):
    trigger = ctx.triggered_id
    
    if not trigger or stored_data is None or flot_config is None:
        raise PreventUpdate
    
    if trigger == "recovery-stats-generate":
        if recovery_date is None:
            raise PreventUpdate
        
        # Generate recovery stats graph
        html_path = _generate_param_BA_stats_report(
            1, recovery_date, stored_data, group_map, flot_config,
            "%Recovery", "%", "Recovery"
        )
        
        # Check if this was triggered by View or Add to PDF
        # We need to check the previous click context
        # For now, we'll show the modal (View behavior)
        modal = html.Div(
            className="modal-overlay",
            children=[
                html.Div(
                    className="large-modal",
                    children=[
                        html.Div(
                            className="large-modal-header",
                            children=[
                                html.H3("Recovery – Statistical Analysis", className="large-modal-title"),
                                html.Div(
                                    className="large-modal-actions",
                                    children=[
                                        html.A(
                                            html.Button("Download", className="btn-download-modal"),
                                            href=f"file://{html_path}",
                                            download="Recovery_Statistical_Analysis.html"
                                        ),
                                        html.Button("✕", id="btn-close-recovery-modal", className="btn-close-modal", n_clicks=0)
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className="large-modal-body",
                            children=[
                                html.Iframe(
                                    srcDoc=open(html_path, "r", encoding="utf-8").read() if os.path.exists(html_path) else "",
                                    style={"width": "100%", "height": "700px", "border": "none"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        
        return modal, no_update
    
    return no_update, no_update


@callback(
    Output("graph-expand-modal-container", "children", allow_duplicate=True),
    Input("btn-close-recovery-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_recovery_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return None


# =======================
# CALLBACKS - PART 9: GENERATE PDF REPORT
# =======================

@callback(
    Output("download-pdf-report", "data"),
    Input("btn-generate-pdf", "n_clicks"),
    State("pdf-charts-list", "data"),
    prevent_initial_call=True
)
def generate_pdf_report(n_clicks, pdf_list):
    if not n_clicks or not pdf_list:
        raise PreventUpdate
    
    try:
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER
        from datetime import datetime
        
        # Temporary: Return message that PDF generation requires ReportLab
        raise PreventUpdate  # Skip PDF generation for now
        
        # Create PDF
        tmp_dir = tempfile.gettempdir()
        pdf_filename = f"Flotation_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(tmp_dir, pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Add custom title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#1e3a8a',
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Add Metso logo
        logo_path = "/mnt/user-data/uploads/MetsoLogo.png"
        if os.path.exists(logo_path):
            logo = Image(logo_path, width=2*inch, height=0.7*inch)
            story.append(logo)
            story.append(Spacer(1, 0.3*inch))
        
        # Add title
        title = Paragraph("Flotation Cell Analysis Report", title_style)
        story.append(title)
        
        # Add date
        date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}"
        date_para = Paragraph(date_text, styles['Normal'])
        story.append(date_para)
        story.append(Spacer(1, 0.5*inch))
        
        # Add each chart
        for chart in pdf_list:
            # Add chart title
            chart_title = Paragraph(f"<b>{chart['title']}</b>", styles['Heading2'])
            story.append(chart_title)
            story.append(Spacer(1, 0.2*inch))
            
            # Note: Converting HTML charts to PDF images is complex
            # For now, we'll add a placeholder
            note = Paragraph(
                f"Chart: {chart['title']}<br/>Source: {chart['html_path']}<br/><br/>"
                "Note: Interactive charts are best viewed in HTML format. "
                "Download individual charts for full interactivity.",
                styles['Normal']
            )
            story.append(note)
            story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        
        return send_file(pdf_path)
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise PreventUpdate

