import os
import math
import tempfile
import dash
from dash import html, dcc, callback, Input, Output, State, ctx, no_update
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

dash.register_page(__name__, path="/plots")

# =======================
# Layout – Modals helpers
# =======================

def build_airflow_stats_modal():
    return dbc.Modal(
        id="airflow-stats-modal",
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("Airflow – Before vs After statistical analysis")
            ),
            dbc.ModalBody(
                [
                    html.H6("Current available time", className="mb-3"),

                    dbc.Row(
                        [
                            dbc.Col(html.Div("From"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="airflow-stats-from",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("To"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="airflow-stats-to",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.H6("Date for analysis", className="mb-2"),

                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.DatePickerSingle(
                                    id="airflow-stats-date",
                                    display_format="DD/MM/YYYY",
                                    className="w-100",
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-4",
                    ),

                    dbc.Button(
                        "Generate plot",
                        id="airflow-stats-generate",
                        color="primary",
                    ),
                    dcc.Download(id="download-airflow-stats"),
                ]
            ),
        ],
    )

def build_froth_stats_modal():
    return dbc.Modal(
        id="froth-stats-modal",
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("Froth Depth – Before vs After statistical analysis")
            ),
            dbc.ModalBody(
                [
                    html.H6("Current available time", className="mb-3"),

                    dbc.Row(
                        [
                            dbc.Col(html.Div("From"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="froth-stats-from",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("To"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="froth-stats-to",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.H6("Date for analysis", className="mb-2"),

                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.DatePickerSingle(
                                    id="froth-stats-date",
                                    display_format="DD/MM/YYYY",
                                    className="w-100",
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-4",
                    ),

                    dbc.Button(
                        "Generate plot",
                        id="froth-stats-generate",
                        color="primary",
                    ),
                    dcc.Download(id="download-froth-stats"),
                ]
            ),
        ],
    )


def build_power_stats_modal():
    return dbc.Modal(
        id="power-stats-modal",
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("Cell Power – Before vs After statistical analysis")
            ),
            dbc.ModalBody(
                [
                    html.H6("Current available time", className="mb-3"),

                    dbc.Row(
                        [
                            dbc.Col(html.Div("From"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="power-stats-from",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("To"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="power-stats-to",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.H6("Date for analysis", className="mb-2"),

                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.DatePickerSingle(
                                    id="power-stats-date",
                                    display_format="DD/MM/YYYY",
                                    className="w-100",
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-4",
                    ),

                    dbc.Button(
                        "Generate plot",
                        id="power-stats-generate",
                        color="primary",
                    ),
                    dcc.Download(id="download-power-stats"),
                ]
            ),
        ],
    )

def build_recovery_stats_modal():
    return dbc.Modal(
        id="recovery-stats-modal",
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("Recovery – Before vs After statistical analysis")
            ),
            dbc.ModalBody(
                [
                    html.H6("Current available time", className="mb-3"),

                    dbc.Row(
                        [
                            dbc.Col(html.Div("From"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="recovery-stats-from",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("To"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="recovery-stats-to",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.H6("Date for analysis", className="mb-2"),

                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.DatePickerSingle(
                                    id="recovery-stats-date",
                                    display_format="DD/MM/YYYY",
                                    className="w-100",
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-4",
                    ),

                    dbc.Button(
                        "Generate plot",
                        id="recovery-stats-generate",
                        color="primary",
                    ),
                    # descarga del HTML con los histogramas
                    dcc.Download(id="download-recovery-stats"),
                ]
            ),
        ],
    )

def build_2d_corr_modal():
    return dbc.Modal(
        id="corr2d-modal",
        is_open=False,
        size="xl",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("2D Correlation – Scatter plot")
            ),
            dbc.ModalBody(
                [
                    html.H6("Variables", className="mb-3"),

                    # Ejes X e Y
                    dbc.Row(
                        [
                            dbc.Col(html.Div("Axis X"), width="auto"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr2d-x",
                                    placeholder="Select X axis variable",
                                    className="w-100",
                                ),
                                width=4,
                            ),
                            dbc.Col(html.Div("Axis Y"), width="auto"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr2d-y",
                                    placeholder="Select Y axis variable",
                                    className="w-100",
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),

                    # Tercera variable como color map
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Checkbox(
                                    id="corr2d-enable-color",
                                    value=False,
                                    label="Enable a third variable as a color map",
                                ),
                                width=12,
                            )
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr2d-color",
                                    placeholder="Select color map variable",
                                    disabled=True,
                                    className="w-100",
                                ),
                                width=4,
                            )
                        ],
                        className="mb-4",
                    ),

                    html.H6("Date Time", className="mb-2"),

                    html.Div("Current available time", className="mb-2"),
                    dbc.Row(
                        [
                            dbc.Col(html.Div("From"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="corr2d-from",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("To"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="corr2d-to",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.Div("Time range for analysis", className="mb-2"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.DatePickerRange(
                                    id="corr2d-range",
                                    display_format="DD/MM/YYYY",
                                    className="w-100",
                                ),
                                width=6,
                            ),
                        ],
                        className="mb-4",
                    ),

                    dbc.Button(
                        "Generate plot",
                        id="corr2d-generate",
                        color="primary",
                    ),
                    dcc.Download(id="download-2d-corr"),
                ]
            ),
        ],
    )

def build_3d_corr_modal():
    return dbc.Modal(
        id="corr3d-modal",
        is_open=False,
        size="xl",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("3D Correlation – Scatter plot")
            ),
            dbc.ModalBody(
                [
                    html.H6("Variables", className="mb-3"),

                    # Ejes X, Y, Z
                    dbc.Row(
                        [
                            dbc.Col(html.Div("Axis X"), width="auto"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr3d-x",
                                    placeholder="Select X axis variable",
                                    className="w-100",
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("Axis Y"), width="auto"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr3d-y",
                                    placeholder="Select Y axis variable",
                                    className="w-100",
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("Axis Z"), width="auto"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr3d-z",
                                    placeholder="Select Z axis variable",
                                    className="w-100",
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    # Cuarta variable como color map
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Checkbox(
                                    id="corr3d-enable-color",
                                    value=False,
                                    label="Enable a fourth variable as a color map",
                                ),
                                width=12,
                            )
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="corr3d-color",
                                    placeholder="Select color map variable",
                                    disabled=True,
                                    className="w-100",
                                ),
                                width=4,
                            )
                        ],
                        className="mb-4",
                    ),

                    html.H6("Date Time", className="mb-2"),

                    html.Div("Current available time", className="mb-2"),
                    dbc.Row(
                        [
                            dbc.Col(html.Div("From"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="corr3d-from",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                            dbc.Col(html.Div("To"), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="corr3d-to",
                                    type="text",
                                    disabled=True,
                                ),
                                width=3,
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.Div("Time range for analysis", className="mb-2"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.DatePickerRange(
                                    id="corr3d-range",
                                    display_format="DD/MM/YYYY",
                                    className="w-100",
                                ),
                                width=6,
                            ),
                        ],
                        className="mb-4",
                    ),

                    dbc.Button(
                        "Generate plot",
                        id="corr3d-generate",
                        color="primary",
                    ),
                    dcc.Download(id="download-3d-corr"),
                ]
            ),
        ],
    )

def build_timeseries_modal():
    return dbc.Modal(
        id="timeseries-modal",
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle("TIME SERIES ANALYSIS")
            ),
            dbc.ModalBody(
                [
                    html.H6("Parameter", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="timeseries-param",
                                    placeholder="Select parameter",
                                    className="w-100",
                                ),
                                width=6,
                            ),
                        ],
                        className="mb-4",
                    ),
                    dbc.Button(
                        "Generate Plot",
                        id="timeseries-generate",
                        color="primary",
                    ),
                    dcc.Download(id="download-timeseries"),
                ]
            ),
        ],
    )

# =======================
# Layout principal
# =======================

layout = html.Div(
    className="main-container",
    children=[
        html.Div(
            className="plots-content",
            children=[
                # ---------- Section A: botones grises izquierdos ----------
                html.Div(
                    className="section-a",
                    children=[
                        html.Button(
                            "Feed",
                            id="btn-feed",
                            n_clicks=0,
                            className="gray-button feed-btn",
                        ),
                        html.Button(
                            "Reagents",
                            id="btn-reagents",
                            n_clicks=0,
                            className="gray-button reagents-btn",
                        ),
                        html.Button(
                            "Mineral Type",
                            id="btn-mineral",
                            n_clicks=0,
                            className="gray-button mineral-btn",
                        ),
                    ],
                ),

                # ---------- Section B: imagen central ----------
                html.Div(
                    className="section-b",
                    children=[
                        html.Img(
                            src="/assets/flotationcell.png",
                            className="flotation-bank-image",
                        )
                    ],
                ),

                # ---------- Section D: botones grises derechos ----------
                html.Div(
                    className="section-d",
                    children=[
                        html.Button(
                            "Concentrate",
                            id="btn-concentrate",
                            n_clicks=0,
                            className="gray-button conc-btn",
                        ),
                        html.Button(
                            "Tails",
                            id="btn-tails",
                            n_clicks=0,
                            className="gray-button tails-btn",
                        ),
                    ],
                ),

                # ---------- Section C: PARAMETERS + CORRELATIONS ----------
                html.Div(
                    className="section-c",
                    children=[
                        html.P("PARAMETERS", className="kpi-label"),

                        # Airflow
                        html.Div(
                            className="param-block",
                            children=[
                                html.P("Airflow, Nm³/h", className="param-name"),
                                html.Div(
                                    className="param-actions",
                                    children=[
                                        html.Button(
                                            "Raw Data",
                                            id="btn-airflow-raw",
                                            n_clicks=0,
                                            className="red-button small-red",
                                        ),
                                        html.Span("→", className="arrow-icon"),
                                        html.Button(
                                            "Statistical Analysis",
                                            id="btn-airflow-stat",
                                            n_clicks=0,
                                            className="red-button param-button",
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # Froth depth
                        html.Div(
                            className="param-block",
                            children=[
                                html.P("Froth Depth, mm", className="param-name"),
                                html.Div(
                                    className="param-actions",
                                    children=[
                                        html.Button(
                                            "Raw Data",
                                            id="btn-froth-raw",
                                            n_clicks=0,
                                            className="red-button small-red",
                                        ),
                                        html.Span("→", className="arrow-icon"),
                                        html.Button(
                                            "Statistical Analysis",
                                            id="btn-froth-stat",
                                            n_clicks=0,
                                            className="red-button small-red",
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # Cell power
                        html.Div(
                            className="param-block",
                            children=[
                                html.P("Cell Power, kW", className="param-name"),
                                html.Div(
                                    className="param-actions",
                                    children=[
                                        html.Button(
                                            "Raw Data",
                                            id="btn-power-raw",
                                            n_clicks=0,
                                            className="red-button small-red",
                                        ),
                                        html.Span("→", className="arrow-icon"),
                                        html.Button(
                                            "Statistical Analysis",
                                            id="btn-power-stat",
                                            n_clicks=0,
                                            className="red-button small-red",
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # Recovery button
                        html.Div(
                            className="param-block",
                            children=[
                                html.Button(
                                    "Recovery Statistical Analysis",
                                    id="btn-recovery-stat",
                                    n_clicks=0,
                                    className="red-button",
                                )
                            ],
                        ),

                        html.P("CORRELATIONS", className="kpi-label"),
                        html.Div(
                            className="blue-buttons-row",
                            children=[
                                html.Img(
                                    src="/assets/2dscatter.png",
                                    id="btn-2d",
                                    className="image-button",
                                ),
                                html.Img(
                                    src="/assets/3dscatter.png",
                                    id="btn-3d",
                                    className="image-button",
                                ),
                                html.Img(
                                    src="/assets/timeseries.png",
                                    id="btn-time",
                                    className="image-button",
                                ),
                            ],
                        ),

                        # Modals
                        build_airflow_stats_modal(),
                        build_froth_stats_modal(),
                        build_power_stats_modal(),
                        # Modal de Recovery
                        build_recovery_stats_modal(),
                        build_2d_corr_modal(),
                        build_3d_corr_modal(),   # <-- añade esta línea
                        build_timeseries_modal(),   # ⬅️ agrega esta línea
                    ],
                ),
            ],
        ),

        # Descargas ocultas
        Download(id="download-group-report"),
        Download(id="download-airflow-raw"),
        Download(id="download-froth-raw"),
        Download(id="download-power-raw"),
    ],
)

# =======================
# Helpers
# =======================

def _get_group_df(stored_data: str, group_map: dict, group_name: str) -> pd.DataFrame:
    """Rebuild dataframe only with the columns belonging to one group."""
    if stored_data is None or group_map is None:
        raise ValueError("No data loaded. Please upload an Excel file on the home page.")

    if group_name not in group_map:
        raise ValueError(f"No variables found for group '{group_name}'. Check your template.")

    df_flat = pd.read_json(stored_data, orient="split")

    variables = group_map.get(group_name, [])
    col_names = [
        f"{group_name}__{v}"
        for v in variables
        if f"{group_name}__{v}" in df_flat.columns
    ]

    if not col_names:
        raise ValueError(f"Group '{group_name}' has no matching columns in the data.")

    df_group = df_flat[col_names].copy()
    # Rename back to clean variable names
    df_group.columns = [
        v for v in variables if f"{group_name}__{v}" in df_flat.columns
    ]

    if df_group.empty:
        raise ValueError(f"Group '{group_name}' has no data rows.")

    return df_group


def build_param_lines(
    df_ops: pd.DataFrame,
    prefix: str,
    n_rows: int,
    cells_per_row: int,
) -> pd.DataFrame:
    """
    A partir de un DF con columnas como 'Airflow 1', 'Airflow 2', ..., 'Airflow N'
    construye un DF con columnas 'Line 1', 'Line 2', ..., una por línea,
    apilando las celdas de cada línea.
    """
    if n_rows is None or cells_per_row is None:
        raise ValueError("Set flotation rows and cells per row on the home page.")

    param_cols = [c for c in df_ops.columns if c.startswith(prefix)]
    if not param_cols:
        raise ValueError(f"No columns found for prefix '{prefix}' in Operational Parameters.")

    def cell_index(col):
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 0

    param_cols = sorted(param_cols, key=cell_index)

    total_cells = len(param_cols)
    expected_cells = n_rows * cells_per_row

    if total_cells != expected_cells:
        raise ValueError(
            f"Configuration {n_rows} x {cells_per_row} = {expected_cells} cells, "
            f"but found {total_cells} '{prefix}' columns."
        )

    per_line = {}
    for line_idx in range(n_rows):
        start = line_idx * cells_per_row
        end = start + cells_per_row
        line_cols = param_cols[start:end]

        serie_line = (
            df_ops[line_cols]
            .stack()
            .reset_index(drop=True)
        )

        per_line[f"Line {line_idx + 1}"] = serie_line

    df_lines = pd.DataFrame(per_line)
    return df_lines


def _build_param_line_boxplots(df_ops, n_rows, cells_per_row,
                               param_prefix, y_label, title):
    """
    Crea una figura con:
    - 1 subplot por línea
    - En cada subplot, 1 boxplot por celda de esa línea
    """
    param_cols = [c for c in df_ops.columns if c.startswith(param_prefix)]
    if not param_cols:
        raise ValueError(f"No columns starting with '{param_prefix}' found in Operational Parameters.")

    def cell_index(col):
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 0

    param_cols = sorted(param_cols, key=cell_index)

    total_cells = len(param_cols)
    expected_cells = n_rows * cells_per_row
    if expected_cells != total_cells:
        raise ValueError(
            f"Configuration {n_rows} x {cells_per_row} = {expected_cells} cells, "
            f"but found {total_cells} '{param_prefix}' columns."
        )

    line_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
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
            title_text=y_label,
            row=line_idx + 1,
            col=1,
        )

    fig.update_layout(
        title_text=title,
        showlegend=True,
        height=250 * n_rows,
    )

    return fig


def _generate_param_raw_report(n_clicks, stored_data, group_map, flot_config,
                               param_prefix, filename, y_label, title):
    """
    Helper genérico para los 3 botones rojos de *Raw Data*:
    Airflow, Froth Depth, Power Motor.
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_data is None or group_map is None or flot_config is None:
        raise PreventUpdate

    try:
        n_rows = flot_config.get("rows")
        cells_per_row = flot_config.get("cells_per_row")
        if not n_rows or not cells_per_row:
            raise ValueError("Flotation configuration (rows / cells per row) is missing.")

        df_ops = _get_group_df(stored_data, group_map, "Operational Parameters")

        fig = _build_param_line_boxplots(
            df_ops=df_ops,
            n_rows=n_rows,
            cells_per_row=cells_per_row,
            param_prefix=param_prefix,
            y_label=y_label,
            title=title,
        )

        tmp_dir = tempfile.gettempdir()
        path = os.path.join(tmp_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")
        return send_file(path)

    except Exception as e:
        print(f"Error generating raw report for {param_prefix}: {e}")
        raise PreventUpdate


def _generate_param_BA_stats_report(
    n_clicks,
    selected_date,
    stored_json,
    group_map,
    flot_config,
    *,
    param_prefix: str,
    y_label: str,
    title: str,
    filename: str,
):
    """
    Genera boxplots Before vs After por línea y celda para un parámetro
    de Operational Parameters (Froth Depth, Power Motor, etc.).
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_json is None or group_map is None or flot_config is None:
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

        df_flat = pd.read_json(stored_json, orient="split")

        time_col = "Time__DateTime_Measured"
        if time_col not in df_flat.columns:
            raise ValueError(f"Time column '{time_col}' not found in data.")

        time_series = pd.to_datetime(df_flat[time_col])

        df_ops = _get_group_df(stored_json, group_map, "Operational Parameters")

        param_cols = [c for c in df_ops.columns if c.startswith(param_prefix)]
        if not param_cols:
            raise ValueError(
                f"No '{param_prefix}' columns found in Operational Parameters."
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
                f"but found {total_cells} '{param_prefix}' columns."
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
                f"No valid '{param_prefix}' data found to build Before/After plot."
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
                        marker=dict(color="#ff7f0e"),
                        offsetgroup="After",
                    ),
                    row=line_idx + 1,
                    col=1,
                )
                if first_after:
                    first_after = False

            fig.update_yaxes(
                title_text=y_label,
                row=line_idx + 1,
                col=1,
            )

        fig.update_layout(
            title_text=title,
            boxmode="group",
            boxgroupgap=0.25,
            boxgap=0.1,
            height=260 * n_rows,
        )

        tmp_dir = tempfile.gettempdir()
        path = os.path.join(tmp_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")

        return send_file(path)

    except Exception as e:
        print(f"Error generating {param_prefix} Before/After report: {e}")
        raise PreventUpdate


def generate_froth_depth_stats_report(
    n_clicks,
    selected_date,
    stored_json,
    group_map,
    flot_config,
):
    return _generate_param_BA_stats_report(
        n_clicks=n_clicks,
        selected_date=selected_date,
        stored_json=stored_json,
        group_map=group_map,
        flot_config=flot_config,
        param_prefix="Froth Depth",
        y_label="Froth Depth, mm",
        title="Froth Depth – Before vs After (per line & cell)",
        filename="Froth_Rougher_BA_Grouped_BoxPlot.html",
    )


def generate_power_stats_report(
    n_clicks,
    selected_date,
    stored_json,
    group_map,
    flot_config,
):
    return _generate_param_BA_stats_report(
        n_clicks=n_clicks,
        selected_date=selected_date,
        stored_json=stored_json,
        group_map=group_map,
        flot_config=flot_config,
        param_prefix="Power Motor",
        y_label="Cell Power, kW",
        title="Cell Power – Before vs After (per line & cell)",
        filename="CellPower_Rougher_BA_Grouped_BoxPlot.html",
    )

# =======================
# Callbacks
# =======================

@callback(
    Output("download-group-report", "data"),
    Output("report-message", "children"),
    Input("btn-feed", "n_clicks"),
    Input("btn-reagents", "n_clicks"),
    Input("btn-mineral", "n_clicks"),
    Input("btn-concentrate", "n_clicks"),
    Input("btn-tails", "n_clicks"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    prevent_initial_call=True,
)
def generate_sweetviz_report(
    n_feed,
    n_reagents,
    n_mineral,
    n_conc,
    n_tails,
    stored_data,
    group_map,
):
    clicks = [n_feed, n_reagents, n_mineral, n_conc, n_tails]
    if all((c is None) or (c == 0) for c in clicks):
        raise PreventUpdate

    if not ctx.triggered_id:
        raise PreventUpdate

    triggered_id = ctx.triggered_id

    if triggered_id == "btn-feed":
        group_name = "Feed"
    elif triggered_id == "btn-reagents":
        group_name = "Reagents"
    elif triggered_id == "btn-mineral":
        group_name = "Mineral type"
    elif triggered_id == "btn-concentrate":
        group_name = "Concentrate"
    elif triggered_id == "btn-tails":
        group_name = "Tails"
    else:
        raise PreventUpdate

    try:
        df_group = _get_group_df(stored_data, group_map, group_name)

        report = sv.analyze(df_group)

        tmp_dir = tempfile.gettempdir()
        filename = f"{group_name.replace(' ', '_')}_report.html"
        file_path = os.path.join(tmp_dir, filename)
        report.show_html(file_path, open_browser=False)

        message = f"{group_name} Sweetviz report generated. Download should start automatically."
        return send_file(file_path), message

    except Exception as e:
        return no_update, f"Error generating report for {group_name}: {e}"


# ---------- Raw Data (botones rojos) ----------

@callback(
    Output("download-airflow-raw", "data"),
    Input("btn-airflow-raw", "n_clicks"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    prevent_initial_call=True,
)
def generate_airflow_raw_report(n_clicks, stored_data, group_map, flot_config):
    return _generate_param_raw_report(
        n_clicks=n_clicks,
        stored_data=stored_data,
        group_map=group_map,
        flot_config=flot_config,
        param_prefix="Airflow",
        filename="Airflow_RawData_Boxplot.html",
        y_label="Airflow, Nm³/h",
        title="Airflow Distribution – Rougher Lines",
    )


@callback(
    Output("download-froth-raw", "data"),
    Input("btn-froth-raw", "n_clicks"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    prevent_initial_call=True,
)
def generate_froth_raw_report(n_clicks, stored_data, group_map, flot_config):
    return _generate_param_raw_report(
        n_clicks=n_clicks,
        stored_data=stored_data,
        group_map=group_map,
        flot_config=flot_config,
        param_prefix="Froth Depth",
        filename="FrothDepth_RawData_Boxplot.html",
        y_label="Froth Depth, mm",
        title="Froth Depth Distribution – Rougher Lines",
    )


@callback(
    Output("download-power-raw", "data"),
    Input("btn-power-raw", "n_clicks"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    prevent_initial_call=True,
)
def generate_power_raw_report(n_clicks, stored_data, group_map, flot_config):
    return _generate_param_raw_report(
        n_clicks=n_clicks,
        stored_data=stored_data,
        group_map=group_map,
        flot_config=flot_config,
        param_prefix="Power Motor",
        filename="PowerMotor_RawData_Boxplot.html",
        y_label="Power Motor, kW",
        title="Power Motor Distribution – Rougher Lines",
    )


# ---------- Modals: abrir/cerrar ----------

@dash.callback(
    Output("airflow-stats-modal", "is_open"),
    Input("btn-airflow-stat", "n_clicks"),
    State("airflow-stats-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_airflow_stats_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


@dash.callback(
    Output("froth-stats-modal", "is_open"),
    Input("btn-froth-stat", "n_clicks"),
    State("froth-stats-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_froth_stats_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


@dash.callback(
    Output("power-stats-modal", "is_open"),
    Input("btn-power-stat", "n_clicks"),
    State("power-stats-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_power_stats_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open

@dash.callback(
    Output("recovery-stats-modal", "is_open"),
    Input("btn-recovery-stat", "n_clicks"),
    State("recovery-stats-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_recovery_stats_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open

@dash.callback(
    Output("corr2d-modal", "is_open"),
    Input("btn-2d", "n_clicks"),
    State("corr2d-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_corr2d_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open

@dash.callback(
    Output("corr3d-modal", "is_open"),
    Input("btn-3d", "n_clicks"),
    State("corr3d-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_corr3d_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open

@dash.callback(
    Output("timeseries-modal", "is_open"),
    Input("btn-time", "n_clicks"),
    State("timeseries-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_timeseries_modal(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open

# ---------- Fechas min/max para los 3 modals ----------

@dash.callback(
    Output("airflow-stats-from", "value"),
    Output("airflow-stats-to", "value"),
    Output("airflow-stats-date", "min_date_allowed"),
    Output("airflow-stats-date", "max_date_allowed"),
    Output("airflow-stats-date", "date"),
    Input("airflow-stats-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_airflow_stats_dates(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    return from_str, to_str, dmin, dmax, dmin


@dash.callback(
    Output("froth-stats-from", "value"),
    Output("froth-stats-to", "value"),
    Output("froth-stats-date", "min_date_allowed"),
    Output("froth-stats-date", "max_date_allowed"),
    Output("froth-stats-date", "date"),
    Input("froth-stats-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_froth_stats_dates(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    return from_str, to_str, dmin, dmax, dmin


@dash.callback(
    Output("power-stats-from", "value"),
    Output("power-stats-to", "value"),
    Output("power-stats-date", "min_date_allowed"),
    Output("power-stats-date", "max_date_allowed"),
    Output("power-stats-date", "date"),
    Input("power-stats-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_power_stats_dates(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    return from_str, to_str, dmin, dmax, dmin

@dash.callback(
    Output("recovery-stats-from", "value"),
    Output("recovery-stats-to", "value"),
    Output("recovery-stats-date", "min_date_allowed"),
    Output("recovery-stats-date", "max_date_allowed"),
    Output("recovery-stats-date", "date"),
    Input("recovery-stats-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_recovery_stats_dates(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")
    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    return from_str, to_str, dmin, dmax, dmin

@dash.callback(
    Output("corr2d-from", "value"),
    Output("corr2d-to", "value"),
    Output("corr2d-range", "min_date_allowed"),
    Output("corr2d-range", "max_date_allowed"),
    Output("corr2d-range", "start_date"),
    Output("corr2d-range", "end_date"),
    Output("corr2d-x", "options"),
    Output("corr2d-y", "options"),
    Output("corr2d-color", "options"),
    Input("corr2d-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_corr2d_modal(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    # Construimos la lista de variables numéricas
    var_options = []
    for col in df_flat.columns:
        if col.startswith("Time__"):
            continue
        series = df_flat[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        label = col.split("__", 1)[1] if "__" in col else col
        var_options.append({"label": label, "value": col})

    return (
        from_str,
        to_str,
        dmin,
        dmax,
        dmin,
        dmax,
        var_options,
        var_options,
        var_options,
    )

@dash.callback(
    Output("corr2d-color", "disabled"),
    Input("corr2d-enable-color", "value"),
)
def toggle_corr2d_color_disabled(enabled):
    # Si el checkbox está marcado, habilitamos el dropdown
    return not bool(enabled)
@dash.callback(
    Output("corr3d-from", "value"),
    Output("corr3d-to", "value"),
    Output("corr3d-range", "min_date_allowed"),
    Output("corr3d-range", "max_date_allowed"),
    Output("corr3d-range", "start_date"),
    Output("corr3d-range", "end_date"),
    Output("corr3d-x", "options"),
    Output("corr3d-y", "options"),
    Output("corr3d-z", "options"),
    Output("corr3d-color", "options"),
    Input("corr3d-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_corr3d_modal(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    # variables numéricas
    var_options = []
    for col in df_flat.columns:
        if col.startswith("Time__"):
            continue
        series = df_flat[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        label = col.split("__", 1)[1] if "__" in col else col
        var_options.append({"label": label, "value": col})

    return (
        from_str,
        to_str,
        dmin,
        dmax,
        dmin,
        dmax,
        var_options,
        var_options,
        var_options,
        var_options,
    )
@dash.callback(
    Output("corr3d-color", "disabled"),
    Input("corr3d-enable-color", "value"),
)
def toggle_corr3d_color_disabled(enabled):
    return not bool(enabled)

@dash.callback(
    Output("timeseries-param", "options"),
    Input("timeseries-modal", "is_open"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def fill_timeseries_modal(is_open, stored_json):
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    var_options = []
    for col in df_flat.columns:
        # saltamos las columnas de tiempo
        if col.startswith("Time__"):
            continue

        series = df_flat[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # label sin el prefijo de grupo
        label = col.split("__", 1)[1] if "__" in col else col
        var_options.append({"label": label, "value": col})

    if not var_options:
        raise PreventUpdate

    return var_options

# ---------- Download Airflow stats (tu función actual, intacta) ----------

@dash.callback(
    Output("download-airflow-stats", "data"),
    Input("airflow-stats-generate", "n_clicks"),
    State("airflow-stats-date", "date"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    prevent_initial_call=True,
)
def generate_airflow_stats_report(
    n_clicks,
    selected_date,
    stored_json,
    group_map,
    flot_config,
):
    """
    Genera boxplots Before vs After por línea y por celda para Airflow,
    y los descarga como un HTML.
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_json is None or group_map is None or flot_config is None:
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

        df_flat = pd.read_json(stored_json, orient="split")

        time_col = "Time__DateTime_Measured"
        if time_col not in df_flat.columns:
            raise ValueError(f"Time column '{time_col}' not found in data.")

        time_series = pd.to_datetime(df_flat[time_col])

        df_ops = _get_group_df(stored_json, group_map, "Operational Parameters")

        airflow_cols = [c for c in df_ops.columns if c.startswith("Airflow")]
        if not airflow_cols:
            raise ValueError(
                "No 'Airflow' columns found in Operational Parameters."
            )

        def cell_index(col):
            m = re.search(r"(\d+)$", col)
            return int(m.group(1)) if m else 0

        airflow_cols = sorted(airflow_cols, key=cell_index)

        total_cells = len(airflow_cols)
        expected_cells = n_rows * cells_per_row
        if total_cells != expected_cells:
            raise ValueError(
                f"Configuration {n_rows} x {cells_per_row} = {expected_cells} cells, "
                f"but found {total_cells} Airflow columns."
            )

        event_date = pd.to_datetime(selected_date).date()
        records = []

        for idx, ts in time_series.items():
            if pd.isna(ts):
                continue
            current_date = ts.date()
            period = "Before" if current_date < event_date else "After"

            for col in airflow_cols:
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
                "No valid Airflow data found to build Before/After plot."
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
                        marker=dict(color="#ff7f0e"),
                        offsetgroup="After",
                    ),
                    row=line_idx + 1,
                    col=1,
                )
                if first_after:
                    first_after = False

            fig.update_yaxes(
                title_text="Airflow, Nm³/h",
                row=line_idx + 1,
                col=1,
            )

        fig.update_layout(
            title_text="Airflow – Before vs After (per line & cell)",
            boxmode="group",
            boxgroupgap=0.25,
            boxgap=0.1,
            height=260 * n_rows,
        )

        tmp_dir = tempfile.gettempdir()
        filename = "Air_Rougher_BA_Grouped_BoxPlot.html"
        path = os.path.join(tmp_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")

        return send_file(path)

    except Exception as e:
        print(f"Error generating Airflow Before/After report: {e}")
        raise PreventUpdate


# ---------- Download Froth & Power stats ----------

@dash.callback(
    Output("download-froth-stats", "data"),
    Input("froth-stats-generate", "n_clicks"),
    State("froth-stats-date", "date"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    prevent_initial_call=True,
)
def cb_froth_stats(
    n_clicks,
    selected_date,
    stored_json,
    group_map,
    flot_config,
):
    return generate_froth_depth_stats_report(
        n_clicks,
        selected_date,
        stored_json,
        group_map,
        flot_config,
    )


@dash.callback(
    Output("download-power-stats", "data"),
    Input("power-stats-generate", "n_clicks"),
    State("power-stats-date", "date"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),
    prevent_initial_call=True,
)
def cb_power_stats(
    n_clicks,
    selected_date,
    stored_json,
    group_map,
    flot_config,
):
    return generate_power_stats_report(
        n_clicks,
        selected_date,
        stored_json,
        group_map,
        flot_config,
    )

@dash.callback(
    Output("download-recovery-stats", "data"),
    Input("recovery-stats-generate", "n_clicks"),
    State("recovery-stats-date", "date"),
    State("stored-data", "data"),
    State("group-map-store", "data"),
    State("flotation-config", "data"),  # <-- añade esta línea
    prevent_initial_call=True,
)

def generate_recovery_stats_report(n_clicks, selected_date,
                                   stored_json, group_map, flot_config):
    """
    Genera histogramas Before/After + tabla de stats por cada columna
    de recuperación (%Rec ...) del grupo 'Process Calculated'.
    El HTML se descarga como archivo.
    """
    if not n_clicks:
        raise PreventUpdate
    if stored_json is None or group_map is None or flot_config is None:
        raise PreventUpdate
    if selected_date is None:
        raise PreventUpdate

    # --- imports extra (solo SciPy puede no estar) ---
    try:
        from scipy import stats
    except ImportError:
        stats = None

    # --- reconstruir DF plano y columna de tiempo ---
    df_flat = pd.read_json(stored_json, orient="split")

    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise ValueError(f"Time column '{time_col}' not found in data.")
    time_series = pd.to_datetime(df_flat[time_col])

    # --- obtener grupo Process Calculated ---
    df_proc = _get_group_df(stored_json, group_map, "Process Calculated")

    # columnas de recuperación (flexible: cualquier cosa que contenga 'Rec')
    rec_cols = [
        c for c in df_proc.columns
        if "rec" in c.lower()
    ]
    if not rec_cols:
        raise ValueError(
            "No recovery columns found in 'Process Calculated' (looking for 'Rec')."
        )
    n_metrics = len(rec_cols)   # número de gráficos
    nbins = 40                  # o el número de bins que quieras usar
    # --- helper: densidad simple a partir del histograma ---
    def _density_curve(values, bins=40):
        values = np.asarray(values, dtype=float)
        if values.size < 2:
            return values, np.zeros_like(values)
        counts, edges = np.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        return centers, counts

    # --- figura con subplots: 1 fila por %Rec, 2 columnas (hist + tabla) ---
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=len(rec_cols),
        cols=2,
        specs=[[{"type": "xy"}, {"type": "table"}] for _ in rec_cols],
        column_widths=[0.65, 0.35],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    event_date = pd.to_datetime(selected_date).date()

    for row_idx, col_name in enumerate(rec_cols, start=1):
        series = df_proc[col_name]

        # máscaras before / after
        valid = series.notna() & time_series.notna()
        before_mask = valid & (time_series.dt.date < event_date)
        after_mask = valid & (time_series.dt.date >= event_date)

        before_vals = series[before_mask].astype(float).values
        after_vals = series[after_mask].astype(float).values

        if before_vals.size == 0 or after_vals.size == 0:
            # si falta info para esa línea, saltamos
            continue

        # --- histograma + curva de densidad (Before / After) ---
        # Colores fijos para todos los plots
        color_before = "#1f77b4"  # azul
        color_after = "#ff7f0e"   # naranja

        # barras
        fig.add_trace(
            go.Histogram(
                x=before_vals,
                name="Before",
                marker_color=color_before,
                opacity=0.5,
                nbinsx=nbins,
                offsetgroup="before",    # <--- clave para agrupar
                histnorm="probability density",
                legendgroup="period",
                showlegend=(row_idx == 1),
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=after_vals,
                name="After",
                marker_color=color_after,
                opacity=0.5,
                histnorm="probability density",
                nbinsx=nbins,
                offsetgroup="after",     # <--- grupo distinto
                legendgroup="period",
                showlegend=(row_idx == 1),
            ),
            row=row_idx,
            col=1,
        )

        # curvas de densidad (sin afectar leyenda)
        x_b, y_b = _density_curve(before_vals)
        x_a, y_a = _density_curve(after_vals)

        fig.add_trace(
            go.Scatter(
                x=x_b,
                y=y_b,
                mode="lines",
                line=dict(color=color_before),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_a,
                y=y_a,
                mode="lines",
                line=dict(color=color_after),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )

        # ejes
        fig.update_xaxes(title_text=col_name, row=row_idx, col=1)
        fig.update_yaxes(
            title_text="Density" if row_idx == 1 else "",
            row=row_idx,
            col=1,
        )

        # --- estadística básica ---
        n_before = int(before_vals.size)
        n_after = int(after_vals.size)

        mean_before = float(np.mean(before_vals))
        mean_after = float(np.mean(after_vals))

        std_before = float(np.std(before_vals, ddof=1)) if n_before > 1 else float("nan")
        std_after = float(np.std(after_vals, ddof=1)) if n_after > 1 else float("nan")

        se_before = std_before / np.sqrt(n_before) if n_before > 0 else float("nan")
        se_after = std_after / np.sqrt(n_after) if n_after > 0 else float("nan")

        # --- estadísticos extra para el bloque de texto (t-test de Welch) ---
        delta = mean_before - mean_after
        ci_low = ci_high = t_stat = p_val = df = None

        if stats is not None and n_before > 1 and n_after > 1:
            var_before = std_before ** 2
            var_after = std_after ** 2
            se_diff = math.sqrt(var_before / n_before + var_after / n_after)

            if se_diff > 0:
                t_stat = delta / se_diff
                # grados de libertad de Welch
                num = (var_before / n_before + var_after / n_after) ** 2
                den = (
                    (var_before ** 2) / (n_before ** 2 * (n_before - 1))
                    + (var_after ** 2) / (n_after ** 2 * (n_after - 1))
                )
                df = num / den if den != 0 else None

                if df is not None and df > 0:
                    p_val = 2 * stats.t.sf(abs(t_stat), df)
                    alpha = 0.05
                    t_crit = stats.t.ppf(1 - alpha / 2, df)
                    ci_low = delta - t_crit * se_diff
                    ci_high = delta + t_crit * se_diff

        # --- tabla: mismo nombre de colores en TODOS los plots ---
        header_values = ["Period", "N", "Mean", "StDev", "SE Mean"]

        period_vals = ["A_before", "B_after"]
        n_vals = [n_before, n_after]
        mean_vals = [mean_before, mean_after]
        stdev_vals = [std_before, std_after]
        se_vals = [se_before, se_after]

        # bloque de texto abajo de la tabla (si pudimos calcularlo)
        if (
            df is not None
            and t_stat is not None
            and p_val is not None
            and ci_low is not None
            and ci_high is not None
        ):
            lines = [
                f"&Delta; = &mu;(A) - &mu;(B) = {delta:+.3f}",
                f"95% CI: ({ci_low:.3f}; {ci_high:.3f})",
                f"t = {t_stat:.2f} (df = {df:.0f})",
                f"p = {p_val:.3f}",
            ]
            text_block = "<br>".join(lines)

            period_vals.append(text_block)
            n_vals.append("")
            mean_vals.append("")
            stdev_vals.append("")
            se_vals.append("")

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header_values,   # << NO ponemos %Rec aquí
                    align="left",
                    fill_color="rgb(230, 230, 230)",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[period_vals, n_vals, mean_vals, stdev_vals, se_vals],
                    align="left",
                    format=[None, "d", ".2f", ".2f", ".2f"],
                ),
                columnwidth=[90, 40, 60, 60, 70],
            ),
            row=row_idx,
            col=2,
        )

    # layout general
    fig.update_layout(
        title_text="Recovery – Before vs After (per line)",
        barmode="group",        # <--- esto evita la superposición
        bargap=0.05,
        bargroupgap=0.02,
        legend_title_text="Period",
        height=280 * n_metrics,
        template="simple_white",
        margin=dict(l=60, r=40, t=80, b=60),
    )

    # guardar y devolver como descarga
    tmp_dir = tempfile.gettempdir()
    filename = "Recovery_Rougher_BA_Line_Hist_Stats.html"
    path = os.path.join(tmp_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn")

    return send_file(path)

@dash.callback(
    Output("download-2d-corr", "data"),
    Input("corr2d-generate", "n_clicks"),
    State("corr2d-range", "start_date"),
    State("corr2d-range", "end_date"),
    State("corr2d-x", "value"),
    State("corr2d-y", "value"),
    State("corr2d-enable-color", "value"),
    State("corr2d-color", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def generate_2d_corr_report(
    n_clicks,
    start_date,
    end_date,
    x_col,
    y_col,
    enable_color,
    color_col,
    stored_json,
):
    if not n_clicks:
        raise PreventUpdate
    if stored_json is None:
        raise PreventUpdate
    if x_col is None or y_col is None:
        # Falta seleccionar ejes
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    time_series = pd.to_datetime(df_flat[time_col])

    # Filtro de fechas
    if start_date is not None and end_date is not None:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (time_series >= start) & (time_series <= end)
        df_sel = df_flat.loc[mask].copy()
    else:
        df_sel = df_flat.copy()

    if df_sel.empty:
        raise PreventUpdate

    # Labels bonitos (sin el prefijo de grupo)
    def _nice_label(col):
        return col.split("__", 1)[1] if "__" in col else col

    x_label = _nice_label(x_col)
    y_label = _nice_label(y_col)

    use_color = bool(enable_color) and color_col is not None

    if use_color:
        fig = px.scatter(
            df_sel,
            x=x_col,
            y=y_col,
            color=color_col,
            color_continuous_scale="Plasma",
        )
    else:
        fig = px.scatter(
            df_sel,
            x=x_col,
            y=y_col,
        )

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        title="SCATTER PLOT",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="simple_white",
        height=700,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    # Guardar como HTML y devolver
    tmp_dir = tempfile.gettempdir()
    filename = "Correlation_2D_Scatter.html"
    path = os.path.join(tmp_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn")

    return send_file(path)

@dash.callback(
    Output("download-3d-corr", "data"),
    Input("corr3d-generate", "n_clicks"),
    State("corr3d-range", "start_date"),
    State("corr3d-range", "end_date"),
    State("corr3d-x", "value"),
    State("corr3d-y", "value"),
    State("corr3d-z", "value"),
    State("corr3d-enable-color", "value"),
    State("corr3d-color", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def generate_3d_corr_report(
    n_clicks,
    start_date,
    end_date,
    x_col,
    y_col,
    z_col,
    enable_color,
    color_col,
    stored_json,
):
    if not n_clicks:
        raise PreventUpdate
    if stored_json is None:
        raise PreventUpdate
    if x_col is None or y_col is None or z_col is None:
        # faltan ejes
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    time_series = pd.to_datetime(df_flat[time_col])

    # Filtro por rango de fechas
    if start_date is not None and end_date is not None:
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
        fig = px.scatter_3d(
            df_sel,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            color_continuous_scale="Plasma",
        )
    else:
        fig = px.scatter_3d(
            df_sel,
            x=x_col,
            y=y_col,
            z=z_col,
        )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        title="SCATTER PLOT 3D",
        scene=dict(
            xaxis=dict(
                title=x_label,
                showbackground=True,
                backgroundcolor="rgba(240, 240, 240, 1)",
                showgrid=True,
                gridcolor="lightgrey",
                zeroline=False,
            ),
            yaxis=dict(
                title=y_label,
                showbackground=True,
                backgroundcolor="rgba(240, 240, 240, 1)",
                showgrid=True,
                gridcolor="lightgrey",
                zeroline=False,
            ),
            zaxis=dict(
                title=z_label,
                showbackground=True,
                backgroundcolor="rgba(240, 240, 240, 1)",
                showgrid=True,
                gridcolor="lightgrey",
                zeroline=False,
            ),
            # esto fuerza la forma de cubo en lugar de un prisma achatado
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

    return send_file(path)

@dash.callback(
    Output("download-timeseries", "data"),
    Input("timeseries-generate", "n_clicks"),
    State("timeseries-param", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def generate_timeseries_report(
    n_clicks,
    param_col,
    stored_json,
):
    if not n_clicks:
        raise PreventUpdate
    if stored_json is None or param_col is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        raise PreventUpdate

    # limpiamos y ordenamos por tiempo
    df_flat[time_col] = pd.to_datetime(df_flat[time_col])
    df_sel = df_flat[[time_col, param_col]].dropna()
    df_sel = df_sel.sort_values(time_col)

    if df_sel.empty:
        raise PreventUpdate

    def _nice_label(col):
        return col.split("__", 1)[1] if "__" in col else col

    y_label = _nice_label(param_col)

    fig = px.line(
        df_sel,
        x=time_col,
        y=param_col,
    )
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

    return send_file(path)
