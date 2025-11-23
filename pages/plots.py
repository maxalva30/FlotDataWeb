import os
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

dash.register_page(__name__, path="/plots")

# =======================
# Layout
# =======================
# ---------- Stats modal helper (Airflow, luego copiamos para otros) ----------
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
                    # descarga del HTML con el boxplot
                    dcc.Download(id="download-airflow-stats"),
                ]
            ),
        ],
    )
layout = html.Div(
    className="main-container",
    children=[
        # Contenido principal de la página
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
                            src="/assets/flotationbank.png",
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
                    # Modal de Airflow
                    build_airflow_stats_modal(),
                    ],
                ),
            ],
        ),

        # Componente oculto para descargar el HTML de Sweetviz
        Download(id="download-group-report"),
        # Componente oculto para descargar el boxplot de Airflow Raw Data
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

    # Filtrar solo las columnas que empiezan con el prefijo (Airflow, Froth Depth, etc.)
    param_cols = [c for c in df_ops.columns if c.startswith(prefix)]
    if not param_cols:
        raise ValueError(f"No columns found for prefix '{prefix}' in Operational Parameters.")

    # Ordenar por el número al final del nombre (Airflow 1, Airflow 2, ...)
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

        # Apilar los valores de las celdas de esa línea en una sola serie
        serie_line = (
            df_ops[line_cols]
            .stack()              # apila columnas una debajo de otra
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
    # Seleccionar columnas de este parámetro
    param_cols = [c for c in df_ops.columns if c.startswith(param_prefix)]
    if not param_cols:
        raise ValueError(f"No columns starting with '{param_prefix}' found in Operational Parameters.")

    # Ordenar según el número al final del nombre de la columna
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

    # Colores: un color por línea
    line_colors = [
        "#1f77b4",  # Line 1
        "#ff7f0e",  # Line 2
        "#2ca02c",  # Line 3
        "#d62728",  # Line 4
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,  # mismo espacio que te gustó en Airflow
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

        # Grupo 'Operational Parameters'
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

def _build_airflow_BA_grouped_boxplot(
    df_ops: pd.DataFrame,
    time_series: pd.Series,
    cutoff_date,
    n_rows: int,
    cells_per_row: int,
):
    """
    Construye una figura de Airflow Before vs After:
    - 1 subplot por línea
    - En cada subplot, 2 boxplots por celda (Before y After), agrupados por celda.
    """

    param_prefix = "Airflow"
    param_cols = [c for c in df_ops.columns if c.startswith(param_prefix)]
    if not param_cols:
        raise ValueError(f"No columns starting with '{param_prefix}' found in Operational Parameters.")

    # Ordenamos por el número al final del nombre de la columna: Airflow 1, Airflow 2, ...
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

    # Máscaras Before / After según la fecha elegida
    ts = pd.to_datetime(time_series)
    cutoff_date = pd.to_datetime(cutoff_date).date()

    mask_before = ts.dt.date < cutoff_date
    mask_after = ts.dt.date >= cutoff_date

    if not mask_before.any() or not mask_after.any():
        raise ValueError("Not enough data Before or After the selected date.")

    # Condiciones: Before / After
    cond_specs = [
        ("Before", mask_before, "rgba(31, 119, 180, 0.7)"),
        ("After",  mask_after,  "rgba(255, 127, 14, 0.7)"),
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"Line {i+1}" for i in range(n_rows)],
    )

    # Recorremos línea por línea
    for line_idx in range(n_rows):
        start = line_idx * cells_per_row
        end = start + cells_per_row
        line_cols = param_cols[start:end]

        for cond_name, cond_mask, color in cond_specs:
            for col in line_cols:
                serie = df_ops.loc[cond_mask, col].dropna()
                if serie.empty:
                    continue

                m = re.search(r"(\d+)$", col)
                cell_label = f"Cell {m.group(1)}" if m else col

                show_legend = (line_idx == 0 and col == line_cols[0])

                fig.add_trace(
                    go.Box(
                        y=serie,
                        x=[cell_label] * len(serie),
                        name=cond_name if show_legend else None,
                        boxmean="sd",
                        marker=dict(color=color),
                        legendgroup=cond_name,
                        offsetgroup=cond_name,
                    ),
                    row=line_idx + 1,
                    col=1,
                )

        fig.update_yaxes(
            title_text="Airflow, Nm³/h",
            row=line_idx + 1,
            col=1,
        )

    fig.update_layout(
        title_text="Airflow – Before vs After (per line & cell)",
        boxmode="group",   # agrupa Before/After por celda
        showlegend=True,
        height=300 * n_rows,
    )

    return fig

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
    """Generate and download a Sweetviz HTML report when a grey group button is clicked."""

    # 1) Si todos los botones están en 0 / None, no hacemos nada
    clicks = [n_feed, n_reagents, n_mineral, n_conc, n_tails]
    if all((c is None) or (c == 0) for c in clicks):
        raise PreventUpdate

    # 2) Si por alguna razón no tenemos triggered_id, tampoco hacemos nada
    if not ctx.triggered_id:
        raise PreventUpdate

    triggered_id = ctx.triggered_id

    if triggered_id == "btn-feed":
        group_name = "Feed"
    elif triggered_id == "btn-reagents":
        group_name = "Reagents"
    elif triggered_id == "btn-mineral":
        group_name = "Mineral type"  # debe coincidir con el nombre del grupo en el Excel
    elif triggered_id == "btn-concentrate":
        group_name = "Concentrate"
    elif triggered_id == "btn-tails":
        group_name = "Tails"
    else:
        raise PreventUpdate

    try:
        df_group = _get_group_df(stored_data, group_map, group_name)

        # Generate Sweetviz report
        report = sv.analyze(df_group)

        # Save HTML to a temporary file
        tmp_dir = tempfile.gettempdir()
        filename = f"{group_name.replace(' ', '_')}_report.html"
        file_path = os.path.join(tmp_dir, filename)
        report.show_html(file_path, open_browser=False)

        message = f"{group_name} Sweetviz report generated. Download should start automatically."
        return send_file(file_path), message

    except Exception as e:
        return no_update, f"Error generating report for {group_name}: {e}"

# ============================
# Callbacks Raw Data (botones rojos)
# ============================

# Airflow – Raw Data
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
        param_prefix="Airflow",                      # nombre de las columnas
        filename="Airflow_RawData_Boxplot.html",     # nombre del html
        y_label="Airflow, Nm³/h",                    # etiqueta eje Y
        title="Airflow Distribution – Rougher Lines" # título del gráfico
    )


# Froth Depth – Raw Data
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
        param_prefix="Froth Depth",                      # p.ej. "Froth Depth 1"
        filename="FrothDepth_RawData_Boxplot.html",
        y_label="Froth Depth, mm",
        title="Froth Depth Distribution – Rougher Lines",
    )


# Power Motor – Raw Data
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
        param_prefix="Power Motor",                     # p.ej. "Power Motor 1"
        filename="PowerMotor_RawData_Boxplot.html",
        y_label="Power Motor, kW",
        title="Power Motor Distribution – Rougher Lines",
    )
    """
    Generate an Airflow-by-cell boxplot report:
    - One subplot per line
    - In each subplot, one boxplot per cell of that line
    """
    if not n_clicks:
        raise PreventUpdate

    if stored_data is None or group_map is None or flot_config is None:
        raise PreventUpdate

    try:
        n_rows = flot_config.get("rows")
        cells_per_row = flot_config.get("cells_per_row")

        # 1) DataFrame del grupo 'Operational Parameters'
        df_ops = _get_group_df(stored_data, group_map, "Operational Parameters")

        # 2) Columnas de Airflow (Airflow 1, Airflow 2, ..., Airflow N)
        param_cols = [c for c in df_ops.columns if c.startswith("Airflow")]
        if not param_cols:
            raise ValueError("No 'Airflow' columns found in Operational Parameters.")

        # Ordenar por número al final
        def cell_index(col):
            m = re.search(r"(\d+)$", col)
            return int(m.group(1)) if m else 0

        param_cols = sorted(param_cols, key=cell_index)

        total_cells = len(param_cols)
        expected_cells = n_rows * cells_per_row
        if total_cells != expected_cells:
            raise ValueError(
                f"Configuration {n_rows} x {cells_per_row} = {expected_cells} cells, "
                f"but found {total_cells} Airflow columns."
            )
        # Paleta de colores: un color por línea
        line_colors = [
            "#1f77b4",  # Line 1
            "#ff7f0e",  # Line 2
            "#2ca02c",  # Line 3
            "#d62728",  # Line 4
        ]

        # 3) Crear subplots: una fila por línea
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,  # 👈 más espacio entre líneas
            subplot_titles=[f"Line {i+1}" for i in range(n_rows)],
        )

        # 4) Para cada línea, añadir boxplots por celda de esa línea
        for line_idx in range(n_rows):
            start = line_idx * cells_per_row
            end = start + cells_per_row
            line_cols = param_cols[start:end]

            # Color para esta línea
            line_color = line_colors[line_idx % len(line_colors)]

            for col in line_cols:
                serie = df_ops[col].dropna()
                if serie.empty:
                    continue

                # Nombre corto para la celda (Cell 1, Cell 2, etc.)
                m = re.search(r"(\d+)$", col)
                cell_label = f"Cell {m.group(1)}" if m else col

                fig.add_trace(
                    go.Box(
                        y=serie,
                        name=cell_label,
                        boxmean="sd",
                        marker=dict(color=line_color),  # 👈 todas las celdas de la línea con el mismo color
                    ),
                    row=line_idx + 1,
                    col=1,
                )

            fig.update_yaxes(
                title_text="Airflow, Nm³/h",
                row=line_idx + 1,
                col=1,
            )

            # Etiqueta de eje Y por línea
            fig.update_yaxes(
                title_text="Airflow, Nm³/h",
                row=line_idx + 1,
                col=1,
            )

        fig.update_layout(
            title_text="Airflow Distribution – Rougher Lines",
            showlegend=True,
            height=250 * n_rows,  # altura proporcional al nº de líneas
        )

        # 5) Guardar HTML y devolver como descarga
        tmp_dir = tempfile.gettempdir()
        filename = "Airflow_RawData_Boxplot.html"
        file_path = os.path.join(tmp_dir, filename)
        fig.write_html(file_path, include_plotlyjs="cdn")

        return send_file(file_path)

    except Exception as e:
        print(f"Error generating Airflow report: {e}")
        raise PreventUpdate
@dash.callback(
    Output("airflow-stats-modal", "is_open"),
    Input("btn-airflow-stat", "n_clicks"),
    State("airflow-stats-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_airflow_stats_modal(n_clicks, is_open):
    # Cada vez que haces click en el botón rojo, se abre/cierra el modal
    if not n_clicks:
        raise PreventUpdate
    return not is_open

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
    # Solo actuamos cuando se abre el modal
    if not is_open or stored_json is None:
        raise PreventUpdate

    df_flat = pd.read_json(stored_json, orient="split")

    # Ajusta este nombre si tu columna de tiempo se llama diferente
    time_col = "Time__DateTime_Measured"
    if time_col not in df_flat.columns:
        # No rompemos la app, solo no hacemos nada
        raise PreventUpdate

    # Convertir a datetime y obtener min/max
    s = pd.to_datetime(df_flat[time_col].dropna())
    if s.empty:
        raise PreventUpdate

    dmin = s.min().date()
    dmax = s.max().date()

    from_str = dmin.strftime("%d/%m/%Y")
    to_str = dmax.strftime("%d/%m/%Y")

    # El calendario solo permite elegir dentro de este rango
    return from_str, to_str, dmin, dmax, dmin

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
        # --- 1) Configuración de flotación ---
        n_rows = flot_config.get("rows")
        cells_per_row = flot_config.get("cells_per_row")
        if not n_rows or not cells_per_row:
            raise ValueError(
                "Flotation configuration (rows / cells per row) is missing."
            )

        # --- 2) Reconstruir DF plano completo y serie de tiempo ---
        df_flat = pd.read_json(stored_json, orient="split")

        time_col = "Time__DateTime_Measured"
        if time_col not in df_flat.columns:
            raise ValueError(f"Time column '{time_col}' not found in data.")

        time_series = pd.to_datetime(df_flat[time_col])

        # --- 3) DF del grupo Operational Parameters (Airflow 1…N) ---
        df_ops = _get_group_df(stored_json, group_map, "Operational Parameters")

        airflow_cols = [c for c in df_ops.columns if c.startswith("Airflow")]
        if not airflow_cols:
            raise ValueError(
                "No 'Airflow' columns found in Operational Parameters."
            )

        # Ordenar columnas de Airflow por número (Airflow 1, Airflow 2, …)
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

        # --- 4) Construimos DF "largo": (time, cell_num, line, period, value) ---
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

        # --- 5) Figura con 1 subplot por línea ---
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=[f"Line {i+1}" for i in range(n_rows)],
        )

        # Para la leyenda: solo mostrar el primer 'Before' y el primer 'After'
        first_before = True
        first_after = True

        for line_idx in range(n_rows):
            line_name = f"Line {line_idx + 1}"
            df_line = df_long[df_long["Line"] == line_name]

            # Ordenar celdas dentro de cada línea
            unique_cells = sorted(df_line["CellNum"].unique())

            for cell_num in unique_cells:
                df_cell = df_line[df_line["CellNum"] == cell_num]
                cell_label = f"Cell {cell_num}"

                before_vals = df_cell[df_cell["Period"] == "Before"]["Value"].dropna()
                after_vals = df_cell[df_cell["Period"] == "After"]["Value"].dropna()

                if before_vals.empty and after_vals.empty:
                    continue

                # --- trace Before ---
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

                # --- trace After ---
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

            # Etiqueta del eje Y por línea
            fig.update_yaxes(
                title_text="Airflow, Nm³/h",
                row=line_idx + 1,
                col=1,
            )

        fig.update_layout(
            title_text="Airflow – Before vs After (per line & cell)",
            boxmode="group",          # Before y After lado a lado
            boxgroupgap=0.25,         # espacio entre grupos de celdas
            boxgap=0.1,               # espacio entre Before y After
            height=260 * n_rows,
        )

        # --- 6) Guardar y enviar como HTML ---
        tmp_dir = tempfile.gettempdir()
        filename = "Air_Rougher_BA_Grouped_BoxPlot.html"
        path = os.path.join(tmp_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")

        return send_file(path)

    except Exception as e:
        print(f"Error generating Airflow Before/After report: {e}")
        raise PreventUpdate
