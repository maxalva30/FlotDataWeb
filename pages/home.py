# pages/home.py
import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import io, base64

dash.register_page(__name__, path="/")


layout = html.Div(
    className="main-container",
    children=[
        html.Div(
            className="content-container",
            children=[
                # Left column: flotation image
                html.Div(
                    className="left-column",
                    children=[
                        html.Img(
                            src="/assets/flotationbank.png",
                            className="flotation-img",
                        )
                    ],
                ),

                # Right column: configuration + data upload
                html.Div(
                    className="right-column",
                    children=[
                        # Flotation configuration
                        html.Div(
                            className="top-right",
                            children=[
                                html.H4(
                                    "Flotation configuration",
                                    className="section-title",
                                ),
                                html.Label(
                                    "Flotation rows:",
                                    className="dropdown-label",
                                ),
                                dcc.Dropdown(
                                    id="dropdown-flotation-rows",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in range(1, 11)
                                    ],
                                    placeholder="Select number of rows",
                                    className="custom-dropdown",
                                ),
                                html.Label(
                                    "Number of cells per row:",
                                    className="dropdown-label",
                                ),
                                dcc.Dropdown(
                                    id="dropdown-cells-per-row",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in range(1, 21)
                                    ],
                                    placeholder="Select cells per row",
                                    className="custom-dropdown",
                                ),
                            ],
                        ),

                        # Data upload
                        html.Div(
                            className="bottom-right",
                            children=[
                                html.H4(
                                    "Data upload",
                                    className="section-title",
                                ),
                                html.Div(
                                    className="upload-row",
                                    children=[
                                        dcc.Upload(
                                            id="upload-data",
                                            children=html.Div(
                                                "Select Excel file",
                                                id="upload-text",
                                            ),
                                            className="upload-box",
                                            style={
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "5px",
                                                "padding": "10px",
                                                "textAlign": "center",
                                                "cursor": "pointer",
                                                "backgroundColor": "#f9f9f9",
                                            },
                                            accept=".xls,.xlsx",
                                            multiple=False,
                                        ),
                                        html.A(
                                            "Go to plots",
                                            href="/plots",
                                            className="go-button",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
    ],
)


@dash.callback(
    Output("stored-data", "data"),
    Output("group-map-store", "data"),
    Output("upload-data", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # Open Excel file
        xls = pd.ExcelFile(io.BytesIO(decoded))

        # Expect a sheet called 'Data'
        if "Data" not in xls.sheet_names:
            raise ValueError("Template mismatch: sheet 'Data' was not found in this file.")

        # Read Excel with two header rows:
        #   row 6 (index 5) -> group name: Feed, Reagents, Mineral type, etc.
        #   row 7 (index 6) -> variable name inside each group.
        df_multi = pd.read_excel(
            xls,
            sheet_name="Data",
            header=[5, 6],
        )

        df_multi.columns = df_multi.columns.set_names(["group", "variable"])

        # Build mapping: group -> list of variables
        group_to_columns = {}
        for grp, var in df_multi.columns:
            if pd.isna(grp) or pd.isna(var):
                continue
            g = str(grp).strip()
            v = str(var).strip()
            group_to_columns.setdefault(g, []).append(v)

        # Flatten columns for JSON storage: "Group__Variable"
        df_flat = df_multi.copy()
        flat_columns = [f"{g}__{v}" for g, v in df_flat.columns]
        df_flat.columns = flat_columns

        # Convert first column (time) to datetime if possible
        if len(df_flat.columns) > 0:
            first_col = df_flat.columns[0]
            df_flat[first_col] = pd.to_datetime(df_flat[first_col], errors="coerce")

        df_json = df_flat.to_json(date_format="iso", orient="split")

        label = filename if filename else "Excel file loaded"
        upload_children = html.Div(
            className="upload-inner",
            children=[
                html.Img(
                    src="/assets/excel-icon.png",
                    className="excel-icon",
                ),
                html.Span(label, className="file-name"),
            ],
        )

        return df_json, group_to_columns, upload_children

    except Exception as e:
        print("Error reading Excel file:", e)

        error_children = html.Div(
            className="upload-error",
            children=[
                html.Span(
                    "Invalid file. Use the Flotation Template: sheet 'Data' "
                    "with row 6 as groups and row 7 as variable names."
                )
            ],
        )

        return dash.no_update, dash.no_update, error_children

    # 🔻 EXACTAMENTE AQUÍ, DEBAJO DE handle_upload 🔻

@dash.callback(
    Output("flotation-config", "data"),
    Input("dropdown-flotation-rows", "value"),
    Input("dropdown-cells-per-row", "value"),
)
def store_flotation_config(n_rows, cells_per_row):
    """
    Save flotation configuration (rows and cells per row)
    so it can be used later in the plots page.
    """
    if n_rows is None or cells_per_row is None:
        raise PreventUpdate

    return {
        "rows": int(n_rows),
        "cells_per_row": int(cells_per_row),
    }
