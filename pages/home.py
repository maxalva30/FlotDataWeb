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
                            src="/assets/Flotationbank.png",
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
    """
    Handle Excel upload:
    - Read the 'Data' sheet using row 5 (groups) and row 6 (variable names) as a MultiIndex header.
    - Build a mapping: group -> list of variable names.
    - Store the dataframe as JSON in 'stored-data' and the mapping in 'group-map-store'.
    - Update the upload box content to show Excel icon + filename (inside the dashed box).
    """
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # Open Excel file
        xls = pd.ExcelFile(io.BytesIO(decoded))

        # We expect a sheet called "Data" (flotation template)
        if "Data" not in xls.sheet_names:
            raise ValueError(
                "Template mismatch: sheet 'Data' was not found in this file."
            )

        # Read Excel with two header rows:
        # row 5 (index 4) -> group name: Feed, Reagents, Mineral type, etc.
        # row 6 (index 5) -> variable name inside each group.
        df = pd.read_excel(
            xls,
            sheet_name="Data",
            header=[4, 5],
        )

        # Name the MultiIndex levels for clarity
        df.columns = df.columns.set_names(["group", "variable"])

        # Try to convert the first column to datetime (usually the time column)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col], errors="coerce")

        # Build mapping: group name -> list of variable names
        group_to_columns = {}
        for grp, var in df.columns:
            if pd.isna(grp) or pd.isna(var):
                continue
            g = str(grp).strip()
            v = str(var).strip()
            group_to_columns.setdefault(g, []).append(v)

        # Serialize dataframe to JSON for storage
        df_json = df.to_json(date_format="iso", orient="split")

        # Upload box content: Excel icon + filename INSIDE the dashed box
        label = filename if filename else "Excel file loaded"
        upload_children = html.Div(
            className="upload-inner",
            children=[
                html.Img(
                    src="/assets/excel-icon.png",
                    className="excel-icon",
                ),
                html.Span(
                    label,
                    className="file-name",
                ),
            ],
        )

        return df_json, group_to_columns, upload_children

    except Exception as e:
        # More informative error: the user probably loaded a file
        # that does not follow the flotation template.
        print("Error reading Excel file:", e)

        error_children = html.Div(
            className="upload-error",
            children=[
                html.Span(
                    "Invalid file. Use the Flotation Template: sheet 'Data' with "
                    "row 5 as groups and row 6 as variable names.",
                )
            ],
        )

        # Do not overwrite previous valid data if there was any
        return dash.no_update, dash.no_update, error_children
