import dash
from dash import html, dcc, callback, Input, Output, State, ctx, ALL, MATCH, no_update
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/plots')

layout = html.Div([
    html.Div([

        # Sección A

        html.Div([
            html.Button("Feed", className="gray-button feed-btn"),
            html.Button("Reagents", className="gray-button reagents-btn"),
            html.Button("Mineral Type", className="gray-button mineral-btn")
        ], className="section-a"),

        # Sección B
        html.Div([
            html.Img(src="/assets/flotationcell.png", className="flotation-image")
        ], className="section-b"),

        # Sección D
        html.Div([
            html.Button("Concentrate", className="gray-button btn-concentrate"),
            html.Button("Tails", className="gray-button btn-tails")
        ], className="section-d"),

        # Sección C
        html.Div([

            html.P("Airflow, Nm3/h"),
            html.Div([
                html.Button("Raw Data", className="red-button"),
                html.Img(src="/assets/arrow.png", className="arrow-icon"),
                html.Button("Statistical Analysis", className="red-button")
            ], className="param-row"),
            html.P("Froth Depth, mm"),
            html.Div([
                html.Button("Raw Data", className="red-button"),
                html.Img(src="/assets/arrow.png", className="arrow-icon"),
                html.Button("Statistical Analysis", className="red-button")
            ], className="param-row"),

            html.P("Cell Power, kW"),
            html.Div([
                html.Button("Raw Data", className="red-button"),
                html.Img(src="/assets/arrow.png", className="arrow-icon"),
                html.Button("Statistical Analysis", className="red-button")
            ], className="param-row"),

            html.Div([
            html.Button("Recovery Statistical Analysis", className="red-button")
            ], className="param-row"),

            html.P("CORRELATIONS", className="kpi-label"),
            html.Div([
                html.Img(src="/assets/2dscatter.png", id="btn-2d", className="image-button"),
                html.Img(src="/assets/3dscatter.png", id="btn-3d", className="image-button"),
                html.Img(src="/assets/timeseries.png", id="btn-time", className="image-button")
            ], className="blue-buttons-row"),
        ], className="section-c")

    ], className="main-container")
])
