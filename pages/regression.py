from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from pages.navbar import create_navbar
from pages.regression_model import RegressionModel
from pages.cleaned_table import create_cleaned_table_layout
from pages.k_means_clustering import create_kmeans_layout


class Regression:
    def __init__(self, df):
        self.df = df
        self.attributes = list(df.columns)
        self.model = RegressionModel(df)

        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()

        print(" Dash-Server wird gestartet. Öffne http://127.0.0.1:8050/ im Browser")

    def setup_layout(self):
        """Erstellt das Layout mit Multi-Page Navigation"""
        self.app.layout = dbc.Container([
            dcc.Location(id="url", refresh=False),
            create_navbar(),
            html.Div(id="page-content", className="mt-4")
        ], fluid=True, className="p-0")

    def setup_callbacks(self):
        """Definiert die Page-Umschaltung"""

        @self.app.callback(
            Output("page-content", "children"),
            Input("url", "pathname")
        )
        def display_page(pathname):
            if pathname == "/cleaned-table":
                return create_cleaned_table_layout(self)
            if pathname == "/kmeans":
                return create_kmeans_layout(self)
            return self._create_regression_layout()

        @self.app.callback(
            [Output("regression-plot", "figure"),
             Output("metrics-output", "children")],
            [Input("x-dropdown", "value"),
             Input("y-dropdown", "value")]
        )
        def update_output(x_attr, y_attr):
            if not x_attr or not y_attr or x_attr == y_attr:
                fig = go.Figure()
                fig.add_annotation(text="Wähle zwei verschiedene Attribute",
                                   xref="paper", yref="paper", x=0.5, y=0.5,
                                   showarrow=False, font={"size": 20})
                return fig, "Bitte zwei verschiedene Attribute auswählen."

            return self.model.calculate_regression(x_attr, y_attr)

    def _create_regression_layout(self):
        """Layout für Regression-Seite"""
        return html.Div(className="container-fluid p-4", children=[
            html.H1("Lineare Regression - Wein Daten", className="text-center mb-4"),
            html.Div(className="row mb-3", children=[
                html.Div(className="col-md-3", children=[
                    html.Div(className="card mb-3", children=[
                        html.Div(className="card-body", children=[
                            html.Label("X-Achse:", className="form-label"),
                            dcc.Dropdown(id="x-dropdown",
                                         options=[{"label": attr, "value": attr} for attr in self.attributes],
                                         placeholder="Wähle X-Achse"),
                            html.Br(), html.Br(),
                            html.Label("Y-Achse:", className="form-label"),
                            dcc.Dropdown(id="y-dropdown",
                                         options=[{"label": attr, "value": attr} for attr in self.attributes],
                                         placeholder="Wähle Y-Achse")
                        ])
                    ])
                ]),
                html.Div(className="col-md-9", children=[
                    html.Div(className="row", children=[
                        html.Div(className="col-lg-8 mb-3", children=[
                            html.Div(className="card", children=[
                                html.Div(className="card-body p-2", children=[
                                    dcc.Graph(id="regression-plot")
                                ])
                            ])
                        ]),
                        html.Div(className="col-lg-4 mb-3", children=[
                            html.Div(className="card", children=[
                                html.Div(className="card-body", children=[
                                    html.H5("Metriken", className="card-title"),
                                    html.Div(id="metrics-output", className="small",
                                             style={"fontFamily": "monospace", "whiteSpace": "pre-wrap",
                                                    "overflowY": "auto", "height": "500px"})
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        ])

    def run(self):
        """Startet den Dash-Server"""
        self.app.run(debug=False)
