from dash import dcc, html, dash_table
from typing import cast
from pages.navbar import create_navbar

def create_cleaned_table_layout(regression_instance):
    """Erstellt das Layout der Subpage mit der bereinigten Tabelle."""
    return html.Div(children=[
        html.Div(className="container-fluid p-4", children=[
            html.H1("Bereinigte Tabelle", className="text-center mb-4"),

            html.Div(className="row g-3 mb-3", children=[
                html.Div(className="col-md-4", children=[
                    html.Div(className="card", children=[
                        html.Div(className="card-body", children=[
                            html.H6("Zeilen", className="card-title text-muted"),
                            html.Div(str(len(regression_instance.df)), className="h3 mb-0")
                        ])
                    ])
                ]),
                html.Div(className="col-md-4", children=[
                    html.Div(className="card", children=[
                        html.Div(className="card-body", children=[
                            html.H6("Spalten", className="card-title text-muted"),
                            html.Div(str(len(regression_instance.df.columns)), className="h3 mb-0")
                        ])
                    ])
                ]),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-body", children=[
                    html.H5("Bereinigte Daten", className="card-title"),
                    dash_table.DataTable(
                        id="cleaned-table",
                        columns=[{"name": col, "id": col} for col in regression_instance.df.columns],
                        data=cast(
                            list[dict[str | int | float, str | int | float | bool]],
                            regression_instance.df.to_dict("records")
                        ),
                        page_size=20,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto", "maxHeight": "700px", "overflowY": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "6px",
                            "whiteSpace": "normal",
                            "fontSize": "13px"
                        },
                        style_header={"fontWeight": "bold"},
                    )
                ])
            ])
        ])
    ])
