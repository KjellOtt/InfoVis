from dash import html, dash_table
from typing import cast
from pages.navbar import create_navbar


def create_cleaned_table_layout(instance):
    """Erstellt das Layout der Subpage mit der bereinigten Tabelle im Dash-Design."""
    df = instance.df

    return html.Div([
        create_navbar(),

        html.Div(className="container-fluid py-4", children=[
            html.Div(className="row mb-4", children=[
                html.Div(className="col-12", children=[
                    html.Div(className="card shadow-sm border-0", children=[
                        html.Div(className="card-body py-4", children=[
                            html.H1(
                                "Bereinigte Tabelle",
                                className="text-center mb-2 fw-bold"
                            ),
                            html.P(
                                "Übersicht über die bereinigten Daten.",
                                className="text-center text-muted mb-0"
                            )
                        ])
                    ])
                ])
            ]),

            html.Div(className="row g-3 mb-4", children=[
                html.Div(className="col-md-4", children=[
                    html.Div(className="card shadow-sm border-0 h-100", children=[
                        html.Div(className="card-body text-center", children=[
                            html.H6("Zeilen", className="card-title text-muted mb-2"),
                            html.Div(str(len(df)), className="display-6 fw-bold text-primary")
                        ])
                    ])
                ]),
                html.Div(className="col-md-4", children=[
                    html.Div(className="card shadow-sm border-0 h-100", children=[
                        html.Div(className="card-body text-center", children=[
                            html.H6("Spalten", className="card-title text-muted mb-2"),
                            html.Div(str(len(df.columns)), className="display-6 fw-bold text-success")
                        ])
                    ])
                ]),
                html.Div(className="col-md-4", children=[
                    html.Div(className="card shadow-sm border-0 h-100", children=[
                        html.Div(className="card-body text-center", children=[
                            html.H6("Nicht-Null-Werte", className="card-title text-muted mb-2"),
                            html.Div(str(int(df.notna().sum().sum())), className="display-6 fw-bold text-warning")
                        ])
                    ])
                ]),
            ]),

            html.Div(className="row", children=[
                html.Div(className="col-12", children=[
                    html.Div(className="card shadow-sm border-0", children=[
                        html.Div(className="card-header bg-white border-0 py-3", children=[
                            html.H5("Bereinigte Daten", className="mb-0 fw-semibold")
                        ]),
                        html.Div(className="card-body p-0", children=[
                            dash_table.DataTable(
                                id="cleaned-table",
                                columns=[{"name": col, "id": col} for col in df.columns],
                                data=cast(
                                    list[dict[str | int | float, str | int | float | bool]],
                                    df.to_dict("records")
                                ),
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                                fixed_rows={"headers": True},
                                style_table={
                                    "overflowX": "auto",
                                    "maxHeight": "720px",
                                    "overflowY": "auto",
                                },
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "8px",
                                    "whiteSpace": "normal",
                                    "fontSize": "13px",
                                    "minWidth": "120px",
                                    "maxWidth": "260px",
                                    "border": "1px solid #e9ecef",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #dee2e6",
                                },
                                style_data={
                                    "backgroundColor": "white",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "#fcfcfd",
                                    }
                                ],
                            )
                        ])
                    ])
                ])
            ])
        ])
    ])
