import dash
from dash import html, dcc, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import bereinigung

BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"

def remove_outliers_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return s[(s >= lower) & (s <= upper)]

def split_outliers_series(s: pd.Series):
    """
    Gibt zwei Serien zurück:
    - clean: Werte innerhalb der IQR-Grenzen
    - outliers: Werte außerhalb der IQR-Grenzen
    """
    if s.empty:
        return s, s

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clean = s[(s >= lower) & (s <= upper)]
    outliers = s[(s < lower) | (s > upper)]
    return clean, outliers

def build_numeric_figure(plot_series: pd.Series, attr: str, outlier_count: int, outlier_mode: str):
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.28, 0.72],
        horizontal_spacing=0.12,
        subplot_titles=(f'Boxplot: {attr}', f'Histogramm: {attr}')
    )

    box_color = 'red' if outlier_mode == 'highlight' and outlier_count > 0 else 'steelblue'
    boxpoints = 'outliers' if outlier_mode != 'hide' else False

    # Boxplot
    fig.add_trace(
        go.Box(
            y=plot_series,
            orientation='v',
            name=attr,
            boxpoints=boxpoints,
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(color=box_color, size=6),
            line=dict(color='steelblue')
        ),
        row=1,
        col=1
    )

    # Histogramm
    fig.add_trace(
        go.Histogram(
            x=plot_series,
            nbinsx=30,
            marker=dict(color='steelblue', line=dict(color='black', width=1)),
            opacity=0.8,
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.update_xaxes(title_text=attr, row=1, col=2)
    fig.update_yaxes(title_text=attr, row=1, col=1)
    fig.update_yaxes(title_text='Häufigkeit', row=1, col=2)
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=30), height=500, showlegend=False)
    return fig

def prepare_dataframe():
    df = bereinigung.clean_dataframe(bereinigung.file_path, verbose=False)
    if df is None:
        raise RuntimeError("Bereinigung fehlgeschlagen.")
    df = df.reset_index(drop=True)
    return df

def build_app(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col].dtype) and col not in numeric_cols:
            numeric_cols.append(col)

    categorical_candidates = [c for c in [
        'Known As', 'Full Name', 'Positions Played', 'Best Position',
        'Nationality', 'Club Name', 'Club Position', 'Preferred Foot',
        'Attacking Work Rate', 'Defensive Work Rate'
    ] if c in df.columns]

    all_attrs = sorted(list(dict.fromkeys(numeric_cols + categorical_candidates)))

    nat_options = ['(alle)'] + sorted(df['Nationality'].dropna().unique().tolist()) if 'Nationality' in df.columns else ['(alle)']
    club_options = ['(alle)'] + sorted(df['Club Name'].dropna().unique().tolist()) if 'Club Name' in df.columns else ['(alle)']

    app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP_CSS])

    app.layout = html.Div([
        # Header
        html.Nav(className="navbar navbar-dark bg-dark mb-3", children=[
            html.Div(className="container-fluid", children=[
                html.Span("InfoVis - Visualisierung", className="navbar-brand mb-0 h1")
            ])
        ]),

        html.Div(className="container-fluid", children=[
            html.Div(className="row", children=[
                # Sidebar / Controls
                html.Div(className="col-md-3", children=[
                    html.Div(className="card mb-3", children=[
                        html.Div(className="card-body", children=[
                            html.H5("Filter", className="card-title"),
                            html.Label("Nationality", className="form-label"),
                            dcc.Dropdown(id='dropdown-nat', options=[{'label': v, 'value': v} for v in nat_options], value='(alle)', clearable=False),
                            html.Br(),
                            html.Label("Club", className="form-label"),
                            dcc.Dropdown(id='dropdown-club', options=[{'label': v, 'value': v} for v in club_options], value='(alle)', clearable=False),
                            html.Br(),
                            html.Label("Attribute (max 2)", className="form-label"),
                            dcc.Dropdown(id='dropdown-attrs', options=[{'label': a, 'value': a} for a in all_attrs],
                                         value=[], multi=True, placeholder='Wähle 1-2 Attribute'),
                            html.Small("Numerische Attribute: Histogramm/Scatter. Kategorial: Balken.", className="form-text text-muted"),
                            html.Br(),
                            html.Div(className="mt-2", children=[
                                html.Label("Ausreißer", className="form-label"),
                                dcc.RadioItems(
                                    id='outlier-mode',
                                    options=[
                                        {'label': 'Ausreißer standardmäßig mit an', 'value': 'show'},
                                        {'label': 'Ausreißer markieren', 'value': 'highlight'},
                                        {'label': 'Ausreißer ausblenden', 'value': 'hide'},
                                    ],
                                    value='show',
                                    inputStyle={'margin-right': '6px', 'margin-left': '10px'}
                                )
                            ])
                        ])
                    ]),
                    html.Div(className="card mb-3", children=[
                        html.Div(className="card-body", children=[
                            html.H6("Informationen", className="card-subtitle mb-2 text-muted"),
                            html.Div(id='info-count', className="h5"),
                            html.Div(id='info-selected', className="small text-muted")
                        ])
                    ]),
                    html.Div(id='ui-warnings')  # for dynamic alerts
                ]),

                # Main content: Plot + compare
                html.Div(className="col-md-9", children=[
                    html.Div(className="card mb-3", children=[
                        html.Div(className="card-body", children=[
                            dcc.Graph(id='main-plot', style={'height':'600px'}),
                            html.Div(id='warning-text', style={'color':'red', 'marginTop':'6px'})
                        ])
                    ]),

                    html.Div(className="card", children=[
                        html.Div(className="card-body", children=[
                            html.H5("Zwei Datenpunkte vergleichen", className="card-title"),

                            html.Div(className="row g-2 align-items-end", children=[
                                html.Div(className="col-md-4", children=[
                                    html.Label("Index 1", className="form-label"),
                                    dcc.Input(
                                        id='idx1',
                                        type='number',
                                        min=0,
                                        max=len(df) - 1,
                                        value=0,
                                        className="form-control"
                                    )
                                ]),
                                html.Div(className="col-md-4", children=[
                                    html.Label("Index 2", className="form-label"),
                                    dcc.Input(
                                        id='idx2',
                                        type='number',
                                        min=0,
                                        max=len(df) - 1,
                                        value=1,
                                        className="form-control"
                                    )
                                ]),
                                html.Div(className="col-md-4", children=[
                                    html.Label("Ausgewähltes Ziel", className="form-label"),
                                    dcc.Dropdown(
                                        id='compare-target',
                                        options=[
                                            {'label': 'Index 1 setzen', 'value': 'idx1'},
                                            {'label': 'Index 2 setzen', 'value': 'idx2'}
                                        ],
                                        value='idx1',
                                        clearable=False
                                    )
                                ]),
                            ]),

                            html.Div(className="mt-3", children=[
                                html.Button("Vergleichen", id='btn-compare', n_clicks=0, className="btn btn-primary")
                            ]),

                            html.Hr(),

                            html.H6("Zeilenübersicht", className="card-subtitle mb-2 text-muted"),
                            html.Div(
                                dash_table.DataTable(
                                    id='row-table',
                                    columns=[{'name': 'Index', 'id': 'Index'}] + [
                                        {'name': col, 'id': col} for col in df.columns[:8]
                                    ],
                                    data=[
                                        {'Index': i, **{col: str(df.iloc[i][col]) for col in df.columns[:8]}}
                                        for i in range(len(df))
                                    ],
                                    row_selectable='single',
                                    selected_rows=[],
                                    page_size=10,
                                    sort_action='native',
                                    filter_action='native',
                                    style_table={'overflowX': 'auto', 'maxHeight': '350px', 'overflowY': 'auto'},
                                    style_cell={
                                        'textAlign': 'left',
                                        'padding': '6px',
                                        'whiteSpace': 'normal',
                                        'fontSize': '13px'
                                    },
                                    style_header={'fontWeight': 'bold'},
                                )
                            ),

                            html.Hr(),
                            html.Div(id='compare-output')
                        ])
                    ])
                ])
            ])
        ])
    ], style={'paddingBottom': '30px'})

    @app.callback(
        Output('idx1', 'value'),
        Output('idx2', 'value'),
        Input('row-table', 'selected_rows'),
        State('compare-target', 'value'),
        prevent_initial_call=True
    )
    def fill_compare_indices(selected_rows, target):
        if not selected_rows:
            raise dash.exceptions.PreventUpdate

        selected_idx = selected_rows[0]

        if target == 'idx2':
            return dash.no_update, selected_idx
        return selected_idx, dash.no_update

    # Update plot + info
    @app.callback(
        Output('main-plot', 'figure'),
        Output('warning-text', 'children'),
        Output('info-count', 'children'),
        Output('info-selected', 'children'),
        Output('ui-warnings', 'children'),
        Input('dropdown-nat', 'value'),
        Input('dropdown-club', 'value'),
        Input('dropdown-attrs', 'value'),
        Input('outlier-mode', 'value')
    )

    def update_plot(nat, club, attrs, outlier_mode):
        df_f = df.copy()
        warnings_list = []

        if nat and nat != '(alle)' and 'Nationality' in df_f.columns:
            df_f = df_f[df_f['Nationality'] == nat]
        if club and club != '(alle)' and 'Club Name' in df_f.columns:
            df_f = df_f[df_f['Club Name'] == club]

        outlier_mode = outlier_mode or 'show'

        # enforce max 2 attributes in the UI sense: show warning if >2 selected and ignore extras
        if attrs and len(attrs) > 2:
            warnings_list.append(html.Div(className="alert alert-warning", children=[
                html.Strong("Hinweis: "), f"Es sind {len(attrs)} Attribute ausgewählt. Es werden nur die ersten 2 verwendet."
            ]))
            attrs = attrs[:2]

        info_count = f"{len(df_f)} Datenpunkte"
        info_selected = f"{len(attrs) or 0} Attribut(e) ausgewählt"

        if not attrs:
            fig = px.scatter()
            fig.update_layout(
                xaxis={'visible': False}, yaxis={'visible': False},
                annotations=[{'text':'Wähle ein Attribut', 'xref':'paper','yref':'paper','showarrow':False, 'x':0.5,'y':0.5}]
            )
            return fig, '', info_count, info_selected, warnings_list

        # Single attribute
        if len(attrs) == 1:
            attr = attrs[0]
            if attr in df_f.select_dtypes(include=[np.number]).columns:
                series = df_f[attr].dropna().astype(float)
                clean_series, outlier_series = split_outliers_series(series)

                if outlier_mode == 'hide':
                    plot_series = clean_series
                else:
                    plot_series = series

                if plot_series.empty:
                    fig = px.scatter()
                    fig.update_layout(
                        annotations=[{
                            'text': 'Keine Daten nach Filterung',
                            'xref': 'paper',
                            'yref': 'paper',
                            'showarrow': False,
                            'x': 0.5,
                            'y': 0.5
                        }],
                        xaxis={'visible': False},
                        yaxis={'visible': False}
                    )
                    return fig, '', info_count, info_selected, warnings_list

                fig = build_numeric_figure(plot_series, attr, len(outlier_series), outlier_mode)
                outlier_msg = f" | Ausreißer: {len(outlier_series)}" if len(outlier_series) > 0 else " | Keine Ausreißer"
                fig.update_layout(title=f'Verteilung: {attr} (n={len(plot_series)}{outlier_msg})')

                warning_text = ''
                if outlier_mode == 'hide' and len(outlier_series) > 0:
                    warning_text = f"{len(outlier_series)} Ausreißer wurden ausgeblendet."
                elif outlier_mode == 'highlight' and len(outlier_series) > 0:
                    warning_text = f"{len(outlier_series)} Ausreißer markiert."

                return fig, warning_text, info_count, info_selected, warnings_list
            else:
                if attr not in df_f.columns:
                    fig = px.scatter()
                    return fig, f"Attribut {attr} nicht in Daten.", info_count, info_selected, warnings_list
                vals = df_f[attr].dropna().astype(str)
                if vals.empty:
                    fig = px.scatter()
                    fig.update_layout(annotations=[
                        {'text': 'Keine Daten nach Filterung', 'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                         'x': 0.5, 'y': 0.5}],
                                      xaxis={'visible': False}, yaxis={'visible': False})
                    return fig, '', info_count, info_selected, warnings_list
                counts = vals.value_counts().nlargest(20).sort_values()
                fig = px.bar(x=counts.index, y=counts.values, labels={'x': attr, 'y': 'Häufigkeit'},
                             title=f'Verteilung (Top {len(counts)}): {attr}')
                fig.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=50, b=80))
                return fig, '', info_count, info_selected, warnings_list

        # Two attributes -> scatter if both numeric
        a1, a2 = attrs[0], attrs[1]
        numeric = df_f.select_dtypes(include=[np.number]).columns
        if a1 not in numeric or a2 not in numeric:
            fig = px.scatter()
            msg = 'Scatterplots sind nur für numerische Attribute möglich.'
            return fig, msg, info_count, info_selected, warnings_list
        data = df_f[[a1, a2]].dropna().astype(float)

        clean_x, out_x = split_outliers_series(data[a1])
        clean_y, out_y = split_outliers_series(data[a2])

        clean_idx = clean_x.index.intersection(clean_y.index)
        out_idx = data.index.difference(clean_idx)

        data_clean = data.loc[clean_idx]
        data_out = data.loc[out_idx]

        if outlier_mode == 'show':
            fig = px.scatter(
                data,
                x=a1,
                y=a2,
                title=f'{a1} vs {a2} (n={len(data)})',
                opacity=0.65
            )
            warning_text = f"{len(data_out)} Ausreißer sichtbar, aber nicht markiert." if len(data_out) > 0 else ''
        elif outlier_mode == 'highlight':
            fig = px.scatter(
                data_clean,
                x=a1,
                y=a2,
                title=f'{a1} vs {a2} (n={len(data)})',
                opacity=0.7
            )
            if not data_out.empty:
                fig.add_scatter(
                    x=data_out[a1],
                    y=data_out[a2],
                    mode='markers',
                    name='Ausreißer',
                    marker=dict(color='red', size=10, symbol='x')
                )
            warning_text = f"{len(data_out)} Ausreißer markiert." if len(data_out) > 0 else ''
        else:
            fig = px.scatter(
                data_clean,
                x=a1,
                y=a2,
                title=f'{a1} vs {a2} (n={len(data_clean)})',
                opacity=0.7
            )
            warning_text = f"{len(data_out)} Ausreißer wurden ausgeblendet." if len(data_out) > 0 else ''

        if data.empty:
            fig = px.scatter()
            fig.update_layout(
                annotations=[{
                    'text': 'Keine Daten nach Filterung',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'x': 0.5,
                    'y': 0.5
                }],
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig, '', info_count, info_selected, warnings_list

        fig.update_layout(margin=dict(l=10, r=10, t=50, b=30))
        return fig, warning_text, info_count, info_selected, warnings_list

    @app.callback(
        Output('compare-output', 'children'),
        Input('btn-compare', 'n_clicks'),
        Input('row-table', 'selected_rows'),
        State('idx1', 'value'),
        State('idx2', 'value')
    )
    def compare_points(n_clicks, selected_rows, idx1, idx2):
        if selected_rows:
            pass

        if n_clicks is None or n_clicks == 0:
            return ''

        try:
            idx1 = int(idx1)
            idx2 = int(idx2)
        except Exception:
            return html.Div(className="alert alert-danger", children="Bitte gültige Indizes eingeben (ganze Zahlen).")

        if idx1 == idx2:
            return html.Div(className="alert alert-warning", children="Wähle zwei verschiedene Indizes.")
        if not (0 <= idx1 < len(df)) or not (0 <= idx2 < len(df)):
            return html.Div(className="alert alert-danger",
                            children=f"Indizes müssen zwischen 0 und {len(df) - 1} liegen.")

        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]

        compare_rows = []
        for col in df.columns:
            v1 = row1[col]
            v2 = row2[col]
            if (
                    pd.notna(v1)
                    and pd.notna(v2)
                    and isinstance(v1, (int, float, np.integer, np.floating))
                    and isinstance(v2, (int, float, np.integer, np.floating))
            ):
                diff = float(v2) - float(v1)
                compare_rows.append({'Attribut': col, 'Index1': v1, 'Index2': v2, 'Differenz': diff})
            else:
                compare_rows.append({'Attribut': col, 'Index1': str(v1), 'Index2': str(v2), 'Differenz': '-'})

        table = dash_table.DataTable(
            data=compare_rows,
            columns=[
                {"name": "Attribut", "id": "Attribut"},
                {"name": f"Index {idx1}", "id": "Index1"},
                {"name": f"Index {idx2}", "id": "Index2"},
                {"name": "Differenz", "id": "Differenz"},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '6px', 'whiteSpace': 'normal'},
            page_size=15,
            sort_action='native',
            filter_action='native'
        )

        return table

    return app

if __name__ == '__main__':
    df = prepare_dataframe()
    app = build_app(df)
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8050)
