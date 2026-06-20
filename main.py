import pandas as pd
from dash import Dash, html, dcc, Input, Output
from pages import übersicht, matrix as matrix_page, baum as baum_page, evaluierung as evaluierung_page

app = Dash(
    __name__,
    external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"],
    suppress_callback_exceptions=True
)

def load_data(csv_path: str = "Daten/Titanic.csv") -> pd.DataFrame:
    """Lädt die CSV-Datei und gibt das Roh-DataFrame zurück."""
    return pd.read_csv(csv_path)

def chose_target_column(df: pd.DataFrame, preferred: str = "Survived") -> str | None:
    if preferred in df.columns:
        return preferred
    lower_cols = {col.lower(): col for col in df.columns if isinstance(col, str)}
    if preferred.lower() in lower_cols:
        return lower_cols[preferred.lower()]
    return None

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Navbar
    html.Nav([
        html.Div([
            html.A("InfoVis Praktikum", className="navbar-brand", href="/"),
            html.Button([
                html.Span(className="navbar-toggler-icon")
            ], className="navbar-toggler", type="button", **{"data-bs-toggle": "collapse", "data-bs-target": "#navbarNav"}),
            html.Div([
                html.Ul([
                    html.Li([
                        dcc.Link("Übersicht", href="/uebersicht", className="nav-link", id="uebersicht-link")
                    ], className="nav-item"),
                    html.Li([
                        dcc.Link("Evaluierung", href="/evaluierung", className="nav-link", id="evaluierung-link")
                    ], className="nav-item"),
                    html.Li([
                        dcc.Link("Matrix", href="/matrix", className="nav-link", id="matrix-link")
                    ], className="nav-item"),
                    html.Li([
                        dcc.Link("Baum", href="/baum", className="nav-link", id="baum-link")
                    ], className="nav-item"),
                ], className="navbar-nav")
            ], className="collapse navbar-collapse", id="navbarNav")
        ], className="container-fluid")
    ], className="navbar navbar-expand-lg navbar-dark bg-dark mb-4"),
    
    # Seiteninhalt
    html.Div(id='page-content', className="container py-4")
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    raw_data = load_data()
    cleaned = übersicht.bereinige_daten(raw_data)
    target_column = chose_target_column(cleaned)
    
    if pathname == '/evaluierung':
        if target_column is None:
            return html.Div("Keine Zielspalte gefunden.", className="alert alert-warning")
        return html.Div([html.H1("Evaluierung"), evaluierung_page.prepare_evaluation_html(cleaned, target_column)])
    
    elif pathname == '/matrix':
        if target_column is None:
            return html.Div("Keine Zielspalte gefunden.", className="alert alert-warning")
        return html.Div([html.H1("Matrix"), matrix_page.render_matrix(cleaned, target_column)])
    
    elif pathname == '/baum':
        if target_column is None:
            return html.Div("Keine Zielspalte gefunden.", className="alert alert-warning")
        return html.Div([html.H1("Baum"), baum_page.render_tree(cleaned, target_column)])
    
    else: # /uebersicht oder / oder unbekannt
        return html.Div([html.H1("Übersicht"), übersicht.zeige_uebersicht(raw_data)])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
