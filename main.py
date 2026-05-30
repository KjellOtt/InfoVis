import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def bereinigen(df="Daten/wein.csv"):
    try:
        df = pd.read_csv(df)
        print(
            f" Datei '{df}' geladen. Start mit {len(df)} Datenpunkten."
        )
    except FileNotFoundError:
        print(f" Fehler: Die Datei '{df}' wurde nicht gefunden.")
        return None
    
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # Alle Spalten in Float umwandeln
    for col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = df[col].str.replace(r"[^0-9\.\-eE+]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f" Bereinigung abgeschlossen. Verbleibende Datenpunkte: {len(df)}")
    return df

class Regression:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.attributes = list(df.columns)
        self.model = None
        self.x_attr = None
        self.y_attr = None
        
        # Dash App erstellen
        self.app = Dash(__name__)
        self.app.layout = self.create_layout()
        
        # Callbacks definieren
        self.setup_callbacks()
        
        # Server starten
        print(" Dash-Server wird gestartet. Öffne http://127.0.0.1:8050/ im Browser")
        self.app.run_server(debug=False, use_reloader=False)
    
    def create_layout(self):
        """Erstellt das Layout der Dash-Anwendung"""
        return html.Div([
            html.H1("Lineare Regression - Wein Daten", style={"textAlign": "center", "marginBottom": 30}),
            
            # Kontrollbereich
            html.Div([
                html.Div([
                    html.Label("X-Achse:"),
                    dcc.Dropdown(
                        id="x-dropdown",
                        options=[{"label": attr, "value": attr} for attr in self.attributes],
                        placeholder="Wähle X-Achse",
                        style={"width": "100%", "minWidth": "420px", "fontSize": "16px", "padding": "10px"}
                    )
                ], style={"display": "inline-block", "marginRight": 40, "width": "160px"}),
                
                html.Div([
                    html.Label("Y-Achse:"),
                    dcc.Dropdown(
                        id="y-dropdown",
                        options=[{"label": attr, "value": attr} for attr in self.attributes],
                        placeholder="Wähle Y-Achse",
                        style={"width": "100%", "minWidth": "160px", "fontSize": "16px", "padding": "10px"}
                    )
                ], style={"display": "inline-block", "width": "420px"})
            ], style={"marginBottom": 30, "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),
            
            # Hauptbereich: Plot und Metriken
            html.Div([
                html.Div([
                    dcc.Graph(id="regression-plot")
                ], style={"width": "70%", "display": "inline-block", "verticalAlign": "top"}),
                
                html.Div([
                    html.Div(id="metrics-output", style={
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "5px",
                        "fontFamily": "monospace",
                        "whiteSpace": "pre-wrap",
                        "overflowY": "auto",
                        "height": "500px"
                    })
                ], style={"width": "28%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "2%"})
            ], style={"marginBottom": 30})
        ], style={"padding": "20px", "maxWidth": "1400px", "margin": "0 auto"})
    
    def setup_callbacks(self):
        """Definiert die Callbacks für interaktive Updates"""
        @self.app.callback(
            [Output("regression-plot", "figure"),
             Output("metrics-output", "children")],
            [Input("x-dropdown", "value"),
             Input("y-dropdown", "value")]
        )
        def update_output(x_attr, y_attr):
            if not x_attr or not y_attr or x_attr == y_attr:
                # Leeres Plot bei fehlender oder gleicher Auswahl
                fig = go.Figure()
                fig.add_annotation(
                    text="Wähle zwei verschiedene Attribute",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font={"size": 20}
                )
                return fig, "Bitte zwei verschiedene Attribute auswählen."
            
            return self.plot_regression(x_attr, y_attr)
    
    def plot_regression(self, x_attr: str, y_attr: str):
        """Berechnet und visualisiert die lineare Regression"""

        # 1. Nur die zwei relevanten Spalten isolieren und Zeilen mit NaNs verwerfen
        temp_df = self.df[[x_attr, y_attr]].dropna()
        
        # Sicherheits-Check: Haben wir nach dem Löschen noch genug Daten?
        if len(temp_df) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Nicht genug gültige Datenpunkte ohne NaNs", 
                               showarrow=False, font={"size": 20})
            return fig, "Zu viele fehlende Werte für diese Kombination."

        # 2. X und y aus dem bereinigten, temporären DataFrame ziehen
        X = temp_df[[x_attr]].values
        y = temp_df[y_attr].values
        # Modell trainieren
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Vorhersagen treffen
        y_pred = self.model.predict(X)
        
        # Metriken berechnen
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        # Regressionslinie berechnen
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = self.model.predict(X_line)
        
        # Plot mit Plotly erstellen
        fig = go.Figure()
        
        # Datenpunkte
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            name='Datenpunkte',
            marker=dict(size=8, opacity=0.6, color='steelblue'),
            hovertemplate='<b>%{customdata[0]}</b><br>%{x}<br>%{y}<extra></extra>',
            customdata=[[f'{x_attr}/{y_attr}'] for _ in range(len(X))]
        ))
        
        # Regressionslinie
        fig.add_trace(go.Scatter(
            x=X_line.flatten(),
            y=y_line,
            mode='lines',
            name='Regressionslinie',
            line=dict(color='red', width=3),
            hovertemplate='%{x:.3f} → %{y:.3f}<extra></extra>'
        ))
        
        # Layout aktualisieren
        fig.update_layout(
            title=f"Lineare Regression: {y_attr} vs {x_attr}",
            xaxis_title=x_attr,
            yaxis_title=y_attr,
            hovermode='closest',
            plot_bgcolor='white',
            height=550,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            font=dict(size=12)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Metriken als Text formatieren
        metrics_text = f"""Regressionsmetriken
{'='*30}

Attribute:
  X-Achse: {x_attr}
  Y-Achse: {y_attr}

Koeffizient:
  Steigung: {self.model.coef_[0]:.6f}
  Konstante: {self.model.intercept_:.6f}

Qualitätsmetriken:
  R² Score: {r2:.6f}
  RMSE: {rmse:.6f}
  MAE: {mae:.6f}
  MSE: {mse:.6f}

Datenpunkte: {len(self.df)}
"""
        
        return fig, metrics_text

        
if __name__ == "__main__":
    df = bereinigen("Daten/wein.csv")
    if df is not None:
        regression = Regression(df)