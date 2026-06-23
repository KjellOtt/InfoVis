import pandas as pd
import numpy as np
import plotly.express as px
from dash import html, dcc, dash_table

def bereinige_daten(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Bereinigt das DataFrame:
    - Entfernt leere Spalten und Zeilen.
    - Strippt Leerzeichen von Spaltennamen.
    - Behandelt fehlende Werte (Imputation).
    - Behandelt Datentypen.
    - Entfernt irrelevante Features.
    Gibt (bereinigtes df, stats_dict) zurück.
    """
    stats = {}
    cleaned = df.copy()
    
    # 0. Missing Values vorab zählen
    stats['missing_before'] = cleaned.isnull().sum().to_dict()
    
    # 1. Spaltennamen bereinigen
    cleaned.columns = [col.strip() if isinstance(col, str) else col for col in cleaned.columns]
    
    # 2. Unnötige Spalten entfernen
    unnamed_cols = [col for col in cleaned.columns if str(col).startswith("Unnamed:")]
    cleaned = cleaned.drop(columns=unnamed_cols)
    
    # 3. Bereinigung: PassengerId, Name, Ticket, Cabin entfernen
    # Diese Spalten sind zu komplex oder redundant
    to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    existing_to_drop = [c for c in to_drop if c in cleaned.columns]
    cleaned = cleaned.drop(columns=existing_to_drop)
    stats['dropped_features'] = existing_to_drop

    # 4. Vollständig leere Zeilen/Spalten entfernen
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    
    # 5. Fehlende Werte behandeln (Imputation)
    # Numerische Spalten mit Median füllen
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        
    # Kategorische Spalten mit Mode füllen
    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if not cleaned[col].mode().empty:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
        else:
            cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned, stats

def zeige_uebersicht(df: pd.DataFrame):
    """Gibt Dash-Komponenten für die Übersicht der bereinigten Daten zurück."""
    cleaned, bereinigungs_stats = bereinige_daten(df)
    row_count, col_count = cleaned.shape
    
    stats_df = cleaned.describe(include='all').transpose().reset_index()
    
    # Missing Values Info
    missing_info = bereinigungs_stats['missing_before']
    missing_elements = [html.Li(f"{col}: {val} fehlende Werte") for col, val in missing_info.items() if val > 0]
    if not missing_elements:
        missing_elements = [html.Li("Keine fehlenden Werte gefunden.")]

    return html.Div([
        html.Div([
            html.Div([
                html.H5("Zusammenfassung der Bereinigung", className="card-title"),
                html.Ul([
                    html.Li(f"Entfernte Spalten (Irrelevant/Viele fehlende Werte): {', '.join(bereinigungs_stats['dropped_features'])}"),
                    html.Li([
                        html.Strong("Ursprünglich fehlende Werte:"),
                        html.Ul(missing_elements)
                    ])
                ]),
                html.P(f"Resultierende Daten: {row_count} Zeilen, {col_count} Spalten", className="mt-2 fw-bold")
            ], className="card-body")
        ], className="card mb-4 border-info shadow-sm"),

        html.Div([
            html.Div([
                html.H5("Analyse der Datenqualität & Verteilung", className="card-title"),
                html.P("Ein Scatterplot hilft dabei, Ausreißer zu identifizieren und den Zusammenhang zwischen Features (z.B. Alter und Ticketpreis) in Bezug auf die Zielvariable zu verstehen.", className="card-text"),
                dcc.Graph(
                    figure=px.scatter(
                        cleaned, 
                        x="Age", 
                        y="Fare", 
                        color=str(target_column_actual) if target_column_actual else None,
                        hover_data=cleaned.columns,
                        title="Zusammenhang: Alter vs. Ticketpreis (nach Überleben)",
                        labels={"Age": "Alter", "Fare": "Ticketpreis (Fare)", "color": "Überlebt"}
                    ).update_layout(margin=dict(l=20, r=20, t=40, b=20))
                )
            ], className="card-body")
        ], className="card mb-4 shadow-sm"),

        html.Div([
            html.Div([
                html.H5("Statistik (Bereinigt)", className="card-title"),
                html.Div([
                    dash_table.DataTable(
                        data=stats_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in stats_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        page_size=15
                    )
                ], className="table-responsive")
            ], className="card-body")
        ], className="card mb-4"),
        
        html.Div([
            html.Div([
                html.H5("Vorschau (Erste 10 Zeilen)", className="card-title"),
                html.Div([
                    dash_table.DataTable(
                        data=cleaned.head(10).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in cleaned.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ], className="table-responsive")
            ], className="card-body")
        ], className="card")
    ])
