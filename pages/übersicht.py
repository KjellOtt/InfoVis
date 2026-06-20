import pandas as pd
import numpy as np
from dash import html, dcc, dash_table

def bereinige_daten(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt das DataFrame:
    - Entfernt leere Spalten und Zeilen.
    - Strippt Leerzeichen von Spaltennamen.
    - Behandelt fehlende Werte (Imputation).
    - Entfernt Duplikate.
    - Behandelt Datentypen.
    """
    cleaned = df.copy()
    
    # 1. Spaltennamen bereinigen
    cleaned.columns = [col.strip() if isinstance(col, str) else col for col in cleaned.columns]
    
    # 2. Unnötige Spalten entfernen
    cleaned = cleaned.loc[:, [not str(col).startswith("Unnamed:") for col in cleaned.columns]]
    
    # 3. Vollständig leere Zeilen/Spalten entfernen
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    
    # 4. Duplikate entfernen
    cleaned = cleaned.drop_duplicates()
    
    # 5. Fehlende Werte behandeln
    # Numerische Spalten mit Median füllen
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        
    # Kategorische Spalten mit Mode füllen
    cat_cols = cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if not cleaned[col].mode().empty:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
        else:
            cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned

def zeige_uebersicht(df: pd.DataFrame):
    """Gibt Dash-Komponenten für die Übersicht der bereinigten Daten zurück."""
    cleaned = bereinige_daten(df)
    row_count, col_count = cleaned.shape
    
    stats = cleaned.describe(include='all').transpose().reset_index()
    
    return html.Div([
        html.Div([
            html.Div([
                html.H5("Datensatz-Statistik", className="card-title"),
                html.P(f"Anzahl Zeilen: {row_count} | Anzahl Spalten: {col_count}"),
                html.Div([
                    dash_table.DataTable(
                        data=stats.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in stats.columns],
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
