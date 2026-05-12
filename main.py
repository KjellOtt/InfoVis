import pandas as pd
import warnings
from datetime import datetime
import numpy as np
import sys
import traceback

file_path = 'Daten/Praktikum-1.csv'

# Text-Spalten 
TEXT_COLUMNS = [
    'Known As', 'Full Name', 'Positions Played', 'Best Position', 
    'Nationality', 'Image Link', 'Club Name', 'Club Position', 
    'Preferred Foot', 'Attacking Work Rate', 'Defensive Work Rate',
    'National Team Name', 'National Team Image Link', 'National Team Position',
    'Contract Until'
]

# Integer-Spalten
INTEGER_COLUMNS = [
    'Overall', 'Potential', 'Age', 'Height(in cm)', 'Weight(in kg)',
    'TotalStats', 'BaseStats', 'Club Jersey Number', 'Weak Foot Rating',
    'Skill Moves', 'International Reputation', 'National Team Jersey Number',
    'Pace Total', 'Shooting Total', 'Passing Total', 'Dribbling Total',
    'Defending Total', 'Physicality Total',
    'Crossing', 'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys',
    'Dribbling', 'Curve', 'Freekick Accuracy', 'LongPassing', 'BallControl',
    'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance',
    'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots',
    'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
    'Composure', 'Marking', 'Standing Tackle', 'Sliding Tackle',
    'Goalkeeper Diving', 'Goalkeeper Handling', 'GoalkeeperKicking',
    'Goalkeeper Positioning', 'Goalkeeper Reflexes',
    'ST Rating', 'LW Rating', 'LF Rating', 'CF Rating', 'RF Rating', 'RW Rating',
    'CAM Rating', 'LM Rating', 'CM Rating', 'RM Rating', 'LWB Rating',
    'CDM Rating', 'RWB Rating', 'LB Rating', 'CB Rating', 'RB Rating', 'GK Rating'
]

# Float-Spalten 
FLOAT_COLUMNS = [
    'Value(in Euro)', 'Wage(in Euro)', 'Release Clause'
]

# Boolean-Spalten
BOOLEAN_COLUMNS = ['On Loan']

# Datum-Spalten
DATE_COLUMNS = ['Joined On']

def parse_currency(value):
    if pd.isna(value) or value == '-':
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def parse_boolean(value):
    if pd.isna(value) or value == '-':
        return False
    if str(value).lower() in ['yes', 'true', '1']:
        return True
    return False

def parse_integer(value):
    if pd.isna(value) or value == '-' or value == '':
        return np.nan
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return np.nan

def parse_date(value):
    if pd.isna(value) or value == '-' or value == '':
        return pd.NaT
    try:
        return pd.to_datetime(value, errors='coerce')
    except:
        return pd.NaT

def parse_text(value):
    if pd.isna(value):
        return ''
    return str(value).strip()

try:
    # Python-basierter Parser, der verschiedene Trennzeichen erkennt
    # Warnungen unterdrücken für saubere Ausgabe
    warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
    df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='warn')

    print("--- Spalten im Datensatz ---")
    print(df.columns.tolist())
    print(f"\nAnzahl Zeilen (vor Bereinigung): {len(df)}")

    # Spaltennamen bereinigen
    df.columns = df.columns.str.replace("'", "").str.replace('"', "").str.strip()
    df.columns = [col.encode("ascii", "ignore").decode("utf-8") if col.startswith("\ufeff") else col for col in df.columns]

    print(f"\n--- Spalten nach Bereinigung ---")
    print(df.columns.tolist())

    # Text-Spalten parsen
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_text)
        else:
            print(f"WARNUNG: Spalte '{col}' nicht gefunden")

    # Integer-Spalten parsen
    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_integer).astype('Int64') 
        else:
            print(f"WARNUNG: Spalte '{col}' nicht gefunden")

    # Float-Spalten parsen
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_currency).astype('Float64')
        else:
            print(f"WARNUNG: Spalte '{col}' nicht gefunden")

    # Boolean-Spalten parsen
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_boolean).astype('bool')
        else:
            print(f"WARNUNG: Spalte '{col}' nicht gefunden")

    # Datum-Spalten parsen
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)
        else:
            print(f"WARNUNG: Spalte '{col}' nicht gefunden")

    print(f"\nAnzahl Zeilen (nach Bereinigung): {len(df)}")
    print(f"\nDatentypen nach Parsing:")
    print(df.dtypes)

    print(f"\nFehlende Werte (NaN) pro Spalte (Spalten mit Lücken):")
    missing = df.isna().sum()
    missing_filtered = missing[missing > 0]
    if len(missing_filtered) > 0:
        print(missing_filtered)
    else:
        print("Keine fehlenden Werte gefunden!")

except Exception as e:
    print(f"FEHLER: {str(e)}")
    traceback.print_exc()