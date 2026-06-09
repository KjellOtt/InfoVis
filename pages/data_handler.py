import pandas as pd

def bereinigen(filepath="Daten/wein.csv"):
    """Bereinigt die CSV-Datei und entfernt Duplikate"""
    try:
        df = pd.read_csv(filepath)
        print(f" Datei '{filepath}' geladen. Start mit {len(df)} Datenpunkten.")
    except FileNotFoundError:
        print(f" Fehler: Die Datei '{filepath}' wurde nicht gefunden.")
        return None

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    for col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = df[col].str.rstrip('.')
        df[col] = df[col].str.replace(r"[^0-9\.\-eE+]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f" Bereinigung abgeschlossen. Verbleibende Datenpunkte: {len(df)}")
    return df
