import pandas as pd


def bereinige_daten(df: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt das DataFrame und entfernt leere Spalten und doppelte Zeilen."""
    cleaned = df.copy()
    cleaned.columns = [col.strip() if isinstance(col, str) else col for col in cleaned.columns]
    cleaned = cleaned.loc[:, [not str(col).startswith("Unnamed:") for col in cleaned.columns]]
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.drop_duplicates()
    return cleaned


def zeige_uebersicht(df: pd.DataFrame) -> None:
    """Gibt die bereinigte Tabelle und Informationen zu Zeilen und Spalten aus."""
    cleaned = bereinige_daten(df)
    row_count, col_count = cleaned.shape

    print("\n=== Übersicht der bereinigten Daten ===")
    print(f"Anzahl Zeilen: {row_count}")
    print(f"Anzahl Spalten: {col_count}")
    print("\nBereinigte Tabelle:\n")
    print(cleaned.to_string(index=False))
    print("\nSpaltennamen:")
    print(", ".join(cleaned.columns.astype(str)))
