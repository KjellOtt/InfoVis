import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd


def zeige_wein_tabelle():
    datei_name = "Daten/wein.csv"

    # 1. Daten einlesen
    try:
        df = pd.read_csv(datei_name)
    except FileNotFoundError:
        root_error = tk.Tk()
        root_error.withdraw()  # Versteckt das Hauptfenster
        messagebox.showerror(
            "Fehler",
            f"Die Datei '{datei_name}' wurde nicht gefunden!\n"
            "Stelle sicher, dass sie im selben Ordner wie dieses Skript liegt.",
        )
        return

    print("=" * 50)
    print(" ÜBERBLICK")
    print("=" * 50)
    print(f"Anzahl der Wein-Datenpunkte (Zeilen): {df.shape[0]}")
    print(f"Anzahl der Attribute (Spalten):       {df.shape[1]}")
    print(f"Attributnamen: {', '.join(df.columns)}\n")

    print("=" * 50)
    print(" DATENTYPEN & FEHLENDE WERTE")
    print("=" * 50)
    print(df.info())
    print("\n")
    # 2. Tkinter Fenster aufsetzen
    root = tk.Tk()
    root.title(f"Übersicht: {datei_name} ({len(df)} Datenpunkte)")
    root.geometry("900x600")  # Startgröße des Fensters

    # Container-Frame für Tabelle und Scrollbars
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 3. Tabelle (Treeview) erstellen
    # 'show="headings"' sorgt dafür, dass die leere erste Gitterschnitt-Spalte ausgeblendet wird
    spalten = list(df.columns)
    tabelle = ttk.Treeview(frame, columns=spalten, show="headings")

    # 4. Scrollbars hinzufügen (essentiell bei vielen Zeilen/Spalten)
    y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tabelle.yview)
    x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tabelle.xview)
    tabelle.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    # Layout per Grid-Manager (damit die Scrollbars sauber am Rand kleben)
    tabelle.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")

    # Gewichtung verteilen, damit sich die Tabelle beim Großziehen des Fensters anpasst
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    # 5. Spaltenüberschriften setzen
    for col in spalten:
        tabelle.heading(col, text=col)
        # 'anchor="center"' zentriert den Text in den Zellen
        tabelle.column(col, width=120, anchor="center")

    # 6. Daten in die Tabelle füllen
    for _, row in df.iterrows():
        # pd.notna(x) sorgt dafür, dass leere Zellen (NaN) einfach als leerer Text angezeigt werden
        zeilen_werte = [str(x) if pd.notna(x) else "" for x in row]
        tabelle.insert("", tk.END, values=zeilen_werte)

    # 7. Interaktions-Sperre (Reiner Anzeige-Modus)
    # Verhindert, dass der Nutzer Zeilen durch Anklicken blau markieren kann
    tabelle.bind("<<TreeviewSelect>>", lambda e: tabelle.selection_remove(tabelle.selection()))

    # Fenster starten
    root.mainloop()


if __name__ == "__main__":
    zeige_wein_tabelle()