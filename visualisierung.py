import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


class VisualizationApp:
    def __init__(app, df: pd.DataFrame):
        # Erstellt eine Kopie des DataFrames
        app.df = df.reset_index(drop=True)
        app.df_filtered = app.df.copy()
        app.selected_attributes = []
        app.num_points = len(app.df)
        
        # Filtert und sortiert numerische Spalten
        app.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        app.all_attributes = sorted(app.numeric_columns)
        
        # GUI-Fenster
        app.root = tk.Tk()
        app.root.title("Datenvisualisierung")
        app.root.geometry("1400x800")
        
        app._build_ui()
        
    def _build_ui(app):
        # Oberer Frame: Filterung
        top_frame = ttk.Frame(app.root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Datenpunkte:").pack(side=tk.LEFT, padx=5)
        app.label_count = ttk.Label(top_frame, text=f"{app.num_points}", font=("Arial", 12, "bold"))
        app.label_count.pack(side=tk.LEFT, padx=5) # Position der Labels zueinander
        
        ttk.Separator(top_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Interaktion: Nationality per Dropdown wählen
        ttk.Label(top_frame, text="Nationality:").pack(side=tk.LEFT, padx=5)
        nat_vals = ['(alle)'] + sorted(app.df['Nationality'].dropna().unique().tolist())
        app.var_nat = tk.StringVar(value='(alle)')
        nat_combo = ttk.Combobox(top_frame, textvariable=app.var_nat, values=nat_vals, width=15, state='readonly') # Dropdown-Menü
        nat_combo.pack(side=tk.LEFT, padx=5)
        nat_combo.bind('<<ComboboxSelected>>', lambda e: app._apply_filters()) # Bei Auswahländerung wird die Filterfunktion ausgeführt
        
        # Interaktion: Club per Dropdown wählen
        ttk.Label(top_frame, text="Club:").pack(side=tk.LEFT, padx=5)
        club_vals = ['(alle)'] + sorted(app.df['Club Name'].dropna().unique().tolist())
        app.var_club = tk.StringVar(value='(alle)')
        club_combo = ttk.Combobox(top_frame, textvariable=app.var_club, values=club_vals, width=15, state='readonly') # Dropdown-Menü
        club_combo.pack(side=tk.LEFT, padx=5)
        club_combo.bind('<<ComboboxSelected>>', lambda e: app._apply_filters())
        
        # Mittlerer Frame: Attribut-Liste und Plot
        main_frame = ttk.Frame(app.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Links: Attribut-Liste
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(left_frame, text="Attribute (max 2):", font=("Arial", 10, "bold")).pack(pady=5)
        app.listbox = tk.Listbox(left_frame, width=25, height=30, selectmode=tk.MULTIPLE)
        app.listbox.pack(fill=tk.BOTH, expand=True)
        for attr in app.all_attributes: # Alle numerischen Attribute in die Box einfügen
            app.listbox.insert(tk.END, attr)
        app.listbox.bind('<<ListboxSelect>>', app._on_attr_select) # Bei Auswahl eines Attributs wird die entsprechende Funktion aufgerufen
        
        # Rechts: Plot-Bereich leer initialisieren
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        app.fig = Figure(figsize=(8, 6), dpi=100)
        app.canvas = FigureCanvasTkAgg(app.fig, master=plot_frame)
        app.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Unterer Frame: Datenpunkt-Vergleich
        bottom_frame = ttk.LabelFrame(app.root, text="Zwei Datenpunkte vergleichen")
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(bottom_frame, text=f"Index 1 (0-{app.num_points-1}):").pack(side=tk.LEFT, padx=5, pady=5)
        app.entry_index1 = ttk.Entry(bottom_frame, width=10)
        app.entry_index1.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(bottom_frame, text=f"Index 2 (0-{app.num_points-1}):").pack(side=tk.LEFT, padx=5, pady=5)
        app.entry_index2 = ttk.Entry(bottom_frame, width=10)
        app.entry_index2.pack(side=tk.LEFT, padx=5)
        
        # Führt per Klick die Vergleichsfunktion aus
        ttk.Button(bottom_frame, text="Vergleichen", command=app._compare_points).pack(side=tk.LEFT, padx=5)
        
    def _apply_filters(app):
        # Filtert Daten nach Nationality und Club. Datenpunkte, die nicht den gewählten Wert haben, werden auf False gesetzt.
        app.df_filtered = app.df.copy()
        
        nat = app.var_nat.get()
        if nat != '(alle)':
            app.df_filtered = app.df_filtered[app.df_filtered['Nationality'] == nat]
        
        club = app.var_club.get()
        if club != '(alle)':
            app.df_filtered = app.df_filtered[app.df_filtered['Club Name'] == club]
        
        # 
        app.label_count.config(text=str(len(app.df_filtered)))
        app._update_plot()
        
    def _on_attr_select(app, event=None):
        # Aktualisiert Plot bei Attribut-Auswahl.
        indices = app.listbox.curselection()
        app.selected_attributes = [app.all_attributes[i] for i in indices]
        
        # Max 2 Attribute können gleichzeitig ausgewählt sein.
        if len(app.selected_attributes) > 2:
            app.listbox.selection_clear(0, tk.END)
            for i, attr in enumerate(app.all_attributes):
                if attr in app.selected_attributes[-2:]:
                    app.listbox.selection_set(app.all_attributes.index(attr))
            app.selected_attributes = app.selected_attributes[-2:]
        
        app._update_plot()
        
    def _update_plot(app):
        # Aktualisiert den Plot basierend auf Auswahl. Histogramm für 1 Attribut und Scatterplot für 2 Attribute.
        app.fig.clear()
        
        if not app.selected_attributes:
            ax = app.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Wähle ein Attribut', ha='center', va='center', fontsize=12)
            ax.axis('off')
            app.canvas.draw()
            return
        
        if len(app.selected_attributes) == 1:
            app._plot_histogram()
        else:
            app._plot_scatter()
            
        app.canvas.draw()
        
    def _plot_histogram(app):
        # Zeigt Histogram für das ausgewählte Attribut.
        attr = app.selected_attributes[0]
        data = app.df_filtered[attr].dropna()
        
        # Falls Datenmenge leer
        if data.empty:
            ax = app.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Keine Daten nach Filterung', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        ax = app.fig.add_subplot(111)
        ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(attr, fontsize=11)
        ax.set_ylabel('Häufigkeit', fontsize=11)
        ax.set_title(f'Verteilung: {attr} (n={len(data)})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        app.fig.tight_layout()
        
    def _plot_scatter(app):
        # Zeigt Scatterplot für zwei ausgewählte Attribute.
        attr1, attr2 = app.selected_attributes[0], app.selected_attributes[1]
        data = app.df_filtered[[attr1, attr2]].dropna()
        
        # Falls Datenmenge leer
        if data.empty:
            ax = app.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Keine Daten nach Filterung', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        ax = app.fig.add_subplot(111)
        ax.scatter(data[attr1], data[attr2], alpha=0.6, s=40, color='steelblue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel(attr1, fontsize=11)
        ax.set_ylabel(attr2, fontsize=11)
        ax.set_title(f'{attr1} vs {attr2} (n={len(data)})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        app.fig.tight_layout()
        
    def _compare_points(app):
        # Vergleicht zwei Datenpunkte nach Index.
        try:
            index1 = int(app.entry_index1.get())
            index2 = int(app.entry_index2.get())
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Indizes eingeben (Zahlen).")
            return
        
        if not (0 <= index1 < app.num_points) or not (0 <= index2 < app.num_points):
            messagebox.showerror("Fehler", f"Indizes müssen zwischen 0 und {app.num_points-1} liegen.")
            return
        
        if index1 == index2:
            messagebox.showwarning("Warnung", "Wähle zwei verschiedene Datenpunkte.")
            return
        
        row1 = app.df.iloc[index1]
        row2 = app.df.iloc[index2]
        
        # Vergleichsfenster
        comp_window = tk.Toplevel(app.root)
        comp_window.title(f"Vergleich: Index {index1} vs {index2}")
        comp_window.geometry("700x500")
        
        ttk.Label(comp_window, text=f"Datenpunkt {index1} vs Datenpunkt {index2}", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Text-Widget für Vergleich
        text_widget = tk.Text(comp_window, height=25, width=80, font=("Courier", 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, f"{'Attribut':<30} {'Index {}':<20} {'Index {}':<20} {'Differenz':<15}\n".format(index1, index2))
        text_widget.insert(tk.END, "-" * 85 + "\n")
        
        for col in app.df.columns:
            val1 = row1[col]
            val2 = row2[col]
            
            # Nur numerische Spalten für Differenz
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and not pd.isna(val1) and not pd.isna(val2):
                diff = val2 - val1
                text_widget.insert(tk.END, f"{col:<30} {str(val1):<20} {str(val2):<20} {str(diff):<15}\n")
            else:
                text_widget.insert(tk.END, f"{col:<30} {str(val1):<20} {str(val2):<20} {'–':<15}\n")
        
        text_widget.config(state=tk.DISABLED)
        
    def run(app):
        # Startet die GUI.
        app.root.mainloop()


def visualize(df: pd.DataFrame, output_dir: str = 'plots'):
    # Startet die interaktive Visualisierungs-GUI.
    app = VisualizationApp(df)
    app.run()
