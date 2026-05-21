import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


class VisualizationApp:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.df_filtered = self.df.copy()
        self.selected_attrs = []
        self.num_points = len(self.df)
        
        # Alle numerischen und geeigneten String-Spalten
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.all_attrs = sorted(self.numeric_cols)
        
        # GUI-Fenster
        self.root = tk.Tk()
        self.root.title("Datenvisualisierung")
        self.root.geometry("1400x800")
        
        self._build_ui()
        
    def _build_ui(self):
        # Oberer Frame: Filterung
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Datenpunkte:").pack(side=tk.LEFT, padx=5)
        self.label_count = ttk.Label(top_frame, text=f"{self.num_points}", font=("Arial", 12, "bold"))
        self.label_count.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(top_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(top_frame, text="Nationality:").pack(side=tk.LEFT, padx=5)
        nat_vals = ['(alle)'] + sorted(self.df['Nationality'].dropna().unique().tolist())
        self.var_nat = tk.StringVar(value='(alle)')
        nat_combo = ttk.Combobox(top_frame, textvariable=self.var_nat, values=nat_vals, width=15, state='readonly')
        nat_combo.pack(side=tk.LEFT, padx=5)
        nat_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        ttk.Label(top_frame, text="Club:").pack(side=tk.LEFT, padx=5)
        club_vals = ['(alle)'] + sorted(self.df['Club Name'].dropna().unique().tolist())
        self.var_club = tk.StringVar(value='(alle)')
        club_combo = ttk.Combobox(top_frame, textvariable=self.var_club, values=club_vals, width=15, state='readonly')
        club_combo.pack(side=tk.LEFT, padx=5)
        club_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())

        self.var_show_outliers = tk.BooleanVar(value=True)
        cb_outliers = ttk.Checkbutton(
            top_frame,
            text="Ausreißer anzeigen",
            variable=self.var_show_outliers,
            command=self._update_plot
        )
        cb_outliers.pack(side=tk.LEFT, padx=5)

        # Mittlerer Frame: Attribut-Liste und Plot
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Links: Attribut-Liste
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(left_frame, text="Attribute (max 2):", font=("Arial", 10, "bold")).pack(pady=5)
        self.listbox = tk.Listbox(left_frame, width=25, height=30, selectmode=tk.MULTIPLE)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        for attr in self.all_attrs:
            self.listbox.insert(tk.END, attr)
        self.listbox.bind('<<ListboxSelect>>', self._on_attr_select)
        
        # Rechts: Plot-Bereich
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Unterer Frame: Datenpunkt-Vergleich
        bottom_frame = ttk.LabelFrame(self.root, text="Zwei Datenpunkte vergleichen")
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(bottom_frame, text=f"Index 1 (0-{self.num_points-1}):").pack(side=tk.LEFT, padx=5, pady=5)
        self.entry_idx1 = ttk.Entry(bottom_frame, width=10)
        self.entry_idx1.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(bottom_frame, text=f"Index 2 (0-{self.num_points-1}):").pack(side=tk.LEFT, padx=5, pady=5)
        self.entry_idx2 = ttk.Entry(bottom_frame, width=10)
        self.entry_idx2.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Vergleichen", command=self._compare_points).pack(side=tk.LEFT, padx=5)
        
    def _apply_filters(self):
        """Filtert Daten nach Nationality und Club."""
        self.df_filtered = self.df.copy()
        
        nat = self.var_nat.get()
        if nat != '(alle)':
            self.df_filtered = self.df_filtered[self.df_filtered['Nationality'] == nat]
        
        club = self.var_club.get()
        if club != '(alle)':
            self.df_filtered = self.df_filtered[self.df_filtered['Club Name'] == club]
        
        self.label_count.config(text=str(len(self.df_filtered)))
        self._update_plot()
        
    def _on_attr_select(self, event=None):
        """Aktualisiert Plot bei Attribut-Auswahl."""
        indices = self.listbox.curselection()
        self.selected_attrs = [self.all_attrs[i] for i in indices]
        
        # Max 2 Attribute
        if len(self.selected_attrs) > 2:
            self.listbox.selection_clear(0, tk.END)
            for i, attr in enumerate(self.all_attrs):
                if attr in self.selected_attrs[-2:]:
                    self.listbox.selection_set(self.all_attrs.index(attr))
            self.selected_attrs = self.selected_attrs[-2:]
        
        self._update_plot()
        
    def _update_plot(self):
        """Aktualisiert den Plot basierend auf Auswahl."""
        self.fig.clear()
        
        if not self.selected_attrs:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Wähle ein Attribut', ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.canvas.draw()
            return
        
        if len(self.selected_attrs) == 1:
            self._plot_histogram()
        else:
            self._plot_scatter()
            
        self.canvas.draw()

    def _remove_outliers(self, series: pd.Series) -> pd.Series:
        """Entfernt Ausreißer aus einer Series mittels Interquartile-Range."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]

    def _plot_histogram(self):
        """Zeigt Histogram für ein Attribut."""
        attr = self.selected_attrs[0]
        data = self.df_filtered[attr].dropna()

        if not self.var_show_outliers.get():
            data = self._remove_outliers(data)

        if data.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Keine Daten nach Filterung', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        ax = self.fig.add_subplot(111)
        ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(attr, fontsize=11)
        ax.set_ylabel('Häufigkeit', fontsize=11)

        title_suffix = "" if self.var_show_outliers.get() else " (Ausreißer ausgeblendet)"
        ax.set_title(f'Verteilung: {attr} (n={len(data)}){title_suffix}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        self.fig.tight_layout()

    def _plot_scatter(self):
        """Zeigt Scatterplot für zwei Attribute."""
        attr1, attr2 = self.selected_attrs[0], self.selected_attrs[1]
        data = self.df_filtered[[attr1, attr2]].dropna()

        if not self.var_show_outliers.get():
            data = data.loc[self._remove_outliers(data[attr1]).index]
            data = data.loc[self._remove_outliers(data[attr2]).index]

        if data.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Keine Daten nach Filterung', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        ax = self.fig.add_subplot(111)
        ax.scatter(data[attr1], data[attr2], alpha=0.6, s=40, color='steelblue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel(attr1, fontsize=11)
        ax.set_ylabel(attr2, fontsize=11)

        title_suffix = "" if self.var_show_outliers.get() else " (Ausreißer ausgeblendet)"
        ax.set_title(f'{attr1} vs {attr2} (n={len(data)}){title_suffix}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        
    def _compare_points(self):
        """Vergleicht zwei Datenpunkte nach Index."""
        try:
            idx1 = int(self.entry_idx1.get())
            idx2 = int(self.entry_idx2.get())
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Indizes eingeben (Zahlen).")
            return
        
        if not (0 <= idx1 < self.num_points) or not (0 <= idx2 < self.num_points):
            messagebox.showerror("Fehler", f"Indizes müssen zwischen 0 und {self.num_points-1} liegen.")
            return
        
        if idx1 == idx2:
            messagebox.showwarning("Warnung", "Wähle zwei verschiedene Datenpunkte.")
            return
        
        row1 = self.df.iloc[idx1]
        row2 = self.df.iloc[idx2]
        
        # Vergleichsfenster
        comp_window = tk.Toplevel(self.root)
        comp_window.title(f"Vergleich: Index {idx1} vs {idx2}")
        comp_window.geometry("700x500")
        
        # Überschrift
        ttk.Label(comp_window, text=f"Datenpunkt {idx1} vs Datenpunkt {idx2}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Text-Widget für Vergleich
        text_widget = tk.Text(comp_window, height=25, width=80, font=("Courier", 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, f"{'Attribut':<30} {'Index {}':<20} {'Index {}':<20} {'Differenz':<15}\n".format(idx1, idx2))
        text_widget.insert(tk.END, "-" * 85 + "\n")
        
        for col in self.df.columns:
            val1 = row1[col]
            val2 = row2[col]
            
            # Nur numerische Spalten für Differenz
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and not pd.isna(val1) and not pd.isna(val2):
                diff = val2 - val1
                text_widget.insert(tk.END, f"{col:<30} {str(val1):<20} {str(val2):<20} {str(diff):<15}\n")
            else:
                text_widget.insert(tk.END, f"{col:<30} {str(val1):<20} {str(val2):<20} {'–':<15}\n")
        
        text_widget.config(state=tk.DISABLED)
        
    def run(self):
        """Startet die GUI."""
        self.root.mainloop()


def visualize(df: pd.DataFrame, output_dir: str = 'plots'):
    """Startet die interaktive Visualisierungs-GUI."""
    app = VisualizationApp(df)
    app.run()
