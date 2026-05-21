import sys
import traceback
import visualisierung
import bereinigung

if __name__ == '__main__':
    # Modusabfrage: bereinigung oder visualisierung
    mode = input("Wähle Modus ('bereinigung (1)' oder 'visualisierung (2)'): ").strip().lower()
    if mode not in ['1', '2']:
        print("Ungültiger Modus. Bitte '1' oder '2' wählen.")
        sys.exit(1)

    if mode == '1':
        # Volle Bereinigung mit Ausgaben
        bereinigung.clean_dataframe(bereinigung.file_path, verbose=True)
    else:
        # Nur bereinigen und Visualisierungen erzeugen
        df_clean = bereinigung.clean_dataframe(bereinigung.file_path, verbose=False)
        if df_clean is None:
            print("Bereinigung fehlgeschlagen. Keine Visualisierung möglich.")
            sys.exit(1)
        try:
            visualisierung.visualize(df_clean, output_dir='plots')
        except Exception as e:
            print(f"FEHLER bei Visualisierung: {e}")
            traceback.print_exc()
    pass