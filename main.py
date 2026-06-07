import sys
import traceback
import bereinigung
import visualisierung_dash

if __name__ == '__main__':
    # Nur bereinigen und Visualisierungen erzeugen
    df_clean = bereinigung.clean_dataframe(bereinigung.file_path, verbose=False)
    if df_clean is None:
        print("Bereinigung fehlgeschlagen. Keine Visualisierung möglich.")
        sys.exit(1)
    try:
        app = visualisierung_dash.build_app(df_clean)
        app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8050)
    except Exception as e:
        print(f"FEHLER bei Visualisierung: {e}")
        traceback.print_exc()