from pages.regression import Regression
from pages.data_handler import bereinigen

BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"

if __name__ == "__main__":
    df = bereinigen("Daten/wein.csv")
    if df is not None:
        app = Regression(df)
        app.run()
