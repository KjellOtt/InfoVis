# Bereinigung:
    duplikate entfernen -> alle werte zu float parsen (text-fehler zu NaN)
Bei Auswahl von Attributen werden Datenpunkte mit NaN gelöscht

## Arten von Werten in der rohen CSV:
- 14.23 und in der selben Spalte: 13.02.
    - also mit Punkt am ende --> hauptproblem --> resultiert mit normalen parsern in NaN
- 02.05. --> führende 0 am anfang

---

# Qualitätsmetriken:
  R² Score: {r2:.6f}    --> Anteil der erklärten Varianz; dimensionslos.
  RMSE: {rmse:.6f}      --> Wurzel aus MSE; gleiche Einheit wie das Ziel. 
  MAE: {mae:.6f}        --> Mittlere absolute Abweichung; weniger empfindlich gegenüber Ausreißern.
  MSE: {mse:.6f}        --> Durchschnitt der quadrierten Fehler; bestraft große Abweichungen stärker.
https://www.techzeitgeist.de/regression-leicht-erklaert-mse-rmse-mae-r%C2%B2-mape-verstaendlich/#kapitel1

# k-Means
- beste K = 4, denn:
  - Silhouette-Score: 0.274 (höchster Wert)
  - Davies-Bouldin Index: 1.08 (niedrigster Wert)