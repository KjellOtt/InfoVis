import base64
import io
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pages import übersicht, matrix as matrix_page, baum as baum_page, evaluierung as evaluierung_page

app = Flask(__name__)

BOOTSTRAP_CSS = (
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
)

CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
}

METRIC_TOOLTIPS = {
    "TN": "True Negative: Anzahl korrekt als negativ klassifizierter Fälle.",
    "FP": "False Positive: Anzahl fälschlich als positiv klassifizierter negativer Fälle.",
    "FN": "False Negative: Anzahl fälschlich als negativ klassifizierter positiver Fälle.",
    "TP": "True Positive: Anzahl korrekt als positiv klassifizierter Fälle.",
    "Accuracy": "Accuracy: Anteil korrekt klassifizierter Fälle insgesamt.",
    "Precision": "Precision: Anteil der richtig positiven Vorhersagen unter allen positiven Vorhersagen.",
    "Recall": "Recall: Anteil der richtig positiven Vorhersagen unter allen tatsächlichen Positiven.",
    "False Positive Rate": "False Positive Rate: Anteil der fälschlich als positiv klassifizierten negativen Fälle.",
    "F1": "F1-Score: harmonisches Mittel aus Precision und Recall.",
    "AUC": "AUC: Fläche unter der ROC-Kurve, Maß für Trennschärfe.",
}


def load_data(csv_path: str = "Daten/Titanic.csv") -> pd.DataFrame:
    """Lädt die CSV-Datei und gibt das Roh-DataFrame zurück."""
    return pd.read_csv(csv_path)


def chose_target_column(df: pd.DataFrame, preferred: str = "Survived") -> str | None:
    if preferred in df.columns:
        return preferred
    lower_cols = {col.lower(): col for col in df.columns if isinstance(col, str)}
    if preferred.lower() in lower_cols:
        return lower_cols[preferred.lower()]
    return None


def stratified_kfold_split(
    df: pd.DataFrame,
    target_column: str,
    n_splits: int = 10,
    random_state: int = 42,
    fold_index: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gibt einen Trainings- und Testdatensatz für einen StratifiedKFold zurück."""
    if target_column not in df.columns:
        raise ValueError(f"Target-Spalte '{target_column}' ist nicht vorhanden.")
    if n_splits < 2 or n_splits > len(df):
        raise ValueError("n_splits muss zwischen 2 und der Anzahl der Zeilen liegen.")

    y = df[target_column].fillna("missing")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        if idx == fold_index:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            return train_df, test_df

    raise ValueError(f"Fold-Index {fold_index} ist außerhalb des Bereichs.")


def bootstrap_632_split(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erzeugt einen Bootstrap-Trainingsdatensatz und den OOB-Testdatensatz."""
    if target_column not in df.columns:
        raise ValueError(f"Target-Spalte '{target_column}' ist nicht vorhanden.")

    n = len(df)
    rng = np.random.default_rng(random_state)
    sampled_indices = rng.choice(n, size=n, replace=True)
    train_df = df.iloc[sampled_indices].reset_index(drop=True)

    unique_train_indices = np.unique(sampled_indices)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[unique_train_indices] = False
    test_df = df.iloc[oob_mask].reset_index(drop=True)

    if test_df.empty:
        raise ValueError(
            "Bootstrap 0.632 erzeugt keinen OOB-Testdatensatz; verwenden Sie größere Datenmenge oder einen anderen Seed."
        )

    return train_df, test_df


def prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column]).copy()
    features = pd.get_dummies(features, drop_first=True)
    if features.empty:
        raise ValueError("Keine Features vorhanden, um das Modell zu trainieren.")
    return features.fillna(0), df[target_column].fillna("missing")


def align_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return X_train.align(X_test, join="outer", axis=1, fill_value=0)


def encode_target(y: pd.Series) -> tuple[pd.Series, list[str]]:
    y_clean = y.fillna("missing").astype(str)
    labels = sorted(y_clean.unique())
    if len(labels) != 2:
        raise ValueError("Nur binäre Zielvariablen werden für die Auswertung unterstützt.")
    mapping = {label: idx for idx, label in enumerate(labels)}
    return y_clean.map(mapping), labels


def plot_roc_curve_png(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Zufall")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def evaluate_classifier(
    classifier_factory,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> dict:
    X_train, y_train = prepare_features(train_df, target_column)
    X_test, y_test = prepare_features(test_df, target_column)
    X_train, X_test = align_features(X_train, X_test)
    y_train_enc, _ = encode_target(y_train)
    y_test_enc, _ = encode_target(y_test)

    clf = classifier_factory()
    clf.fit(X_train, y_train_enc)
    y_pred = clf.predict(X_test)

    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(X_test)

    cm = confusion_matrix(y_test_enc, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_manual = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    roc_values = None
    auc_score = None
    if y_prob is not None and len(np.unique(y_test_enc)) == 2:
        try:
            fpr_values, tpr_values, _ = roc_curve(y_test_enc, y_prob)
            roc_values = (fpr_values, tpr_values)
            auc_score = auc(fpr_values, tpr_values)
        except ValueError:
            roc_values = None
            auc_score = None

    return {
        "classifier": clf.__class__.__name__,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_test_enc, y_pred),
        "precision": precision_score(y_test_enc, y_pred, zero_division=0),
        "recall": recall_score(y_test_enc, y_pred, zero_division=0),
        "f1": f1_score(y_test_enc, y_pred, zero_division=0),
        "false_positive_rate": fpr_manual,
        "roc_values": roc_values,
        "auc": auc_score,
    }


def render_metrics_table(metrics: list[dict]) -> str:
    rows = []
    for metric in metrics:
        rows.append(
            f"""
            <tr>
              <td>{metric['classifier']}</td>
              <td>{metric['tn']}</td>
              <td>{metric['fp']}</td>
              <td>{metric['fn']}</td>
              <td>{metric['tp']}</td>
              <td>{metric['accuracy']:.3f}</td>
              <td>{metric['precision']:.3f}</td>
              <td>{metric['recall']:.3f}</td>
              <td>{metric['false_positive_rate']:.3f}</td>
              <td>{metric['f1']:.3f}</td>
              <td>{metric['auc']:.3f}</td>
            </tr>
            """
        )
    header_cells = ''.join(
        f"<th data-bs-toggle=\"tooltip\" data-bs-placement=\"top\" title=\"{METRIC_TOOLTIPS.get(col, '')}\">{col}</th>"
        for col in [
            "Klassifikator",
            "TN",
            "FP",
            "FN",
            "TP",
            "Accuracy",
            "Precision",
            "Recall",
            "False Positive Rate",
            "F1",
            "AUC",
        ]
    )
    return f"""
      <div class=\"table-responsive mb-4\">
        <table class=\"table table-sm table-hover table-bordered\">
          <thead class=\"table-dark\">
            <tr>
              {header_cells}
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    """


def prepare_evaluation_html(cleaned: pd.DataFrame, target_column: str) -> str:
    train_kf, test_kf = stratified_kfold_split(cleaned, target_column)
    train_bts, test_bts = bootstrap_632_split(cleaned, target_column)

    evaluation_html = ""
    for method_name, (train_df, test_df) in {
        "Stratified 10-Fold": (train_kf, test_kf),
        "Bootstrap 0.632": (train_bts, test_bts),
    }.items():
        metrics = []
        roc_curves = {}
        for display_name, factory in CLASSIFIERS.items():
            result = evaluate_classifier(factory, train_df, test_df, target_column)
            metrics.append({
                "classifier": display_name,
                "tn": result["tn"],
                "fp": result["fp"],
                "fn": result["fn"],
                "tp": result["tp"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "false_positive_rate": result["false_positive_rate"],
                "f1": result["f1"],
                "auc": result["auc"] if result["auc"] is not None else 0.0,
            })
            if result["roc_values"] is not None:
                roc_curves[display_name] = result["roc_values"]

        roc_html = ""
        if roc_curves:
            roc_png = plot_roc_curve_png(roc_curves, f"ROC-Kurve ({method_name})")
            roc_html = f"<div class=\"mb-3\"><img src=\"data:image/png;base64,{roc_png}\" class=\"img-fluid rounded\" alt=\"ROC {method_name}\"></div>"

        evaluation_html += f"""
          <div class=\"mb-4\">
            <h4>{method_name}</h4>
            {render_metrics_table(metrics)}
            {roc_html}
          </div>
        """

    return evaluation_html


def render_page(title: str, body_html: str, active: str = "uebersicht") -> str:
    html = f"""
    <!doctype html>
    <html lang="de">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <link href="{BOOTSTRAP_CSS}" rel="stylesheet" crossorigin="anonymous">
      </head>
      <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
          <div class="container-fluid">
            <a class="navbar-brand" href="/">InfoVis</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
              <ul class="navbar-nav">
                <li class="nav-item">
                  <a class="nav-link{' active' if active == 'uebersicht' else ''}" href="/uebersicht">Übersicht</a>
                </li>
                <li class="nav-item">
                  <a class="nav-link{' active' if active == 'evaluierung' else ''}" href="/evaluierung">Evaluierung</a>
                </li>
                <li class="nav-item">
                  <a class="nav-link{' active' if active == 'matrix' else ''}" href="/matrix">Matrix</a>
                </li>
                <li class="nav-item">
                  <a class="nav-link{' active' if active == 'baum' else ''}" href="/baum">Baum</a>
                </li>
              </ul>
            </div>
          </div>
        </nav>
        <div class="container py-4">
          <h1 class="mb-4">{title}</h1>
          {body_html}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" crossorigin="anonymous"></script>
        <script>
          document.addEventListener('DOMContentLoaded', function () {{
            var tooltipTriggerList = [].slice.call(document.querySelectorAll("[data-bs-toggle='tooltip']"))
            tooltipTriggerList.map(function (tooltipTriggerEl) {{
              return new bootstrap.Tooltip(tooltipTriggerEl)
            }})
          }})
        </script>
      </body>
    </html>
    """
    return html


@app.route("/")
@app.route("/uebersicht")
def index() -> str:
    raw_data = load_data()
    cleaned = übersicht.bereinige_daten(raw_data)
    row_count, col_count = cleaned.shape
    table_html = cleaned.to_html(
        classes="table table-striped table-bordered",
        index=False,
        border=0,
        justify="left",
        escape=False,
    )
    columns = ", ".join(cleaned.columns.astype(str))

    target_column = chose_target_column(cleaned)
    split_info = ""
    evaluation_html = ""

    if target_column is not None:
        evaluation_html = prepare_evaluation_html(cleaned, target_column)
        train_kf, test_kf = stratified_kfold_split(cleaned, target_column)
        train_bts, test_bts = bootstrap_632_split(cleaned, target_column)
        split_info = f"""
        <div class=\"mb-4\">
          <h5>Train/Test-Split</h5>
          <p>Verwendete Zielspalte: <strong>{target_column}</strong></p>
          <ul>
            <li>Stratified 10-Fold: Trainingsdaten = {len(train_kf)} Zeilen, Testdaten = {len(test_kf)} Zeilen</li>
            <li>Bootstrap 0.632: Trainingsdaten = {len(train_bts)} Zeilen, Testdaten = {len(test_bts)} Zeilen</li>
          </ul>
        </div>
        """
    else:
        split_info = "<div class=\"alert alert-warning\" role=\"alert\">Keine Zielspalte gefunden. Kreuzvalidierung und Bootstrap-Splits werden nicht berechnet.</div>"

    body_html = f"""
      <div class=\"mb-3\">
        <span class=\"badge bg-primary\">Zeilen: {row_count}</span>
        <span class=\"badge bg-secondary\">Spalten: {col_count}</span>
      </div>
      <div class=\"mb-4\">
        <h5>Spaltennamen</h5>
        <p>{columns}</p>
      </div>
      {split_info}
      {evaluation_html}
      <div class=\"table-responsive\">
        {table_html}
      </div>
    """

    return render_template_string(render_page("Übersicht", body_html, active="uebersicht"))


@app.route("/evaluierung")
def evaluierung() -> str:
    raw_data = load_data()
    cleaned = übersicht.bereinige_daten(raw_data)
    target_column = chose_target_column(cleaned)
    if target_column is None:
        body_html = "<div class=\"alert alert-warning\" role=\"alert\">Keine Zielspalte gefunden. Evaluierung kann nicht durchgeführt werden.</div>"
    else:
        body_html = evaluierung_page.prepare_evaluation_html(cleaned, target_column)
    return render_template_string(render_page("Evaluierung", body_html, active="evaluierung"))


@app.route("/matrix")
def matrix() -> str:
    raw_data = load_data()
    cleaned = übersicht.bereinige_daten(raw_data)
    target_column = chose_target_column(cleaned)
    if target_column is None:
        body_html = "<div class=\"alert alert-warning\" role=\"alert\">Keine Zielspalte gefunden. Konfusionsmatrix kann nicht erstellt werden.</div>"
    else:
        body_html = matrix_page.render_matrix(cleaned, target_column)
    return render_template_string(render_page("Matrix", body_html, active="matrix"))


@app.route("/baum")
def baum() -> str:
    raw_data = load_data()
    cleaned = übersicht.bereinige_daten(raw_data)
    target_column = chose_target_column(cleaned)
    if target_column is None:
        body_html = "<div class=\"alert alert-warning\" role=\"alert\">Keine Zielspalte gefunden. Baum kann nicht erstellt werden.</div>"
    else:
        body_html = baum_page.render_tree(cleaned, target_column)
    return render_template_string(render_page("Baum", body_html, active="baum"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
