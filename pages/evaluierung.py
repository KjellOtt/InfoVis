import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    y = df[target_column].fillna("missing")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        if idx == fold_index:
            return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    raise ValueError(f"Fold-Index {fold_index} ist außerhalb des Bereichs.")


def prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column]).copy()
    features = pd.get_dummies(features, drop_first=True)
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
    if y_prob is not None:
        fpr_values, tpr_values, _ = roc_curve(y_test_enc, y_prob)
        roc_values = (fpr_values, tpr_values)
        auc_score = auc(fpr_values, tpr_values)

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
    header_cells = ''.join(
        f"<th data-bs-toggle='tooltip' data-bs-placement='top' title='{METRIC_TOOLTIPS.get(col, '')}'>{col}</th>"
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
    return f"""
      <div class='table-responsive mb-4'>
        <table class='table table-sm table-hover table-bordered'>
          <thead class='table-dark'>
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
    train_df, test_df = stratified_kfold_split(cleaned, target_column)
    metrics = []
    roc_curves = {}
    for display_name, factory in CLASSIFIERS.items():
        result = evaluate_classifier(factory, train_df, test_df, target_column)
        metrics.append(result)
        if result['roc_values'] is not None:
            roc_curves[display_name] = result['roc_values']

    roc_html = ""
    if roc_curves:
        roc_png = plot_roc_curve_png(roc_curves, 'ROC-Kurve Evaluierung')
        roc_html = f"<div class='mb-4'><img src='data:image/png;base64,{roc_png}' class='img-fluid rounded' alt='ROC Kurve'></div>"

    return f"""
      <div class='mb-4'>
        <p>Die Evaluierung zeigt Metriken und ROC-Kurve für alle Klassifikatoren.</p>
      </div>
      {render_metrics_table(metrics)}
      {roc_html}
    """
