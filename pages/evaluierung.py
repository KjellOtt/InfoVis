import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
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


def bootstrap_632_split(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erzeugt einen Bootstrap-Trainingsdatensatz und den OOB-Testdatensatz."""
    n = len(df)
    rng = np.random.default_rng(random_state)
    sampled_indices = rng.choice(n, size=n, replace=True)
    train_df = df.iloc[sampled_indices].reset_index(drop=True)

    unique_train_indices = np.unique(sampled_indices)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[unique_train_indices] = False
    test_df = df.iloc[oob_mask].reset_index(drop=True)

    if test_df.empty:
        return train_df, df.sample(frac=0.3, random_state=random_state)

    return train_df, test_df


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


def plot_roc_curve_plotly(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str):
    fig = go.Figure()
    for name, (fpr, tpr) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Zufall', line=dict(dash='dash', color='black')))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.7, y=0.1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig


def evaluate_classifier(
    classifier_factory,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    method_name: str = "CV"
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
        "method": method_name,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": round(accuracy_score(y_test_enc, y_pred), 3),
        "precision": round(precision_score(y_test_enc, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test_enc, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_test_enc, y_pred, zero_division=0), 3),
        "false_positive_rate": round(fpr_manual, 3),
        "roc_values": roc_values,
        "auc": round(auc_score, 3) if auc_score is not None else 0.0,
    }


def render_metrics_table(metrics: list[dict]):
    table_data = []
    for m in metrics:
        table_data.append({
            "Methode": m["method"],
            "Klassifikator": m["classifier"],
            "TN": m["tn"],
            "FP": m["fp"],
            "FN": m["fn"],
            "TP": m["tp"],
            "Acc": m["accuracy"],
            "Prec": m["precision"],
            "Rec": m["recall"],
            "FPR": m["false_positive_rate"],
            "F1": m["f1"],
            "AUC": m["auc"]
        })

    return html.Div([
        dash_table.DataTable(
            data=table_data,
            columns=[{"name": i, "id": i} for i in table_data[0].keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ], className="table-responsive mb-4")


def prepare_evaluation_html(cleaned: pd.DataFrame, target_column: str):
    train_kf, test_kf = stratified_kfold_split(cleaned, target_column)
    train_bs, test_bs = bootstrap_632_split(cleaned, target_column)

    metrics = []
    roc_curves = {}
    
    for display_name, factory in CLASSIFIERS.items():
        # CV
        res_cv = evaluate_classifier(factory, train_kf, test_kf, target_column, "10-Fold CV")
        metrics.append(res_cv)
        if res_cv['roc_values'] is not None:
            roc_curves[f"{display_name} (CV)"] = res_cv['roc_values']
            
        # Bootstrap
        res_bs = evaluate_classifier(factory, train_bs, test_bs, target_column, "Bootstrapping")
        metrics.append(res_bs)
        if res_bs['roc_values'] is not None:
            roc_curves[f"{display_name} (BS)"] = res_bs['roc_values']

    roc_graph = html.Div()
    if roc_curves:
        fig = plot_roc_curve_plotly(roc_curves, 'ROC-Kurven Vergleich')
        roc_graph = html.Div([
            html.Div([
                html.H5("ROC-Kurven", className="card-title"),
                dcc.Graph(figure=fig)
            ], className="card-body text-center")
        ], className="card mb-4")

    return html.Div([
        html.Div([
            html.P([
                "Die Evaluierung vergleicht die Performance der Klassifikatoren mittels ",
                html.Strong("10-Fold Cross-Validation"),
                " und ",
                html.Strong("Bootstrapping 0.632"),
                "."
            ])
        ], className="alert alert-info"),
        render_metrics_table(metrics),
        roc_graph
    ])
