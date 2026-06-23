import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict


CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=42), # Vereinheitlicht auf max_depth=4
    "K-Nearest Neighbor (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
}

METRIC_TOOLTIPS = {
    "TN": "True Negative: Anzahl korrekt als negativ klassifizierter Fälle (Durchschnitt über Folds).",
    "FP": "False Positive: Anzahl fälschlich als positiv klassifizierter negativer Fälle.",
    "FN": "False Negative: Anzahl fälschlich als negativ klassifizierter positiver Fälle.",
    "TP": "True Positive: Anzahl korrekt als positiv klassifizierter Fälle.",
    "Accuracy": "Accuracy: Anteil korrekt klassifizierter Fälle insgesamt (Mittelwert ± Std-Abw).",
    "Balanced Accuracy": "Balanced Accuracy: Arithmetisches Mittel aus Sensitivity (Recall) und Specificity.",
    "Precision": "Precision: Anteil der richtig positiven Vorhersagen unter allen positiven Vorhersagen.",
    "Recall": "Recall (Sensitivity): Anteil der richtig positiven Vorhersagen unter allen tatsächlichen Positiven.",
    "Specificity": "Specificity: Anteil der richtig negativen Vorhersagen unter allen tatsächlichen Negativen.",
    "FPR": "False Positive Rate: Anteil der fälschlich als positiv klassifizierten negativen Fälle.",
    "F1": "F1-Score: harmonisches Mittel aus Precision und Recall.",
    "AUC": "AUC: Fläche unter der ROC-Kurve, Maß für Trennschärfe.",
}

def get_pipeline(classifier_factory, num_features, cat_features):
    """Erstellt eine Pipeline mit Skalierung und Encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier_factory())
    ])


def load_data(csv_path: str = "Daten/Titanic.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def chose_target_column(df: pd.DataFrame, preferred: str = "Survived") -> str | None:
    if preferred in df.columns:
        return preferred
    lower_cols = {col.lower(): col for col in df.columns if isinstance(col, str)}
    if preferred.lower() in lower_cols:
        return lower_cols[preferred.lower()]
    return None


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


def evaluate_cv(classifier_factory, df: pd.DataFrame, target_column: str, n_splits=10):
    """Führt eine 10-fold CV durch und berechnet aggregierte Metriken."""
    X = df.drop(columns=[target_column])
    y, labels = encode_target(df[target_column])
    
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    pipeline = get_pipeline(classifier_factory, num_features, cat_features)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)
    
    # Vorhersagen für Konfusionsmatrix
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Normalisierung der CM auf Durchschnitt pro Fold
    tn_avg, fp_avg, fn_avg, tp_avg = tn/n_splits, fp/n_splits, fn/n_splits, tp/n_splits
    
    # Spezifität berechnen (avg)
    spec_avg = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC für alle Daten (Out-of-fold Probabilities)
    y_probas = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')[:, 1]
    fpr_vals, tpr_vals, _ = roc_curve(y, y_probas)
    
    return {
        "classifier": classifier_factory().__class__.__name__,
        "method": f"{n_splits}-Fold CV",
        "tn": round(tn_avg, 1),
        "fp": round(fp_avg, 1),
        "fn": round(fn_avg, 1),
        "tp": round(tp_avg, 1),
        "accuracy": f"{cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}",
        "balanced_accuracy": round(cv_results['test_balanced_accuracy'].mean(), 3),
        "precision": round(cv_results['test_precision'].mean(), 3),
        "recall": round(cv_results['test_recall'].mean(), 3),
        "specificity": round(spec_avg, 3),
        "f1": round(cv_results['test_f1'].mean(), 3),
        "false_positive_rate": round(fp / (fp + tn), 3) if (fp + tn) > 0 else 0,
        "roc_values": (fpr_vals, tpr_vals),
        "auc": round(cv_results['test_roc_auc'].mean(), 3),
    }

def evaluate_bootstrap_0632(classifier_factory, df: pd.DataFrame, target_column: str, n_iterations=10):
    """
    Berechnet die .632 Bootstrap-Schätzung über mehrere Iterationen.
    Err_0.632 = 0.368 * Err_train + 0.632 * Err_oob
    """
    X = df.drop(columns=[target_column])
    y, labels = encode_target(df[target_column])
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    n = len(df)
    rng = np.random.default_rng(42)
    
    acc_scores = []
    # Für CM und ROC nutzen wir beispielhaft den letzten Split oder aggregieren
    all_tn, all_fp, all_fn, all_tp = [], [], [], []
    
    last_pipeline = None
    last_X_oob = None
    last_y_oob = None

    for i in range(n_iterations):
        sampled_indices = rng.choice(n, size=n, replace=True)
        unique_indices = np.unique(sampled_indices)
        oob_indices = np.setdiff1d(np.arange(n), unique_indices)
        
        if len(oob_indices) == 0: continue
        
        X_train, y_train = X.iloc[sampled_indices], y.iloc[sampled_indices]
        X_oob, y_oob = X.iloc[oob_indices], y.iloc[oob_indices]
        
        pipeline = get_pipeline(classifier_factory, num_features, cat_features)
        pipeline.fit(X_train, y_train)
        
        acc_train = accuracy_score(y_train, pipeline.predict(X_train))
        acc_oob = accuracy_score(y_oob, pipeline.predict(X_oob))
        
        acc_0632 = 0.368 * acc_train + 0.632 * acc_oob
        acc_scores.append(acc_0632)
        
        cm = confusion_matrix(y_oob, pipeline.predict(X_oob), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        all_tn.append(tn); all_fp.append(fp); all_fn.append(fn); all_tp.append(tp)
        
        last_pipeline = pipeline
        last_X_oob = X_oob
        last_y_oob = y_oob

    if last_pipeline is None:
        return {"classifier": classifier_factory().__class__.__name__, "method": "Bootstrapping .632", "accuracy": "N/A", "roc_values": None, "auc": 0, "tn": 0, "fp": 0, "fn": 0, "tp": 0, "precision": 0, "recall": 0, "f1": 0}

    # ROC (beispielhaft für letzten Split)
    y_probas_oob = last_pipeline.predict_proba(last_X_oob)[:, 1]
    fpr_vals, tpr_vals, _ = roc_curve(last_y_oob, y_probas_oob)

    return {
        "classifier": classifier_factory().__class__.__name__,
        "method": "Bootstrapping .632",
        "tn": round(np.mean(all_tn), 1),
        "fp": round(np.mean(all_fp), 1),
        "fn": round(np.mean(all_fn), 1),
        "tp": round(np.mean(all_tp), 1),
        "accuracy": f"{np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}",
        "balanced_accuracy": "-", # .632 meist für Accuracy definiert
        "precision": "-", 
        "recall": "-",
        "specificity": "-",
        "f1": "-",
        "false_positive_rate": "-",
        "roc_values": (fpr_vals, tpr_vals),
        "auc": "-",
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
            "Accuracy": m["accuracy"],
            "B-Acc": m.get("balanced_accuracy", "-"),
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Spec": m.get("specificity", "-"),
            "F1-Score": m["f1"],
            "AUC": m["auc"]
        })

    return html.Div([
        dash_table.DataTable(
            data=table_data,
            columns=[{"name": i, "id": i} for i in table_data[0].keys()],
            tooltip_header=METRIC_TOOLTIPS,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold',
                'textDecoration': 'underline',
                'textDecorationStyle': 'dotted',
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
    metrics = []
    roc_curves = {}
    
    for display_name, factory in CLASSIFIERS.items():
        # CV
        res_cv = evaluate_cv(factory, cleaned, target_column)
        metrics.append(res_cv)
        if res_cv['roc_values'] is not None:
            roc_curves[f"{display_name} (CV)"] = res_cv['roc_values']
            
        # Bootstrap
        res_bs = evaluate_bootstrap_0632(factory, cleaned, target_column)
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
