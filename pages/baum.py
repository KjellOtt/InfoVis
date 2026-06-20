import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dash import html
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
}


def stratified_kfold_split(
    df: pd.DataFrame,
    target_column: str,
    n_splits: int = 10,
    random_state: int = 42,
    fold_index: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_column not in df.columns:
        raise ValueError(f"Target-Spalte '{target_column}' ist nicht vorhanden.")

    y = df[target_column].fillna("missing")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        if idx == fold_index:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            return train_df, test_df

    raise ValueError(f"Fold-Index {fold_index} ist außerhalb des Bereichs.")


def prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column]).copy()
    features = pd.get_dummies(features, drop_first=True)
    return features.fillna(0), df[target_column].fillna("missing")


def plot_tree_png(clf: DecisionTreeClassifier, feature_names: list[str], class_names: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=False,
        ax=ax,
        fontsize=10,
        precision=2,
        impurity=False,
        label='all'
    )
    ax.set_title("Entscheidungsbaum (max_depth=4 für bessere Visualisierung)", fontsize=16, fontweight='bold')
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_tree(cleaned: pd.DataFrame, target_column: str):
    train_df, test_df = stratified_kfold_split(cleaned, target_column)
    X_train, y_train = prepare_features(train_df, target_column)
    
    X_train_clean = X_train.copy()
    y_train = y_train.astype(str)

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    clf.fit(X_train_clean, y_train)

    feature_names = X_train_clean.columns.tolist()
    class_names = sorted(y_train.unique())
    tree_png = plot_tree_png(clf, feature_names, class_names)
    text_repr = export_text(clf, feature_names=feature_names)

    return html.Div([
        html.Div([
            html.P([
                "Der Entscheidungsbaum wurde mit ",
                html.Strong("Stratified 10-Fold CV"),
                " (Fold 0) trainiert. Die Tiefe wurde auf 4 Ebenen begrenzt, um eine optimale Visualisierung und Interpretierbarkeit zu gewährleisten."
            ])
        ], className="alert alert-info mb-4"),
        html.Div([
            html.Div([
                html.H5("Grafische Baumstruktur", className="card-title mb-0")
            ], className="card-header bg-primary text-white"),
            html.Div([
                html.Img(src=f"data:image/png;base64,{tree_png}", className="img-fluid rounded", alt="Entscheidungsbaum")
            ], className="card-body text-center bg-light")
        ], className="card mb-4 shadow-sm"),
        html.Div([
            html.Div([
                html.H5("Textuelle Baumrepräsentation", className="card-title mb-0")
            ], className="card-header bg-secondary text-white"),
            html.Div([
                html.Pre(text_repr, style={
                    "whiteSpace": "pre-wrap",
                    "wordBreak": "break-word",
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "fontFamily": "monospace"
                })
            ], className="card-body")
        ], className="card shadow-sm")
    ])
