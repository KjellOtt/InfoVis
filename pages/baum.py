import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        ax=ax,
        fontsize=10,
    )
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_tree(cleaned: pd.DataFrame, target_column: str) -> str:
    train_df, test_df = stratified_kfold_split(cleaned, target_column)
    X_train, y_train = prepare_features(train_df, target_column)
    y_train = y_train.astype(str)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()
    class_names = sorted(y_train.unique())
    tree_png = plot_tree_png(clf, feature_names, class_names)
    text_repr = export_text(clf, feature_names=feature_names)

    return f"""
      <div class=\"mb-4\">
        <p>Der Entscheidungsbaum wurde mit einem Trainingsdatensatz aus Stratified 10-Fold erstellt.</p>
      </div>
      <div class=\"card mb-4\">
        <div class=\"card-body\">
          <h5 class=\"card-title\">Baumstruktur</h5>
          <img src=\"data:image/png;base64,{tree_png}\" class=\"img-fluid rounded\" alt=\"Entscheidungsbaum\">
        </div>
      </div>
      <div class=\"card\">
        <div class=\"card-body\">
          <h5 class=\"card-title\">Textuelle Baumrepräsentation</h5>
          <pre style=\"white-space: pre-wrap; word-break: break-word;\">{text_repr}</pre>
        </div>
      </div>
    """
