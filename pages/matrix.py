import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


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


def plot_confusion_matrix_png(cm: np.ndarray, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negativ", "Positiv"])
    ax.set_yticklabels(["Negativ", "Positiv"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=14)

    fig.colorbar(im, ax=ax)
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_matrix(cleaned: pd.DataFrame, target_column: str) -> str:
    train_df, test_df = stratified_kfold_split(cleaned, target_column)

    rows = []
    figures = []
    for name, factory in CLASSIFIERS.items():
        X_train, y_train = prepare_features(train_df, target_column)
        X_test, y_test = prepare_features(test_df, target_column)
        clf = factory()
        clf.fit(X_train, y_train.astype(str))
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test.astype(str), y_pred, labels=sorted(y_test.astype(str).unique()))
        png = plot_confusion_matrix_png(cm, name)
        figures.append(f"<div class='col-md-4 mb-4'><div class='card'><div class='card-body'><h5 class='card-title'>{name}</h5><img src='data:image/png;base64,{png}' class='img-fluid rounded' alt='Konfusionsmatrix {name}'></div></div></div>")

    return f"""
      <div class=\"row mb-4\">
        <div class=\"col\"><p>Die Konfusionsmatrix zeigt für jeden Klassifikator die vier Quadranten der Vorhersagegüte.</p></div>
      </div>
      <div class=\"row\">
        {''.join(figures)}
      </div>
    """
