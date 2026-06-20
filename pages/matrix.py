import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc
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


def plot_confusion_matrix_plotly(cm: np.ndarray, title: str):
    classes = ["Negativ", "Positiv"]
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="True Label", color="Anzahl"),
        x=classes,
        y=classes,
        title=title
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        coloraxis_showscale=False
    )
    return fig


def render_matrix(cleaned: pd.DataFrame, target_column: str):
    train_df, test_df = stratified_kfold_split(cleaned, target_column)

    figures = []
    X_train, y_train = prepare_features(train_df, target_column)
    X_test, y_test = prepare_features(test_df, target_column)
    
    # Feature Alignment
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
    
    y_train_str = y_train.astype(str)
    y_test_str = y_test.astype(str)
    labels = sorted(y_train_str.unique())

    for name, factory in CLASSIFIERS.items():
        clf = factory()
        clf.fit(X_train, y_train_str)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test_str, y_pred, labels=labels)
        fig = plot_confusion_matrix_plotly(cm, name)
        
        figures.append(
            html.Div([
                html.Div([
                    html.Div([
                        html.H5(name, className='card-title'),
                        dcc.Graph(figure=fig, config={'displayModeBar': False})
                    ], className='card-body text-center')
                ], className='card h-100 shadow-sm')
            ], className='col-md-6 col-lg-4 mb-4')
        )

    return html.Div([
        html.Div([
            html.P([
                "Die Konfusionsmatrizen basieren auf einem ",
                html.Strong("10-Fold Cross-Validation Split"),
                " (Fold 0). Sie zeigen die Vorhersagegüte der einzelnen Klassifikatoren im Detail."
            ])
        ], className="alert alert-info mb-4"),
        html.Div(figures, className="row justify-content-center")
    ])
