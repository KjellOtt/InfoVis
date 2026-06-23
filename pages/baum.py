import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dash import html
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=42),
    "K-Nearest Neighbor (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
}


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
    X = cleaned.drop(columns=[target_column])
    y = cleaned[target_column].fillna("missing").astype(str)
    
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    X_transformed = preprocessor.fit_transform(X)
    
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
    feature_names = num_features + cat_feature_names

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    clf.fit(X_transformed, y)

    class_names = [str(c) for c in sorted(y.unique())]
    tree_png = plot_tree_png(clf, feature_names, class_names)
    text_repr = export_text(clf, feature_names=feature_names)

    return html.Div([
        html.Div([
            html.P([
                "Der Entscheidungsbaum visualisiert die gelernten Regeln des Modells. ",
                "Die Daten wurden vorab skaliert und kodiert, um konsistent mit den anderen Klassifikatoren zu sein. ",
                "Die Tiefe wurde auf 4 Ebenen begrenzt, um die Interpretierbarkeit zu gewährleisten."
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
