import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=42),
    "K-Nearest Neighbor (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
}

def get_pipeline(classifier_factory, num_features, cat_features):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier_factory())
    ])

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
    X = cleaned.drop(columns=[target_column])
    y = cleaned[target_column].fillna("missing").astype(str)
    labels = sorted(y.unique())
    
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    figures = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, factory in CLASSIFIERS.items():
        pipeline = get_pipeline(factory, num_features, cat_features)
        y_pred = cross_val_predict(pipeline, X, y, cv=skf)
        cm = confusion_matrix(y, y_pred, labels=labels)
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
                "Die Konfusionsmatrizen zeigen die aggregierten Vorhersagen einer ",
                html.Strong("10-Fold Cross-Validation"),
                ". Dies bietet eine robustere Übersicht über die Modellleistung über den gesamten Datensatz hinweg."
            ])
        ], className="alert alert-info mb-4"),
        html.Div(figures, className="row justify-content-center")
    ])
