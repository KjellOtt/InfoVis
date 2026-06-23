import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc, callback, Input, Output
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=None, random_state=42),
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

def calculate_matrices(cleaned: pd.DataFrame, target_column: str, method: str):
    X = cleaned.drop(columns=[target_column])
    y_raw = cleaned[target_column].fillna("missing").astype(str)
    labels = sorted(y_raw.unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = y_raw.map(mapping)
    
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    figures = []

    if method == "cv":
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for name, factory in CLASSIFIERS.items():
            pipeline = get_pipeline(factory, num_features, cat_features)
            y_pred = cross_val_predict(pipeline, X, y, cv=skf)
            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            # Durchschnitt pro Fold wie in Evaluierung
            cm_avg = cm / 10.0
            fig = plot_confusion_matrix_plotly(cm_avg, name)
            figures.append(fig)
    else:
        # Bootstrapping .632
        rng = np.random.default_rng(42)
        n = len(cleaned)
        n_iterations = 10
        
        for name, factory in CLASSIFIERS.items():
            all_cms = []
            for _ in range(n_iterations):
                sampled_indices = rng.choice(n, size=n, replace=True)
                unique_indices = np.unique(sampled_indices)
                oob_indices = np.setdiff1d(np.arange(n), unique_indices)
                
                if len(oob_indices) == 0: continue
                
                X_train, y_train = X.iloc[sampled_indices], y.iloc[sampled_indices]
                X_oob, y_oob = X.iloc[oob_indices], y.iloc[oob_indices]
                
                pipeline = get_pipeline(factory, num_features, cat_features)
                pipeline.fit(X_train, y_train)
                y_pred_oob = pipeline.predict(X_oob)
                
                all_cms.append(confusion_matrix(y_oob, y_pred_oob, labels=[0, 1]))
            
            cm_avg = np.mean(all_cms, axis=0) if all_cms else np.zeros((2, 2))
            fig = plot_confusion_matrix_plotly(cm_avg, name)
            figures.append(fig)
            
    return figures

def render_matrix(cleaned: pd.DataFrame, target_column: str):
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Validierungsmethode:", className="fw-bold me-2"),
                dcc.RadioItems(
                    id='matrix-method-toggle',
                    options=[
                        {'label': ' 10-Fold Cross-Validation', 'value': 'cv'},
                        {'label': ' Bootstrapping 0.632', 'value': 'bootstrap'}
                    ],
                    value='cv',
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                )
            ], className="col-md-12"),
        ], className="row mb-3 p-3 border rounded bg-light"),

        html.Div(id='matrix-info-alert', className="alert alert-info mb-4"),
        
        html.Div(id='matrix-container', className="row justify-content-center"),
        
        dcc.Store(id='matrix-data-store', data={'target': target_column})
    ])

@callback(
    [Output('matrix-container', 'children'),
     Output('matrix-info-alert', 'children')],
    [Input('matrix-method-toggle', 'value'),
     Input('matrix-data-store', 'data')]
)
def update_matrix_view(method, store_data):
    from main import load_data
    from pages.übersicht import bereinige_daten
    
    raw_data = load_data()
    cleaned, _ = bereinige_daten(raw_data)
    target_column = store_data['target']
    
    figures = calculate_matrices(cleaned, target_column, method)
    
    method_name = "10-Fold Cross-Validation" if method == "cv" else "Bootstrapping 0.632"
    info_text = [
        f"Die Konfusionsmatrizen zeigen die durchschnittlichen Vorhersagen (pro Fold/Iteration) einer ",
        html.Strong(method_name),
        "."
    ]
    
    cards = []
    classifier_names = list(CLASSIFIERS.keys())
    for i, fig in enumerate(figures):
        name = classifier_names[i]
        cards.append(
            html.Div([
                html.Div([
                    html.Div([
                        html.H5(name, className='card-title'),
                        dcc.Graph(figure=fig, config={'displayModeBar': False})
                    ], className='card-body text-center')
                ], className='card h-100 shadow-sm')
            ], className='col-md-6 col-lg-4 mb-4')
        )
        
    return cards, info_text
