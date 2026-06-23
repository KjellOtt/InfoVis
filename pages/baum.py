import base64
import io
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
from dash import html, dcc, callback, Input, Output
import pages.übersicht as übersicht
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, _tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


CLASSIFIERS = {
    "Logistische Regression": lambda: LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=None, random_state=42),
    "K-Nearest Neighbor (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
}


def plot_tree_plotly(clf: DecisionTreeClassifier, feature_names: list[str], class_names: list[str]) -> go.Figure:
    """
    Erstellt eine interaktive Visualisierung des Entscheidungsbaums mit Plotly.
    """
    tree_ = clf.tree_
    
    def get_node_info(node_id):
        samples = tree_.n_node_samples[node_id]
        values = tree_.value[node_id][0]
        if class_names:
            class_idx = np.argmax(values)
            class_label = class_names[class_idx]
        else:
            class_label = "N/A"
            
        hover_info = f"Samples: {samples}<br>Klasse: {class_label}<br>Werte: {values.tolist()}"
        
        display_label = class_label
        
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            hover_info = f"Split: {name} <= {threshold:.2f}<br>" + hover_info
            display_label = f"{name}\n<= {threshold:.2f}"
            
        return hover_info, display_label

    positions = {}
    
    def calc_position(node_id, depth, left, right):
        x = (left + right) / 2
        y = -depth
        positions[node_id] = (x, y)
        
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            calc_position(tree_.children_left[node_id], depth + 1, left, x)
            calc_position(tree_.children_right[node_id], depth + 1, x, right)

    calc_position(0, 0, 0, 1)

    edge_x = []
    edge_y = []
    for node_id, pos in positions.items():
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]
            
            edge_x.extend([pos[0], positions[left_child][0], None])
            edge_y.extend([pos[1], positions[left_child][1], None])
            
            edge_x.extend([pos[0], positions[right_child][0], None])
            edge_y.extend([pos[1], positions[right_child][1], None])

    node_x = []
    node_y = []
    node_hover_text = []
    node_display_text = []
    node_color = []
    
    for node_id, pos in positions.items():
        node_x.append(pos[0])
        node_y.append(pos[1])
        h_info, d_label = get_node_info(node_id)
        node_hover_text.append(h_info)
        node_display_text.append(d_label)
        
        # Farbe basierend auf der Klasse (für Klassifikation)
        class_idx = np.argmax(tree_.value[node_id][0])
        node_color.append(class_idx)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_display_text,
        hovertext=node_hover_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            colorscale='Viridis',
            size=30,
            color=node_color,
            line_width=2)
    ))

    fig.update_layout(
        title="Interaktiver Entscheidungsbaum",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig


def get_tree_content(use_preprocessing: bool, validation_method: str = "cv"):
    raw_data = pd.read_csv("Daten/Titanic.csv")
    cleaned, _ = übersicht.bereinige_daten(raw_data)
    
    def chose_target_column(df: pd.DataFrame, preferred: str = "Survived") -> str | None:
        if preferred in df.columns:
            return preferred
        lower_cols = {col.lower(): col for col in df.columns if isinstance(col, str)}
        if preferred.lower() in lower_cols:
            return lower_cols[preferred.lower()]
        return None

    target_column = chose_target_column(cleaned)
    if target_column is None:
        return html.Div("Zielspalte nicht gefunden.", className="alert alert-danger")

    X = cleaned.drop(columns=[target_column])
    y_raw = cleaned[target_column].fillna("missing").astype(str)
    labels = sorted(y_raw.unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = y_raw.map(mapping)

    rng = np.random.default_rng(42)
    n = len(cleaned)
    
    if validation_method == "cv":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        train_idx, _ = next(skf.split(X, y))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        method_text = "Dies ist ein Beispielbaum, trainiert auf einem der 10 Folds."
    else:
        sampled_indices = rng.choice(n, size=n, replace=True)
        X_train, y_train = X.iloc[sampled_indices], y.iloc[sampled_indices]
        method_text = "Dies ist ein Beispielbaum, trainiert auf einem Bootstrapping-Sample."

    if use_preprocessing:
        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ])
        X_transformed = preprocessor.fit_transform(X_train)
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
        feature_names = num_features + cat_feature_names
        prep_text = "Die Daten wurden standardisiert/kodiert."
    else:
        X_transformed = X_train.copy()
        for col in X_transformed.select_dtypes(exclude=[np.number]).columns:
            if col.lower() == 'sex':
                X_transformed[col] = X_transformed[col].map({'female': 1, 'male': 0})
            else:
                X_transformed[col] = X_transformed[col].astype('category').cat.codes
        feature_names = X.columns.tolist()
        prep_text = "Die Daten sind im Originalzustand."

    clf = DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, random_state=42)
    clf.fit(X_transformed, y_train)

    class_names_raw = sorted(y_raw.unique())
    class_mapping = {"0": "Not Survived", "1": "Survived"}
    class_names = [class_mapping.get(str(c), str(c)) for c in class_names_raw]
    
    tree_fig = plot_tree_plotly(clf, feature_names, class_names)
    text_repr = export_text(clf, feature_names=feature_names)

    legend_items = []
    colors = px.colors.sequential.Viridis
    for i, cls in enumerate(class_names):
        color_idx = int(i * (len(colors)-1) / (len(class_names)-1)) if len(class_names) > 1 else 0
        color = colors[color_idx]
        legend_items.append(html.Div([
            html.Span(style={
                "backgroundColor": color,
                "width": "15px",
                "height": "15px",
                "display": "inline-block",
                "marginRight": "5px",
                "borderRadius": "3px"
            }),
            html.Span(f"Klasse: {cls}", className="me-3")
        ], style={"display": "inline-block"}))

    return html.Div([
        html.Div([
            html.P([html.B("Validierung: "), method_text]),
            html.P([html.B("Vorverarbeitung: "), prep_text])
        ], className="alert alert-info mb-4"),
        html.Div([
            html.Div([
                html.H5("Interaktive Baumstruktur", className="card-title mb-0")
            ], className="card-header bg-primary text-white"),
            html.Div([
                html.Div([
                    html.B("Farberklärung: "),
                    html.Div(legend_items, style={"display": "inline-block"})
                ], className="p-2 border-bottom bg-white"),
                dcc.Graph(figure=tree_fig, config={'displayModeBar': True})
            ], className="card-body p-0 bg-light")
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


def render_tree(cleaned: pd.DataFrame, target_column: str):
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Validierungsmethode:", className="fw-bold me-2"),
                dcc.RadioItems(
                    id='tree-method-toggle',
                    options=[
                        {'label': ' 10-Fold Cross-Validation', 'value': 'cv'},
                        {'label': ' Bootstrapping 0.632', 'value': 'bootstrap'}
                    ],
                    value='cv',
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                )
            ], className="col-md-6"),
            html.Div([
                html.Label("Darstellungsmodus:", className="fw-bold me-2"),
                dcc.RadioItems(
                    id='tree-toggle',
                    options=[
                        {'label': ' Standardisiert', 'value': True},
                        {'label': ' Originalwerte', 'value': False}
                    ],
                    value=True,
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                )
            ], className="col-md-6"),
        ], className="row mb-3 p-3 border rounded bg-light"),

        html.Div(id='tree-container', children=get_tree_content(True, 'cv'))
    ])


@callback(
    Output('tree-container', 'children'),
    [Input('tree-toggle', 'value'),
     Input('tree-method-toggle', 'value')]
)
def update_tree(use_preprocessing, validation_method):
    return get_tree_content(use_preprocessing, validation_method)
