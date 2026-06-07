
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics


def create_kmeans_layout(regression_instance):
	"""Erstellt eine K-Means Analyse-Seite.

	- Standardisiert numerische Merkmale
	- Führt PCA durch (2D-Projektion + Scree)
	- Berechnet KMeans für k=2..5
	- Zeichnet Scatterplots (PCA-Projektion) farbig nach Cluster
	- Zeichnet Scree-Plot und Metriken (Silhouette, Davies-Bouldin)
	"""
	df = regression_instance.df.copy()

	# Nur numerische Spalten nutzen
	num_df = df.select_dtypes(include=[np.number]).copy()
	num_df = num_df.dropna().reset_index(drop=True)

	if num_df.shape[0] == 0 or num_df.shape[1] == 0:
		return html.Div("Keine numerischen Daten verfügbar für K-Means.")

	X_raw = num_df.values

	# Standardisieren
	scaler = StandardScaler()
	X = scaler.fit_transform(X_raw)

	# PCA für Scree (voll) und 2D-Projektion
	pca_full = PCA()
	pca_full.fit(X)
	explained = pca_full.explained_variance_ratio_

	pca2 = PCA(n_components=2)
	X_pca = pca2.fit_transform(X)

	ks = [2, 3, 4, 5]
	results = {}
	sil_scores = []
	db_scores = []

	for k in ks:
		kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
		labels = kmeans.fit_predict(X)
		# Metriken
		sil = metrics.silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float('nan')
		db = metrics.davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float('nan')
		sil_scores.append(sil)
		db_scores.append(db)
		results[k] = {"labels": labels, "kmeans": kmeans}

	# Erzeuge Scatterplots (je k ein Graph)
	scatter_graphs = []
	for k in ks:
		labels = results[k]["labels"]
		fig = px.scatter(
			x=X_pca[:, 0], y=X_pca[:, 1], color=labels.astype(str),
			labels={"x": "PC1", "y": "PC2"},
			title=f"K-Means (k={k}) - PCA Projektion",
			color_discrete_sequence=px.colors.qualitative.Safe,
		)
		fig.update_traces(marker=dict(size=6, opacity=0.8))
		fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend_title_text='Cluster')
		scatter_graphs.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

	# Scree-Plot
	scree_fig = go.Figure()
	scree_fig.add_trace(go.Bar(x=list(range(1, len(explained) + 1)), y=explained, name='Explained Variance'))
	scree_fig.update_layout(title="Scree Plot (Erklärte Varianzanteile)", xaxis_title="Principal Component", yaxis_title="Explained Variance Ratio")

	# Metriken: Silhouette und Davies-Bouldin im Vergleich
	metrics_fig = make_subplots(specs=[[{"secondary_y": True}]])
	metrics_fig.add_trace(go.Scatter(x=ks, y=sil_scores, mode='lines+markers', name='Silhouette Score'), secondary_y=False)
	metrics_fig.add_trace(go.Scatter(x=ks, y=db_scores, mode='lines+markers', name='Davies-Bouldin Index'), secondary_y=True)
	metrics_fig.update_xaxes(title_text='Anzahl Cluster (k)')
	metrics_fig.update_yaxes(title_text='Silhouette Score (höher besser)', secondary_y=False)
	metrics_fig.update_yaxes(title_text='Davies-Bouldin (niedriger besser)', secondary_y=True)
	metrics_fig.update_layout(title='Metriken für verschiedene k')

	# Bestimme empfohlene k anhand Silhouette (höchster) und Davies-Bouldin (niedrigster)
	best_k_sil = ks[int(np.nanargmax(sil_scores))]
	best_k_db = ks[int(np.nanargmin(db_scores))]

	summary = (
		f"Optimale Anzahl Cluster: Silhouette best: k={best_k_sil} (Score={max(sil_scores):.4f}), "
		f"Davies-Bouldin best: k={best_k_db} (Index={min(db_scores):.4f})"
	)

	# Layout
	return html.Div(children=[
		html.Div(className="container-fluid p-4", children=[
			html.H1("K-Means Clustering - Wein Daten", className="text-center mb-4"),

			html.Div(className="row mb-3", children=[
				html.Div(className="col-lg-6 mb-3", children=[
					html.Div(className="card", children=[
						html.Div(className="card-body p-2", children=[
							scatter_graphs[0]
						])
					])
				]),
				html.Div(className="col-lg-6 mb-3", children=[
					html.Div(className="card", children=[
						html.Div(className="card-body p-2", children=[
							scatter_graphs[1]
						])
					])
				]),
				html.Div(className="col-lg-6 mb-3", children=[
					html.Div(className="card", children=[
						html.Div(className="card-body p-2", children=[
							scatter_graphs[2]
						])
					])
				]),
				html.Div(className="col-lg-6 mb-3", children=[
					html.Div(className="card", children=[
						html.Div(className="card-body p-2", children=[
							scatter_graphs[3]
						])
					])
				]),
			]),

			html.Div(className="row g-3", children=[
				html.Div(className="col-md-6", children=[
					html.Div(className="card", children=[
						html.Div(className="card-body", children=[
							html.H5("Scree Plot", className="card-title"),
							dcc.Graph(figure=scree_fig, config={"displayModeBar": False})
						])
					])
				]),
				html.Div(className="col-md-6", children=[
					html.Div(className="card", children=[
						html.Div(className="card-body", children=[
							html.H5("Vergleich der Metriken", className="card-title"),
							dcc.Graph(figure=metrics_fig, config={"displayModeBar": False}),
							html.P(summary, className="mt-2 small", style={"fontFamily": "monospace"})
						])
					])
				])
			])
		])
	])

