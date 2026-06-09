from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import base64
import pickle


def create_kmeans_plot(decoded_data, show_boundaries):
	"""Erstellt K-Means Plots basierend auf den gespeicherten Daten.

	Args:
		decoded_data: Dict mit X_pca, X, pca2_components, pca2_mean, ks, results
		show_boundaries: Boolean, ob Trennbereiche angezeigt werden sollen

	Returns:
		List von Plotly-Figuren für k=2,3,4,5
	"""
	X_pca = np.array(decoded_data["X_pca"])
	X = np.array(decoded_data["X"])
	pca2_components = np.array(decoded_data["pca2_components"])
	pca2_mean = np.array(decoded_data["pca2_mean"])

	class SimplePCA:
		def __init__(self, components, mean):
			self.components_ = components
			self.mean_ = mean

		def inverse_transform(self, X):
			return X @ self.components_ + self.mean_

	pca2 = SimplePCA(pca2_components, pca2_mean)

	figs = []
	for k in decoded_data["ks"]:
		k_str = str(k)
		labels = np.array(decoded_data["results"][k_str]["labels"])
		cluster_centers = np.array(decoded_data["results"][k_str]["cluster_centers"])

		h = 0.05
		x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
		y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		mesh_points = np.c_[xx.ravel(), yy.ravel()]
		mesh_points_original = pca2.inverse_transform(mesh_points)

		distances = np.vstack([np.sum((mesh_points_original - center) ** 2, axis=1) for center in cluster_centers])
		mesh_labels = np.argmin(distances, axis=0).reshape(xx.shape)

		fig = go.Figure()

		if show_boundaries:
			fig.add_trace(go.Contour(
				z=mesh_labels,
				x=xx[0],
				y=yy[:, 0],
				colorscale=px.colors.qualitative.Safe[:k],
				showscale=False,
				hoverinfo='skip',
				contours=dict(showlabels=False),
				opacity=0.3,
				name='Cluster Bereich'
			))

		for cluster_id in range(k):
			cluster_mask = labels == cluster_id
			fig.add_trace(go.Scatter(
				x=X_pca[cluster_mask, 0],
				y=X_pca[cluster_mask, 1],
				mode='markers',
				marker=dict(size=6, opacity=0.8, color=px.colors.qualitative.Safe[cluster_id % len(px.colors.qualitative.Safe)]),
				name=f'Cluster {cluster_id}',
				hovertemplate=f'<b>Cluster {cluster_id}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
			))

		centers_pca = (cluster_centers - pca2_mean) @ pca2_components.T
		fig.add_trace(go.Scatter(
			x=centers_pca[:, 0],
			y=centers_pca[:, 1],
			mode='markers',
			marker=dict(size=15, color='black', symbol='x', line=dict(width=2)),
			name='Clusterzentren',
			hovertemplate='<b>Zentrum:</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
		))

		fig.update_layout(
			title=f"K-Means (k={k}) - PCA Projektion",
			xaxis_title='PC1',
			yaxis_title='PC2',
			margin=dict(t=40, b=10, l=10, r=10),
			legend_title_text='Cluster',
			hovermode='closest'
		)

		figs.append(fig)

	return figs


def create_kmeans_layout(regression_instance):
	"""Erstellt eine K-Means Analyse-Seite.

	- Standardisiert numerische Merkmale
	- Führt PCA durch (2D-Projektion + Scree)
	- Berechnet KMeans für k=2..5
	- Zeichnet Scatterplots (PCA-Projektion) farbig nach Cluster
	- Zeichnet Scree-Plot und Metriken (Silhouette, Davies-Bouldin)
	- Zeigt Clusterzentren an
	- Zeigt kumulierte Varianz an
	- Ermöglicht Toggle der Trennbereiche
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
	cumulative_explained = np.cumsum(explained)

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

	# Erzeuge Scatterplots mit Trennbereichen (je k ein Graph)
	def create_plot_with_decision_boundary(k, X_pca, X, pca2, kmeans, labels, k_value, show_boundaries=True):
		"""Erstellt einen Scatterplot mit Decision Boundaries und Clusterzentren."""
		h = 0.05  # Schrittweite für Mesh-Grid
		x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
		y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		
		mesh_points = np.c_[xx.ravel(), yy.ravel()]
		mesh_points_original = pca2.inverse_transform(mesh_points)
		
		mesh_labels = kmeans.predict(mesh_points_original).reshape(xx.shape)
		
		fig = go.Figure()
		
		if show_boundaries:
			fig.add_trace(go.Contour(
				z=mesh_labels,
				x=xx[0],
				y=yy[:, 0],
				colorscale=px.colors.qualitative.Safe[:k_value],
				showscale=False,
				hoverinfo='skip',
				contours=dict(showlabels=False),
				opacity=0.3,
				name='Cluster Bereich'
			))

		for cluster_id in range(k_value):
			cluster_mask = labels == cluster_id
			fig.add_trace(go.Scatter(
				x=X_pca[cluster_mask, 0],
				y=X_pca[cluster_mask, 1],
				mode='markers',
				marker=dict(size=6, opacity=0.8, color=px.colors.qualitative.Safe[cluster_id % len(px.colors.qualitative.Safe)]),
				name=f'Cluster {cluster_id}',
				hovertemplate=f'<b>Cluster {cluster_id}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
			))
		
		centers_pca = pca2.transform(kmeans.cluster_centers_)
		fig.add_trace(go.Scatter(
			x=centers_pca[:, 0],
			y=centers_pca[:, 1],
			mode='markers',
			marker=dict(size=15, color='black', symbol='x', line=dict(width=2)),
			name='Clusterzentren',
			hovertemplate='<b>Zentrum:</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
		))

		fig.update_layout(
			title=f"K-Means (k={k_value}) - PCA Projektion",
			xaxis_title='PC1',
			yaxis_title='PC2',
			margin=dict(t=40, b=10, l=10, r=10),
			legend_title_text='Cluster',
			hovermode='closest'
		)
		
		return fig
	
	scatter_graphs = []
	for k in ks:
		kmeans = results[k]["kmeans"]
		labels = results[k]["labels"]
		fig = create_plot_with_decision_boundary(k, X_pca, X, pca2, kmeans, labels, k, show_boundaries=True)
		scatter_graphs.append(dcc.Graph(figure=fig, config={"displayModeBar": False}, id=f"kmeans-plot-{k}"))

	# Scree-Plot mit kumulierter Varianz
	scree_fig = make_subplots(specs=[[{"secondary_y": True}]])
	scree_fig.add_trace(
		go.Bar(x=list(range(1, len(explained) + 1)), y=explained, name='Explained Variance',
		       marker=dict(color='steelblue')),
		secondary_y=False
	)
	scree_fig.add_trace(
		go.Scatter(x=list(range(1, len(cumulative_explained) + 1)), y=cumulative_explained,
		          mode='lines+markers', name='Kumulierte Varianz',
		          line=dict(color='red', width=2), marker=dict(size=6)),
		secondary_y=True
	)
	scree_fig.update_xaxes(title_text="Principal Component")
	scree_fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
	scree_fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
	scree_fig.update_layout(title="Scree Plot mit kumulierter Varianz", hovermode='x unified')

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

	# Speichere Daten für Callback (für Toggle-Funktionalität)
	kmeans_data = {
		"X_pca": X_pca.tolist(),
		"X": X.tolist(),
		"pca2_components": pca2.components_.tolist(),
		"pca2_mean": pca2.mean_.tolist(),
		"ks": ks,
		"results": {
			str(k): {
				"labels": results[k]["labels"].tolist(),
				"cluster_centers": results[k]["kmeans"].cluster_centers_.tolist()
			}
			for k in ks
		}
	}
	encoded_data = base64.b64encode(pickle.dumps(kmeans_data)).decode()

	# Layout
	return html.Div(children=[
		dcc.Store(id='kmeans-data-store', data=encoded_data),
		dcc.Store(id='kmeans-show-boundaries-store', data=True),
		html.Div(className="container-fluid p-4", children=[
			html.H1("K-Means Clustering - Wein Daten", className="text-center mb-4"),

			html.Div(className="row mb-3", children=[
				html.Div(className="col-12", children=[
					html.Button(
						"Trennbereiche togglen",
						id="kmeans-toggle-button",
						className="btn btn-primary mb-3"
					)
				])
			]),

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
							html.H5("Scree Plot mit kumulierter Varianz", className="card-title"),
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

