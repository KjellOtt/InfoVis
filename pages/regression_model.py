import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RegressionModel:
    """Verwaltet Regressions-Berechnung und Visualisierung"""

    def __init__(self, df):
        self.df = df
        self.model = None

    def calculate_regression(self, x_attr: str, y_attr: str):
        """Berechnet lineare Regression und gibt Figure + Metriken zurück"""

        # Bereinige Daten
        temp_df = self.df[[x_attr, y_attr]].dropna()

        if len(temp_df) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Nicht genug gültige Datenpunkte ohne NaNs",
                               showarrow=False, font={"size": 20})
            return fig, "Zu viele fehlende Werte für diese Kombination."

        # Modell trainieren
        X = temp_df[[x_attr]].values
        y = temp_df[y_attr].values

        self.model = LinearRegression()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        # Metriken
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)

        # Plot
        fig = self._create_plot(X, y, x_attr, y_attr, rmse, mae, mse, r2)
        metrics_text = self._format_metrics(x_attr, y_attr, r2, rmse, mae, mse, len(self.df))

        return fig, metrics_text

    def _create_plot(self, X, y, x_attr, y_attr, rmse, mae, mse, r2):
        """Erstellt das Plotly-Diagramm"""
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = self.model.predict(X_line)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y, mode='markers', name='Datenpunkte',
            marker=dict(size=8, opacity=0.6, color='steelblue'),
            hovertemplate='<b>%{customdata[0]}</b><br>%{x}<br>%{y}<extra></extra>',
            customdata=[[f'{x_attr}/{y_attr}'] for _ in range(len(X))]
        ))

        fig.add_trace(go.Scatter(
            x=X_line.flatten(), y=y_line, mode='lines', name='Regressionslinie',
            line=dict(color='red', width=3),
            hovertemplate='%{x:.3f} → %{y:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Lineare Regression: {y_attr} vs {x_attr}",
            xaxis_title=x_attr, yaxis_title=y_attr,
            hovermode='closest', plot_bgcolor='white', height=550,
            showlegend=True, legend=dict(x=0.02, y=0.98),
            font=dict(size=12)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        return fig

    def _format_metrics(self, x_attr, y_attr, r2, rmse, mae, mse, data_points):
        """Formatiert Metriken als Text"""
        return f"""Regressionsmetriken
{'=' * 30}

Attribute:
  X-Achse: {x_attr}
  Y-Achse: {y_attr}

Koeffizient:
  Steigung: {self.model.coef_[0]:.6f}
  Konstante: {self.model.intercept_:.6f}

Qualitätsmetriken:
  R² Score: {r2:.6f}
  RMSE: {rmse:.6f}
  MAE: {mae:.6f}
  MSE: {mse:.6f}

Datenpunkte: {data_points}
"""
