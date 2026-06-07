from dash import html
import dash_bootstrap_components as dbc

def create_navbar():
    """Erstellt eine Navigation Bar mit Links zu Regression und bereinigte Tabelle"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Regression", href="/regression", active="exact")),
            dbc.NavItem(dbc.NavLink("Bereinigte Tabelle", href="/cleaned-table", active="exact")),
        ],
        brand="InfoVis - Wein Analyse",
        brand_href="/",
        color="dark",
        dark=True,
    )
