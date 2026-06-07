import dash_bootstrap_components as dbc

def create_navbar():
    """Erstellt eine Navigation Bar mit den Unterseiten Analyse und Tabelle"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink("Analyse", href="/analyse", active="exact")
            ),
            dbc.NavItem(
                dbc.NavLink("Tabelle", href="/tabelle", active="exact")
            ),
        ],
        brand="InfoVis - FIFA Analyse",
        brand_href="/",
        color="dark",
        dark=True,
    )
