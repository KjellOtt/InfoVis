import dash_bootstrap_components as dbc

def create_navbar():
    """Erstellt eine Navigation Bar mit den Unterseiten Analyse, Tabelle und H2H."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink("Analyse", href="/analyse", active="exact")
            ),
            dbc.NavItem(
                dbc.NavLink("Tabelle", href="/tabelle", active="exact")
            ),
            dbc.NavItem(
                dbc.NavLink("H2H", href="/h2h", active="exact")
            ),
        ],
        brand="InfoVis - FIFA Analyse",
        brand_href="/",
        color="dark",
        dark=True,
    )
