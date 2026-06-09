from dash import html, dash_table
import numpy as np
import pandas as pd

from pages.navbar import create_navbar
from functools import lru_cache
import base64
import requests

PLAYER_COLORS = ["#0d6efd", "#fd7e14"]

@lru_cache(maxsize=512)
def fetch_image_data_url(image_url: str) -> str | None:
    if not image_url:
        return None
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "image/png")
        encoded = base64.b64encode(response.content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    except Exception:
        return None


def _attribute_scales(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    scales = {}
    for col in df.columns:
        series = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index)
        if pd.notna(series).any():
            scales[col] = (float(series.min()), float(series.max()))
    return scales


def _scaled_width(value, min_v: float, max_v: float) -> float:
    if _is_missing(value):
        return 0.0
    v = float(value)
    if max_v == min_v:
        return 100.0
    return ((v - min_v) / (max_v - min_v)) * 100.0


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _serialise_value(value):
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text_value(value):
    if _is_missing(value):
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not _is_missing(value)


def _player_label(row: pd.Series, fallback: str) -> str:
    for column in ("Known As", "Full Name", "Name"):
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    return fallback


def _player_meta(row: pd.Series) -> list[tuple[str, str]]:
    fields = ["Club Name", "Nationality", "Best Position", "Preferred Foot"]
    meta = []
    for field in fields:
        if field in row.index:
            value = _text_value(row[field])
            if value != "-":
                meta.append((field, value))
    return meta


def _player_image(row: pd.Series, name: str) -> html.Div | None:
    if "Image Link" not in row.index:
        return None

    image_url = row["Image Link"]
    if _is_missing(image_url) or not str(image_url).strip():
        return None

    cached_src = fetch_image_data_url(str(image_url))

    return html.Div(
        className="text-center mb-3",
        children=[
            html.Img(
                src=cached_src,
                alt=name,
                style={
                    "width": "120px",
                    "height": "120px",
                    "objectFit": "contain",
                    "borderRadius": "16px",
                    "backgroundColor": "#f8f9fa",
                    "padding": "8px",
                },
            ),
            html.Div(name, className="mt-2 fw-semibold"),
        ],
    )


def _bar(width: float, color: str) -> html.Div:
    return html.Div(
        style={
            "height": "10px",
            "backgroundColor": "#e9ecef",
            "borderRadius": "999px",
            "overflow": "hidden",
            "width": "100%",
            "marginTop": "6px",
        },
        children=html.Div(
            style={
                "width": f"{max(0.0, min(100.0, width)):.1f}%",
                "height": "100%",
                "backgroundColor": color,
                "borderRadius": "999px",
            }
        ),
    )


def _value_block(value, color: str, width: float | None = None) -> html.Div:
    if width is not None:
        return html.Div(
            [
                html.Div(
                    _text_value(value),
                    className="fw-semibold mb-1",
                    style={"color": color},
                ),
                _bar(width, color),
            ]
        )
    return html.Div(
        html.Div(
            _text_value(value),
            className="badge rounded-pill",
            style={"backgroundColor": color, "color": "white"},
        )
    )


def _attribute_card(attr: str, value1, value2, player1: str, player2: str, scale=None) -> html.Div:
    numeric = _is_numeric(value1) and _is_numeric(value2)

    if numeric:
        v1 = float(value1)
        v2 = float(value2)
        if scale and attr in scale:
            min_v, max_v = scale[attr]
            w1 = _scaled_width(v1, min_v, max_v)
            w2 = _scaled_width(v2, min_v, max_v)
            scale_label = f"Globale Skala: {min_v:g} – {max_v:g}"
        else:
            max_v = max(v1, v2)
            if max_v == 0:
                w1 = w2 = 100.0
            else:
                w1 = (v1 / max_v) * 100.0
                w2 = (v2 / max_v) * 100.0
            scale_label = "Relative Skala zwischen den beiden Spielern"

        content = html.Div([
            html.Div(attr, className="fw-semibold"),
            html.Div(scale_label, className="small text-muted mb-2"),
            html.Div([
                html.Div([
                    html.Span(player1, className="text-muted small"),
                    html.Div(_text_value(value1), className="fw-semibold", style={"color": PLAYER_COLORS[0]}),
                    _bar(w1, PLAYER_COLORS[0]),
                ], className="mb-3"),
                html.Div([
                    html.Span(player2, className="text-muted small"),
                    html.Div(_text_value(value2), className="fw-semibold", style={"color": PLAYER_COLORS[1]}),
                    _bar(w2, PLAYER_COLORS[1]),
                ]),
            ])
        ])
    else:
        content = html.Div([
            html.Div(attr, className="fw-semibold"),
            html.Div("Kategorial", className="small text-muted mb-2"),
            html.Div([
                html.Div([
                    html.Span(player1, className="text-muted small"),
                    _value_block(value1, PLAYER_COLORS[0]),
                ], className="mb-3"),
                html.Div([
                    html.Span(player2, className="text-muted small"),
                    _value_block(value2, PLAYER_COLORS[1]),
                ]),
            ])
        ])

    return html.Div(
        className="card shadow-sm border-0 mb-3",
        children=html.Div(className="card-body py-3", children=content),
    )


def _player_card(row: pd.Series, idx: int) -> html.Div:
    name = _player_label(row, f"Spieler {idx}")
    meta = _player_meta(row)
    image_block = _player_image(row, name)

    body_children = [
        html.Div(
            className="d-flex justify-content-between align-items-start mb-2",
            children=[
                html.Span(
                    f"Spieler {idx + 1}",
                    className="badge text-bg-dark",
                ),
            ],
        ),
    ]

    if image_block is not None:
        body_children.extend([
            image_block,
            html.Div(f"Zeile {idx}", className="text-muted small mb-3 text-center"),
        ])
    else:
        body_children.extend([
            html.H5(name, className="card-title mb-1"),
            html.Div(f"Zeile {idx}", className="text-muted small mb-3"),
        ])

    body_children.append(
        html.Div(
            [
                html.Div(
                    [
                        html.Span(f"{field}: ", className="text-muted"),
                        html.Strong(value),
                    ],
                    className="small mb-1",
                )
                for field, value in meta
            ]
        )
    )

    return html.Div(
        className="col-md-6",
        children=html.Div(
            className="card shadow-sm border-0 h-100",
            children=[
                html.Div(
                    className="card-body",
                    children=body_children,
                )
            ],
        ),
    )


def build_h2h_sections(df: pd.DataFrame, selected_row_ids):
    selected_ids: list[int] = []
    for row_id in selected_row_ids or []:
        try:
            idx = int(row_id)
        except Exception:
            continue
        if idx not in selected_ids:
            selected_ids.append(idx)

    if len(selected_ids) < 2:
        info = html.Div()
        compare = html.Div()
        return info, compare

    used_ids = selected_ids[:2]
    if max(used_ids) >= len(df) or min(used_ids) < 0:
        info = html.Div(
            className="alert alert-danger mb-3",
            children="Mindestens ein ausgewählter Spieler liegt außerhalb des gültigen Bereichs.",
        )
        compare = html.Div()
        return info, compare

    row1 = df.iloc[used_ids[0]]
    row2 = df.iloc[used_ids[1]]
    name1 = _player_label(row1, f"Spieler {used_ids[0]}")
    name2 = _player_label(row2, f"Spieler {used_ids[1]}")

    scales = _attribute_scales(df)

    selection_note = ""
    if len(selected_ids) > 2:
        selection_note = " Es werden nur die ersten zwei markierten Spieler verglichen."

    info = html.Div(
        className="alert alert-success mb-3",
        children=f"Vergleich aktiv: {name1} vs. {name2}.{selection_note}",
    )

    compare = html.Div(
        children=[
            html.Div(
                className="row g-3 mb-4",
                children=[
                    _player_card(row1, used_ids[0]),
                    _player_card(row2, used_ids[1]),
                ],
            ),
            html.Div(
                className="mb-2 text-muted small",
                children="Vertikaler Vergleich aller Attribute",
            ),
            html.Div(
                children=[
                    _attribute_card(attr, row1[attr], row2[attr], name1, name2, scales)
                    for attr in df.columns
                ]
            ),
        ]
    )

    return info, compare


def create_h2h_layout(instance):
    df = instance.df

    table_records = []
    for idx, row in df.reset_index(drop=True).iterrows():
        record = {"id": idx}
        for col in df.columns:
            record[col] = _serialise_value(row[col])
        table_records.append(record)

    return html.Div(
        [
            create_navbar(),
            html.Div(
                className="container-fluid py-4",
                children=[
                    html.Div(
                        className="row mb-4",
                        children=[
                            html.Div(
                                className="col-12",
                                children=html.Div(
                                    className="card shadow-sm border-0",
                                    children=html.Div(
                                        className="card-body py-4",
                                        children=[
                                            html.H1(
                                                "H2H Vergleich",
                                                className="text-center mb-2 fw-bold",
                                            )
                                        ],
                                    ),
                                ),
                            )
                        ],
                    ),
                    html.Div(),
                    html.Div(
                        className="row mb-4",
                        children=[
                            html.Div(
                                className="col-12",
                                children=html.Div(
                                    className="card shadow-sm border-0",
                                    children=[
                                        html.Div(
                                            className="card-header bg-white border-0 py-3",
                                            children=html.H5("Spieler auswählen", className="mb-0 fw-semibold"),
                                        ),
                                        html.Div(
                                            className="card-body p-0",
                                            children=dash_table.DataTable(
                                                id="h2h-table",
                                                columns=[{"name": col, "id": col} for col in df.columns],
                                                data=table_records,
                                                row_selectable="multi",
                                                selected_row_ids=[],
                                                page_size=15,
                                                sort_action="native",
                                                virtualization=True,
                                                filter_action="native",
                                                fixed_rows={"headers": True},
                                                style_table={
                                                    "overflowX": "auto",
                                                    "maxHeight": "720px",
                                                    "overflowY": "auto",
                                                },
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "8px",
                                                    "whiteSpace": "normal",
                                                    "fontSize": "13px",
                                                    "minWidth": "120px",
                                                    "maxWidth": "260px",
                                                    "border": "1px solid #e9ecef",
                                                },
                                                style_header={
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#f8f9fa",
                                                    "border": "1px solid #dee2e6",
                                                },
                                                style_data={
                                                    "backgroundColor": "white",
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"state": "selected"},
                                                        "backgroundColor": "#dbeafe",
                                                        "border": "1px solid #0d6efd",
                                                    },
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "#fcfcfd",
                                                    },
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),
                    html.Div(id="h2h-selection-info"),
                    html.Div(id="h2h-compare"),
                ],
            ),
        ]
    )



