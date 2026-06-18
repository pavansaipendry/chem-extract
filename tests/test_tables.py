"""Deterministic pieces of table extraction + table-aware layout — no API key.

The vision call (extract_tables) is exercised manually with credentials present.
Here we pin the result model to the real Page 57 shape — the electrode
temperature test, a (time, temperature) log with an observations column — and the
table-placement helpers the UI relies on (a "table" layout row references a table
by index; placed_table_indices reports which tables got positioned so the rest
fall to a leftover strip instead of vanishing).
"""
from chemextract.layout import (
    LayoutItem,
    LayoutRow,
    PageLayout,
    layout_to_text,
    placed_table_indices,
)
from chemextract.tables import Table, TablesOutput


def _temp_table() -> Table:
    return Table(
        title="Electrode Temperature test: hot plate at 30 °C",
        columns=["Time", "Temp", "Observation"],
        rows=[
            ["0 min", "22.4 °C", "Film looks grey + dull"],
            ["1 min", "23.1 °C", "XRD min peak at"],
            ["5 min", "25.6 C", "2θ = 2.1° (low intens.)"],
            ["10 min", "27.9 C", "Shoulder at 2θ = 4.7°"],
            ["20 min", "30.1 C", ""],
            ["40 min", "31.5 C", ""],
            ["1 hr", "32.0 C", ""],
            ["1 hr 30 min", "32.6 C", ""],
        ],
        note="two side-by-side (time, temperature) blocks merged into one column pair; observations aligned to the first four rows",
        confidence=0.82,
    )


def test_table_model_parses_real_shape():
    t = _temp_table()
    assert t.columns == ["Time", "Temp", "Observation"]
    assert len(t.rows) == 8
    # cells kept verbatim, units and symbols intact
    assert t.rows[0] == ["0 min", "22.4 °C", "Film looks grey + dull"]
    assert "2θ = 2.1°" in t.rows[2][2]


def test_tables_output_empty_is_valid():
    out = TablesOutput(tables=[])
    assert out.tables == []


def test_table_row_kind_places_a_table_by_index():
    lay = PageLayout(
        rows=[
            LayoutRow(kind="text", text="Electrode Temperature test:", indent=0),
            LayoutRow(kind="table", table_index=0),
        ],
        confidence=0.8,
    )
    assert placed_table_indices(lay) == {0}


def test_placed_table_indices_ignores_other_rows():
    lay = PageLayout(
        rows=[
            LayoutRow(kind="diagram", items=[LayoutItem(kind="structure", structure_index=1)]),
            LayoutRow(kind="text", text="some prose", indent=1),
        ],
        confidence=0.5,
    )
    assert placed_table_indices(lay) == set()


def test_layout_to_text_marks_table_rows():
    lay = PageLayout(
        rows=[
            LayoutRow(kind="text", text="header line", indent=0),
            LayoutRow(kind="table", table_index=2),
        ],
        confidence=0.7,
    )
    flat = layout_to_text(lay)
    assert "header line" in flat
    assert "[table 2]" in flat
