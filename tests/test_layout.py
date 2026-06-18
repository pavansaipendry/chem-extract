"""Deterministic pieces of the layout module — no API key, no RDKit/Cairo.

The vision call (refine_layout) is exercised manually with credentials present.
Here we pin down what must hold without it: the PageLayout result model parses
the shape the model returns, and the two pure helpers — placed_indices (so the UI
knows which drawings still need a leftover strip) and layout_to_text (the flatten
used for logging/regression) — behave on the real Page 57 shape: an interleaving
of indented text rows and a reaction band where "[Li(12-crown-4)]+ + e- -> Li +
12-crown-4" reads left->right with its connectors restored.
"""
from chemextract.layout import (
    LayoutItem,
    LayoutRow,
    PageLayout,
    layout_to_text,
    placed_indices,
)


def _page57_layout() -> PageLayout:
    """A small layout in the shape Page 57 produces: an electrolyte line with the
    diglyme drawing (structure 0) under it, then the reduction scheme as one
    diagram band — the crown complex (1), '+ e⁻', the arrow, 'Li +', the free
    crown (2) — with LiTFSI (3) drawn below."""
    return PageLayout(
        rows=[
            LayoutRow(kind="text", text="Electrolyte: 1[M] LiTFSI in diglyme : EtOH", indent=1),
            LayoutRow(kind="diagram", items=[
                LayoutItem(kind="structure", structure_index=0),
            ]),
            LayoutRow(kind="text", text="Deposition run 240604-B1", indent=0),
            LayoutRow(kind="diagram", items=[
                LayoutItem(kind="structure", structure_index=1),
                LayoutItem(kind="text", text="+ e⁻"),
                LayoutItem(kind="text", text="⟶  -0.45 V vs Ag/AgCl"),
                LayoutItem(kind="text", text="Li +"),
                LayoutItem(kind="structure", structure_index=2),
            ]),
            LayoutRow(kind="diagram", items=[
                LayoutItem(kind="structure", structure_index=3),
                LayoutItem(kind="text", text="LiTFSI"),
            ]),
        ],
        corrections=["restored '+ e⁻' and the arrow voltage in the reduction scheme",
                     "moved diglyme under the electrolyte line"],
        confidence=0.86,
    )


def test_pagelayout_parses_real_shape():
    lay = _page57_layout()
    assert len(lay.rows) == 5
    assert lay.rows[0].kind == "text" and lay.rows[0].indent == 1
    band = lay.rows[3]
    assert band.kind == "diagram"
    # the reaction reads structure, connector(s), structure — connectors restored
    kinds = [it.kind for it in band.items]
    assert kinds == ["structure", "text", "text", "text", "structure"]


def test_placed_indices_collects_every_drawn_structure():
    lay = _page57_layout()
    assert placed_indices(lay) == {0, 1, 2, 3}


def test_placed_indices_ignores_text_items():
    lay = PageLayout(
        rows=[LayoutRow(kind="diagram", items=[
            LayoutItem(kind="text", text="+ e⁻"),
            LayoutItem(kind="structure", structure_index=2),
        ])],
        confidence=0.5,
    )
    assert placed_indices(lay) == {2}


def test_layout_to_text_restores_reaction_connectors():
    lay = _page57_layout()
    flat = layout_to_text(lay)
    # the dropped reaction part is back in the flattened view
    assert "+ e⁻" in flat
    assert "[structure 1]" in flat and "[structure 2]" in flat
    # text rows survive verbatim and indentation is reflected
    assert "Electrolyte: 1[M] LiTFSI" in flat
    assert "  Electrolyte" in flat  # indent=1 -> two leading spaces
    assert "Deposition run 240604-B1" in flat


def test_empty_layout_is_safe():
    lay = PageLayout(rows=[], confidence=0.0)
    assert placed_indices(lay) == set()
    assert layout_to_text(lay) == ""
