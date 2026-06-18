"""Spatial layout / placement refinement — the last pass before the document is shown.

The rest of the pipeline reads the page as a *stream*: a verbatim transcript plus
a flat list of recognised structures. That stream loses the page's two dimensions.
When the UI rebuilds the document it can only stack every drawing in one left
column and walk the text top-to-bottom, so on a real notebook page two things go
wrong (both visible on Page 57):

  1. Diagrams that sit at different places on the page — diglyme under the
     electrolyte line, the crown-ether/LiTFSI reaction scheme lower down — all
     collapse into a single straight column, detached from where they belong.
  2. A reaction drawn left-to-right ("[Li(12-crown-4)]+  + e-  --(-0.45 V)-->  Li
     + 12-crown-4") loses its connectors: the "+ e-" and the arrow label never
     make it into the linear transcript, so the scheme reads wrong.

This pass fixes both with one vision call. It is given the original page image,
a clean canonical drawing of every structure already recognised, and the final
fully-specified text, and it returns an explicit top-to-bottom LAYOUT: an ordered
list of rows where each row is either a line of text (with its indentation) or a
horizontal band of structures with the reaction connectors restored between them.

It is placement only. It never rewrites the chemistry text — text lines are copied
verbatim (brackets and all) so the green/black review highlighting survives — and
it may only reference structures the earlier legs already recognised, by index.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field

# StructureVerdict / render_png come from structures.py and Table from tables.py;
# both import extract.py lazily — so importing them here does not create a cycle and
# lets schema.py pull PageLayout in alongside the other result models.
from .structures import StructureVerdict, render_png
from .tables import Table


# --- Result models -----------------------------------------------------------

class LayoutItem(BaseModel):
    """One element in a horizontal diagram band: either a placed structure or a
    bit of connecting text (a reaction operator, a species label, a caption)."""
    kind: str = Field(
        description='"structure" for a drawn molecule, or "text" for a connector/'
        'label drawn alongside it such as "+ e⁻", "⟶", "Li +", "LiTFSI"'
    )
    structure_index: Optional[int] = Field(
        default=None,
        description="index into the structures list (the manifest) when kind=='structure'; null otherwise",
    )
    text: Optional[str] = Field(
        default=None, description="the connector/label text when kind=='text'; null otherwise"
    )


class LayoutRow(BaseModel):
    """One row of the page, read top-to-bottom. Either a line of document text, a
    horizontal band of drawn structures, or a data table placed where it sits. A
    reaction scheme is a single diagram row with structures and operator text
    interleaved left->right."""
    kind: str = Field(
        description='"text" for a line of prose/data, "diagram" for a band of drawn '
        'structures, or "table" for a data table placed at this position'
    )
    text: Optional[str] = Field(
        default=None,
        description="the line of final text VERBATIM (keep any [bracketed] insertions exactly) when kind=='text'",
    )
    indent: int = Field(
        default=0, description="0-6, the line's left-indentation on the page so the layout mirrors the original"
    )
    items: list[LayoutItem] = Field(
        default=[], description="left->right contents of a diagram row when kind=='diagram'"
    )
    table_index: Optional[int] = Field(
        default=None, description="index into the tables list (the table manifest) when kind=='table'; null otherwise"
    )


class PageLayout(BaseModel):
    rows: list[LayoutRow] = Field(
        description="the page rebuilt top-to-bottom, every diagram placed where it actually sits"
    )
    corrections: list[str] = Field(
        default=[],
        description="concretely what the linear transcript got wrong that you fixed — diagrams "
        "repositioned, missing reaction parts (e.g. '+ e⁻') restored",
    )
    confidence: float = Field(description="0-1 confidence in the spatial reconstruction")


# --- Pure helpers (no LLM) ---------------------------------------------------

def placed_indices(layout: PageLayout) -> set[int]:
    """Structure indices the layout actually drops onto the page, so the UI can
    show whatever's left over in a fallback strip instead of silently losing it."""
    return {
        it.structure_index
        for row in layout.rows
        for it in row.items
        if it.kind == "structure" and it.structure_index is not None
    }


def placed_table_indices(layout: PageLayout) -> set[int]:
    """Table indices the layout places, so any table it doesn't position can still
    be shown in a fallback section rather than silently dropped."""
    return {
        row.table_index
        for row in layout.rows
        if row.kind == "table" and row.table_index is not None
    }


def layout_to_text(layout: PageLayout) -> str:
    """Flatten a layout back to plain text (diagram rows shown as their inline
    connectors). Handy for logging/regression tests; the UI renders rows directly."""
    out = []
    for row in layout.rows:
        if row.kind == "diagram":
            parts = [
                (it.text or "") if it.kind == "text" else f"[structure {it.structure_index}]"
                for it in row.items
            ]
            out.append("  " * row.indent + " ".join(p for p in parts if p))
        elif row.kind == "table":
            out.append("  " * row.indent + f"[table {row.table_index}]")
        else:
            out.append("  " * row.indent + (row.text or ""))
    return "\n".join(out)


# --- The vision pass ---------------------------------------------------------

LAYOUT_SYSTEM = """You are the layout component of a chemistry document extraction \
system. You are shown the ORIGINAL page image, a clean canonical drawing of every \
chemical structure already recognised on it, and the system's best fully-specified \
transcript of the text. Earlier stages read the page as a stream and lost its 2-D \
layout: the drawings got stacked in one column and reaction schemes lost their \
connectors. Your job is to rebuild the page faithfully in space.

Return `rows`, top-to-bottom, exactly as they appear on the page:

- TEXT row (kind="text"): copy one non-empty line of the provided transcript into \
`text` VERBATIM — keep every [bracketed] insertion exactly, do NOT reword, add, \
correct, or drop any content. Set `indent` (0-6) to mirror that line's left margin \
on the page.

- DIAGRAM row (kind="diagram"): wherever the page shows one or more drawn \
structures, emit a diagram row AT THAT VERTICAL POSITION (between the surrounding \
text rows — not all collected together). List its contents left->right in `items`, \
matching the page's arrangement:
    * a drawn molecule -> item kind="structure" with the `structure_index` from the \
      manifest below that it depicts;
    * any connector or species the drawing shows between molecules -> item \
      kind="text", e.g. "+ e⁻", the arrow with its "-0.45 V vs Ag/AgCl" label, \
      "Li +". This is where you RESTORE the reaction parts the linear transcript \
      dropped, so the scheme reads correctly left->right.

- TABLE row (kind="table"): where the page shows a data table that has already been \
extracted (see the table manifest below), emit a table row with `table_index` at the \
vertical position the table occupies, and do NOT also re-emit its cells as text rows.

Rules — follow exactly:
1. Reference structures and tables ONLY by an index that exists in the manifests. \
Never invent one. If the same molecule is drawn more than once (e.g. the crown ether \
both free and as the Li complex), reuse the closest matching index.
2. Include every non-empty transcript line exactly once, in reading order, \
interleaved with the diagram/table rows at the right places — EXCEPT lines that are \
just the cells of a table you placed with a table row; those are covered by the table.
3. You place and connect; you do NOT rewrite chemistry. Text lines stay verbatim.
4. In `corrections`, state concretely what you fixed (diagrams repositioned, \
"+ e⁻" restored to the reduction scheme, a table placed in line, etc.). If nothing \
needed fixing, return an empty list.
5. `confidence` is how sure you are of the spatial reconstruction (0-1)."""


def _structures_manifest(structures: list[StructureVerdict]) -> str:
    if not structures:
        return "(no drawn structures were recognised on this page)"
    lines = []
    for i, s in enumerate(structures):
        bits = [f"[{i}]"]
        if s.label:
            bits.append(f'label="{s.label}"')
        if s.name:
            bits.append(f'name="{s.name}"')
        if s.structure_smiles:
            bits.append(f"smiles={s.structure_smiles}")
        if s.recognized and s.recognized.is_reaction:
            bits.append("(drawn as part of a reaction scheme)")
        lines.append(" ".join(bits))
    return "\n".join(lines)


def _tables_manifest(tables: list[Table]) -> str:
    if not tables:
        return "(no data tables were extracted from this page)"
    lines = []
    for i, t in enumerate(tables):
        cols = ", ".join(t.columns) if t.columns else "?"
        title = t.title or "untitled"
        lines.append(f'[{i}] "{title}" — {len(t.rows)} row(s), columns: {cols}')
    return "\n".join(lines)


def _structure_image_blocks(structures: list[StructureVerdict]) -> list[dict]:
    """Attach a clean raster of each structure so the model can tie the page's
    rough hand drawings to a known identity. Best-effort: skips any structure we
    can't rasterise (no SMILES, or RDKit/Cairo unavailable)."""
    blocks: list[dict] = []
    for i, s in enumerate(structures):
        png = render_png(s.structure_smiles)
        if not png:
            continue
        label = s.name or s.label or "?"
        blocks.append({"type": "text", "text": f"Canonical drawing of structure [{i}] = {label}:"})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode("utf-8"),
            },
        })
    return blocks


def refine_layout(
    image_path: str | Path,
    final_text: str,
    structures: list[StructureVerdict],
    tables: list[Table] | None = None,
    client: Anthropic | None = None,
) -> PageLayout:
    """One vision call: page image + canonical structure drawings + final text (+ any
    extracted tables) -> an explicit top-to-bottom layout with diagrams and tables
    placed where they belong and reaction connectors restored."""
    from .extract import MODEL, _image_block  # lazy: avoids a schema<->extract cycle

    tables = tables or []
    client = client or Anthropic()
    user_text = (
        "Rebuild this page's layout. Place each drawn structure and data table where "
        "it actually sits and restore any reaction connectors the transcript dropped.\n\n"
        "Structures recognised on the page (reference these by index):\n"
        f"{_structures_manifest(structures)}\n\n"
        "Tables already extracted from the page (place each with a table row by index, "
        "do not re-type their cells):\n"
        f"{_tables_manifest(tables)}\n\n"
        "Fully-specified transcript (copy text lines verbatim, keep [brackets]):\n"
        "---\n"
        f"{final_text}\n"
        "---"
    )
    content: list[dict] = [_image_block(image_path)]
    content += _structure_image_blocks(structures)
    content.append({"type": "text", "text": user_text})

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=LAYOUT_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_format=PageLayout,
    )
    return response.parsed_output
