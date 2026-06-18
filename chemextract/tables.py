"""Table detection — find the genuine tables on a page and return them structured.

A lab page often records data in a grid that the linear transcript flattens into
prose. Page 57's electrode-temperature test is the case in point: two side-by-side
(time, temperature) column-pairs plus a column of observations. Read as a stream,
those cells scatter across lines and the grid is lost — "0 min 22.4 °C 20 min
30.1 C Film looks grey" reads as one run-on line instead of a table.

This pass uses the same vision API to find the ACTUAL tables on the page and return
each as title + columns + rows, every cell verbatim, so the UI can present them as
tables again. It extracts only real tabular layouts — it never forces prose, a lone
equation, or a list of steps into a grid — and it invents nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field


class Table(BaseModel):
    """One genuine table read off the page, cells verbatim."""
    title: Optional[str] = Field(
        description="the table's caption/title on the page, verbatim, or a short inferred label; null if none fits"
    )
    columns: list[str] = Field(
        description="column headers — verbatim when the page writes them, else a short inferred header per column"
    )
    rows: list[list[str]] = Field(
        description="one inner list per row, left->right, one cell per column. Each cell VERBATIM, keeping "
        "numbers/units/symbols exactly ('22.4 °C', '30.1 C', '2θ = 2.1°'); '' for a blank cell"
    )
    note: str = Field(
        default="",
        description="anything notable: headers were inferred, side-by-side blocks merged, an observations column, etc.",
    )
    confidence: float = Field(description="0-1 confidence this is a faithful reading of a real table")


class TablesOutput(BaseModel):
    tables: list[Table] = Field(description="every genuine table on the page; empty list if there are none")


TABLES_SYSTEM = """You are the table-extraction component of a chemistry document \
extraction system. You are shown a page — often a handwritten lab notebook. Find \
every GENUINE table on it and return each as structured rows and columns.

A table is data laid out in a grid: aligned columns with a repeating row structure \
(a time/temperature log, a reagent table, a results matrix). It may have ruled lines \
or merely consistent alignment. Do NOT turn ordinary prose, a single equation, or a \
list of procedure steps into a table — only actual tabular layouts.

For each table:
1. `columns`: a header for each column. Use the page's headers verbatim when they are \
written. If the table has no explicit headers but each column clearly means something \
(e.g. elapsed time vs temperature), supply a short inferred header and say so in `note`.
2. `rows`: one inner list per row, left->right, exactly one cell per column. Copy each \
cell VERBATIM — keep the number, unit and symbol exactly as written ("22.4 °C", \
"30.1 C", "2θ = 2.1°"). Use "" for a blank cell. Every row must have the same number \
of cells as there are columns.
3. If the page lays the same fields out in several side-by-side blocks to save space \
(e.g. two (time, temperature) pairs running across the page), and that is clearly one \
logical table split for space, MERGE them into a single taller table with one set of \
columns. If a block (such as a column of observations) doesn't align cleanly, keep it \
as its own column or its own table rather than forcing the rows.
4. `title`: the caption written above or beside the table, verbatim, or a short \
inferred label describing it.
5. Invent nothing and reorder nothing. Read faithfully; do not compute, total, or \
sort the data. If the page has no tables at all, return an empty list."""


def extract_tables(
    image_path: str | Path, transcript: str | None = None, client: Anthropic | None = None
) -> list[Table]:
    """Vision pass: page image (+ optional transcript context) -> structured tables.
    Returns an empty list when the page has no genuine table."""
    from .extract import MODEL, _image_block  # lazy: avoids a schema<->extract cycle

    client = client or Anthropic()
    user_text = "Find every real table on this page and return it as structured rows and columns."
    if transcript:
        user_text += f"\n\nVerbatim transcript of the page (context only):\n{transcript}"
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=TABLES_SYSTEM,
        messages=[{
            "role": "user",
            "content": [_image_block(image_path), {"type": "text", "text": user_text}],
        }],
        output_format=TablesOutput,
    )
    return response.parsed_output.tables
