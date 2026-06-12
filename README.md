# chem-extract

**Extract structured data from chemistry documents — without ever guessing.**

A lab note says `heated to 85`. Eighty-five *what*? °C is the obvious read — but "obvious" is how
a materials scientist loses three weeks chasing a wrong number. The core design rule of this
pipeline is: **extract verbatim, validate deterministically, flag ambiguity, and keep every
inference in a separate, clearly-labeled layer that never overwrites what the document actually
says.** A confident wrong answer is the only unacceptable output.

## See it in 30 seconds — no install, no API key

**Input:** a handwritten lab note. Three quantities are missing their units — `heated to 85`,
`added 0.5 HCl`, `yield: 1.8`:

<img src="eval/images/handwritten_note.png" width="440" alt="handwritten lab note">

**Output** (`python cli.py show results/handwritten_note.json` — this exact result is saved in
the repo): every quantity extracted verbatim with its source quote; the three ambiguous fields
flagged for review — and for each, a *separate* inferred unit with calibrated confidence and
chemistry-grounded reasoning. Clean fields pass green; nothing is silently guessed:

![CLI output: extraction table with flags and unit inference](docs/cli_output.svg)

Note what it did **not** do: it did not write "85 °C" into the data. The page doesn't say °C.
It says `85`, flagged `missing_unit`, with °C proposed alongside — citing the `5 c` written
elsewhere in the same note, the chemistry (aqueous recrystallization stays below 100 °C), and
why Kelvin and Fahrenheit lose (85 K = −188 °C is not "heating").

## What it does

Give it an image or PDF of a chemistry document (handwritten lab notes, typed procedures,
scanned pages). It returns structured observations like this — from a real handwritten note:

```json
{
  "quantity_type": "temperature",
  "value": "85",
  "unit": null,
  "quote": "heated to 85, stirred",
  "confidence": 0.97,
  "flags": ["missing_unit"],
  "needs_review": true,
  "inferred": {
    "unit": "°C",
    "confidence": 0.85,
    "reasoning": "The document explicitly uses Celsius elsewhere ('cooled in ice bath
      to 5 c'). 85 °C is plausible for an aqueous recrystallization (below boiling).
      Kelvin is implausible (85 K = -188 °C is not 'heating'). Fahrenheit is
      inconsistent with the explicit 'c' used for the cooling step."
  }
}
```

The raw extraction (`value`, `unit`, `quote`) is always verbatim. The unit was not on the page,
so `unit` is `null` and the field is flagged for human review. The *inference* — °C, with
calibrated confidence and a chemistry-grounded justification a reviewer can evaluate — lives in
its own sub-object. The system proposes; the human disposes.

## Pipeline

```
image/PDF ─► vision extraction ×N ─► self-consistency check ─► deterministic validation
                (verbatim, schema-       (fields that don't        (unit whitelists, range
                 enforced, Claude)        reproduce get flagged)    checks — no LLM, pure fns)
                                                                          │
            human review ◄─ reconstruction ◄─ unit inference ◄─ flagged fields only
              (web UI)        (canonical rewrite;  (full-document context +
                               unresolved spans     physical plausibility table;
                               shown as competing   never overwrites raw)
                               interpretations)
```

- **Self-consistency:** every page is read N times; observations that don't reproduce across
  runs get `self_consistency_mismatch`.
- **Deterministic validation** (`chemextract/validate.py`): pure functions, no LLM. Unit
  whitelists and plausibility ranges for 25+ quantity types (temperature, volume, wavelength,
  molar mass, potential, …). Catches missing units, unknown units, and unit-confusion misreads
  (`70 K` as a heating temperature converts to −203 °C → `implausible_value`).
- **Grounded inference** (`chemextract/infer.py`): for flagged fields only. The LLM sees the
  full transcript *plus* a deterministic table of which candidate units are physically
  plausible. Honest "can't tell" (null unit, low confidence) beats a confident guess.
- **Reconstruction** (`chemextract/reconstruct.py`): a fully-specified rewrite with inferred
  insertions marked (`70[°C]`); spans the system can't settle are returned as competing
  interpretations for a human to choose between — never silently picked.
- **RAG layer** (`chemextract/rag.py`): reviewed documents are indexed into ChromaDB; questions
  are answered only from indexed sources, with citations, and refused otherwise.

## Evaluation

200 synthetic lab-note images (1,082 ground-truth fields) with deliberately seeded ambiguities —
missing units, unit confusion, smudged digits (`dataset/chem_notes_dataset/`):

| Metric | Result |
|---|---|
| Field detection | 1078/1082 = **99.6%** |
| Extraction accuracy (matched fields, verbatim) | **100.0%** |
| **Flag recall** (ambiguous fields correctly flagged) | 406/407 = **99.8%** |
| Unflagged accuracy (fields the system said were clean) | 610/610 = **100.0%** |

Flag recall is the #1 metric: a missed flag is a silent guess reaching a scientist.
Unflagged accuracy is the trust contract: if the system doesn't flag it, it must be right.

Reproduce with `eval/run_dataset_eval.py`. Unit tests: `python -m pytest tests/` (28 tests,
validation layer is fully deterministic so no API calls needed).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# CLI: process one image
python cli.py process eval/images/handwritten_note.png

# Web UI: drag-and-drop, watch each stage stream live
uvicorn webapp.server:app --port 8000   # → http://localhost:8000
```

## Why this shape

Document extraction in chemistry isn't an OCR problem — it's a trust problem. The same
architecture applies anywhere an AI interprets scientific data and a human acts on the result:
mining synthesis conditions from the literature, instrument-data interpretation, lab-notebook
digitization. The handwriting case here is a stand-in; the data-integrity discipline —
verbatim + validate + calibrated confidence + flag-don't-guess — is the product.
