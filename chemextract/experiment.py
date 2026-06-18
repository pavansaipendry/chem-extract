"""Level 4: experiment understanding + DETERMINISTIC calculation checking.

Reading a page is necessary but not the point. The point is reconstructing what
the experiment *was* — goal, conditions, procedure, results — and, where the page
shows a calculation, independently RE-DERIVING it and checking the scientist's
arithmetic. A confident wrong number on a notebook page is exactly the failure
this whole system exists to catch, so the math gets verified, not transcribed.

Two halves, same split as the rest of the system:
  - understand_experiment(): an LLM reasoning pass over the transcript +
    extracted quantities + recognised structures -> a structured ExperimentRecord.
    This is interpretation; it is labelled as such.
  - verify_experiment(): pure functions, no LLM. Each calculation on the page is
    recomputed from its inputs via a known physical relation and compared to the
    stated result. Mismatches are FLAGGED, never silently corrected.

Worked on the real Page 57 (Li electrodeposition, Faraday's law):
    I = J·A      = 0.50 mA/cm² · 0.3 cm² = 1.5e-4 A
    Q = I·t      = 1.5e-4 A · 5400 s      = 0.81 C
    n = Q/(z·F)  = 0.81 C / (1 · 96485)   = 8.4e-6 mol
    m = n·M      = 8.4e-6 mol · 6.94 g/mol = 5.8e-5 g
Every step the page asserts, the verifier re-derives and confirms (or flags).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field

# MODEL is imported lazily inside understand_experiment() (not at module top) so
# this module doesn't pull in extract.py — lets schema.py import these models
# without an import cycle.

FARADAY = 96485.0  # C/mol, Faraday constant


# --- Structured experiment record (LLM output, labelled as interpretation) ----

class CalcInput(BaseModel):
    """One named input to a calculation, in the canonical unit the relation expects."""
    name: str = Field(description="canonical key, e.g. current_A, time_s, area_cm2, charge_C, electrons, moles, molar_mass_g_mol")
    value: float
    unit: str = Field(description="the unit this value is in (should match the relation's canonical unit)")


class Calculation(BaseModel):
    """A calculation written on the page, structured so it can be re-derived."""
    relation: str = Field(description="one of: current_from_density, faraday_charge, faraday_moles, mass_from_moles")
    description: str = Field(description="what is being computed, plain language")
    inputs: list[CalcInput]
    stated_result: Optional[float] = Field(description="the result the page asserts, in canonical units, or null")
    result_unit: Optional[str] = None
    quote: str = Field(description="the source text of the calculation, verbatim")


class ProcedureStep(BaseModel):
    order: int
    action: str = Field(description="one step of the procedure, as performed")
    conditions: list[str] = Field(default=[], description="conditions attached to this step (temp, time, atmosphere)")


class ExperimentRecord(BaseModel):
    goal: Optional[str] = Field(description="the stated objective of the experiment")
    conditions: list[str] = Field(default=[], description="setup/conditions: electrodes, electrolyte, atmosphere, temperature target")
    procedure: list[ProcedureStep] = Field(default=[])
    results: list[str] = Field(default=[], description="observed outcomes/measurements")
    calculations: list[Calculation] = Field(default=[])
    notes: Optional[str] = None


# --- Deterministic calculation checker (no LLM) ------------------------------

def _current_from_density(i: dict[str, float]) -> float:
    return i["current_density_A_cm2"] * i["area_cm2"]          # I = J·A


def _faraday_charge(i: dict[str, float]) -> float:
    return i["current_A"] * i["time_s"]                        # Q = I·t


def _faraday_moles(i: dict[str, float]) -> float:
    return i["charge_C"] / (i["electrons"] * FARADAY)          # n = Q/(z·F)


def _mass_from_moles(i: dict[str, float]) -> float:
    return i["moles"] * i["molar_mass_g_mol"]                  # m = n·M


# relation key -> (function, human formula, required input names)
RELATIONS: dict[str, tuple] = {
    "current_from_density": (_current_from_density, "I = J · A", ("current_density_A_cm2", "area_cm2")),
    "faraday_charge": (_faraday_charge, "Q = I · t", ("current_A", "time_s")),
    "faraday_moles": (_faraday_moles, "n = Q / (z·F)", ("charge_C", "electrons")),
    "mass_from_moles": (_mass_from_moles, "m = n · M", ("moles", "molar_mass_g_mol")),
}


class CalcCheck(BaseModel):
    relation: str
    formula: Optional[str] = None
    recomputed: Optional[float] = None
    stated: Optional[float] = None
    rel_error: Optional[float] = None
    agree: Optional[bool] = None
    flags: list[str] = []
    note: str = ""


def verify_calculation(calc: Calculation, rel_tol: float = 0.03) -> CalcCheck:
    """Re-derive a calculation from its inputs and check it against the stated
    result. Pure arithmetic — this is the trustworthy half. A mismatch is flagged
    (the scientist may have slipped a unit or dropped the electron count), never
    rewritten."""
    spec = RELATIONS.get(calc.relation)
    if spec is None:
        return CalcCheck(relation=calc.relation, flags=["unknown_relation"],
                         note=f"No known physical relation '{calc.relation}' to check against.")
    fn, formula, required = spec
    inputs = {ci.name: ci.value for ci in calc.inputs}

    missing = [k for k in required if k not in inputs]
    if missing:
        return CalcCheck(relation=calc.relation, formula=formula, flags=["missing_inputs"],
                         note=f"Cannot check {formula}: missing input(s) {missing}.")

    try:
        recomputed = fn(inputs)
    except ZeroDivisionError:
        return CalcCheck(relation=calc.relation, formula=formula, flags=["division_by_zero"],
                         note=f"{formula} divides by zero with the given inputs.")

    if calc.stated_result is None:
        return CalcCheck(relation=calc.relation, formula=formula, recomputed=recomputed,
                         flags=["no_stated_result"],
                         note=f"{formula} → {recomputed:.4g} (page stated no result to check).")

    denom = abs(calc.stated_result) or 1.0
    rel_error = abs(recomputed - calc.stated_result) / denom
    agree = rel_error <= rel_tol
    flags = [] if agree else ["calculation_mismatch"]
    note = (f"{formula} → {recomputed:.4g} vs stated {calc.stated_result:.4g} "
            f"({'agrees' if agree else f'OFF by {rel_error:.1%} — check the page'}).")
    return CalcCheck(relation=calc.relation, formula=formula, recomputed=recomputed,
                     stated=calc.stated_result, rel_error=rel_error, agree=agree,
                     flags=flags, note=note)


def verify_experiment(record: ExperimentRecord, rel_tol: float = 0.03) -> list[CalcCheck]:
    """Check every calculation in the record. Empty list if the page had none."""
    return [verify_calculation(c, rel_tol=rel_tol) for c in record.calculations]


# --- LLM reasoning pass (interpretation, labelled as such) -------------------

UNDERSTAND_SYSTEM = """You are the experiment-understanding component of a chemistry \
document extraction system. Given a verbatim transcript of a lab-notebook page plus \
the quantities and structures already extracted from it (and, when available, the \
page image itself), reconstruct what the experiment was.

Produce: the goal; the setup/conditions; the procedure as ordered steps; the results; \
and any calculations written on the page, structured so they can be re-derived.

Rules:
1. Use ONLY what the page supports. Do not invent steps, results, or numbers. If \
something is unclear, omit it rather than guess.
2. For each calculation on the page, fill `relation` with the matching key \
(current_from_density: I=J·A; faraday_charge: Q=I·t; faraday_moles: n=Q/(z·F); \
mass_from_moles: m=n·M), give every input in the CANONICAL unit named by the key \
(amperes, seconds, coulombs, square centimetres, moles, g/mol; electrons = z, the \
number of electrons transferred), and record the result the page asserts in \
`stated_result`. A deterministic checker re-derives it — your job is to read the \
numbers faithfully, not to compute.
3. Quote the source text for each calculation verbatim.
4. Convert units into the canonical ones for the inputs (e.g. 0.50 mA/cm² → 5e-4 \
A/cm²; 90 min → 5400 s), but never alter the stated result — record it as written.
5. When the page image is provided, use it to read diagrams, reaction schemes and \
the spatial layout the transcript can't convey — e.g. an electrochemical reduction \
drawn as a scheme, or which conditions belong to which step."""


def understand_experiment(
    transcript: str,
    observations_summary: str = "",
    structures_summary: str = "",
    image_path: str | Path | None = None,
    client: Anthropic | None = None,
) -> ExperimentRecord:
    """LLM pass: transcript (+ context, + optional page image) -> structured
    ExperimentRecord. The calculations it returns are then checked deterministically
    by verify_experiment. Passing image_path grounds the reading in what the page
    actually shows (diagrams, schemes, layout), not just the flattened transcript."""
    from .extract import MODEL, _image_block   # lazy: avoids a schema<->extract import cycle
    client = client or Anthropic()
    prompt = f"""Verbatim transcript of the page:
---
{transcript}
---
"""
    if observations_summary:
        prompt += f"\nQuantities already extracted:\n{observations_summary}\n"
    if structures_summary:
        prompt += f"\nStructures already recognised:\n{structures_summary}\n"
    prompt += "\nReconstruct the experiment. Structure every calculation so it can be re-derived."

    content: list[dict] = []
    if image_path is not None:
        content.append(_image_block(image_path))
    content.append({"type": "text", "text": prompt})

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=UNDERSTAND_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_format=ExperimentRecord,
    )
    return response.parsed_output


def analyze_experiment(
    transcript: str,
    observations_summary: str = "",
    structures_summary: str = "",
    image_path: str | Path | None = None,
    rel_tol: float = 0.03,
    client: Anthropic | None = None,
) -> tuple[ExperimentRecord, list[CalcCheck]]:
    """Understand the experiment, then verify its arithmetic. Returns both the
    interpreted record and the deterministic calculation checks."""
    record = understand_experiment(
        transcript, observations_summary, structures_summary, image_path=image_path, client=client
    )
    return record, verify_experiment(record, rel_tol=rel_tol)
