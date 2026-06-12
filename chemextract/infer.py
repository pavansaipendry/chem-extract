"""Unit inference pass for flagged fields.

Second LLM call with full-document context, grounded by the deterministic
plausibility table. Output is stored in the separate `inferred` sub-object —
it never overwrites the raw extraction.
"""
import json

from anthropic import Anthropic

from .extract import MODEL
from .schema import InferredUnit, Observation
from .validate import parse_value, plausible_units

INFER_SYSTEM = """You are the unit-inference component of a chemistry document extraction \
system. A value was extracted from a document WITHOUT a visible unit. Your job is to \
propose the most likely intended unit — clearly labeled as an inference, never presented \
as fact.

Rules:
1. Use the full document transcript: if other values of the same kind carry units, that \
is strong evidence.
2. Use the physical-plausibility analysis provided: units marked implausible would put \
the value outside sane bench-chemistry ranges (e.g. 70 K = -203 °C is not "heating").
3. Use chemical context (e.g. reflux in ethanol implies ~78 °C; 'heated' contradicts \
room temperature).
4. `reasoning` must be a human-readable justification a reviewer can evaluate.
5. If no unit is clearly more likely, return unit null with low confidence — an honest \
"can't tell" beats a confident guess.
6. Calibrate confidence: 0.9+ only when the document itself almost settles it (same \
unit used everywhere else); 0.6-0.8 for strong contextual evidence; below 0.5 when \
genuinely ambiguous."""


def infer_unit(observation: Observation, transcript: str, client: Anthropic | None = None) -> InferredUnit:
    client = client or Anthropic()

    num = parse_value(observation.value)
    plausibility = plausible_units(observation.quantity_type, num) if num is not None else {}

    prompt = f"""Full document transcript:
---
{transcript}
---

Flagged field:
- quantity type: {observation.quantity_type.value}
- value (verbatim): {observation.value}
- source quote: "{observation.quote}"
- unit: MISSING

Physical plausibility per candidate unit (deterministic range check):
{json.dumps(plausibility, indent=2)}

Propose the most likely intended unit, with confidence and reasoning."""

    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=INFER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=InferredUnit,
    )
    return response.parsed_output
