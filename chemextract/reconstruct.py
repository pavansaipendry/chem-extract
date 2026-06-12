"""Reconstruction pass: rebuild the most accurate, fully-specified version of the
document, using chemistry first principles to resolve ambiguity.

Two outputs:
  - canonical_text: the best single rewrite, with inferred insertions marked.
  - ambiguities: spans chemistry could NOT settle — offered as 2+ competing
    versions for a human to choose, never silently picked.

The principle is unchanged from the rest of the system: where domain knowledge
makes a reading near-certain, fill it in and STATE THE FACT used; where it is
genuinely ambiguous, surface multiple valid versions and flag for human choice.
"""
from anthropic import Anthropic

from .extract import MODEL
from .schema import DocumentExtraction, Observation, Reconstruction

# Compact grounding facts (1 atm). The model knows more chemistry than this, but
# pinning the common ones makes the reasoning consistent and auditable.
REFERENCE_FACTS = """Reference physical facts at 1 atm (use to ground unit inference,
and cite the specific fact in your reasoning):
- Water boils at 100 °C, freezes at 0 °C.
- Boiling points: ethanol 78, methanol 65, acetone 56, diethyl ether 35,
  toluene 111, water 100, DMSO 189, chloroform 61 (all °C).
- "Reflux" means heating at the solvent's boiling point.
- Room temperature ≈ 20-25 °C; ice bath ≈ 0 °C; dry ice/acetone ≈ -78 °C;
  liquid nitrogen ≈ -196 °C; body temp / incubator ≈ 37 °C.
- pH: 7 neutral, 0-1 strong acid, 13-14 strong base (unitless, 0-14).
- A temperature in K for a bench reaction is almost always implausible
  (e.g. 70 K = -203 °C); °F rarely appears in lab notes outside the US.
- 100 °F ≈ 38 °C (not boiling); 100 K ≈ -173 °C (cryogenic)."""

SYSTEM = """You are the reconstruction component of a chemistry document extraction
system. You are given a verbatim transcript of a (possibly messy, handwritten)
document plus the quantities already extracted from it, some missing units.

Your job: produce the most accurate possible fully-specified version of the text.

RULES — follow exactly:
1. Resolve a missing or ambiguous unit ONLY when chemistry makes it near-certain.
   When you do, you MUST cite the physical fact you used. Example: "boiling water
   at 100" -> 100 °C, because water boils at 100 °C at 1 atm; 100 °F (38 °C) is not
   boiling and 100 K is cryogenic.
2. In canonical_text, mark every value you inserted or corrected with square
   brackets so a human can see what you changed, e.g. "boiling water at 100[°C]".
   Leave text that was already complete untouched.
3. When a span is GENUINELY ambiguous and chemistry does NOT settle it (e.g. a
   number that could be 5 mL or 5 g of a reagent, or a digit you cannot read with
   certainty), DO NOT pick one. Add it to `ambiguities` with 2+ interpretations,
   each a complete valid rewrite of that span, ranked most-likely first, each with
   chemistry-grounded reasoning. Set resolved=false. Each interpretation must be a
   genuinely DIFFERENT reading — never the same reading twice with different
   wording or formatting. Mark inserted units in interpretation text with square
   brackets exactly as in canonical_text (e.g. "added 5[g] of NaCl").
4. If one reading clearly dominates on chemistry grounds, you may still record it
   in ambiguities with resolved=true and the winning interpretation first — but
   only when you are confident.
5. NEVER invent quantities that are not in the document. NEVER present a guess as
   fact. Honest "two possible versions" beats a confident wrong rewrite.
6. canonical_text confidence reflects the whole rewrite; lower it if unresolved
   ambiguities remain."""


def _obs_summary(observations: list[Observation]) -> str:
    lines = []
    for o in observations:
        unit = o.unit if o.unit else "MISSING"
        inf = ""
        if o.inferred and o.inferred.unit:
            inf = f"  (deterministic inference: {o.inferred.unit} @ {o.inferred.confidence:.2f})"
        flags = f"  flags={o.flags}" if o.flags else ""
        lines.append(f"- {o.quantity_type.value}: '{o.value}' unit={unit} from \"{o.quote}\"{flags}{inf}")
    return "\n".join(lines)


def reconstruct(doc: DocumentExtraction, client: Anthropic | None = None) -> Reconstruction:
    client = client or Anthropic()
    prompt = f"""{REFERENCE_FACTS}

Verbatim transcript of the document:
---
{doc.transcript}
---

Quantities already extracted:
{_obs_summary(doc.observations)}

Produce the corrected, fully-specified reconstruction. Fill in units that
chemistry makes near-certain (citing the fact); for anything genuinely ambiguous,
give multiple valid versions for a human to choose."""

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=Reconstruction,
    )
    return response.parsed_output
