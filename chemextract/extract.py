"""Vision LLM extraction. One call per page, schema-enforced via structured outputs.

The prompt is the safety-critical part: extract verbatim, never invent units.
Self-consistency: run N times and let the pipeline compare runs.
"""
import base64
import mimetypes
from pathlib import Path

from anthropic import Anthropic

from .schema import RawExtraction
from .symbols import LEGEND

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a chemistry document extraction system. You will be shown an \
image of a handwritten or printed chemistry document (lab notebook page, test report, \
experiment sheet).

Rules — these are absolute:
1. Extract VERBATIM. The `value`, `unit`, and `quote` fields must contain exactly what \
is written on the page, character for character.
2. NEVER invent a unit. If no unit is visible next to a value, set `unit` to null — \
even if the intended unit seems obvious from context. Someone else infers; you only read.
3. Do not normalize, convert, or correct anything. "70" stays "70", not "70.0" or "70°C".
4. `quote` is the exact source phrase the value appears in, so a human can find it.
5. `transcript` is your best full verbatim transcription of the page.
6. `confidence` reflects only how certain you are that you READ the characters correctly \
(legibility), not whether the value makes chemical sense.
7. Extract every physical quantity you can see: temperatures, volumes, masses, times, \
pH values, concentrations, pressures, yields. Classify each with the closest \
quantity_type; use "other" only when none fits.
8. Several quantities are unitless — pH, absorbance (A), pKa, equilibrium constants (K, \
Ka), refractive index. A number for these is complete without a unit (unit null is \
correct, not missing).

""" + LEGEND


def _image_block(image_path: str | Path) -> dict:
    path = Path(image_path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def extract_page(image_path: str | Path, n_runs: int = 2, client: Anthropic | None = None) -> list[RawExtraction]:
    """Run extraction n_runs times for self-consistency checking."""
    client = client or Anthropic()
    runs = []
    for _ in range(n_runs):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    _image_block(image_path),
                    {"type": "text", "text": "Extract all physical quantities from this document."},
                ],
            }],
            output_format=RawExtraction,
        )
        runs.append(response.parsed_output)
    return runs
