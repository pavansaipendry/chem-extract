"""End-to-end pipeline: image -> extraction (xN) -> self-consistency ->
validation -> inference for flagged fields -> DocumentExtraction.
"""
from pathlib import Path

from anthropic import Anthropic

from .extract import extract_page
from .infer import infer_unit
from .schema import DocumentExtraction, Observation, RawExtraction, RawObservation
from .validate import validate_observation

CONFIDENCE_THRESHOLD = 0.9


def _obs_key(o: RawObservation) -> tuple:
    """Identity of an observation for cross-run comparison."""
    return (o.quantity_type, o.value.strip(), (o.unit or "").strip().lower())


def _consistency_flags(primary: RawExtraction, others: list[RawExtraction]) -> set[tuple]:
    """Keys present in the primary run but not reproduced in every other run."""
    mismatched: set[tuple] = set()
    for other in others:
        other_keys = {_obs_key(o) for o in other.observations}
        for o in primary.observations:
            if _obs_key(o) not in other_keys:
                mismatched.add(_obs_key(o))
    return mismatched


def process_image(
    image_path: str | Path,
    n_runs: int = 2,
    run_inference: bool = True,
    run_reconstruction: bool = False,
    client: Anthropic | None = None,
) -> DocumentExtraction:
    client = client or Anthropic()
    runs = extract_page(image_path, n_runs=n_runs, client=client)
    primary, others = runs[0], runs[1:]
    mismatched = _consistency_flags(primary, others)

    observations: list[Observation] = []
    for raw in primary.observations:
        flags = validate_observation(raw)
        if _obs_key(raw) in mismatched:
            flags.append("self_consistency_mismatch")
        if raw.confidence < CONFIDENCE_THRESHOLD:
            flags.append("low_confidence")

        obs = Observation(
            quantity_type=raw.quantity_type,
            value=raw.value,
            unit=raw.unit,
            quote=raw.quote,
            confidence=raw.confidence,
            flags=flags,
            needs_review=bool(flags),
        )

        if run_inference and "missing_unit" in flags:
            obs.inferred = infer_unit(obs, primary.transcript, client=client)

        observations.append(obs)

    doc = DocumentExtraction(
        source_image=str(image_path),
        transcript=primary.transcript,
        observations=observations,
    )

    if run_reconstruction:
        from .reconstruct import reconstruct
        doc.reconstruction = reconstruct(doc, client=client)

    return doc
