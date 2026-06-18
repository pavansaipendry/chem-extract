"""End-to-end pipeline: image -> extraction (xN) -> self-consistency ->
validation -> inference for flagged fields -> DocumentExtraction.
"""
from pathlib import Path

from anthropic import Anthropic

from .experiment import analyze_experiment
from .extract import extract_page
from .infer import infer_unit
from .notation import notation_flags
from .schema import DocumentExtraction, Observation, RawExtraction, RawObservation
from .structures import analyze_structures
from .validate import validate_observation

CONFIDENCE_THRESHOLD = 0.9


def _obs_summary(observations: list[Observation]) -> str:
    return "\n".join(
        f"- {o.quantity_type.value}: '{o.value}' {o.unit or '(no unit)'} from \"{o.quote}\""
        for o in observations
    )


def _struct_summary(structures: list) -> str:
    return "\n".join(
        f"- {s.name or 'unidentified'}: {s.structure_smiles or '(no structure)'}"
        f" [{', '.join(s.flags) if s.flags else 'trusted'}]"
        for s in structures
    )


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
    run_structures: bool = True,
    run_experiment: bool = True,
    run_tables: bool = True,
    run_reconstruction: bool = False,
    run_layout: bool = False,
    client: Anthropic | None = None,
) -> DocumentExtraction:
    """End-to-end across all five levels:
      L1 text  -> extract_page (xN, self-consistency)
      L2 symbols -> validate + notation_flags (deterministic)
      L3 chemistry -> infer units; analyze_structures (recognise + cross-check)
      L4 experiment -> analyze_experiment (understand + verify the calculations)
      L5 layout -> refine_layout (place diagrams in 2-D, restore reaction connectors)
    """
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
        flags += notation_flags(raw.value, raw.quote)   # L2: dropped exponent etc.

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

    # L3: recognise + cross-check the hand-drawn structures.
    if run_structures:
        doc.structures = analyze_structures(image_path, transcript=primary.transcript, client=client)

    # L4: reconstruct the experiment and deterministically verify its calculations.
    # The page image grounds the reading (diagrams/schemes/layout), not just the text.
    if run_experiment:
        record, checks = analyze_experiment(
            primary.transcript,
            observations_summary=_obs_summary(observations),
            structures_summary=_struct_summary(doc.structures),
            image_path=image_path,
            client=client,
        )
        doc.experiment = record
        doc.calc_checks = checks

    # Genuine data tables on the page, read back into structured rows/columns.
    if run_tables:
        from .tables import extract_tables
        doc.tables = extract_tables(image_path, transcript=primary.transcript, client=client)

    if run_reconstruction:
        from .reconstruct import reconstruct
        doc.reconstruction = reconstruct(doc, client=client)

    # L5: place the diagrams and tables where they belong; restore reaction connectors.
    if run_layout:
        from .layout import refine_layout
        final_text = doc.reconstruction.canonical_text if doc.reconstruction else doc.transcript
        doc.layout = refine_layout(image_path, final_text, doc.structures, tables=doc.tables, client=client)

    return doc
