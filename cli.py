"""POC CLI: process an image and view the extraction with flags.

  python cli.py process path/to/image.png          # extract + save JSON
  python cli.py show results/<name>.json           # re-display a saved result
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chemextract.pipeline import process_image
from chemextract.schema import DocumentExtraction

load_dotenv()  # pick up ANTHROPIC_API_KEY from a local .env (gitignored) if present

RESULTS_DIR = Path(__file__).parent / "results"
console = Console()


def render(doc: DocumentExtraction) -> None:
    console.print(f"\n[bold]Source:[/bold] {doc.source_image}")
    console.print(f"[bold]Transcript:[/bold] {doc.transcript}\n")

    table = Table(title="Extracted quantities")
    table.add_column("Type")
    table.add_column("Value (verbatim)")
    table.add_column("Unit")
    table.add_column("Conf")
    table.add_column("Flags")
    table.add_column("Inferred unit")
    table.add_column("Review?")

    for o in doc.observations:
        inferred = ""
        if o.inferred:
            inferred = f"{o.inferred.unit} ({o.inferred.confidence:.2f})"
        style = "yellow" if o.needs_review else "green"
        table.add_row(
            o.quantity_type.value,
            o.value,
            o.unit or "[red]∅[/red]",
            f"{o.confidence:.2f}",
            ", ".join(o.flags) or "—",
            inferred,
            "⚠ yes" if o.needs_review else "no",
            style=style,
        )
    console.print(table)

    for o in doc.observations:
        if o.inferred:
            console.print(
                f"[dim]Inference for '{o.quote}': {o.inferred.unit} "
                f"(conf {o.inferred.confidence:.2f}) — {o.inferred.reasoning}[/dim]"
            )

    # --- L3: hand-drawn structures ---
    if doc.structures:
        stable = Table(title="Hand-drawn structures")
        stable.add_column("Label")
        stable.add_column("Identified as")
        stable.add_column("Structure (SMILES)")
        stable.add_column("Confirms")
        stable.add_column("Flags")
        for s in doc.structures:
            style = "yellow" if s.needs_review else "green"
            stable.add_row(
                s.label or "—",
                s.name or "[red]unidentified[/red]",
                s.structure_smiles or "[red]∅[/red]",
                str(s.confirmations),
                ", ".join(s.flags) or "✓ trusted",
                style=style,
            )
        console.print(stable)

    # --- L4: experiment + deterministic calculation checks ---
    if doc.experiment:
        console.print(f"\n[bold]Experiment goal:[/bold] {doc.experiment.goal}")
    if doc.calc_checks:
        ctable = Table(title="Calculation verification (re-derived from inputs)")
        ctable.add_column("Relation")
        ctable.add_column("Formula")
        ctable.add_column("Re-derived")
        ctable.add_column("Page says")
        ctable.add_column("Verdict")
        for c in doc.calc_checks:
            ok = c.agree
            verdict = "✓ agrees" if ok else ("? " + ", ".join(c.flags) if ok is None else "✗ MISMATCH")
            ctable.add_row(
                c.relation,
                c.formula or "—",
                f"{c.recomputed:.4g}" if c.recomputed is not None else "—",
                f"{c.stated:.4g}" if c.stated is not None else "—",
                verdict,
                style="green" if ok else "yellow",
            )
        console.print(ctable)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_process = sub.add_parser("process")
    p_process.add_argument("image")
    p_process.add_argument("--runs", type=int, default=2)
    p_process.add_argument("--no-infer", action="store_true")

    p_show = sub.add_parser("show")
    p_show.add_argument("result_json")

    args = parser.parse_args()

    if args.cmd == "process":
        doc = process_image(args.image, n_runs=args.runs, run_inference=not args.no_infer)
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / (Path(args.image).stem + ".json")
        out.write_text(doc.model_dump_json(indent=2))
        render(doc)
        console.print(f"\n[bold green]Saved:[/bold green] {out}")
    elif args.cmd == "show":
        doc = DocumentExtraction.model_validate_json(Path(args.result_json).read_text())
        render(doc)


if __name__ == "__main__":
    sys.exit(main())
