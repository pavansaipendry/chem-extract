"""POC CLI: process an image and view the extraction with flags.

  python cli.py process path/to/image.png          # extract + save JSON
  python cli.py show results/<name>.json           # re-display a saved result
"""
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chemextract.pipeline import process_image
from chemextract.schema import DocumentExtraction

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
