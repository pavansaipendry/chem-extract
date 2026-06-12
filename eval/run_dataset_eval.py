"""Batch eval against the synthetic handwritten-notes dataset.

  python eval/run_dataset_eval.py --limit 12 --runs 2
  python eval/run_dataset_eval.py --all --runs 1        # full sweep

Matches each ground-truth field to a predicted observation by raw_value, then
reports the three metrics the dataset README asks for:

  extraction accuracy   raw_value read correctly (among matched fields)
  flag recall           of ambiguous=true fields, % we flagged for review   <- #1
  unflagged accuracy    of fields we did NOT flag, % fully correct (value+unit)

Per-image pipeline results are cached under results/dataset/ so re-runs are free.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from anthropic import Anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemextract.pipeline import process_image  # noqa: E402
from chemextract.schema import QuantityType, RawObservation  # noqa: E402
from chemextract.validate import normalize_unit, validate_observation  # noqa: E402

DATA = ROOT / "dataset" / "chem_notes_dataset"
CACHE = ROOT / "results" / "dataset"

# dataset unit vocabulary -> the verbatim token a reader would see on the page
UNIT_ALIASES = {
    "degc": "c", "degree c": "c", "celsius": "c", "°c": "c",
    "degf": "f", "fahrenheit": "f",
    "kelvin": "k",
    "minute": "min", "minutes": "min", "sec": "sec", "second": "s", "seconds": "s",
    "hour": "h", "hours": "h",
    "milliliter": "ml", "millilitre": "ml", "liter": "l", "litre": "l",
    "gram": "g", "milligram": "mg", "microgram": "ug", "kilogram": "kg",
    "molar": "m", "millimolar": "mm",
    "percent": "%",
    "dimensionless": "", "none": "", "unitless": "",
}


def canon_unit(u: str | None) -> str:
    if not u:
        return ""
    n = normalize_unit(u)
    return UNIT_ALIASES.get(n, n)


def _revalidate(obs: dict) -> dict:
    """Re-run the (possibly updated) validation layer on a cached extraction,
    without re-calling the vision API. Lets us measure validation-layer fixes
    against frozen extractions. Inference sub-objects are left untouched."""
    try:
        qtype = QuantityType(obs["quantity_type"])
    except ValueError:
        qtype = QuantityType.OTHER
    raw = RawObservation(
        quantity_type=qtype, value=obs["value"], unit=obs["unit"],
        quote=obs.get("quote", ""), confidence=obs["confidence"],
    )
    flags = validate_observation(raw)
    # preserve consistency/confidence flags from the original run (not value-derived)
    for f in obs.get("flags", []):
        if f in ("self_consistency_mismatch", "low_confidence") and f not in flags:
            flags.append(f)
    obs = dict(obs)
    obs["flags"] = flags
    obs["needs_review"] = bool(flags)
    return obs


def load_predictions(image_id: str, image_path: Path, runs: int, client: Anthropic,
                     revalidate: bool = False) -> list[dict]:
    cache_file = CACHE / f"{image_id}.json"
    if cache_file.exists():
        observations = json.loads(cache_file.read_text())["observations"]
        return [_revalidate(o) for o in observations] if revalidate else observations
    doc = process_image(image_path, n_runs=runs, run_inference=True, client=client)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(doc.model_dump_json(indent=2))
    return json.loads(doc.model_dump_json())["observations"]


def match(gt_field: dict, preds: list[dict], used: set[int]) -> tuple[int, dict] | None:
    """Match a GT field to an unused prediction by raw_value."""
    for i, p in enumerate(preds):
        if i in used:
            continue
        if p["value"].strip() == gt_field["raw_value"].strip():
            return i, p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--revalidate", action="store_true",
                    help="re-score cached extractions through the current validation layer (no API calls)")
    args = ap.parse_args()

    manifest = json.loads((DATA / "manifest.json").read_text())
    if not args.all:
        manifest = manifest[: args.limit]

    client = Anthropic()

    # aggregate counters
    n_fields = n_matched = n_value_ok = 0
    n_ambig = n_ambig_flagged = 0
    n_unflagged = n_unflagged_ok = 0
    failures: list[str] = []

    for entry in manifest:
        image_id = Path(entry["image"]).stem
        label = json.loads((DATA / entry["label"]).read_text())
        try:
            preds = load_predictions(image_id, DATA / entry["image"], args.runs, client,
                                     revalidate=args.revalidate)
        except Exception as e:
            failures.append(f"{image_id}: PIPELINE ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        used: set[int] = set()
        for gt in label["fields"]:
            n_fields += 1
            is_ambig = gt["ambiguous"]
            if is_ambig:
                n_ambig += 1

            m = match(gt, preds, used)
            if m is None:
                failures.append(f"{image_id}: MISSED {gt['raw_value']} ({gt['type']}) '{gt['written_text']}'")
                # a missed ambiguous field is an unflagged error of the worst kind
                continue
            idx, p = m
            used.add(idx)
            n_matched += 1

            value_ok = p["value"].strip() == gt["raw_value"].strip()
            if value_ok:
                n_value_ok += 1

            unit_ok = canon_unit(p["unit"]) == canon_unit(gt["written_unit"])

            if is_ambig:
                if p["needs_review"]:
                    n_ambig_flagged += 1
                else:
                    failures.append(
                        f"{image_id}: UNFLAGGED AMBIGUITY {gt['raw_value']} '{gt['written_text']}' "
                        f"<-- silent guess, the failure that matters"
                    )

            if not p["needs_review"]:
                n_unflagged += 1
                if value_ok and unit_ok:
                    n_unflagged_ok += 1
                else:
                    failures.append(
                        f"{image_id}: UNFLAGGED WRONG {gt['raw_value']} "
                        f"got unit={p['unit']!r} expected written_unit={gt['written_unit']!r}"
                    )

    print(f"\nImages evaluated:     {len(manifest)}  (runs={args.runs})")
    print(f"GT fields:            {n_fields}")
    print(f"Detection rate:       {n_matched}/{n_fields} = {pct(n_matched, n_fields)}")
    print(f"Extraction accuracy:  {n_value_ok}/{n_matched} = {pct(n_value_ok, n_matched)}  (raw_value among matched)")
    print(f"Flag recall:          {n_ambig_flagged}/{n_ambig} = {pct(n_ambig_flagged, n_ambig)}   <- #1 metric")
    print(f"Unflagged accuracy:   {n_unflagged_ok}/{n_unflagged} = {pct(n_unflagged_ok, n_unflagged)}  (target >= 99%)")

    if failures:
        print(f"\n{len(failures)} issues (first 40):")
        for f in failures[:40]:
            print(f"  - {f}")
    return 0


def pct(a: int, b: int) -> str:
    return f"{a/b:.1%}" if b else "n/a"


if __name__ == "__main__":
    sys.exit(main())
