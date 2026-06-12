"""Field-level eval against hand-labeled ground truth.

  python eval/run_eval.py eval/ground_truth/worked_example.json results/worked_example.json

Metrics:
  detection rate       % of ground-truth fields the system found at all
  raw accuracy         among found fields, % where value AND unit match verbatim
  flag recall          % of fields that SHOULD be flagged that were flagged  <- the #1 metric
  unflagged accuracy   among fields the system did NOT flag, % fully correct
                       (an unflagged-wrong field is the only true failure mode)
"""
import json
import sys
from pathlib import Path


def norm_unit(u):
    return (u or "").strip().lower().rstrip(".")


def match(gt_obs, predictions):
    """Match a GT field to a prediction by value (and type when GT specifies one)."""
    candidates = [p for p in predictions if p["value"].strip() == gt_obs["value"]]
    if gt_obs.get("quantity_type"):
        typed = [p for p in candidates if p["quantity_type"] == gt_obs["quantity_type"]]
        if typed:
            candidates = typed
    return candidates[0] if candidates else None


def main(gt_path: str, result_path: str) -> int:
    gt = json.loads(Path(gt_path).read_text())
    result = json.loads(Path(result_path).read_text())
    preds = result["observations"]

    found = 0
    correct = 0
    should_flag = 0
    flagged_of_should = 0
    unflagged_total = 0
    unflagged_correct = 0
    failures = []

    for g in gt["observations"]:
        p = match(g, preds)
        if p is None:
            failures.append(f"MISSED: {g['value']} ({g.get('quantity_type')})")
            if g["expect_flag"]:
                should_flag += 1  # missed ambiguous field = unflagged error
            continue
        found += 1

        value_ok = p["value"].strip() == g["value"]
        unit_ok = norm_unit(p["unit"]) == norm_unit(g["unit"])
        if value_ok and unit_ok:
            correct += 1

        if g["expect_flag"]:
            should_flag += 1
            if p["needs_review"]:
                flagged_of_should += 1
            else:
                failures.append(f"UNFLAGGED AMBIGUITY: {g['value']} — should have been flagged")

        if not p["needs_review"]:
            unflagged_total += 1
            if value_ok and unit_ok:
                unflagged_correct += 1
            else:
                failures.append(
                    f"UNFLAGGED WRONG: value={p['value']} unit={p['unit']} "
                    f"expected unit={g['unit']}  <-- the failure mode that matters"
                )

    n = len(gt["observations"])
    print(f"Ground-truth fields:     {n}")
    print(f"Detection rate:          {found}/{n} = {found/n:.1%}")
    print(f"Raw accuracy (found):    {correct}/{found} = {correct/found:.1%}" if found else "Raw accuracy: n/a")
    if should_flag:
        print(f"Flag recall:             {flagged_of_should}/{should_flag} = {flagged_of_should/should_flag:.1%}   <- #1 metric")
    if unflagged_total:
        print(f"Unflagged accuracy:      {unflagged_correct}/{unflagged_total} = {unflagged_correct/unflagged_total:.1%}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
