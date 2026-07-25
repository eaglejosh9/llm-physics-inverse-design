"""Scoring stage: read generations_<model>.jsonl, run each parsed design
through the frozen surrogate, and write scored_<model>.jsonl with achieved
performance + per-constraint / overall satisfaction.

Kept separate from generate.py so the (slow, GPU-bound) LLM stage and the
(fast, CPU-only) surrogate stage can be re-run independently -- e.g. to
retune tolerances without regenerating.
"""
import argparse
import json

import numpy as np

from common import (
    DISP_ABS_TOL, DISP_REL_TOL, INPUT_COLS, MASS_ABS_TOL, MASS_REL_TOL,
    MATERIAL_COLS, OUTPUT_DIR,
)
from surrogate import Surrogate


def tol_for(target: str, value: float) -> float:
    if target == "Model Mass":
        return max(MASS_ABS_TOL, MASS_REL_TOL * abs(value))
    return max(DISP_ABS_TOL, DISP_REL_TOL * abs(value))


def achieved_for(target: str, mass: float, disp: float) -> float:
    # Specs phrase displacement in magnitude terms ("deflection of X m"),
    # since the sign only encodes FEA load direction, not stiffness.
    return mass if target == "Model Mass" else abs(disp)


def constraint_satisfied(target: str, ctype: str, value: float, achieved: float) -> bool:
    tol = tol_for(target, value)
    if ctype == "point":
        return abs(achieved - value) <= tol
    if ctype == "max":
        return achieved <= value + tol
    if ctype == "min":
        return achieved >= value - tol
    raise ValueError(ctype)


def onehot_validity(parsed: dict) -> bool:
    material_sum = sum(parsed[c] for c in MATERIAL_COLS)
    bools_ok = all(parsed[c] in (0, 1) for c in ["SSB_Include", "CSB_Include"])
    return material_sum == 1 and bools_ok


def score_record(rec: dict, surrogate: Surrogate) -> dict:
    out = dict(rec)  # carries test_idx, split_seed, model, condition, spec_text, constraints, parse_ok
    if not rec["parse_ok"]:
        out["mass_pred"] = None
        out["disp_pred"] = None
        out["onehot_valid"] = None
        out["per_constraint"] = []
        out["all_satisfied"] = False
        return out

    parsed = rec["parsed"]
    mass_pred, disp_pred = surrogate.predict_one({c: parsed[c] for c in INPUT_COLS})
    out["mass_pred"] = mass_pred
    out["disp_pred"] = disp_pred
    out["onehot_valid"] = onehot_validity(parsed)

    per_constraint = []
    for c in rec["constraints"]:
        achieved = achieved_for(c["target"], mass_pred, disp_pred)
        sat = constraint_satisfied(c["target"], c["type"], c["value"], achieved)
        per_constraint.append({
            "target": c["target"], "type": c["type"], "requested": c["value"],
            "achieved": achieved, "abs_err": abs(achieved - c["value"]), "satisfied": sat,
        })
    out["per_constraint"] = per_constraint
    out["all_satisfied"] = all(c["satisfied"] for c in per_constraint)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    in_path = OUTPUT_DIR / f"generations_{args.model}.jsonl"
    out_path = OUTPUT_DIR / f"scored_{args.model}.jsonl"

    surrogate = Surrogate()
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            scored = score_record(rec, surrogate)
            fout.write(json.dumps(scored) + "\n")
            n += 1
    print(f"scored {n} records -> {out_path}")


if __name__ == "__main__":
    main()
