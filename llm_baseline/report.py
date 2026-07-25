"""Aggregate scored_<model>.jsonl files into the headline comparison table:
zero-shot vs few-shot vs (if run) a second model, side by side.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import MODEL_IDS, OUTPUT_DIR


def load_scored(model: str) -> list[dict]:
    path = OUTPUT_DIR / f"scored_{model}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    parse_ok = [r for r in records if r["parse_ok"]]
    n_parse_fail = n - len(parse_ok)
    n_satisfied = sum(1 for r in records if r["all_satisfied"])

    mass_errs, disp_errs = [], []
    for r in parse_ok:
        for c in r["per_constraint"]:
            if c["target"] == "Model Mass":
                mass_errs.append(c["abs_err"])
            else:
                disp_errs.append(c["abs_err"])

    onehot_valid = [r["onehot_valid"] for r in parse_ok if r["onehot_valid"] is not None]

    return {
        "n": n,
        "parse_failure_rate": round(n_parse_fail / n, 4),
        "constraint_satisfaction_rate": round(n_satisfied / n, 4),
        "mass_spec_MAE_kg": round(sum(mass_errs) / len(mass_errs), 4) if mass_errs else None,
        "disp_spec_MAE_m": round(sum(disp_errs) / len(disp_errs), 5) if disp_errs else None,
        "onehot_validity_rate": round(sum(onehot_valid) / len(onehot_valid), 4) if onehot_valid else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    args = ap.parse_args()

    rows = []
    for model in args.models:
        records = load_scored(model)
        if not records:
            continue
        for condition in ["zero-shot", "few-shot"]:
            cond_records = [r for r in records if r["condition"] == condition]
            if not cond_records:
                continue
            summary = summarize(cond_records)
            rows.append({"model": model, "condition": condition, **summary})

    if not rows:
        print("no scored records found -- run generate.py then score.py first")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_csv = OUTPUT_DIR / "report.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
