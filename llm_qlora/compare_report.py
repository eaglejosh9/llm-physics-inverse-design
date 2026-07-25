"""Comparison report: baseline (report.csv, already computed) vs QLoRA
(scored_qlora.jsonl, scored here the same way report.py does it) -- plus the
breakdowns the pooled numbers hide:
  - satisfaction by constraint type (point / min / max)
  - MAE already split by target (report.summarize does this)
  - satisfaction in the target distribution's tails vs its middle, since the
    baseline was found to still fail on very light/heavy or very
    stiff/compliant specs even where it succeeds in the middle.
"""
import numpy as np
import pandas as pd

from common_ft import MODEL_NAME as QLORA_MODEL_NAME

from common import OUTPUT_DIR
from report import load_scored, summarize
from split import load_full_df

TAIL_PCTILE = 10  # below p10 / above p90 counts as a tail


def satisfaction_by_type(records: list[dict]) -> pd.DataFrame:
    """Per-constraint (not per-record) satisfaction rate, split out by
    constraint type, since a single record can mix e.g. a point-mass
    constraint with a max-displacement constraint."""
    rows = []
    for target in ["Model Mass", "Sim 1 Dropout Y Disp."]:
        for ctype in ["point", "min", "max"]:
            total, satisfied = 0, 0
            for r in records:
                if not r["parse_ok"]:
                    continue
                for c in r["per_constraint"]:
                    if c["target"] == target and c["type"] == ctype:
                        total += 1
                        satisfied += int(c["satisfied"])
            rows.append({
                "target": target, "constraint_type": ctype, "n": total,
                "satisfaction_rate": round(satisfied / total, 4) if total else None,
            })
    return pd.DataFrame(rows)


def tail_vs_middle(records: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    """Bucket each individual constraint by where its requested value falls
    in the FULL dataset's distribution for that target (low tail / middle /
    high tail), report satisfaction rate per bucket."""
    thresholds = {
        "Model Mass": (
            np.percentile(df["Model Mass"], TAIL_PCTILE),
            np.percentile(df["Model Mass"], 100 - TAIL_PCTILE),
        ),
        "Sim 1 Dropout Y Disp.": (
            np.percentile(df["Sim 1 Dropout Y Disp."].abs(), TAIL_PCTILE),
            np.percentile(df["Sim 1 Dropout Y Disp."].abs(), 100 - TAIL_PCTILE),
        ),
    }

    buckets = {}  # (target, bucket) -> [n, satisfied]
    for r in records:
        if not r["parse_ok"]:
            continue
        for c in r["per_constraint"]:
            lo, hi = thresholds[c["target"]]
            v = c["requested"]
            bucket = "low tail" if v < lo else ("high tail" if v > hi else "middle")
            key = (c["target"], bucket)
            n, sat = buckets.get(key, (0, 0))
            buckets[key] = (n + 1, sat + int(c["satisfied"]))

    rows = []
    for target in ["Model Mass", "Sim 1 Dropout Y Disp."]:
        for bucket in ["low tail", "middle", "high tail"]:
            n, sat = buckets.get((target, bucket), (0, 0))
            rows.append({
                "target": target, "bucket": bucket, "n": n,
                "satisfaction_rate": round(sat / n, 4) if n else None,
            })
    return pd.DataFrame(rows)


def main():
    baseline_csv = OUTPUT_DIR / "report.csv"
    baseline_df = pd.read_csv(baseline_csv) if baseline_csv.exists() else pd.DataFrame()

    qlora_records = load_scored(QLORA_MODEL_NAME)
    if not qlora_records:
        print(f"no scored_{QLORA_MODEL_NAME}.jsonl found -- run generate_qlora.py then "
              f"llm_baseline/score.py --model {QLORA_MODEL_NAME} first")
        return

    qlora_conditions = [c for c in ["fine-tuned", "fine-tuned-fewshot"]
                        if any(r["condition"] == c for r in qlora_records)]
    qlora_rows = pd.DataFrame([
        {"model": "qwen2.5-7b-qlora", "condition": c,
         **summarize([r for r in qlora_records if r["condition"] == c])}
        for c in qlora_conditions
    ])

    print("=" * 100)
    print("HEADLINE COMPARISON")
    print("=" * 100)
    full_table = pd.concat([baseline_df, qlora_rows], ignore_index=True)
    print(full_table.to_string(index=False))
    full_table.to_csv(OUTPUT_DIR / "report_with_qlora.csv", index=False)

    print()
    print("=" * 100)
    print("QLoRA: satisfaction by constraint type (point / min / max), per target")
    print("=" * 100)
    for c in qlora_conditions:
        print(f"\n-- {c} --")
        print(satisfaction_by_type([r for r in qlora_records if r["condition"] == c]).to_string(index=False))

    print()
    print("=" * 100)
    print(f"Tails (<p{TAIL_PCTILE} or >p{100 - TAIL_PCTILE} of the full dataset) vs middle")
    print("=" * 100)
    df = load_full_df()
    for model_name in ["qwen2.5-7b-instruct", QLORA_MODEL_NAME]:
        records = load_scored(model_name)
        if not records:
            continue
        for condition in sorted({r["condition"] for r in records}):
            cond_records = [r for r in records if r["condition"] == condition]
            print(f"\n-- {model_name} / {condition} --")
            print(tail_vs_middle(cond_records, df).to_string(index=False))

    print(f"\nwrote {OUTPUT_DIR / 'report_with_qlora.csv'}")


if __name__ == "__main__":
    main()
