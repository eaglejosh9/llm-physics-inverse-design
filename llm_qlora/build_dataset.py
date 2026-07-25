"""Build the QLoRA SFT dataset: for each TRAIN row, N_VARIANTS natural-language
specs (reusing prompts.make_spec_for_test_row's point/min/max/single-target
mixing) paired with that row's actual 39 params as the target completion.

TRAIN rows are further split into a train/val subset (held out from the
*training* rows only -- the baseline's 405-row test set is never touched, and
of course never trained on). Writes two flat JSONL files; each record has
just enough to reconstruct the exact Qwen chat messages at train time
(system prompt is regenerated from prompts.system_prompt(None) rather than
duplicated per row, so there's a single source of truth for its text).
"""
import json

from sklearn.model_selection import train_test_split

from common_ft import N_VARIANTS, TRAIN_DATASET_PATH, TRAIN_VAL_SPLIT_PATH, VAL_DATASET_PATH, VAL_ROW_FRACTION

from common import SEED
from prompts import make_spec_for_test_row, row_to_json_str
from split import load_full_df, load_or_build_split, train_test_frames

MAX_VARIANTS_PER_ROW = 10  # must stay small: composite index below is row_idx * MAX_VARIANTS_PER_ROW + variant


def build_records(df, row_indices) -> list[dict]:
    records = []
    for row_idx in row_indices:
        row = df.loc[row_idx]
        json_str = row_to_json_str(row)
        for variant in range(N_VARIANTS):
            composite_idx = row_idx * MAX_VARIANTS_PER_ROW + variant
            spec = make_spec_for_test_row(row, test_idx=composite_idx, seed=SEED)
            records.append({
                "row_idx": int(row_idx),
                "variant": variant,
                "spec_text": spec["text"],
                "constraints": spec["constraints"],
                "json_str": json_str,
            })
    return records


def write_jsonl(records: list[dict], path) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    assert N_VARIANTS <= MAX_VARIANTS_PER_ROW

    df = load_full_df()
    split = load_or_build_split(df)
    train_df, test_df = train_test_frames(df, split)

    train_rows, val_rows = train_test_split(
        sorted(train_df.index.tolist()), test_size=VAL_ROW_FRACTION, random_state=SEED,
    )
    train_rows, val_rows = set(train_rows), set(val_rows)
    assert train_rows.isdisjoint(set(test_df.index))
    assert val_rows.isdisjoint(set(test_df.index))
    assert train_rows.isdisjoint(val_rows)

    train_records = build_records(df, sorted(train_rows))
    val_records = build_records(df, sorted(val_rows))

    write_jsonl(train_records, TRAIN_DATASET_PATH)
    write_jsonl(val_records, VAL_DATASET_PATH)
    TRAIN_VAL_SPLIT_PATH.write_text(json.dumps({
        "seed": SEED,
        "source_split": "llm_baseline/outputs/split.json",
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_variants_per_row": N_VARIANTS,
        "train_row_idx": sorted(train_rows),
        "val_row_idx": sorted(val_rows),
    }))

    print(f"train rows={len(train_rows)} -> {len(train_records)} examples -> {TRAIN_DATASET_PATH}")
    print(f"val rows={len(val_rows)} -> {len(val_records)} examples -> {VAL_DATASET_PATH}")


if __name__ == "__main__":
    main()
