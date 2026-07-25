"""90/10 train/test split, identical to surrogate_model.ipynb.

sklearn's train_test_split derives its shuffle purely from (n_samples,
random_state) — the array's values never enter into it — so splitting
np.arange(n) reproduces the exact row indices used to train the surrogate,
without needing to re-run that notebook.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_CSV, INPUT_COLS, OUTPUT_DIR, SEED, TARGET_COLS

SPLIT_PATH = OUTPUT_DIR / "split.json"


def load_full_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV).drop(columns=["Unnamed: 0", "batch"])
    assert df.isna().sum().sum() == 0
    return df


def build_split(df: pd.DataFrame) -> dict:
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.10, random_state=SEED)
    return {
        "seed": SEED,
        "n_rows": len(df),
        "train_idx": sorted(int(i) for i in train_idx),
        "test_idx": sorted(int(i) for i in test_idx),
    }


def load_or_build_split(df: pd.DataFrame) -> dict:
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text())
        assert split["seed"] == SEED and split["n_rows"] == len(df), "stale split.json"
        return split
    split = build_split(df)
    SPLIT_PATH.write_text(json.dumps(split))
    return split


def train_test_frames(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.loc[split["train_idx"]], df.loc[split["test_idx"]]


if __name__ == "__main__":
    df = load_full_df()
    split = load_or_build_split(df)
    train_df, test_df = train_test_frames(df, split)
    print(f"seed={split['seed']}  train={len(train_df)}  test={len(test_df)}")
    print(f"wrote {SPLIT_PATH}")
