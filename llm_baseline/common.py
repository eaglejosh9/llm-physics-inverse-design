"""Shared constants for the no-fine-tuning LLM baseline.

INPUT_COLS / TARGET_COLS / SEED mirror surrogate_model.ipynb exactly so that
the train/test split and the surrogate's expected input order stay in sync
with the trained checkpoint in checkpoints/surrogate_mlp.pth.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "FRAMED Dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "llm_baseline" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = DATA_DIR / "all_structural_data_oh.csv"

SEED = 42  # must match surrogate_model.ipynb's SEED

INPUT_COLS = [
    "Material=Steel", "Material=Aluminum", "Material=Titanium",
    "SSB_Include", "CSB_Include",
    "CS Length", "BB Drop", "Stack", "SS E", "ST Angle", "BB OD", "TT OD",
    "HT OD", "DT OD", "CS OD", "SS OD", "ST OD", "CS F", "HT LX", "ST UX",
    "HT UX", "HT Angle", "HT Length", "ST Length", "BB Length",
    "Dropout Offset", "SSB OD", "CSB OD", "SSB Offset", "CSB Offset", "SS Z",
    "SS Thickness", "CS Thickness", "TT Thickness", "BB Thickness",
    "HT Thickness", "ST Thickness", "DT Thickness", "DT Length",
]
assert len(INPUT_COLS) == 39

TARGET_COLS = ["Model Mass", "Sim 1 Dropout Y Disp."]

MATERIAL_COLS = ["Material=Steel", "Material=Aluminum", "Material=Titanium"]
BOOL_COLS = ["SSB_Include", "CSB_Include"]
ONEHOT_BOOL_COLS = MATERIAL_COLS + BOOL_COLS
CONTINUOUS_COLS = [c for c in INPUT_COLS if c not in ONEHOT_BOOL_COLS]

MODEL_IDS = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

# Tolerances for constraint satisfaction, chosen from the empirical spread of
# each target (see EDA in surrogate_model.ipynb): mass std=2.18kg, IQR=3.07kg;
# disp std=0.05m, IQR=0.054m (heavy-tailed). A pure relative tolerance blows
# up near zero, so each is max(absolute floor, relative fraction of target).
MASS_ABS_TOL = 0.25   # kg
MASS_REL_TOL = 0.05
DISP_ABS_TOL = 0.01   # m
DISP_REL_TOL = 0.15
