"""Shared setup for the QLoRA fine-tuning stage.

Bootstraps sys.path so this stage can import llm_baseline's common.py,
prompts.py, split.py, surrogate.py, generate.py (for its parse helpers) and
score.py directly -- no copy-pasting, no re-implementing the split, the
prompt templates, the scoring tolerances, or the surrogate loader. Those
stay owned by llm_baseline; this stage only adds what's new (dataset
construction + QLoRA training + adapter-based generation).
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LLM_BASELINE_DIR = BASE_DIR / "llm_baseline"
sys.path.insert(0, str(LLM_BASELINE_DIR))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADAPTER_DIR = Path(__file__).resolve().parent / "adapter_model"

TRAIN_DATASET_PATH = OUTPUT_DIR / "sft_train.jsonl"
VAL_DATASET_PATH = OUTPUT_DIR / "sft_val.jsonl"
TRAIN_VAL_SPLIT_PATH = OUTPUT_DIR / "train_val_split.json"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "qlora"  # generations_qlora.jsonl / scored_qlora.jsonl, matching llm_baseline's naming convention

N_VARIANTS = 4          # spec variants generated per TRAIN row
VAL_ROW_FRACTION = 0.05  # held-out fraction of TRAIN rows (not test rows) for val loss
