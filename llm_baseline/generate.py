"""Zero-shot / few-shot generation stage.

For each TEST row, build a natural-language performance spec, prompt the LLM
for a full 39-param JSON design, and persist every attempt to
outputs/generations_<model>.jsonl — independent of the scoring stage, which
reads that file back in and re-runs are cheap to resume (already-written
(condition, test_idx) pairs are skipped).

Decoding: JSON-schema-constrained generation via Outlines guarantees ~100%
syntactic parse rate (every key present, correct type) — it does NOT guarantee
physically sensible values, which is exactly the thing this baseline measures.
If constrained generation isn't available for a model, generation falls back
to plain sampling + lenient regex/json extraction, and parse failures are
counted rather than hidden.
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

from common import BOOL_COLS, INPUT_COLS, MATERIAL_COLS, MODEL_IDS, OUTPUT_DIR
from prompts import make_spec_for_test_row, select_examples, system_prompt
from split import load_full_df, load_or_build_split, train_test_frames

MAX_NEW_TOKENS = 512


def build_schema() -> str:
    props = {}
    for c in INPUT_COLS:
        if c in MATERIAL_COLS or c in BOOL_COLS:
            props[c] = {"type": "integer", "enum": [0, 1]}
        else:
            props[c] = {"type": "number"}
    schema = {
        "type": "object",
        "properties": props,
        "required": INPUT_COLS,
        "additionalProperties": False,
    }
    return json.dumps(schema)


def lenient_parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate_parsed(d: dict | None) -> bool:
    if not isinstance(d, dict):
        return False
    if set(d.keys()) != set(INPUT_COLS):
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in d.values())


def load_existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    keys.add((rec["condition"], rec["test_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--condition", default="both", choices=["zero-shot", "few-shot", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="cap on #test rows, for smoke tests")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--constrained", action="store_true", default=True)
    ap.add_argument("--no-constrained", dest="constrained", action="store_false")
    args = ap.parse_args()

    conditions = ["zero-shot", "few-shot"] if args.condition == "both" else [args.condition]

    df = load_full_df()
    split = load_or_build_split(df)
    train_df, test_df = train_test_frames(df, split)
    if args.limit:
        test_df = test_df.iloc[: args.limit]

    examples = select_examples(train_df)
    sys_prompts = {
        "zero-shot": system_prompt(None),
        "few-shot": system_prompt(examples),
    }

    model_id = MODEL_IDS[args.model]
    print(f"loading {model_id} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=args.device)

    gen = None
    if args.constrained:
        om = outlines.from_transformers(hf_model, tok)
        gen = outlines.Generator(om, outlines.json_schema(build_schema()))

    out_path = OUTPUT_DIR / f"generations_{args.model}.jsonl"
    done = load_existing_keys(out_path)
    print(f"{len(done)} (condition, test_idx) pairs already present in {out_path}, skipping those")

    n_written = 0
    with open(out_path, "a") as f:
        for condition in conditions:
            sys_prompt = sys_prompts[condition]
            for test_idx, row in test_df.iterrows():
                if (condition, int(test_idx)) in done:
                    continue
                spec = make_spec_for_test_row(row, test_idx, split["seed"])
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": spec["text"]},
                ]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                t0 = time.time()
                parsed, raw, err = None, None, None
                try:
                    if gen is not None:
                        raw = gen(prompt, max_new_tokens=args.max_new_tokens)
                        parsed = json.loads(raw)
                    else:
                        inputs = tok(prompt, return_tensors="pt").to(hf_model.device)
                        out_ids = hf_model.generate(
                            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                        raw = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        parsed = lenient_parse(raw)
                except Exception as e:  # noqa: BLE001 - persist the failure, keep going
                    err = f"{type(e).__name__}: {e}"
                gen_time_s = time.time() - t0
                parse_ok = validate_parsed(parsed)

                record = {
                    "test_idx": int(test_idx),
                    "split_seed": split["seed"],
                    "model": args.model,
                    "condition": condition,
                    "spec_text": spec["text"],
                    "constraints": spec["constraints"],
                    "raw_output": raw,
                    "parsed": parsed if parse_ok else None,
                    "parse_ok": parse_ok,
                    "error": err,
                    "gen_time_s": round(gen_time_s, 3),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                n_written += 1
                if n_written % 25 == 0:
                    print(f"[{args.model}/{condition}] {n_written} written, last test_idx={test_idx}, "
                          f"parse_ok={parse_ok}, {gen_time_s:.1f}s")

    print(f"done. wrote {n_written} new records to {out_path}")


if __name__ == "__main__":
    main()
