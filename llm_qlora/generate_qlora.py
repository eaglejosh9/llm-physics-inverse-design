"""Fine-tuned generation stage: same 405 TEST specs as the baseline, same
constrained-JSON decoding via Outlines, but with the base model + trained
LoRA adapter instead of prompted in-context examples.

Two conditions are available:
  "fine-tuned"          -- system prompt + spec, no in-context exemplars. The
                           default: the whole point of fine-tuning is that the
                           model shouldn't need them.
  "fine-tuned-fewshot"  -- the same 3 exemplars the baseline's few-shot arm
                           uses (prompts.select_examples, TRAIN rows only),
                           to test whether exemplars still help post-tuning.

Mirrors llm_baseline/generate.py's structure and output-file conventions
(independently re-runnable, resumes from existing records) so scoring and
reporting can reuse the baseline's code unmodified.
"""
import argparse
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import outlines

from common_ft import ADAPTER_DIR, MODEL_ID, MODEL_NAME
from common import OUTPUT_DIR
from generate import MAX_NEW_TOKENS, build_schema, lenient_parse, load_existing_keys, validate_parsed
from prompts import make_spec_for_test_row, select_examples, system_prompt
from split import load_full_df, load_or_build_split, train_test_frames

CONDITIONS = ["fine-tuned", "fine-tuned-fewshot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="fine-tuned", choices=CONDITIONS + ["both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="cap on #test rows, for smoke tests")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--constrained", action="store_true", default=True)
    ap.add_argument("--no-constrained", dest="constrained", action="store_false")
    args = ap.parse_args()

    conditions = CONDITIONS if args.condition == "both" else [args.condition]

    df = load_full_df()
    split = load_or_build_split(df)
    train_df, test_df = train_test_frames(df, split)
    if args.limit:
        test_df = test_df.iloc[: args.limit]

    sys_prompts = {
        # zero-shot-equivalent: fine-tuned model needs no in-context examples
        "fine-tuned": system_prompt(None),
        # same exemplars as llm_baseline's few-shot arm, so the two are comparable
        "fine-tuned-fewshot": system_prompt(select_examples(train_df)),
    }

    print(f"loading {MODEL_ID} (4-bit NF4) + adapter {ADAPTER_DIR} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, dtype=torch.bfloat16, device_map=args.device,
    )
    hf_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    hf_model.eval()

    gen = None
    if args.constrained:
        om = outlines.from_transformers(hf_model, tok)
        gen = outlines.Generator(om, outlines.json_schema(build_schema()))

    out_path = OUTPUT_DIR / f"generations_{MODEL_NAME}.jsonl"
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
                    "model": MODEL_NAME,
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
                    print(f"[{MODEL_NAME}/{condition}] {n_written} written, last test_idx={test_idx}, "
                          f"parse_ok={parse_ok}, {gen_time_s:.1f}s")

    print(f"done. wrote {n_written} new records to {out_path}")


if __name__ == "__main__":
    main()
