"""QLoRA SFT stage: fine-tune Qwen2.5-7B-Instruct via token-decoding
cross-entropy on (spec -> design JSON) pairs built by build_dataset.py.

No regression head, no surrogate in the loss -- the model just learns to
emit the same JSON text format the baseline was prompted for, with loss
masked to the assistant completion only. The surrogate is reused later,
purely for evaluation (score.py), exactly as in the baseline.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainerCallback, TrainingArguments,
)

from common_ft import (
    ADAPTER_DIR, MODEL_ID, TRAIN_DATASET_PATH, VAL_DATASET_PATH,
)
from generate import lenient_parse, validate_parsed
from prompts import system_prompt

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
MAX_SEQ_LEN = 1536  # observed prompt+completion is ~1250 tokens; headroom, not a truncation target
N_PARSE_CHECK_SAMPLES = 8
PARSE_CHECK_MAX_NEW_TOKENS = 460


class SFTDataset(torch.utils.data.Dataset):
    """Tokenizes prompt+completion, masking every prompt token to -100.

    The prompt string (system+user, chat-templated with add_generation_prompt)
    is verified to be an exact token prefix of the full prompt+completion
    string before masking -- if a tokenizer boundary ever merges across the
    "...assistant\\n" / "{...}" split this raises loudly instead of silently
    training on a mis-masked example.
    """

    def __init__(self, jsonl_path, tokenizer, system_prompt_text):
        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.system_prompt_text = system_prompt_text

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        messages = [
            {"role": "system", "content": self.system_prompt_text},
            {"role": "user", "content": rec["spec_text"]},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion = rec["json_str"] + self.tokenizer.eos_token
        full = prompt + completion

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prompt_ids)] == prompt_ids, (
            f"tokenizer boundary mismatch at record {i} (row_idx={rec['row_idx']}, variant={rec['variant']})"
        )
        if len(full_ids) > MAX_SEQ_LEN:
            full_ids = full_ids[:MAX_SEQ_LEN]

        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)
            attn.append(b["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class ParseRateCallback(TrainerCallback):
    """Every eval, greedy-decode a few held-out val specs with the
    in-training model and report the JSON parse-success rate. This should
    hit ~100% within the first handful of evals; if it doesn't, the
    completion format or the prompt-masking is off."""

    def __init__(self, tokenizer, val_records, system_prompt_text, n_samples=N_PARSE_CHECK_SAMPLES):
        self.tokenizer = tokenizer
        self.samples = val_records[:n_samples]
        self.system_prompt_text = system_prompt_text

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return  # avoid redundant generation + duplicate log lines on every rank under DDP

        was_training = model.training
        was_caching = model.config.use_cache
        model.eval()
        model.config.use_cache = True

        n_ok = 0
        for rec in self.samples:
            messages = [
                {"role": "system", "content": self.system_prompt_text},
                {"role": "user", "content": rec["spec_text"]},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=PARSE_CHECK_MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            raw = self.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if validate_parsed(lenient_parse(raw)):
                n_ok += 1

        rate = n_ok / len(self.samples)
        model.config.use_cache = was_caching
        model.train(was_training)
        print(f"[parse-rate check] step={state.global_step} rate={rate:.2f} ({n_ok}/{len(self.samples)})")
        if metrics is not None:
            metrics["val_json_parse_rate"] = rate


def build_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Rank-aware: under torchrun each process must load its shard of the quantized
    # model onto its own GPU (LOCAL_RANK), not all processes piling onto GPU 0.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, dtype=torch.bfloat16, device_map={"": local_rank},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)  # effective batch 16
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs, for smoke tests")
    ap.add_argument("--limit-train", type=int, default=None, help="cap on #train examples, for smoke tests")
    ap.add_argument("--limit-val", type=int, default=None, help="cap on #val examples, for smoke tests")
    ap.add_argument("--adapter-dir", default=None, help="override save location, for smoke tests")
    args = ap.parse_args()

    model, tokenizer = build_model_and_tokenizer(MODEL_ID)
    sys_prompt_text = system_prompt(None)  # same zero-shot-style prompt used at eval time

    train_ds = SFTDataset(TRAIN_DATASET_PATH, tokenizer, sys_prompt_text)
    val_ds = SFTDataset(VAL_DATASET_PATH, tokenizer, sys_prompt_text)
    if args.limit_train:
        train_ds.records = train_ds.records[: args.limit_train]
    if args.limit_val:
        val_ds.records = val_ds.records[: args.limit_val]
    with open(VAL_DATASET_PATH) as f:
        val_records = [json.loads(line) for line in f][: args.limit_val or None]

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR.parent / "checkpoints_tmp"),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        label_names=["labels"],
        ddp_find_unused_parameters=False,  # every LoRA param is used on every forward pass
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[ParseRateCallback(tokenizer, val_records, sys_prompt_text)],
    )

    trainer.train()

    if trainer.is_world_process_zero():
        adapter_dir = Path(args.adapter_dir) if args.adapter_dir else ADAPTER_DIR
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"saved adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()
