# LLM Physics — Bicycle Frame Inverse Design

**Can a language model design a bicycle frame that actually hits a target weight
and stiffness?**

The idea in one paragraph: you give the model a request in plain English — *"design
a bicycle frame weighing under 4.70 kg, with dropout deflection of at least 0.004 m"*
— and it replies with a complete frame described as 39 numbers (tube diameters, wall
thicknesses, angles, material). Those numbers go into a small neural network that
has been trained to imitate a finite-element simulation, which predicts what the
frame would actually weigh and how far it would flex. If the prediction lands inside
the tolerance the request asked for, the design passes.

The experiment compares two ways of getting a model to do this: **prompting** it
(including showing it worked examples) versus **fine-tuning** it with QLoRA on
13,832 training examples. It is tested on 405 held-out frames that no model saw
during training.

This is Phase 2 of a two-part project. Phase 1 is
[llm-cad-benchmark](https://github.com/eaglejosh9/llm-cad-benchmark).

---

## Results

Every row below uses the same 405 requests, the same decoder settings, and the
same grader. A design passes only if **every** constraint in its request is met
(±5% or 0.25 kg for mass; ±15% or 1 cm for deflection).

| Model | Condition | Passed | Mass error | Deflection error | Valid JSON |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | zero-shot | 26.9% | 8.13 kg | 0.0498 m | 1.00 |
| Qwen2.5-7B-Instruct | few-shot (3 examples) | 43.0% | 2.02 kg | 0.0283 m | 1.00 |
| Qwen2.5-Coder-7B | zero-shot | 18.8% | 72.85 kg | 0.3409 m | 1.00 |
| Qwen2.5-Coder-7B | few-shot (3 examples) | 42.0% | 2.66 kg | 0.0570 m | 1.00 |
| **Qwen2.5-7B + QLoRA** | **fine-tuned, zero-shot** | **56.3%** | **0.47 kg** | **0.0137 m** | **1.00** |
| Qwen2.5-7B + QLoRA | fine-tuned, few-shot (3 examples) | 41.5% | 0.94 kg | 0.0299 m | 1.00 |

Three things to read off this table:

- **Fine-tuning wins.** It passes most often (56.3%) and its average mass error is
  4.3× smaller than the best prompted model.
- **Worked examples help a prompted model a lot** — roughly doubling the pass rate
  for both baselines.
- **But they hurt the fine-tuned model**, dropping it from 56.3% to 41.5%. Once the
  model has learned the task, the three examples pull it off course.

### How close are the misses?

On the 120 requests that ask for one specific mass:

| | Best prompted model | Fine-tuned |
|---|---|---|
| Median error | 1.36 kg | **0.34 kg** |
| 90th percentile | 3.75 kg | **1.02 kg** |
| Worst case | 9.10 kg | **2.99 kg** |
| Within 1 kg of target | 40.0% | **89.2%** |

### What masses does each model produce?

Real bicycle frames in the dataset mostly weigh 2–8 kg. Comparing that against
what each model outputs across all 405 requests:

| | 10th pct | Median | 90th pct |
|---|---|---|---|
| Real frames (the target) | 2.12 kg | 4.38 kg | 7.50 kg |
| **Fine-tuned** | **1.94 kg** | **4.47 kg** | **7.40 kg** |
| Prompted, few-shot | 1.29 kg | 3.27 kg | 4.97 kg |

The fine-tuned model's outputs line up with real frames. The prompted model stays
bunched near the three examples it was shown and rarely produces a heavy frame.

### The grader

The neural network doing the grading was trained once and then frozen. Its accuracy
on held-out frames:

| Predicting | R² | Error (RMSE) |
|---|---|---|
| Mass | 0.968 | 0.391 kg |
| Dropout deflection | 0.845 | 0.0187 m |

Note that "passes" throughout means *this network thinks it passes*, not that a full
simulation was run.

---

## How to run

Requires Python 3.11 and a CUDA GPU for the fine-tuning stage.

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
```

Then download the FRAMED dataset into `FRAMED Dataset/` — see
[`data/README.md`](data/README.md).

**1. Train the grader** (optional — a trained one is already in `checkpoints/`):

```bash
jupyter nbconvert --execute surrogate_model.ipynb
```

**2. Run the prompted baselines:**

```bash
cd llm_baseline
python generate.py --model qwen2.5-7b-instruct       --condition both --device cuda:0
python generate.py --model qwen2.5-coder-7b-instruct --condition both --device cuda:0
python score.py  --model qwen2.5-7b-instruct
python score.py  --model qwen2.5-coder-7b-instruct
python report.py                                     # -> outputs/report.csv
```

**3. Fine-tune and compare:**

```bash
cd llm_qlora
python build_dataset.py
torchrun --nproc_per_node=2 train.py                 # ~7.5 h on 2 GPUs
python generate_qlora.py --condition both --device cuda:0
python ../llm_baseline/score.py --model qlora
python compare_report.py                             # -> outputs/report_with_qlora.csv
```

Generation is resumable — rerunning the same command skips work already done.

The trained adapter weights (646 MB) are not committed because they exceed GitHub's
file size limit, but `train.py` regenerates them and
`llm_qlora/adapter_model/adapter_config.json` records the exact settings.

---

## Layout

```
surrogate_model.ipynb     Trains the grader network; defines the 90/10 data split
checkpoints/              Trained grader + scalers (committed)
data/README.md            How to obtain FRAMED (not redistributed here)

llm_baseline/             Prompt-based baselines + shared library
  common.py               The 39 input columns, targets, tolerances, model ids
  split.py                The 90/10 train/test split
  prompts.py              System prompt, request templating, example selection
  surrogate.py            Loads the frozen grader network
  generate.py             Zero-shot / few-shot generation
  score.py                Runs designs through the grader, checks constraints
  report.py               Comparison table
  outputs/                Generations, scored records and reports (committed)

llm_qlora/                The fine-tuning stage
  common_ft.py            Imports llm_baseline so both stages share one source of truth
  build_dataset.py        Builds the fine-tuning dataset from training rows
  train.py                QLoRA fine-tuning
  generate_qlora.py       Base model + adapter, same evaluation harness
  compare_report.py       Baseline vs fine-tuned comparison
  outputs/logs/           Training and generation logs (committed)
```

---

## Credits

FRAMED dataset — Regenwetter, Weaver & Ahmed, MIT DeCoDE Lab. See
[`data/README.md`](data/README.md) for citation and download instructions.

Base model — Qwen2.5-7B-Instruct (Alibaba).
