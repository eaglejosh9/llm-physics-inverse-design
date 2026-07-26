# LLM Physics — Bicycle Frame Inverse Design

**Can an LLM design a mechanical part to hit a target physical performance — and
does fine-tuning teach it something a prompt can't?**

Give a model a spec in plain English — *"design a bicycle frame weighing under
4.70 kg, with dropout deflection of at least 0.004 m"* — and ask for a complete
39-parameter frame design as JSON. Then run that design through a frozen neural
surrogate of a finite-element simulation and check whether the physics actually
comes out where the spec asked.

Headline: **a QLoRA fine-tune beats every prompt-based baseline, zero-shot, and
its mean mass error is 4.3× lower.**

This is **Phase 2** of a two-part summer project. Phase 1
([llm-cad-benchmark](https://github.com/eaglejosh9/llm-cad-benchmark)) found that
open-source LLMs write runnable CAD code reliably but score **0% on prompts
requiring physical reasoning**. This repo scopes that failure down to one
mechanical system where physics can be graded automatically in milliseconds, so
the question becomes measurable.

---

## Results

405 held-out frames, one spec each. Identical specs, decoder, tolerance band and
grader in every row — see [Making it fair](#making-it-fair).

| Model | Condition | Constraint satisfaction | Mass MAE | Deflection MAE | JSON valid |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | zero-shot | 26.9% | 8.13 kg | 0.0498 m | 1.00 |
| Qwen2.5-7B-Instruct | few-shot (3 ex.) | 43.0% | 2.02 kg | 0.0283 m | 1.00 |
| Qwen2.5-Coder-7B | zero-shot | 18.8% | 72.85 kg | 0.3409 m | 1.00 |
| Qwen2.5-Coder-7B | few-shot (3 ex.) | 42.0% | 2.66 kg | 0.0570 m | 1.00 |
| **Qwen2.5-7B + QLoRA** | **fine-tuned, zero-shot** | **56.3%** | **0.47 kg** | **0.0137 m** | **1.00** |
| Qwen2.5-7B + QLoRA | fine-tuned, few-shot (3 ex.) | 41.5% | 0.94 kg | 0.0299 m | 1.00 |

A design passes only if *every* constraint in its spec is satisfied within
tolerance (±5% or 0.25 kg for mass; ±15% or 1 cm for deflection).

### Few-shot examples actively *hurt* the fine-tuned model

Handing the fine-tuned model the same 3 worked examples that nearly doubled both
prompted baselines costs it 14.8 points — dropping it back to roughly baseline
territory (41.5% vs the best prompted 43.0%), and doubling its mass error.

The mechanism is visible in the mass distribution. The three exemplars weigh
**1.04 kg, 22.83 kg and 4.39 kg** — two of the three sit outside the dataset's
own p10–p90 band of 2.12–7.50 kg, because they were selected to be *diverse*
(light-and-stiff, heavy, typical), which is what makes them informative to a
model that knows nothing:

| | p10 | median | p90 | max |
|---|---|---|---|---|
| Real FRAMED frames | 2.12 kg | 4.38 kg | 7.50 kg | 22.83 kg |
| QLoRA zero-shot | 1.94 kg | 4.47 kg | 7.40 kg | 11.20 kg |
| QLoRA + few-shot | 1.43 kg | 3.96 kg | 6.65 kg | 18.83 kg |

Conditioning on the exemplars drags the fine-tuned model's output distribution
lighter and wider — away from the dataset prior it had already learned. The
per-constraint-type breakdown confirms the direction: satisfaction on *at-least*
mass constraints collapses from 64.5% to 39.8%, while *at-most* constraints
barely move (83.2% → 80.5%) — exactly what a downward mass bias produces.

So the exemplars aren't adding information any more; they're overriding a better
prior with three unrepresentative points. Fine-tuning didn't just make examples
unnecessary — it made them counterproductive. (One exemplar set, one model; a
representative-sample exemplar set might well behave differently.)

### The satisfaction rate is the least interesting number here

56.3% is a threshold-crossing metric with a deliberately tight band — for a 3 kg
frame, passing means landing within a quarter kilogram. What actually changed is
the whole error distribution. Taking just the 120 specs that request a **point**
mass target:

| | best few-shot baseline | QLoRA fine-tuned |
|---|---|---|
| median abs. error | 1.36 kg | **0.34 kg** |
| 90th percentile | 3.75 kg | **1.02 kg** |
| worst case (of 120) | 9.10 kg | **2.99 kg** |
| within 1 kg of target | 40.0% | **89.2%** |

The fine-tuned model doesn't just cross the pass line more often — its misses are
near-misses, where the baseline's misses are structural.

### It learned the design distribution, not the exemplars

| | p10 | median | p90 |
|---|---|---|---|
| Real FRAMED frames | 2.12 kg | 4.38 kg | 7.50 kg |
| **QLoRA predicted masses** | **1.94 kg** | **4.47 kg** | **7.40 kg** |
| Few-shot baseline | 1.29 kg | 3.27 kg | 4.97 kg |

The few-shot model is anchored on the three worked examples it was shown and
can't leave that neighbourhood. The fine-tuned model reproduces the real design
space. That, more than the pass rate, is the evidence that something
transferable was learned.

### And a failure mode only a physics check catches

Zero-shot, the *coding-specialized* model is off by **72.85 kg** on average, on
frames that really weigh 2–4 kg — with one design the surrogate reads as nearly
8,000 kg. That output is 100% valid, schema-conformant JSON. Knowing code syntax
does not tell you what a reasonable tube wall thickness is, and no formatting
check would ever flag it.

---

## How it works

```
  spec sentence          LLM           39-param JSON      frozen surrogate       pass / fail
"~2.3 kg, 0.02 m"  ->  (+ LoRA)   ->   {"BB OD": ...}  ->   MLP (39 -> 2)   ->  vs tolerance
```

### The parameterization: one frame = 39 numbers

This is the design decision that makes everything else measurable.

| Group | n | What |
|---|---|---|
| Material one-hot | 3 | Steel / Aluminum / Titanium, exactly one set |
| Structural flags | 2 | Seat-stay & chain-stay bridges present (0/1) |
| Tube outer diameters | 9 | BB, TT, HT, DT, CS, SS, ST, SSB, CSB — in metres |
| Wall thicknesses | 7 | One per tube — where the mass/stiffness trade lives |
| Frame geometry | 18 | Lengths, offsets, stack, head & seat angles |

Why this is a valid setup rather than a convenient simplification:

- **It isn't ours.** These are the columns FRAMED ships, which are BikeCAD's own
  parameters. All 4,046 real designs in the dataset are points in exactly this
  space, so "a valid design" means what a real frame-design tool says it means.
- **The output space *is* the input space.** The vector the LLM emits is
  literally the vector the surrogate consumes — no compile step, no geometry
  kernel, no translation layer where the model could be right and the pipeline
  still fail. That deliberately deletes Phase 1's dominant failure mode so that
  what remains to measure is only the physics.
- **Format failure is impossible.** JSON-schema-constrained decoding (Outlines)
  gives a 1.00 parse rate in every condition, so a failure can only ever be a
  physics failure.

The price: **topology is fixed.** The model re-parameterizes one frame layout; it
cannot invent a new one. That's the main reason these results don't
automatically generalize — and also what buys physics feedback in milliseconds
instead of hours.

### The specs: real frames, templated

Every spec is built from a real FRAMED frame's *own* simulated results
([`prompts.py`](llm_baseline/prompts.py)). Verbatim examples from the test set:

```
"Design a bicycle frame mass of roughly 2.59 kg, and targeting roughly
 0.172 m of rear-dropout displacement."                        [point + point]

"Design a bicycle frame weighing under 4.70 kg, and dropout deflection
 of at least 0.004 m."                                            [max + min]

"Design a bicycle frame mass of 3.08 kg or more, and targeting roughly
 0.044 m of rear-dropout displacement."                         [min + point]
```

Three things vary, from an RNG seeded by `(split seed, row index)` so the 405
specs are fixed, reproducible, and byte-identical for every model:

- constraint type per target — point / at-most / at-least
- phrasing — 2–3 interchangeable wordings per type
- coverage — 65% of specs constrain both targets, 35% only one

**Why this is a fair test:** because every number comes from a real frame's own
simulation, each spec is *achievable by construction* — at least one real design
satisfies it, and the at-least/under variants only relax that. We never ask for a
0.5 kg frame nobody has built, and we can't accidentally make the task trivial,
because the dataset's distribution chooses the targets, not us. The source frame
is never shown to the model, and many parameter vectors satisfy any one spec — so
this grades physics, not recall.

### The grader: a frozen surrogate

A small MLP (39 → 128 → 128 → 64 → 2, GELU, dropout 0.1) trained once in
[`surrogate_model.ipynb`](surrogate_model.ipynb) and then frozen. On held-out
frames:

| Target | R² | RMSE |
|---|---|---|
| Model Mass | 0.968 | 0.391 kg |
| Sim 1 Dropout Y Disp. | 0.845 | 0.0187 m |

(It beat XGBoost on both.) Mass is nearly a direct function of geometry and
material, so high accuracy is expected; deflection is noisier and more sensitive,
so 0.845 is the honest ceiling, not a bug. Fast enough to score every candidate
instantly, accurate enough that the score means something.

**Standing caveat:** "physics-conforming" throughout this repo means *the
surrogate agrees*, not *FEA agrees*. That error is baked equally into every
number — it largely cancels when comparing models, but it doesn't vanish from the
absolute values.

---

## Making it fair

Every headline number depends on these controls:

- **One split, seed 42, 90/10** — and it is the *same split the surrogate was
  trained under* ([`split.py`](llm_baseline/split.py) reproduces the notebook's
  indices exactly). So the 405 test frames are unseen by the grader as well as by
  the models.
- **The same 405 specs** — same numbers, same wording, generated once for
  everyone.
- **The same decoder** — identical schema-constrained generation and token budget,
  which is why the parse rate is 1.00 across the board. No model loses points for
  formatting or gains them for tidiness.
- **The same grader** — one frozen surrogate, one tolerance band, one scoring
  script ([`score.py`](llm_baseline/score.py)). Nothing is retuned per model.
- **No leakage** — the 3 few-shot exemplars are the same for both baselines and
  come only from TRAIN rows; QLoRA trained only on TRAIN rows (4 spec variants
  each), and `build_dataset.py` asserts disjointness from the test set.

What *isn't* controlled, stated rather than hidden: surrogate error is common to
all Phase 2 numbers, and pretraining data differs between models and is
unobservable. We control the evaluation, not what each model already saw.

---

## The training objective, and its limitation

Worth being explicit about, because it's the real methodological weakness here.

The loss is **token-level cross-entropy on the assistant completion only** —
prompt tokens masked to `-100` ([`train.py`](llm_qlora/train.py)). The model is
trained to reproduce the reference frame's *digit string*, one token at a time.

**Why isn't the surrogate in the loss?** It's a differentiable MLP, so
backpropagating physics error into the adapter looks natural. But getting from
the model's output to 39 numbers requires an argmax (or a sample) and a
string→float parse. Both are non-differentiable — the gradient dies there:

```
spec -> LoRA-adapted LLM -> token logits  --[ argmax + parse ]->  39 floats -> surrogate -> error
|<---------- cross-entropy gradient ---->|      NO GRADIENT      |<------ forward only ------>|
```

So "the surrogate says this frame is 0.4 kg too heavy" has no path back to the
weights without extra machinery: RL with the surrogate's verdict as reward, a
Gumbel-softmax / straight-through relaxation, or a numeric regression head. None
of those are what this repo runs.

Three consequences:

1. **The signal is textual imitation, not physics.** Every digit is weighted the
   same, so a one-token slip costing 0.5 kg looks much like one costing 50 kg.
2. **Correct-but-different designs are punished.** Many parameter vectors satisfy
   a given spec; only the reference frame's exact tokens score zero loss.
3. **Validation loss is the wrong early-stopping signal.** Eval loss bottomed at
   **epoch 1.16** and rose after — yet the saved epoch-3 checkpoint, well past
   that point, still produced the best physics of any run here. The two metrics
   simply aren't measuring the same thing.

The cheap fix for a next run: select checkpoints by scoring them with the
surrogate directly (one forward pass over 405 designs, seconds) rather than by
loss. **The 56.3% is therefore a floor, not a tuned ceiling** — the epoch-1.16
checkpoint has not been evaluated.

### QLoRA setup

| | |
|---|---|
| Base model | Qwen2.5-7B-Instruct, 4-bit NF4, double-quantized, frozen |
| Adapter | LoRA r=64, α=16, dropout 0.05, on all attention + MLP projections |
| Trainable | 161,480,704 params — ~2% of 7.78 B |
| Data | 13,832 train / 732 val examples (TRAIN rows × 4 spec variants) |
| Schedule | 3 epochs, lr 2e-4 constant+warmup, effective batch 16, paged AdamW 8-bit |
| Hardware | 2× RTX 6000 Ada, ~7.5 h |

Adapter weights (646 MB) are **not committed** — past GitHub's file limit.
`llm_qlora/adapter_model/adapter_config.json` is tracked, so the exact setup is
documented, and `train.py` regenerates the adapter.

---

## Reproducing

**Setup** — Python 3.11, CUDA GPU for the QLoRA stage:

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
```

Then download FRAMED into `FRAMED Dataset/` — see [`data/README.md`](data/README.md).

**1. Train the surrogate** (or skip — a trained one is committed in
`checkpoints/`):

```bash
jupyter nbconvert --execute surrogate_model.ipynb
```

**2. Baselines** — zero-shot and few-shot, both models:

```bash
cd llm_baseline
python generate.py --model qwen2.5-7b-instruct       --condition both --device cuda:0
python generate.py --model qwen2.5-coder-7b-instruct --condition both --device cuda:0
python score.py  --model qwen2.5-7b-instruct
python score.py  --model qwen2.5-coder-7b-instruct
python report.py                                     # -> outputs/report.csv
```

**3. QLoRA** — build data, train, generate, score, compare:

```bash
cd llm_qlora
python build_dataset.py
torchrun --nproc_per_node=2 train.py                 # ~7.5 h on 2 GPUs
python generate_qlora.py --device cuda:0
python ../llm_baseline/score.py --model qlora
python compare_report.py                             # -> outputs/report_with_qlora.csv
```

Generation resumes: already-written `(condition, test_idx)` pairs are skipped, so
an interrupted run can be restarted with the same command. Every stage writes
JSONL, and the slow GPU stage is fully decoupled from the fast CPU scoring stage
— tolerances can be retuned without regenerating anything.

---

## Layout

```
surrogate_model.ipynb     Trains the frozen surrogate; defines the 90/10 split
checkpoints/              Trained surrogate + input/target scalers (committed)
data/README.md            How to obtain FRAMED (not redistributed here)

llm_baseline/             Prompt-based baselines + the shared library
  common.py               39 input cols, targets, tolerances, model ids
  split.py                The 90/10 split, reproducing the notebook's indices
  prompts.py              System prompt, spec templating, exemplar selection
  surrogate.py            Loads the frozen MLP as a grader
  generate.py             Zero-/few-shot generation, schema-constrained
  score.py                Runs designs through the surrogate, checks constraints
  report.py               Headline comparison table
  outputs/                All generations + scored records + reports (committed)

llm_qlora/                The fine-tuning stage
  common_ft.py            Puts llm_baseline on sys.path — one source of truth for
                          the split, prompts, tolerances and scoring
  build_dataset.py        TRAIN rows x 4 spec variants -> SFT JSONL
  train.py                QLoRA SFT, completion-only loss masking
  generate_qlora.py       Base model + adapter, same eval harness as the baseline
  compare_report.py       Baseline vs QLoRA, plus per-constraint-type and
                          tail-vs-middle breakdowns
  outputs/logs/           Full training and generation logs (committed)
```

`llm_qlora` deliberately imports the baseline's modules rather than copying them,
so the split, the prompt templates, the tolerances and the scoring rules cannot
drift between the two stages.

---

## What this does and doesn't show

**Does:** where the output space is fixed and physics can be graded in
milliseconds, a 2%-parameter adapter on a 7B model moves output from *plausible*
to *quantitatively close* — and reproduces the real design distribution rather
than hugging its exemplars. On one machine, in hours.

**Doesn't:** this is not physics reasoning. It's a strong conditional prior over
one fixed 39-column parameterization, learned by imitating text. Nothing here
transfers to a new topology, a new load case, or free-form CAD. The 39 columns
are simultaneously why it works and why it doesn't generalize.

The bottleneck for physics-conforming CAD isn't model scale — it's the loop:
**paired data** (geometry + text + simulation, at scale), **a cheap grader**
(which requires a fixed parameterization first), and **a closed loop** feeding
the grader's verdict back into training. Phase 1 is what you get with none of the
three; Phase 2 is what you get with all three, for exactly one system.

---

## Credits

FRAMED dataset — Regenwetter, Weaver & Ahmed, MIT DeCoDE Lab. See
[`data/README.md`](data/README.md) for citation and download.
Base model — Qwen2.5-7B-Instruct (Alibaba).
