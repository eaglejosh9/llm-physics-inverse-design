# Data

This project uses the **FRAMED** dataset, which is **not redistributed here** —
it belongs to the DeCoDE Lab at MIT. Download it yourself and drop it in place.

## What to download

> **FRAMED: An AutoML Approach for Structural Performance Prediction of Bicycle
> Frames**
> Lyle Regenwetter, Colin Weaver, Faez Ahmed — MIT DeCoDE Lab
> <https://decode.mit.edu/projects/framed/>

FRAMED is ~4,000 real bicycle frames designed by the BikeCAD community, each run
through finite-element simulation under standardized load cases. It is not
synthetic geometry, and the parameterization is BikeCAD's own — which is the
reason this project treats "a valid design" as something the dataset defines
rather than something we defined.

## Where to put it

Unzip so that the CSVs sit in a folder named exactly `FRAMED Dataset/` at the
repo root:

```
LLM_Physics/
├── FRAMED Dataset/
│   ├── all_structural_data_oh.csv      <- the only file the code reads
│   ├── all_structural_data.csv
│   ├── validity.csv
│   └── ...
├── llm_baseline/
└── llm_qlora/
```

The path is set once, in [`llm_baseline/common.py`](../llm_baseline/common.py):

```python
DATA_DIR = BASE_DIR / "FRAMED Dataset"
DATA_CSV = DATA_DIR / "all_structural_data_oh.csv"
```

Change it there if you keep the data elsewhere.

## What the code actually reads

Only `all_structural_data_oh.csv` — the one-hot-encoded version of the
community-design table, 4,046 rows. Two columns are dropped on load
(`Unnamed: 0` and `batch`), leaving:

- **39 design parameters** — the model's output space. See the README's
  parameterization section.
- **2 performance targets** — `Model Mass` (kg) and `Sim 1 Dropout Y Disp.` (m),
  the values the surrogate learns to predict and the specs are built from.

The `_aug` variants (which add AI-generated frames) and the `validity` tables are
part of the download but are **not** used here — every design in this project
traces back to a real community submission.

## Citation

```bibtex
@article{regenwetter2023framed,
  title   = {FRAMED: An AutoML Approach for Structural Performance
             Prediction of Bicycle Frames},
  author  = {Regenwetter, Lyle and Weaver, Colin and Ahmed, Faez},
  journal = {Computer-Aided Design},
  year    = {2023}
}
```
