"""System prompt, few-shot exemplar selection, and per-row spec templating."""
import json
import random

import numpy as np
import pandas as pd

from common import CONTINUOUS_COLS, INPUT_COLS, MATERIAL_COLS, TARGET_COLS

PARAM_GROUP_TEXT = """\
A bicycle frame here is fully described by 39 numeric design parameters, in five groups:

1. MATERIAL (one-hot, exactly one of these three is 1 and the other two are 0):
   "Material=Steel", "Material=Aluminum", "Material=Titanium".
2. STRUCTURAL BOOLEANS (0 or 1): "SSB_Include" (seat-stay bridge present),
   "CSB_Include" (chain-stay bridge present) — small cross-braces that stiffen
   the rear triangle when present.
3. TUBE OUTER DIAMETERS, in meters (keys ending "OD"): BB OD, TT OD, HT OD,
   DT OD, CS OD, SS OD, ST OD, SSB OD, CSB OD — one per named tube (BB=bottom
   bracket, TT=top tube, HT=head tube, DT=down tube, CS=chain stay, SS=seat
   stay, ST=seat tube, SSB/CSB=the two bridges). Typical range ~0.005-0.13 m.
4. TUBE WALL THICKNESSES, in meters (keys ending "Thickness"): SS, CS, TT, BB,
   HT, ST, DT Thickness. Thinner walls save mass but flex more. Typical range
   ~0.0005-0.01 m.
5. FRAME GEOMETRY: lengths, offsets and positions in meters, angles in degrees.
   These fix the shape/layout of the frame: CS Length, BB Drop, Stack, SS E,
   ST Angle, CS F, HT LX, ST UX, HT UX, HT Angle, HT Length, ST Length,
   BB Length, Dropout Offset, SSB Offset, CSB Offset, SS Z, DT Length.

Every one of these 39 keys must appear in your output exactly once."""

TARGET_TEXT = """\
Two structural performance values summarize how a frame behaves under load:

- "Model Mass" — total mass of the frame in kilograms (kg). Lower is lighter.
- "Sim 1 Dropout Y Disp." — vertical displacement of the rear dropouts under
  an in-plane load, in meters (m). This is a deflection: it is usually a small
  negative number, and a LARGER MAGNITUDE (further from zero, e.g. -0.08 vs
  -0.02) means the frame flexes more there, i.e. it is LESS stiff. A magnitude
  closer to zero means a stiffer frame."""

INSTRUCTION_TEXT = """\
Given a target performance spec, propose one full bicycle frame design that you
believe would achieve it. Respond with ONLY a single valid JSON object — no
markdown code fences, no commentary before or after. The object must have
exactly these 39 keys, in this exact order, each mapped to a numeric value:

{key_list}"""


def system_prompt(examples: list[dict] | None) -> str:
    parts = [
        "You are a bicycle-frame structural design assistant.",
        PARAM_GROUP_TEXT,
        TARGET_TEXT,
    ]
    if examples:
        parts.append("Here are worked examples of a spec and a design that achieves it:")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}\nSpec: {ex['spec']}\nJSON: {ex['json_str']}")
    parts.append(INSTRUCTION_TEXT.format(key_list=json.dumps(INPUT_COLS)))
    return "\n\n".join(parts)


def row_to_ordered_dict(row: pd.Series) -> dict:
    d = {}
    for c in INPUT_COLS:
        v = float(row[c])
        d[c] = int(round(v)) if c not in CONTINUOUS_COLS else round(v, 4)
    return d


def row_to_json_str(row: pd.Series) -> str:
    return json.dumps(row_to_ordered_dict(row))


def select_examples(train_df: pd.DataFrame) -> list[dict]:
    """Pick 3 diverse TRAIN rows: light & stiff, heavy, and typical."""
    mass = train_df["Model Mass"]
    disp_abs = train_df["Sim 1 Dropout Y Disp."].abs()
    mass_z = (mass - mass.mean()) / mass.std()
    disp_z = (disp_abs - disp_abs.mean()) / disp_abs.std()

    light_stiff_idx = (mass_z + disp_z).idxmin()
    heavy_idx = mass.idxmax()
    typical_idx = ((mass - mass.median()).abs() / mass.std() + (disp_abs - disp_abs.median()).abs() / disp_abs.std()).idxmin()

    picks = [
        ("a light, stiff", light_stiff_idx),
        ("a heavy", heavy_idx),
        ("a typical", typical_idx),
    ]
    examples = []
    for label, idx in picks:
        row = train_df.loc[idx]
        spec = spec_for_row(row, kind="point", include_mass=True, include_disp=True, rng=random.Random(0))
        examples.append({
            "row_idx": int(idx),
            "label": label,
            "spec": spec["text"],
            "json_str": row_to_json_str(row),
        })
    return examples


def _fmt(v: float, ndigits: int) -> str:
    return f"{v:.{ndigits}f}"


def spec_for_row(row: pd.Series, kind: str, include_mass: bool, include_disp: bool, rng: random.Random) -> dict:
    """Build one natural-language spec + machine-readable constraints for a row.

    kind: "point" | "mixed" (each target independently point/max/min).
    """
    mass_v = float(row["Model Mass"])
    disp_v = float(row["Sim 1 Dropout Y Disp."])
    disp_abs = abs(disp_v)

    constraints = []
    clauses = []

    if include_mass:
        ctype = "point" if kind == "point" else rng.choice(["point", "max", "min"])
        mv = round(mass_v, 2)
        if ctype == "point":
            phr = rng.choice([
                f"weighing about {_fmt(mv, 2)} kg",
                f"a target mass near {_fmt(mv, 2)} kg",
                f"mass of roughly {_fmt(mv, 2)} kg",
            ])
        elif ctype == "max":
            phr = rng.choice([
                f"weighing under {_fmt(mv, 2)} kg",
                f"mass no more than {_fmt(mv, 2)} kg",
            ])
        else:
            phr = rng.choice([
                f"weighing at least {_fmt(mv, 2)} kg",
                f"mass of {_fmt(mv, 2)} kg or more",
            ])
        clauses.append(phr)
        constraints.append({"target": "Model Mass", "type": ctype, "value": mv})

    if include_disp:
        ctype = "point" if kind == "point" else rng.choice(["point", "max", "min"])
        dv = round(disp_abs, 4)
        if ctype == "point":
            phr = rng.choice([
                f"with about {_fmt(dv, 3)} m of dropout deflection under load",
                f"targeting roughly {_fmt(dv, 3)} m of rear-dropout displacement",
            ])
        elif ctype == "max":
            phr = rng.choice([
                f"dropout deflection no more than {_fmt(dv, 3)} m (i.e. fairly stiff)",
                f"keeping rear-dropout displacement under {_fmt(dv, 3)} m",
            ])
        else:
            phr = rng.choice([
                f"dropout deflection of at least {_fmt(dv, 3)} m",
                f"rear-dropout displacement of {_fmt(dv, 3)} m or more (a more compliant frame)",
            ])
        clauses.append(phr)
        constraints.append({"target": "Sim 1 Dropout Y Disp.", "type": ctype, "value": dv})

    if len(clauses) == 2:
        text = f"Design a bicycle frame {clauses[0]}, and {clauses[1]}."
    else:
        text = f"Design a bicycle frame {clauses[0]}."
    return {"text": text, "constraints": constraints}


def make_spec_for_test_row(row: pd.Series, test_idx: int, seed: int) -> dict:
    """Deterministic per-row spec: varies phrasing, constraint type, and
    whether both targets or just one is specified, seeded by (seed, test_idx)
    so re-runs are reproducible."""
    rng = random.Random(f"{seed}:{int(test_idx)}")
    # 65% both targets specified, 35% only one (picked at random)
    if rng.random() < 0.65:
        include_mass, include_disp = True, True
    elif rng.random() < 0.5:
        include_mass, include_disp = True, False
    else:
        include_mass, include_disp = False, True
    return spec_for_row(row, kind="mixed", include_mass=include_mass, include_disp=include_disp, rng=rng)
