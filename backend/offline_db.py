"""
offline_db.py
==============
The real offline database of 13,122 real DFT-computed elastic tensors
that ships with the original ElATools repo (db/Cijs.binery +
db/All_2ID_cop.csv), reverse-engineered and validated.

How the format was decoded (see README.md "Validation notes" for the
full story): `Cijs.binery` turned out to be one IEEE-754 single-precision
float per line, written as a right-justified 16-character hex field
(Fortran `Z16` edit descriptor on a REAL(4) variable) -- NOT a 6x6 block
with a fixed number of lines per material as it first appeared. Every
material occupies exactly 36 consecutive lines (6 rows x 6 columns,
row-major), matched in order to material IDs in `All_2ID_cop.csv`.

Validation performed before trusting this:
  - All 13,122 decoded 6x6 matrices are EXACTLY symmetric (to float
    precision) -- real elastic tensors are symmetric by construction,
    so this would be extremely unlikely by chance if the byte alignment
    were wrong.
  - 86.5% are positive-definite (mechanically stable), consistent with
    published literature noting some fraction of the Materials Project's
    elasticity dataset (of this era) contains unreliable/unstable entries.
  - Spot checks against well-known Materials Project IDs match published
    values closely: mp-66 (diamond) decodes to C11=1054, C12=126,
    C44=562 GPa (literature: ~1076, ~125, ~562-577 GPa); mp-13 (BCC iron)
    decodes to C11=247, C12=150, C44=97 GPa, in the expected range.

This module loads the pre-extracted, human-readable CSV
(data/mp_elastic_db.csv, built once from the above) rather than
re-parsing the original hex file at request time.
"""
from __future__ import annotations
import csv
import os
import numpy as np

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mp_elastic_db.csv")

_TRIU_IDX = [(i, j) for i in range(6) for j in range(i, 6)]

_records: list[dict] | None = None


def _load():
    global _records
    if _records is not None:
        return _records
    records = []
    with open(_DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    _records = records
    return _records


def count() -> int:
    return len(_load())


def search(query: str, limit: int = 25) -> list[dict]:
    """Case-insensitive substring search over material IDs."""
    query = query.strip().lower()
    records = _load()
    if not query:
        results = records[:limit]
    else:
        results = [r for r in records if query in r["material_id"].lower()][:limit]
    return [{"material_id": r["material_id"], "stable": r["stable"] == "True"} for r in results]


def get_matrix(material_id: str) -> dict:
    records = _load()
    for r in records:
        if r["material_id"] == material_id:
            C = np.zeros((6, 6))
            for (i, j) in _TRIU_IDX:
                val = float(r[f"C{i+1}{j+1}"])
                C[i, j] = val
                C[j, i] = val
            return {
                "material_id": material_id,
                "cij": C.tolist(),
                "stable": r["stable"] == "True",
            }
    raise KeyError(f"'{material_id}' not found in the offline database.")
