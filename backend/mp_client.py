"""
mp_client.py
============
Thin client for the official Materials Project REST API, used to look up
real elastic tensors by formula or material ID.

Why this instead of the ~13,000-entry database bundled in the original
ElATools repo (db/Cijs.binery + db/All_2ID_cop.csv)? That file is an
undocumented, custom hex-float-per-line format with variable-length
blocks (its total line count isn't an exact multiple of 36, meaning
block length isn't constant, most likely because it packs fewer values
for higher-symmetry entries) and no header describing the encoding.
Reverse-engineering it with full confidence would need `db/api.bin`
correlated against `All_2ID_cop.csv` line-by-line, and any mistake would
silently produce wrong "reference" numbers for real materials -- worse
than not shipping a database at all. The Materials Project's own REST API
is documented, versioned, and officially supported, so this app queries
it live instead. It needs the user's own free API key (from
https://materialsproject.org/api after login) passed per-request; this
app never stores it server-side.

Docs: https://docs.materialsproject.org/downloading-data/using-the-api
Endpoint used: https://api.materialsproject.org/materials/elasticity/
"""
from __future__ import annotations
import requests

MP_BASE = "https://api.materialsproject.org"


class MPError(Exception):
    pass


def _headers(api_key: str) -> dict:
    return {"X-API-KEY": api_key}


def search_materials(api_key: str, formula: str, limit: int = 10) -> list[dict]:
    """Find material IDs with elasticity data for a given formula/chemsys."""
    try:
        resp = requests.get(
            f"{MP_BASE}/materials/summary/",
            headers=_headers(api_key),
            params={
                "formula": formula,
                "has_props": "elasticity",
                "_fields": "material_id,formula_pretty,structure",
                "_limit": limit,
            },
            timeout=15,
        )
    except requests.RequestException as e:
        raise MPError(f"Could not reach the Materials Project API: {e}")

    if resp.status_code == 401:
        raise MPError("Invalid or missing Materials Project API key.")
    if not resp.ok:
        raise MPError(f"Materials Project API error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json().get("data", [])
    return [
        {"material_id": d.get("material_id"), "formula_pretty": d.get("formula_pretty")}
        for d in data
    ]


def get_elastic_tensor(api_key: str, material_id: str) -> dict:
    """Fetch the 6x6 elastic tensor (IEEE-formatted, GPa) for a material ID."""
    try:
        resp = requests.get(
            f"{MP_BASE}/materials/elasticity/",
            headers=_headers(api_key),
            params={
                "material_ids": material_id,
                "_fields": "material_id,formula_pretty,elastic_tensor",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        raise MPError(f"Could not reach the Materials Project API: {e}")

    if resp.status_code == 401:
        raise MPError("Invalid or missing Materials Project API key.")
    if not resp.ok:
        raise MPError(f"Materials Project API error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json().get("data", [])
    if not data:
        raise MPError(f"No elasticity data found for '{material_id}'.")

    entry = data[0]
    tensor = entry.get("elastic_tensor", {})
    # `ieee_format` is the tensor in the IEEE/standard crystallographic
    # frame -- the correct one to use (confirmed with MP maintainers: see
    # https://matsci.org/t/migrating-from-old-v2-api-to-new-a-few-questions/53639)
    matrix = tensor.get("ieee_format") or tensor.get("raw")
    if not matrix:
        raise MPError(f"Elasticity entry for '{material_id}' has no tensor data.")

    return {
        "material_id": entry.get("material_id"),
        "formula_pretty": entry.get("formula_pretty"),
        "cij": matrix,
    }
