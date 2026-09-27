"""
main.py - ElATools-Web backend (FastAPI)

Endpoints:
    GET  /api/health
    GET  /api/examples                 -> list of built-in 3D example materials
    GET  /api/examples/{name}          -> raw Cij text for one 3D example
    POST /api/analyze                  -> 3D scalar Voigt-Reuss-Hill properties
    POST /api/surface                  -> 3D directional-property mesh (for Plotly)
    POST /api/parse_keyed              -> parse 'C11=... C12=...' text (elast.output-style)
    POST /api/report.csv               -> downloadable CSV of 3D scalar properties
    POST /api/report.json              -> downloadable JSON (3D scalar properties)
    POST /api/standalone_html          -> downloadable self-contained HTML report+plot

    GET  /api/examples_2d              -> list of built-in 2D example materials
    GET  /api/examples_2d/{name}       -> raw Cij-2D text for one 2D example
    POST /api/analyze_2d               -> 2D scalar properties
    POST /api/curve_2d                 -> 2D angular (polar) property curves

    POST /api/mp/search                -> search Materials Project by formula
    POST /api/mp/fetch                 -> fetch a real elastic tensor by material ID

Static frontend is served from /  (frontend/ directory).
"""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import elastic_core as ec
import elastic_core_2d as ec2d
import elastic_core_1d as ec1d
import offline_db
import mp_client

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(APP_DIR, "..", "examples")
FRONTEND_DIR = os.path.join(APP_DIR, "..", "frontend")

app = FastAPI(title="ElATools-Web API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class CijPayload(BaseModel):
    cij_text: str = Field(..., description="6x6 stiffness matrix as whitespace-separated text, GPa")
    density: Optional[float] = Field(None, description="g/cm^3, optional, enables Clarke min. thermal conductivity")
    avg_atomic_mass: Optional[float] = Field(None, description="average mass per atom (u), optional, enables Clarke min. thermal conductivity")


class SurfacePayload(CijPayload):
    property: str = Field(..., description="One of: " + ", ".join(ec.PROPERTY_FUNCS.keys()))
    resolution: Optional[str] = Field("medium", description="low | medium | high")


class PlaneSlicePayload(CijPayload):
    property: str
    plane: str = Field("xy", description="xy | xz | yz")
    n_points: Optional[int] = 180


class HklPayload(CijPayload):
    h: float
    k: float
    l: float


class KeyedPayload(BaseModel):
    text: str = Field(..., description="e.g. 'C11=118.8 C12=53.8 C44=59.4'")
    symmetry: str = Field(..., description="cubic | tetragonal | hexagonal | orthorhombic | trigonal")


class Cij2DPayload(BaseModel):
    cij_text: str = Field(..., description="3x3 in-plane stiffness matrix as text, N/m")


class Curve2DPayload(Cij2DPayload):
    n_phi: Optional[int] = Field(180, description="number of angular sample points (0-360deg)")


class Velocity2DPayload(Cij2DPayload):
    areal_density: float = Field(..., description="2D (areal) mass density, kg/m^2")
    n_phi: Optional[int] = Field(180)


class Cij1DPayload(BaseModel):
    cij_text: str = Field(..., description="2x2 axial stiffness matrix as text")


class MPSearchPayload(BaseModel):
    api_key: str
    formula: str


class MPFetchPayload(BaseModel):
    api_key: str
    material_id: str


class OfflineDBSearchPayload(BaseModel):
    query: str = ""
    limit: Optional[int] = 25


class OfflineDBGetPayload(BaseModel):
    material_id: str


_RESOLUTIONS = {
    "low": (30, 60),
    "medium": (50, 100),
    "high": (80, 160),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _parse_or_400(cij_text: str):
    try:
        return ec.parse_cij_text(cij_text)
    except ec.InvalidCijError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _parse_2d_or_400(cij_text: str):
    try:
        return ec2d.parse_cij_2d_text(cij_text)
    except ec2d.Invalid2DError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _scalar_dict(C, density=None, avg_atomic_mass=None):
    return asdict(ec.compute_scalar_properties(C, density=density, avg_atomic_mass=avg_atomic_mass))


def _scalar_2d_dict(C):
    return asdict(ec2d.compute_scalar_2d_properties(C))


SCALAR_ROWS = [
    ("Bulk modulus", "K_V", "K_R", "K_H", "GPa"),
    ("Shear modulus", "G_V", "G_R", "G_H", "GPa"),
    ("Young's modulus", "E_V", "E_R", "E_H", "GPa"),
    ("Poisson's ratio", "nu_V", "nu_R", "nu_H", ""),
    ("P-wave modulus", "M_V", "M_R", "M_H", "GPa"),
    ("Pugh's ratio (K/G)", "pugh_V", "pugh_R", "pugh_H", ""),
    ("Lame's first parameter", "La1_V", "La1_R", "La1_H", "GPa"),
    ("Lame's second parameter", "La2_V", "La2_R", "La2_H", "GPa"),
    ("Machinability index", "machinability_V", "machinability_R", "machinability_H", ""),
    ("Gruneisen constant", "gruneisen_V", "gruneisen_R", "gruneisen_H", ""),
    ("Thermal expansion coeff.", "thermal_exp_V", "thermal_exp_R", "thermal_exp_H", "1/K"),
]

HARDNESS_ROWS = [
    ("Hardness H_1a (Teter, via G)", "H_a1_V", "H_a1_R", "H_a1_H"),
    ("Hardness H_1b (Teter, via E)", "H_b1_V", "H_b1_R", "H_b1_H"),
    ("Hardness H_2 (Tian, via G)", "H_2_V", "H_2_R", "H_2_H"),
    ("Hardness H_3 (via E)", "H_3_V", "H_3_R", "H_3_H"),
    ("Hardness H_4 (via K, nu)", "H_4_V", "H_4_R", "H_4_H"),
    ("Hardness H_5 (Chen, via G, K/G)", "H_5_V", "H_5_R", "H_5_H"),
]


# --------------------------------------------------------------------------
# 3D example materials
# --------------------------------------------------------------------------

_EXAMPLE_FILES = {
    "GaAs (cubic)": "GaAs_Cij.dat",
    "Tetragonal example": "Tetragonal_Cij.dat",
    "Orthorhombic example": "Orthorhombic_Cij.dat",
}

_EXAMPLE_FILES_2D = {
    "Graphene (illustrative, isotropic)": "Graphene_Cij2D.dat",
}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/examples")
def list_examples():
    return {"examples": list(_EXAMPLE_FILES.keys())}


@app.get("/api/examples/{name}")
def get_example(name: str):
    fname = _EXAMPLE_FILES.get(name)
    if not fname:
        raise HTTPException(status_code=404, detail="Unknown example")
    with open(os.path.join(EXAMPLES_DIR, fname)) as f:
        return {"name": name, "cij_text": f.read()}


@app.get("/api/examples_2d")
def list_examples_2d():
    return {"examples": list(_EXAMPLE_FILES_2D.keys())}


@app.get("/api/examples_2d/{name}")
def get_example_2d(name: str):
    fname = _EXAMPLE_FILES_2D.get(name)
    if not fname:
        raise HTTPException(status_code=404, detail="Unknown example")
    with open(os.path.join(EXAMPLES_DIR, fname)) as f:
        return {"name": name, "cij_text": f.read()}


# --------------------------------------------------------------------------
# 3D Analysis
# --------------------------------------------------------------------------

@app.post("/api/analyze")
def analyze(payload: CijPayload):
    C = _parse_or_400(payload.cij_text)
    result = _scalar_dict(C, density=payload.density, avg_atomic_mass=payload.avg_atomic_mass)
    result["cij"] = C.tolist()
    return result


@app.post("/api/surface")
def surface(payload: SurfacePayload):
    if payload.property not in ec.PROPERTY_FUNCS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown property. Choose one of: {list(ec.PROPERTY_FUNCS.keys())}",
        )
    C = _parse_or_400(payload.cij_text)
    n_theta, n_phi = _RESOLUTIONS.get(payload.resolution, _RESOLUTIONS["medium"])
    try:
        mesh = ec.build_directional_surface(
            C, payload.property, n_theta=n_theta, n_phi=n_phi, density=payload.density
        )
    except np.linalg.LinAlgError:
        raise HTTPException(status_code=400, detail="Stiffness matrix is singular / not invertible.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return mesh


@app.post("/api/parse_keyed")
def parse_keyed(payload: KeyedPayload):
    try:
        C = ec.parse_keyed_cij(payload.text, payload.symmetry)
    except ec.InvalidCijError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rows = "\n".join(" ".join(f"{v:12.4f}" for v in row) for row in C.tolist())
    return {"cij_text": rows, "cij": C.tolist()}


@app.post("/api/hkl")
def hkl_lookup(payload: HklPayload):
    C = _parse_or_400(payload.cij_text)
    try:
        n = ec.hkl_to_direction(payload.h, payload.k, payload.l)
        result = ec.evaluate_at_direction(C, n, density=payload.density)
    except (ValueError, np.linalg.LinAlgError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["note"] = (
        "Exact for orthogonal crystal systems (cubic, tetragonal, orthorhombic), "
        "where crystallographic axes align with Cartesian x,y,z. An approximation "
        "for non-orthogonal systems (hexagonal, trigonal, monoclinic, triclinic)."
    )
    return result


@app.post("/api/plane_slice")
def plane_slice(payload: PlaneSlicePayload):
    C = _parse_or_400(payload.cij_text)
    try:
        return ec.build_plane_slice(
            C, payload.property, plane=payload.plane,
            n_points=payload.n_points or 180, density=payload.density,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/export/wrl")
def export_wrl(payload: SurfacePayload):
    C = _parse_or_400(payload.cij_text)
    n_theta, n_phi = _RESOLUTIONS.get(payload.resolution, _RESOLUTIONS["medium"])
    try:
        mesh = ec.build_directional_surface(C, payload.property, n_theta=n_theta, n_phi=n_phi, density=payload.density)
        wrl = ec.export_surface_wrl(mesh)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(
        iter([wrl]), media_type="model/vrml",
        headers={"Content-Disposition": f"attachment; filename={payload.property}.wrl"},
    )


@app.post("/api/export/gnu")
def export_gnu(payload: SurfacePayload):
    C = _parse_or_400(payload.cij_text)
    n_theta, n_phi = _RESOLUTIONS.get(payload.resolution, _RESOLUTIONS["medium"])
    try:
        mesh = ec.build_directional_surface(C, payload.property, n_theta=n_theta, n_phi=n_phi, density=payload.density)
        gnu = ec.export_surface_gnu(mesh)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(
        iter([gnu]), media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={payload.property}.gnu"},
    )


@app.post("/api/export/agr")
def export_agr(payload: SurfacePayload):
    """Exports the 3D property as an angular (xy plane) slice, since Grace/
    xmgrace .agr files are fundamentally 2D XY line-plot projects."""
    C = _parse_or_400(payload.cij_text)
    try:
        slice_data = ec.build_plane_slice(C, payload.property, plane="xy", n_points=180, density=payload.density)
        agr = ec.export_curve_agr(
            slice_data["angle_deg"], slice_data["value"],
            title=f"{slice_data['label']} ({slice_data['unit']}), xy plane",
            ylabel=slice_data["unit"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(
        iter([agr]), media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={payload.property}.agr"},
    )


@app.post("/api/export/agr_2d")
def export_agr_2d(payload: Curve2DPayload):
    """Same idea, for a 2D-material angular property curve."""
    C = _parse_2d_or_400(payload.cij_text)
    curve = ec2d.directional_2d_properties(C, n_phi=payload.n_phi or 180)
    prop = "youngs_modulus"
    agr = ec.export_curve_agr(curve["phi_deg"], curve[prop], title="Young's Modulus (N/m)", ylabel="N/m")
    return StreamingResponse(
        iter([agr]), media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=young_2d.agr"},
    )


# --------------------------------------------------------------------------
# 1D materials
# --------------------------------------------------------------------------

@app.post("/api/analyze_1d")
def analyze_1d(payload: Cij1DPayload):
    try:
        C = ec1d.parse_cij_1d_text(payload.cij_text)
    except ec1d.Invalid1DError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = asdict(ec1d.compute_scalar_1d_properties(C))
    result["cij"] = C.tolist()
    return result


# --------------------------------------------------------------------------
# 2D wave velocities
# --------------------------------------------------------------------------

@app.post("/api/velocity_2d")
def velocity_2d(payload: Velocity2DPayload):
    C = _parse_2d_or_400(payload.cij_text)
    if payload.areal_density <= 0:
        raise HTTPException(status_code=400, detail="areal_density must be positive.")
    return ec2d.velocity_curve_2d(C, payload.areal_density, n_phi=payload.n_phi or 180)


# --------------------------------------------------------------------------
# Offline database (real Materials Project elastic tensors, bundled)
# --------------------------------------------------------------------------

@app.post("/api/offline_db/search")
def offline_db_search(payload: OfflineDBSearchPayload):
    return {"count": offline_db.count(), "results": offline_db.search(payload.query, limit=payload.limit or 25)}


@app.post("/api/offline_db/get")
def offline_db_get(payload: OfflineDBGetPayload):
    try:
        data = offline_db.get_matrix(payload.material_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    rows = "\n".join(" ".join(f"{v:12.4f}" for v in row) for row in data["cij"])
    return {"material_id": data["material_id"], "stable": data["stable"], "cij_text": rows}


# --------------------------------------------------------------------------
# 3D Downloads
# --------------------------------------------------------------------------

@app.post("/api/report.csv")
def report_csv(payload: CijPayload):
    C = _parse_or_400(payload.cij_text)
    props = _scalar_dict(C, density=payload.density, avg_atomic_mass=payload.avg_atomic_mass)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Property", "Voigt", "Reuss", "Hill", "Unit"])
    for label, v_key, r_key, h_key, unit in SCALAR_ROWS:
        writer.writerow([label, props[v_key], props[r_key], props[h_key], unit])
    writer.writerow([])
    writer.writerow(["Hardness model (GPa)", "Voigt", "Reuss", "Hill"])
    for label, v_key, r_key, h_key in HARDNESS_ROWS:
        writer.writerow([label, props[v_key], props[r_key], props[h_key]])
    writer.writerow([])
    writer.writerow(["Recommended hardness model", props["hardness_recommendation"]])
    writer.writerow(["Kleinman parameter", props["kleinman"]])
    writer.writerow(["Bond character", props["bond_character"]])
    writer.writerow(["Universal anisotropy index (AU)", props["AU"]])
    writer.writerow(["Log-Euclidean anisotropy parameter (AL)", props["AL"]])
    writer.writerow(["Chung-Buessem anisotropy index (Ac)", props["Ac"]])
    writer.writerow(["Cauchy pressure Pc (GPa)", props["Pc"]])
    writer.writerow(["Mechanically stable", props["stable"]])
    writer.writerow(["Ductility (Pugh/Poisson criterion)", props["ductility"]])
    writer.writerow(["Bonding character (Cauchy pressure sign)", props["bonding"]])
    if props.get("thermal_conductivity_min") is not None:
        writer.writerow(["Clarke min. thermal conductivity (W/m.K)", props["thermal_conductivity_min"]])
    if props["warnings"]:
        writer.writerow([])
        for w in props["warnings"]:
            writer.writerow(["Warning", w])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=elatools_report.csv"},
    )


@app.post("/api/report.json")
def report_json(payload: CijPayload):
    C = _parse_or_400(payload.cij_text)
    result = _scalar_dict(C)
    result["cij"] = C.tolist()
    data = json.dumps(result, indent=2)
    return StreamingResponse(
        iter([data]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=elatools_report.json"},
    )


@app.post("/api/standalone_html")
def standalone_html(payload: SurfacePayload):
    """A single self-contained HTML file: property tables + an interactive
    Plotly 3D surface, viewable offline with no server needed."""
    C = _parse_or_400(payload.cij_text)
    props = _scalar_dict(C)
    n_theta, n_phi = _RESOLUTIONS.get(payload.resolution, _RESOLUTIONS["medium"])
    try:
        mesh = ec.build_directional_surface(
            C, payload.property, n_theta=n_theta, n_phi=n_phi, density=payload.density
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    html = _render_standalone_html(props, mesh)
    return StreamingResponse(
        iter([html]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=elatools_report.html"},
    )


def _render_standalone_html(props: dict, mesh: dict) -> str:
    rows_html = "".join(
        f"<tr><td>{label}</td><td>{props[v]:.4f}</td><td>{props[r]:.4f}</td><td>{props[h]:.4f}</td><td>{unit}</td></tr>"
        for label, v, r, h, unit in SCALAR_ROWS
    )
    hardness_html = "".join(
        f"<tr><td>{label}</td><td>{props[v]:.3f}</td><td>{props[r]:.3f}</td><td>{props[h]:.3f}</td></tr>"
        for label, v, r, h in HARDNESS_ROWS
    )
    mesh_json = json.dumps(mesh)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ElATools Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #1c1c1c; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
  th {{ background: #f2f2f2; }}
  td:first-child, th:first-child {{ text-align: left; }}
  #plot {{ width: 100%; height: 600px; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.05rem; }}
</style>
</head>
<body>
<h1>ElATools Elastic Property Report</h1>
<table>
<tr><th>Property</th><th>Voigt</th><th>Reuss</th><th>Hill</th><th>Unit</th></tr>
{rows_html}
</table>
<h2>Hardness models (GPa)</h2>
<table>
<tr><th>Model</th><th>Voigt</th><th>Reuss</th><th>Hill</th></tr>
{hardness_html}
</table>
<p>Kleinman parameter: {props['kleinman']:.4f} ({props['bond_character']})</p>
<p>Universal anisotropy index (AU): {props['AU']:.4f} &nbsp;|&nbsp;
   AL (Kube): {props['AL']:.4f} &nbsp;|&nbsp;
   Chung-Buessem (Ac): {props['Ac']:.4f} &nbsp;|&nbsp;
   Cauchy pressure: {props['Pc']:.4f} GPa</p>
<p>{mesh['label']} ({mesh['unit']}) &mdash; min {mesh['min']:.3f}, max {mesh['max']:.3f}</p>
<div id="plot"></div>
<script>
  const mesh = {mesh_json};
  Plotly.newPlot('plot', [{{
    type: 'surface',
    x: mesh.x, y: mesh.y, z: mesh.z,
    surfacecolor: mesh.value,
    colorscale: 'Viridis',
    colorbar: {{ title: mesh.unit }}
  }}], {{
    margin: {{l:0,r:0,t:30,b:0}},
    scene: {{ aspectmode: 'data' }},
    title: mesh.label
  }});
</script>
</body>
</html>"""


# --------------------------------------------------------------------------
# 2D materials
# --------------------------------------------------------------------------

@app.post("/api/analyze_2d")
def analyze_2d(payload: Cij2DPayload):
    C = _parse_2d_or_400(payload.cij_text)
    result = _scalar_2d_dict(C)
    result["cij"] = C.tolist()
    return result


@app.post("/api/curve_2d")
def curve_2d(payload: Curve2DPayload):
    C = _parse_2d_or_400(payload.cij_text)
    try:
        return ec2d.directional_2d_properties(C, n_phi=payload.n_phi or 180)
    except np.linalg.LinAlgError:
        raise HTTPException(status_code=400, detail="Matrix is singular / not invertible.")


# --------------------------------------------------------------------------
# Materials Project live lookup
# --------------------------------------------------------------------------

@app.post("/api/mp/search")
def mp_search(payload: MPSearchPayload):
    try:
        results = mp_client.search_materials(payload.api_key, payload.formula)
    except mp_client.MPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": results}


@app.post("/api/mp/fetch")
def mp_fetch(payload: MPFetchPayload):
    try:
        data = mp_client.get_elastic_tensor(payload.api_key, payload.material_id)
    except mp_client.MPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rows = "\n".join(" ".join(f"{v:12.4f}" for v in row) for row in data["cij"])
    return {"material_id": data["material_id"], "formula_pretty": data["formula_pretty"], "cij_text": rows}


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
