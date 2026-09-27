"""
elastic_core_1d.py
===================
Elastic properties of 1D materials (nanotubes, nanowires) from a 2x2
axial stiffness matrix. Formulas transcribed directly from the original
Fortran source (soc/Eatools_1D_proelast.f90), which in turn cites:
  Ekuma & Liu, "An Automated Toolkit for Elastic and Mechanical
  Properties of Tubular 2D-Based Nanostructures and Nanotubes."

Matrix convention: C = [[C11, C12], [C12, C22]]
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


class Invalid1DError(ValueError):
    pass


def parse_cij_1d_text(text: str) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(vals)
        if len(rows) == 2:
            break
    if len(rows) < 2:
        raise Invalid1DError(
            f"Could not find 2 rows of 2 numbers each (a 2x2 Cij-1D matrix); "
            f"found {len(rows)} valid row(s)."
        )
    C = np.array(rows, dtype=float)
    return 0.5 * (C + C.T)


@dataclass
class Scalar1DProperties:
    Ex: float   # axial Young's modulus
    Gxy: float  # torsional/shear-like modulus
    vxy: float  # Poisson's ratio
    stable: bool
    warnings: list = field(default_factory=list)


def compute_scalar_1d_properties(C: np.ndarray) -> Scalar1DProperties:
    warnings = []
    C = 0.5 * (C + C.T)
    eigvals = np.linalg.eigvalsh(C)
    stable = bool(np.min(eigvals) > 0)
    if not stable:
        warnings.append(
            "The 2x2 stiffness matrix is not positive definite -- this "
            "1D structure would be mechanically unstable."
        )

    Ex = C[1, 1]
    Gxy = (C[1, 1] - C[0, 1]) / 2.0
    vxy = -C[0, 1] / C[1, 1]

    return Scalar1DProperties(Ex=Ex, Gxy=Gxy, vxy=vxy, stable=stable, warnings=warnings)
