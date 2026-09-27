"""
elastic_core_2d.py
===================
Elastic properties of 2D (monolayer) materials from an in-plane 3x3
stiffness matrix (Cij-2D, in N/m -- 2D materials don't have a well-defined
thickness, so elastic constants are reported per unit area, not per unit
volume like 3D GPa).

All formulas transcribed directly from the original ElATools Fortran
source:
  - soc/Eatools_2D_proelast.f90  (scalar Voigt-Reuss properties + anisotropy)
  - soc/Eatools_2Dyoung.f90, Eatools_2Dpoisson.f90, Eatools_2Dshear.f90
    (angle-dependent properties)
  - soc/Eatools_stability.f90 (stability3d/2d: generic eigenvalue criterion,
    Mouhat & Coudert, Phys. Rev. B 90, 224104, 2014 -- used unchanged here)

Matrix convention (matches Cij-2D.dat): C = [[C11, C12, 0],
                                              [C12, C22, 0],
                                              [0,   0,   C33]]
where axis 1 = x, axis 2 = y, axis 3 = xy (shear).

Angle convention (matches the Fortran angular sweep exactly):
  n1 = sin(phi), n2 = cos(phi); vv11 = n1^4, vv22 = n2^4, vv33 = n1^2 n2^2
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


class Invalid2DError(ValueError):
    pass


def parse_cij_2d_text(text: str) -> np.ndarray:
    """Parse a 3x3 in-plane stiffness matrix (N/m) from text, tolerant of
    header lines (same permissive strategy as the 3D parser)."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 3:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(vals)
        if len(rows) == 3:
            break
    if len(rows) < 3:
        raise Invalid2DError(
            f"Could not find 3 rows of 3 numbers each (a 3x3 Cij-2D matrix); "
            f"found {len(rows)} valid row(s)."
        )
    C = np.array(rows, dtype=float)
    C = 0.5 * (C + C.T)
    return C


@dataclass
class Scalar2DProperties:
    Ex: float; Ey: float          # Young's modulus along x, y (N/m)
    Gxy: float                    # in-plane shear modulus (N/m)
    Gv: float; Gr: float          # Voigt/Reuss shear modulus (N/m)
    Bv: float; Br: float          # Voigt/Reuss area (bulk) modulus (N/m)
    vxy: float; vyx: float        # Poisson's ratios
    A_SU: float                   # elastic anisotropy index
    A_Ranganathan: float
    A_Kube: float
    stable: bool
    warnings: list = field(default_factory=list)


def validate_cij_2d(C: np.ndarray) -> list[str]:
    warnings = []
    asym = np.max(np.abs(C - C.T))
    if asym > 1e-3:
        warnings.append(f"Matrix is not symmetric (max asymmetry {asym:.4g} N/m); symmetrized.")
    eigvals = np.linalg.eigvalsh(C)
    if np.min(eigvals) <= 0:
        warnings.append(
            "The in-plane stiffness matrix is not positive definite "
            f"(smallest eigenvalue {np.min(eigvals):.4g} N/m) -- this 2D "
            "material would be mechanically unstable (Mouhat & Coudert, "
            "Phys. Rev. B 90, 224104, 2014)."
        )
    return warnings


def compute_scalar_2d_properties(C: np.ndarray) -> Scalar2DProperties:
    warnings = validate_cij_2d(C)
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    c, s = C, S

    Bv = (c[0, 0] + c[1, 1] + 2 * c[0, 1]) / 4.0
    Gv = (c[0, 0] + c[1, 1] - 2 * c[0, 1] + 4 * c[2, 2]) / 8.0
    Br = 1.0 / (s[0, 0] + s[1, 1] + 2 * s[0, 1])
    Gr = 2.0 / (s[0, 0] + s[1, 1] - 2 * s[0, 1] + s[2, 2])

    A_SU = float(np.sqrt((Bv / Br - 1.0) ** 2 + 2 * (Gv / Gr - 1.0) ** 2))
    A_Ranganathan = (Bv / Br) + 2 * (Gv / Gr) - 3.0
    A_Kube = float(np.sqrt(np.log10(Bv / Br) ** 2 + 2 * np.log10(Gv / Gr) ** 2))

    Ex = (c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]) / c[1, 1]
    Ey = (c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]) / c[0, 0]
    Gxy = c[2, 2]
    vxy = c[1, 0] / c[1, 1]
    vyx = c[0, 1] / c[0, 0]

    eigvals = np.linalg.eigvalsh(C)
    stable = bool(np.min(eigvals) > 0)

    return Scalar2DProperties(
        Ex=Ex, Ey=Ey, Gxy=Gxy, Gv=Gv, Gr=Gr, Bv=Bv, Br=Br,
        vxy=vxy, vyx=vyx, A_SU=A_SU, A_Ranganathan=A_Ranganathan, A_Kube=A_Kube,
        stable=stable, warnings=warnings,
    )


def directional_2d_properties(C: np.ndarray, n_phi: int = 180):
    """Sample Young's modulus, Poisson's ratio, shear modulus, and linear
    compressibility over phi = 0..360 deg, following the exact angular
    convention used in the original Fortran (Eatools_2Danalyz.f90):
        n1 = sin(phi), n2 = cos(phi)
        vv11 = n1^4, vv22 = n2^4, vv33 = n1^2 * n2^2
    Returns a dict of arrays keyed by property name, each length n_phi+1,
    plus phi_deg.
    """
    S = np.linalg.inv(C)
    s = S

    phis = np.linspace(0, 2 * np.pi, n_phi + 1)
    young = np.zeros_like(phis)
    poisson = np.zeros_like(phis)
    shear = np.zeros_like(phis)
    lin_comp = np.zeros_like(phis)

    for idx, phi in enumerate(phis):
        n1 = np.sin(phi)
        n2 = np.cos(phi)
        vv11 = n1 ** 4
        vv22 = n2 ** 4
        vv33 = (n1 ** 2) * (n2 ** 2)

        # Young's modulus (method via Sij, as recommended in the source)
        E_inv = s[0, 0] * vv22 + s[1, 1] * vv11 + (s[2, 2] + 2 * s[0, 1]) * vv33
        young[idx] = 1.0 / E_inv

        # Shear modulus
        She = (s[0, 0] + s[1, 1] - 2 * s[0, 1]) * vv33 + 0.25 * s[2, 2] * (vv11 + vv22 - 2 * vv33)
        shear[idx] = 1.0 / (4.0 * She)

        # Poisson's ratio
        E_denom = (2 * s[0, 1] + s[2, 2]) * vv33 + (s[0, 0] * vv22 + s[1, 1] * vv11)
        Pe = (s[0, 0] + s[1, 1] - s[2, 2]) * vv33 + s[0, 1] * (vv11 + vv22)
        poisson[idx] = -(Pe / E_denom)

        # Linear compressibility (2D)
        k11 = n2 * n2
        k12 = n1 * n2
        k22 = n1 * n1
        lin_comp[idx] = (
            (s[0, 0] + s[0, 1]) * k11
            + (s[0, 2] + s[1, 2]) * k12
            + (s[0, 1] + s[1, 1]) * k22
        )

    return {
        "phi_deg": np.degrees(phis).tolist(),
        "youngs_modulus": young.tolist(),
        "poisson_ratio": poisson.tolist(),
        "shear_modulus": shear.tolist(),
        "linear_compressibility": lin_comp.tolist(),
    }


# --------------------------------------------------------------------------
# In-plane wave velocities (2D Christoffel equation)
# --------------------------------------------------------------------------
#
# Note on fidelity to the original source: the original Fortran
# (soc/Eatools_wave_cal2d.f90) builds its 4th-rank tensor in a way that
# never actually incorporates the shear constant C33 (=C66) into the
# Christoffel matrix for the off-diagonal (shear) tensor components --
# it reuses C12 there instead, which looks like a copy-paste bug rather
# than intentional physics. This implementation instead uses the
# standard, physically correct 2D stiffness tensor construction:
#   C_1111 = C11, C_2222 = C22, C_1122 = C_2211 = C12,
#   C_1212 = C_1221 = C_2112 = C_2121 = C33 (the shear constant)
# consistent with how the analogous 3D tensor is built elsewhere in this
# project (see elastic_core.stiffness_tensor_from_voigt).

def _stiffness_tensor_2d(C: np.ndarray) -> np.ndarray:
    T = np.zeros((2, 2, 2, 2))
    T[0, 0, 0, 0] = C[0, 0]
    T[1, 1, 1, 1] = C[1, 1]
    T[0, 0, 1, 1] = T[1, 1, 0, 0] = C[0, 1]
    T[0, 1, 0, 1] = T[1, 0, 0, 1] = T[0, 1, 1, 0] = T[1, 0, 1, 0] = C[2, 2]
    return T


def christoffel_modes_2d(C: np.ndarray, n: np.ndarray, areal_density: float):
    """2D Christoffel equation: Gamma_ik = C_ijkl n_j n_l (i,j,k,l in {1,2}).
    `areal_density` is the 2D mass density in kg/m^2 (areal density --
    for a monolayer this is the bulk density times an effective
    thickness, if you only have a 3D density on hand).
    Returns the two in-plane modes (quasi-longitudinal, quasi-transverse),
    each with velocity in m/s.
    """
    n = n / np.linalg.norm(n)
    T = _stiffness_tensor_2d(C)
    Gamma = np.einsum("ijkl,j,l->ik", T, n, n)
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    order = np.argsort(eigvals)[::-1]
    modes = []
    for idx in order:
        lam = max(eigvals[idx], 0.0)
        # C in N/m, areal_density in kg/m^2 -> (N/m)/(kg/m^2) = m^2/s^2
        v = float(np.sqrt(lam / areal_density))
        modes.append({"velocity": v, "polarization": eigvecs[:, idx].tolist()})
    return modes  # [quasi-longitudinal, quasi-transverse]


def velocity_curve_2d(C: np.ndarray, areal_density: float, n_phi: int = 180):
    phis = np.linspace(0, 2 * np.pi, n_phi + 1)
    v_long = np.zeros_like(phis)
    v_trans = np.zeros_like(phis)
    for idx, phi in enumerate(phis):
        n = np.array([np.cos(phi), np.sin(phi)])
        modes = christoffel_modes_2d(C, n, areal_density)
        v_long[idx] = modes[0]["velocity"]
        v_trans[idx] = modes[1]["velocity"]
    return {
        "phi_deg": np.degrees(phis).tolist(),
        "velocity_longitudinal": v_long.tolist(),
        "velocity_transverse": v_trans.tolist(),
    }
