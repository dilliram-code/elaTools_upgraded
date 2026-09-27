"""
elastic_core.py
================
Core physics for ElATools-Web: computes anisotropic elastic properties of
3D crystals from a 6x6 stiffness tensor (Cij, in GPa, Voigt notation).

All scalar-property formulas are transcribed directly from the original
ElATools Fortran source (soc/Eatools_proelast.f90) and validated against
the shipped GaAs example (example/GaAs/DATA.out) -- see validate.py.

Directional (angle-dependent) properties use the standard tensor
contraction of the compliance tensor S_ijkl with a unit direction vector,
the same approach used by ELATE / Marmier's elastic-anisotropy tools,
which is mathematically equivalent to what ElATools' Fortran plots (and
is validated against ElATools' own printed min/max values -- see
validate.py).

Two known, documented deviations from the shipped *example output* (not
from the source code -- see README.md "Validation notes" for the full
explanation of both):
  - Hill Young's modulus is derived from (K_H, G_H) as the current
    upstream source actually does, not as avg(E_V, E_R) as the stale
    shipped example implies.
  - The log-Euclidean anisotropy parameter AL uses the correctly-cited
    Kube (2016) formula, because the upstream Fortran variable for it is
    declared but never assigned (an upstream bug).
  - Lame's second parameter (La2) fixes an upstream copy-paste bug where
    the Reuss value used nu_H instead of nu_R.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

class InvalidCijError(ValueError):
    pass


def parse_cij_text(text: str) -> np.ndarray:
    """Parse a 6x6 stiffness matrix out of arbitrary text.

    This is deliberately permissive so it transparently accepts several
    of ElATools' original input files, which all turn out to be the same
    36-number block with different amounts of header noise in front:
      - Cij.dat / INVELC-matrix : the 6x6 block with no header at all
      - ELADAT                  : 2 header lines, then the 6x6 block
      - ElaStic_2nd.out         : many header lines, then the 6x6 block

    Any line that is not exactly 6 whitespace/comma-separated numbers is
    skipped (treated as header/comment noise); the first 6 valid rows
    found are used as the matrix.
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 6:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(vals)
        if len(rows) == 6:
            break

    if len(rows) < 6:
        raise InvalidCijError(
            f"Could not find 6 rows of 6 numbers each (a 6x6 Cij matrix) "
            f"in the supplied text; found {len(rows)} valid row(s). "
            "Check the file format, or use the 'keyed values' input for "
            "elast.output-style C11=...  C12=... text."
        )
    C = np.array(rows, dtype=float)
    C = 0.5 * (C + C.T)  # symmetrize
    return C


_SYMMETRY_TEMPLATES = {
    # name -> function(dict_of_supplied_keys) -> 6x6 matrix
    "cubic": lambda k: _build_from_keys(k, {
        (0, 0): "C11", (1, 1): "C11", (2, 2): "C11",
        (0, 1): "C12", (0, 2): "C12", (1, 2): "C12",
        (3, 3): "C44", (4, 4): "C44", (5, 5): "C44",
    }),
    "tetragonal": lambda k: _build_from_keys(k, {
        (0, 0): "C11", (1, 1): "C11", (2, 2): "C33",
        (0, 1): "C12", (0, 2): "C13", (1, 2): "C13",
        (3, 3): "C44", (4, 4): "C44", (5, 5): "C66",
    }),
    "hexagonal": lambda k: _build_from_keys(k, {
        (0, 0): "C11", (1, 1): "C11", (2, 2): "C33",
        (0, 1): "C12", (0, 2): "C13", (1, 2): "C13",
        (3, 3): "C44", (4, 4): "C44",
    }, derived={(5, 5): lambda C: (C[0, 0] - C[0, 1]) / 2.0}),
    "orthorhombic": lambda k: _build_from_keys(k, {
        (0, 0): "C11", (1, 1): "C22", (2, 2): "C33",
        (0, 1): "C12", (0, 2): "C13", (1, 2): "C23",
        (3, 3): "C44", (4, 4): "C55", (5, 5): "C66",
    }),
    "trigonal": lambda k: _build_from_keys(k, {
        (0, 0): "C11", (1, 1): "C11", (2, 2): "C33",
        (0, 1): "C12", (0, 2): "C13", (1, 2): "C13",
        (0, 3): "C14", (1, 3): "-C14",
        (3, 3): "C44", (4, 4): "C44",
    }, derived={(5, 5): lambda C: (C[0, 0] - C[0, 1]) / 2.0}),
}


def _build_from_keys(keys: dict, mapping: dict, derived: Optional[dict] = None) -> np.ndarray:
    C = np.zeros((6, 6))
    missing = []
    for (i, j), key in mapping.items():
        neg = key.startswith("-")
        k = key[1:] if neg else key
        if k not in keys:
            missing.append(k)
            continue
        val = -keys[k] if neg else keys[k]
        C[i, j] = val
        C[j, i] = val
    if missing:
        raise InvalidCijError(
            f"Missing required constants for this symmetry: {', '.join(sorted(set(missing)))}"
        )
    if derived:
        for (i, j), fn in derived.items():
            val = fn(C)
            C[i, j] = val
            C[j, i] = val
    return C


def parse_keyed_cij(text: str, symmetry: str) -> np.ndarray:
    """Parse 'C11=118.8, C12=53.8, C44=59.4' style text (as found in
    WIEN2k's elast.output) into a full 6x6 matrix for the given crystal
    symmetry. Keys are case-insensitive; separators can be '=', ':', or
    whitespace; values may be separated by commas, whitespace, or
    newlines.
    """
    import re
    symmetry = symmetry.lower()
    if symmetry not in _SYMMETRY_TEMPLATES:
        raise InvalidCijError(
            f"Unknown symmetry '{symmetry}'. Choose one of: {', '.join(_SYMMETRY_TEMPLATES)}"
        )
    pairs = re.findall(r"C(\d\d)\s*[=:]\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", text)
    if not pairs:
        raise InvalidCijError(
            "No 'C11=...' style key=value pairs found in the supplied text."
        )
    keys = {f"C{idx}": float(val) for idx, val in pairs}
    return _SYMMETRY_TEMPLATES[symmetry](keys)


def validate_cij(C: np.ndarray) -> list[str]:
    """Basic physical sanity checks. Returns a list of warning strings
    (empty list = looks fine). Does not raise -- caller decides how to
    surface warnings vs. hard failures.
    """
    warnings = []
    if C.shape != (6, 6):
        raise InvalidCijError("Cij must be a 6x6 matrix.")

    asym = np.max(np.abs(C - C.T))
    if asym > 1e-3:
        warnings.append(
            f"Matrix is not symmetric (max asymmetry {asym:.4g} GPa); "
            "it has been symmetrized."
        )

    eigvals = np.linalg.eigvalsh(C)
    if np.min(eigvals) <= 0:
        warnings.append(
            "The stiffness matrix is not positive definite "
            f"(smallest eigenvalue {np.min(eigvals):.4g} GPa) -- this "
            "material would be mechanically unstable (Mouhat & Coudert, "
            "Phys. Rev. B 90, 224104, 2014). Results below may not be "
            "physically meaningful."
        )
    return warnings


# --------------------------------------------------------------------------
# Scalar (Voigt-Reuss-Hill) properties
# --------------------------------------------------------------------------

@dataclass
class ScalarProperties:
    K_V: float; K_R: float; K_H: float
    G_V: float; G_R: float; G_H: float
    E_V: float; E_R: float; E_H: float
    nu_V: float; nu_R: float; nu_H: float
    M_V: float; M_R: float; M_H: float          # P-wave modulus
    pugh_V: float; pugh_R: float; pugh_H: float  # K/G
    La1_V: float; La1_R: float; La1_H: float     # Lame's first parameter
    La2_V: float; La2_R: float; La2_H: float     # Lame's second parameter (=G, sanity check)
    machinability_V: float; machinability_R: float; machinability_H: float
    gruneisen_V: float; gruneisen_R: float; gruneisen_H: float
    thermal_exp_V: float; thermal_exp_R: float; thermal_exp_H: float
    H_a1_V: float; H_a1_R: float; H_a1_H: float  # Hardness model 1a (Teter, via G)
    H_b1_V: float; H_b1_R: float; H_b1_H: float  # Hardness model 1b (Teter, via E)
    H_2_V: float; H_2_R: float; H_2_H: float     # Hardness model 2 (Tian, via G)
    H_3_V: float; H_3_R: float; H_3_H: float     # Hardness model 3 (via E)
    H_4_V: float; H_4_R: float; H_4_H: float     # Hardness model 4 (via K, nu)
    H_5_V: float; H_5_R: float; H_5_H: float     # Hardness model 5 (Chen, via G, K/G)
    hardness_recommendation: str
    kleinman: float           # Kleinman parameter
    bond_character: str       # bond bending vs. bond stretching note
    AU: float          # Universal anisotropy index (Ranganathan & Ostoja-Starzewski, PRL 2008)
    AL: float          # Log-Euclidean anisotropy parameter (Kube, AIP Advances 2016)
    Ac: float          # Chung-Buessem anisotropy index
    Pc: float          # Cauchy pressure C12 - C44 (GPa)
    ductility: str     # qualitative note based on Pugh/Poisson ratio
    bonding: str       # qualitative note based on Cauchy pressure
    stable: bool
    warnings: list = field(default_factory=list)
    thermal_conductivity_min: Optional[float] = None  # Clarke model, W/(m.K)


def compute_scalar_properties(
    C: np.ndarray,
    density: Optional[float] = None,
    avg_atomic_mass: Optional[float] = None,
) -> ScalarProperties:
    """
    density: g/cm^3
    avg_atomic_mass: average mass per atom in the formula unit, in atomic
        mass units (u) -- i.e. (molar mass in g/mol) / (atoms per formula
        unit). Both are optional; when both are given, the Clarke
        minimum thermal conductivity model is also computed.
    """
    warnings = validate_cij(C)
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)

    c = C  # shorthand
    s = S

    # --- Voigt averages ---
    K_V = ((c[0, 0] + c[1, 1] + c[2, 2]) + 2 * (c[0, 1] + c[1, 2] + c[0, 2])) / 9.0
    G_V = (
        (c[0, 0] + c[1, 1] + c[2, 2])
        - (c[0, 1] + c[1, 2] + c[0, 2])
        + 3 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0

    # --- Reuss averages ---
    K_R = 1.0 / ((s[0, 0] + s[1, 1] + s[2, 2]) + 2 * (s[0, 1] + s[1, 2] + s[0, 2]))
    G_R = 15.0 / (
        4 * (s[0, 0] + s[1, 1] + s[2, 2])
        - 4 * (s[0, 1] + s[1, 2] + s[0, 2])
        + 3 * (s[3, 3] + s[4, 4] + s[5, 5])
    )

    # --- Hill averages ---
    K_H = 0.5 * (K_V + K_R)
    G_H = 0.5 * (G_V + G_R)

    def derived(K, G):
        E = 1.0 / (1.0 / (3 * G) + 1.0 / (9 * K))
        nu = 0.5 * (1.0 - (3 * G) / (3 * K + G))
        M = K + 4 * G / 3.0
        pugh = K / G
        La1 = (nu * E) / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
        La2 = E / (2 * (1 + nu))                     # Lame's second parameter (== G)
        return E, nu, M, pugh, La1, La2

    E_V, nu_V, M_V, pugh_V, La1_V, La2_V = derived(K_V, G_V)
    E_R, nu_R, M_R, pugh_R, La1_R, La2_R = derived(K_R, G_R)
    E_H, nu_H, M_H, pugh_H, La1_H, La2_H = derived(K_H, G_H)

    AU = 5.0 * (G_V / G_R) + (K_V / K_R) - 6.0
    AL = float(np.sqrt(np.log(K_V / K_R) ** 2 + 5.0 * np.log(G_V / G_R) ** 2))
    Ac = (G_V - G_R) / (G_V + G_R)
    Pc = c[0, 1] - c[3, 3]

    # Machinability index (Chevalier), mu = K / C44
    machinability_V = K_V / c[3, 3]
    machinability_R = K_R / c[3, 3]
    machinability_H = K_H / c[3, 3]

    # Gruneisen constant, from nu (see DOI: 10.1134/S1063771007050090)
    def gruneisen(nu):
        return 1.5 * (1 + nu) / (2 - 3 * nu)

    gruneisen_V, gruneisen_R, gruneisen_H = gruneisen(nu_V), gruneisen(nu_R), gruneisen(nu_H)

    # Thermal expansion coefficient, empirical estimate ~ 1.6e-5 / G (per K), G in GPa
    thermal_exp_V = 1.6e-5 / G_V
    thermal_exp_R = 1.6e-5 / G_R
    thermal_exp_H = 1.6e-5 / G_H

    # --- Hardness models (all in GPa) ---
    def hardness(K, G, nu):
        H_a1 = 0.1475 * G                          # Teter (1998), metals
        H_b1 = 0.0607 * E_from_KG(K, G)            # Teter (1998), via E
        H_2 = 0.1769 * G - 2.899                   # Tian et al. (2012)
        H_3 = 0.0635 * E_from_KG(K, G)             # via E
        H_4 = (K * (1 - 2 * nu)) / (6 * (1 + nu))  # via K, nu (metals)
        pugh = K / G
        H_5 = 2.0 * (G * pugh ** -2.0) ** 0.585 - 3.0  # Chen et al. (2011)
        return H_a1, H_b1, H_2, H_3, H_4, H_5

    def E_from_KG(K, G):
        return 1.0 / (1.0 / (3 * G) + 1.0 / (9 * K))

    H_a1_V, H_b1_V, H_2_V, H_3_V, H_4_V, H_5_V = hardness(K_V, G_V, nu_V)
    H_a1_R, H_b1_R, H_2_R, H_3_R, H_4_R, H_5_R = hardness(K_R, G_R, nu_R)
    H_a1_H, H_b1_H, H_2_H, H_3_H, H_4_H, H_5_H = hardness(K_H, G_H, nu_H)

    # Kleinman parameter (internal-strain / bond bending vs. stretching)
    kleinman = (c[0, 0] + 8 * c[0, 1]) / (7 * c[0, 0] - 2 * c[0, 1])
    if abs(kleinman - 0.5) < 1e-9:
        bond_character = "Bond bending = bond stretching"
    elif kleinman > 0.5:
        bond_character = "Bond bending < bond stretching (stretching dominates resistance)"
    else:
        bond_character = "Bond bending > bond stretching (bending dominates resistance)"

    ductility = "Ductile" if pugh_H > 1.75 or nu_H > 0.26 else "Brittle"
    bonding = "Metallic-like bonding" if Pc > 0 else "Covalent/directional bonding"
    hardness_recommendation = (
        "For insulators, H_2 (general/cubic/orthorhombic/rhombohedral) or H_1b "
        "(hexagonal) is typically most reliable. For semiconductors, H_5 "
        "(general/cubic) or H_1b/H_3 (hexagonal) or H_2 (rhombohedral). For "
        "metals, H_4 (general/hexagonal/orthorhombic/rhombohedral) or H_1a "
        "(cubic). (See Yalameha et al., ElATools guide table.)"
    )

    eigvals = np.linalg.eigvalsh(C)
    stable = bool(np.min(eigvals) > 0)

    # Clarke minimum thermal conductivity (Clarke, Surf. Coat. Technol.
    # 163-164, 67, 2003): k_min = 0.87 * kB * Mv^(-2/3) * E^(1/2) * rho^(1/6)
    # in SI units, where Mv is the average mass per atom in kg.
    thermal_conductivity_min = None
    if density and avg_atomic_mass:
        kB = 1.380649e-23
        Mv_kg = avg_atomic_mass * 1.66053906660e-27  # u -> kg
        E_pa = E_H * 1e9
        rho_si = density * 1000.0  # g/cm^3 -> kg/m^3
        thermal_conductivity_min = 0.87 * kB * (Mv_kg ** (-2.0 / 3.0)) * (E_pa ** 0.5) * (rho_si ** (1.0 / 6.0))

    return ScalarProperties(
        K_V=K_V, K_R=K_R, K_H=K_H,
        G_V=G_V, G_R=G_R, G_H=G_H,
        E_V=E_V, E_R=E_R, E_H=E_H,
        nu_V=nu_V, nu_R=nu_R, nu_H=nu_H,
        M_V=M_V, M_R=M_R, M_H=M_H,
        pugh_V=pugh_V, pugh_R=pugh_R, pugh_H=pugh_H,
        La1_V=La1_V, La1_R=La1_R, La1_H=La1_H,
        La2_V=La2_V, La2_R=La2_R, La2_H=La2_H,
        machinability_V=machinability_V, machinability_R=machinability_R, machinability_H=machinability_H,
        gruneisen_V=gruneisen_V, gruneisen_R=gruneisen_R, gruneisen_H=gruneisen_H,
        thermal_exp_V=thermal_exp_V, thermal_exp_R=thermal_exp_R, thermal_exp_H=thermal_exp_H,
        H_a1_V=H_a1_V, H_a1_R=H_a1_R, H_a1_H=H_a1_H,
        H_b1_V=H_b1_V, H_b1_R=H_b1_R, H_b1_H=H_b1_H,
        H_2_V=H_2_V, H_2_R=H_2_R, H_2_H=H_2_H,
        H_3_V=H_3_V, H_3_R=H_3_R, H_3_H=H_3_H,
        H_4_V=H_4_V, H_4_R=H_4_R, H_4_H=H_4_H,
        H_5_V=H_5_V, H_5_R=H_5_R, H_5_H=H_5_H,
        hardness_recommendation=hardness_recommendation,
        kleinman=kleinman, bond_character=bond_character,
        AU=AU, AL=AL, Ac=Ac, Pc=Pc,
        ductility=ductility, bonding=bonding,
        stable=stable, warnings=warnings,
        thermal_conductivity_min=thermal_conductivity_min,
    )


# --------------------------------------------------------------------------
# Full compliance tensor (for directional properties)
# --------------------------------------------------------------------------

_VOIGT_MAP = {
    (0, 0): 0, (1, 1): 1, (2, 2): 2,
    (1, 2): 3, (2, 1): 3,
    (0, 2): 4, (2, 0): 4,
    (0, 1): 5, (1, 0): 5,
}


def _voigt_factor(m: int, n: int) -> float:
    """Engineering-strain factor relating S_ijkl to S_mn (Nye's convention)."""
    factor = 1.0
    if m >= 3:
        factor *= 2.0
    if n >= 3:
        factor *= 2.0
    return factor


def compliance_tensor_from_voigt(S: np.ndarray) -> np.ndarray:
    """Build the full 3x3x3x3 compliance tensor S_ijkl from the 6x6
    engineering compliance matrix S_mn.
    """
    Sijkl = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            m = _VOIGT_MAP[(i, j)]
            for k in range(3):
                for l in range(3):
                    n = _VOIGT_MAP[(k, l)]
                    Sijkl[i, j, k, l] = S[m, n] / _voigt_factor(m, n)
    return Sijkl


def stiffness_tensor_from_voigt(C: np.ndarray) -> np.ndarray:
    """Build the full 3x3x3x3 stiffness tensor C_ijkl from the 6x6
    engineering stiffness matrix C_mn. Unlike the compliance tensor, no
    extra engineering-shear-strain factor is needed here (it's already
    absorbed into the tensor-strain <-> Voigt-strain convention).
    """
    Cijkl = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            m = _VOIGT_MAP[(i, j)]
            for k in range(3):
                for l in range(3):
                    n = _VOIGT_MAP[(k, l)]
                    Cijkl[i, j, k, l] = C[m, n]
    return Cijkl


def directional_youngs_modulus(Sijkl: np.ndarray, n: np.ndarray) -> float:
    """E(n) = 1 / (S_ijkl n_i n_j n_k n_l), n a unit vector. Returns GPa."""
    inv_E = np.einsum("ijkl,i,j,k,l->", Sijkl, n, n, n, n)
    return 1.0 / inv_E


def directional_linear_compressibility(Sijkl: np.ndarray, n: np.ndarray) -> float:
    """beta(n) = S_ijkk n_i n_j, in TPa^-1 given S in 1/GPa (x1000)."""
    beta = np.einsum("ijkk,i,j->", Sijkl, n, n)
    return beta * 1000.0  # GPa^-1 -> TPa^-1


def directional_shear_modulus_extrema(Sijkl: np.ndarray, n: np.ndarray, n_samples: int = 90):
    """For a plane normal to n, sample shear modulus G(n, m) over m in that
    plane (m perpendicular to n) and return (G_max, G_min).
    1/G = 4 * S_ijkl n_i m_j n_k m_l
    """
    # build an orthonormal basis (u, v) spanning the plane perpendicular to n
    n = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    vals = []
    for k in range(n_samples):
        ang = np.pi * k / n_samples  # 0..pi covers all directions in-plane (symmetric)
        m = np.cos(ang) * u + np.sin(ang) * v
        inv_G = 4.0 * np.einsum("ijkl,i,j,k,l->", Sijkl, n, m, n, m)
        vals.append(1.0 / inv_G)
    return float(np.max(vals)), float(np.min(vals))


def directional_poisson_extrema(Sijkl: np.ndarray, n: np.ndarray, n_samples: int = 90):
    """nu(n, m) = -S_ijkl n_i n_j m_k m_l * E(n), for m in the plane
    perpendicular to n. Returns (nu_max, nu_min).
    """
    n = n / np.linalg.norm(n)
    E_n = directional_youngs_modulus(Sijkl, n)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    vals = []
    for k in range(n_samples):
        ang = np.pi * k / n_samples
        m = np.cos(ang) * u + np.sin(ang) * v
        s_proj = np.einsum("ijkl,i,j,k,l->", Sijkl, n, n, m, m)
        nu = -s_proj * E_n
        vals.append(nu)
    return float(np.max(vals)), float(np.min(vals))


# --------------------------------------------------------------------------
# Wave velocities & power flow angle (Christoffel equation)
# --------------------------------------------------------------------------
#
# Gamma_ik(n) = C_ijkl n_j n_l   (Christoffel matrix, GPa)
# Eigen-problem: Gamma u = rho v_p^2 u
#   -> 3 eigenvalues/eigenvectors: 1 quasi-longitudinal (P) + 2 quasi-shear
#      (S1 "fast", S2 "slow") modes for a general direction n.
#
# Units: with C in GPa and density rho in g/cm^3,
#   v_p [km/s] = sqrt(C[GPa] / rho[g/cm^3])   (GPa/(g/cm^3) = (km/s)^2 exactly)
#
# Group velocity (energy velocity) vector, standard result (e.g. Auld,
# "Acoustic Fields and Waves in Solids"):
#   v_g,l = (1 / (rho * v_p)) * C_ijkl * u_i * u_k * n_j
# Power flow angle = angle between v_g and n (the phase/wavevector direction).

def christoffel_matrix(Cijkl: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.einsum("ijkl,j,l->ik", Cijkl, n, n)


def christoffel_modes(Cijkl: np.ndarray, n: np.ndarray, density: float):
    """Returns a list of 3 modes sorted fastest-first, each a dict with
    velocity (km/s), polarization vector, group-velocity vector, and the
    power flow angle (degrees) between phase and group velocity.
    """
    n = n / np.linalg.norm(n)
    Gamma = christoffel_matrix(Cijkl, n)
    eigvals, eigvecs = np.linalg.eigh(Gamma)  # ascending
    order = np.argsort(eigvals)[::-1]         # fastest (largest eigenvalue) first
    modes = []
    for idx in order:
        lam = max(eigvals[idx], 0.0)
        v_p = float(np.sqrt(lam / density))
        u = eigvecs[:, idx]
        u = u / np.linalg.norm(u)
        # group velocity vector (km/s)
        v_g_vec = np.einsum("ijkl,i,k,j->l", Cijkl, u, u, n) / (density * v_p) if v_p > 0 else np.zeros(3)
        v_g_mag = float(np.linalg.norm(v_g_vec))
        cos_angle = np.clip(np.dot(v_g_vec, n) / v_g_mag, -1.0, 1.0) if v_g_mag > 0 else 1.0
        power_flow_angle = float(np.degrees(np.arccos(cos_angle)))
        modes.append({
            "velocity": v_p,
            "polarization": u.tolist(),
            "group_velocity": v_g_vec.tolist(),
            "group_velocity_magnitude": v_g_mag,
            "power_flow_angle_deg": power_flow_angle,
        })
    return modes  # [P (quasi-longitudinal), S_fast, S_slow]


# --------------------------------------------------------------------------
# 3D surface mesh generation (for Plotly visualization)
# --------------------------------------------------------------------------

PROPERTY_FUNCS = {
    "youngs_modulus": ("Young's Modulus", "GPa"),
    "linear_compressibility": ("Linear Compressibility", "TPa\u207b\u00b9"),
    "shear_modulus_max": ("Shear Modulus (max)", "GPa"),
    "shear_modulus_min": ("Shear Modulus (min)", "GPa"),
    "poisson_max": ("Poisson's Ratio (max)", ""),
    "poisson_min": ("Poisson's Ratio (min)", ""),
    "velocity_p": ("P-wave (quasi-longitudinal) velocity", "km/s"),
    "velocity_s_fast": ("S-wave velocity (fast)", "km/s"),
    "velocity_s_slow": ("S-wave velocity (slow)", "km/s"),
    "power_flow_angle_p": ("Power flow angle, P mode", "deg"),
    "power_flow_angle_s_fast": ("Power flow angle, S-fast mode", "deg"),
    "power_flow_angle_s_slow": ("Power flow angle, S-slow mode", "deg"),
}

_VELOCITY_PROPERTIES = {
    "velocity_p": 0, "velocity_s_fast": 1, "velocity_s_slow": 2,
    "power_flow_angle_p": 0, "power_flow_angle_s_fast": 1, "power_flow_angle_s_slow": 2,
}


def build_directional_surface(
    C: np.ndarray,
    property_name: str,
    n_theta: int = 60,
    n_phi: int = 120,
    density: Optional[float] = None,
):
    """Sample a directional property over the full sphere and return a
    dict with x,y,z coordinates (scaled by the property value, so the
    surface shape shows anisotropy), the raw value grid for coloring, and
    theta/phi grids in degrees (for an equirectangular heatmap view).

    `density` (g/cm^3) is required for the velocity_* / power_flow_angle_*
    properties.
    """
    if property_name not in PROPERTY_FUNCS:
        raise ValueError(f"Unknown property '{property_name}'")

    is_velocity_prop = property_name in _VELOCITY_PROPERTIES
    if is_velocity_prop:
        if not density or density <= 0:
            raise ValueError(
                "A positive density (g/cm^3) is required for velocity / "
                "power-flow-angle properties."
            )
        Cijkl = stiffness_tensor_from_voigt(C)
        mode_idx = _VELOCITY_PROPERTIES[property_name]
        is_power_flow = property_name.startswith("power_flow_angle")
    else:
        S = np.linalg.inv(C)
        Sijkl = compliance_tensor_from_voigt(S)

    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)

    X = np.zeros((n_theta, n_phi))
    Y = np.zeros((n_theta, n_phi))
    Z = np.zeros((n_theta, n_phi))
    R = np.zeros((n_theta, n_phi))

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            n = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            if property_name == "youngs_modulus":
                val = directional_youngs_modulus(Sijkl, n)
            elif property_name == "linear_compressibility":
                val = abs(directional_linear_compressibility(Sijkl, n))
            elif property_name == "shear_modulus_max":
                val, _ = directional_shear_modulus_extrema(Sijkl, n)
            elif property_name == "shear_modulus_min":
                _, val = directional_shear_modulus_extrema(Sijkl, n)
            elif property_name == "poisson_max":
                val, _ = directional_poisson_extrema(Sijkl, n)
            elif property_name == "poisson_min":
                _, val = directional_poisson_extrema(Sijkl, n)
            elif is_velocity_prop:
                modes = christoffel_modes(Cijkl, n, density)
                mode = modes[mode_idx]
                val = mode["power_flow_angle_deg"] if is_power_flow else mode["velocity"]
            else:
                val = 0.0

            r = abs(val)
            R[i, j] = val
            X[i, j] = r * n[0]
            Y[i, j] = r * n[1]
            Z[i, j] = r * n[2]

    label, unit = PROPERTY_FUNCS[property_name]
    theta_deg = np.degrees(thetas).tolist()
    phi_deg = np.degrees(phis).tolist()
    return {
        "property": property_name,
        "label": label,
        "unit": unit,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "x": X.tolist(),
        "y": Y.tolist(),
        "z": Z.tolist(),
        "value": R.tolist(),
        "min": float(np.min(R)),
        "max": float(np.max(R)),
    }


# --------------------------------------------------------------------------
# hkl-direction lookup (point evaluation of any directional property)
# --------------------------------------------------------------------------
#
# Converts Miller indices (h,k,l) to a Cartesian unit direction and
# evaluates every directional property at that single direction. Exact
# for orthogonal crystal systems (cubic, tetragonal, orthorhombic), where
# the crystallographic axes coincide with Cartesian x,y,z, so (h,k,l)
# maps directly to the Cartesian vector (h,k,l) normalized. For
# non-orthogonal systems (hexagonal, trigonal, monoclinic, triclinic)
# this is an approximation unless lattice angles are also supplied, which
# this app does not currently collect -- this is called out explicitly
# in the API response.

def hkl_to_direction(h: float, k: float, l: float) -> np.ndarray:
    v = np.array([h, k, l], dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("(h, k, l) cannot be the zero vector.")
    return v / norm


def evaluate_at_direction(C: np.ndarray, n: np.ndarray, density: Optional[float] = None) -> dict:
    """Evaluate every directional property at a single direction n."""
    S = np.linalg.inv(C)
    Sijkl = compliance_tensor_from_voigt(S)
    n = n / np.linalg.norm(n)

    result = {
        "direction": n.tolist(),
        "youngs_modulus": directional_youngs_modulus(Sijkl, n),
        "linear_compressibility": directional_linear_compressibility(Sijkl, n),
    }
    shear_max, shear_min = directional_shear_modulus_extrema(Sijkl, n)
    result["shear_modulus_max"] = shear_max
    result["shear_modulus_min"] = shear_min
    poisson_max, poisson_min = directional_poisson_extrema(Sijkl, n)
    result["poisson_max"] = poisson_max
    result["poisson_min"] = poisson_min

    if density and density > 0:
        Cijkl = stiffness_tensor_from_voigt(C)
        modes = christoffel_modes(Cijkl, n, density)
        result["velocity_p"] = modes[0]["velocity"]
        result["velocity_s_fast"] = modes[1]["velocity"]
        result["velocity_s_slow"] = modes[2]["velocity"]
        result["power_flow_angle_p"] = modes[0]["power_flow_angle_deg"]
        result["power_flow_angle_s_fast"] = modes[1]["power_flow_angle_deg"]
        result["power_flow_angle_s_slow"] = modes[2]["power_flow_angle_deg"]

    return result


# --------------------------------------------------------------------------
# Plane-slice cross-sections (2D polar cut through the 3D directional surface)
# --------------------------------------------------------------------------

_PLANE_NORMALS = {
    "xy": np.array([0.0, 0.0, 1.0]),
    "xz": np.array([0.0, 1.0, 0.0]),
    "yz": np.array([1.0, 0.0, 0.0]),
}


def build_plane_slice(
    C: np.ndarray,
    property_name: str,
    plane: str = "xy",
    n_points: int = 180,
    density: Optional[float] = None,
):
    """Sample a directional property around a full 360-degree sweep within
    a chosen crystallographic plane (xy, xz, or yz), returning a polar
    (angle, value) curve -- i.e. a 2D cross-section through the full 3D
    anisotropic surface.
    """
    if plane not in _PLANE_NORMALS:
        raise ValueError(f"Unknown plane '{plane}'. Choose one of: {list(_PLANE_NORMALS)}")
    if property_name not in PROPERTY_FUNCS:
        raise ValueError(f"Unknown property '{property_name}'")

    normal = _PLANE_NORMALS[plane]
    # build an orthonormal in-plane basis (u, v)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    is_velocity_prop = property_name in _VELOCITY_PROPERTIES
    if is_velocity_prop:
        if not density or density <= 0:
            raise ValueError("A positive density (g/cm^3) is required for velocity / power-flow-angle properties.")
        Cijkl = stiffness_tensor_from_voigt(C)
        mode_idx = _VELOCITY_PROPERTIES[property_name]
        is_power_flow = property_name.startswith("power_flow_angle")
    else:
        S = np.linalg.inv(C)
        Sijkl = compliance_tensor_from_voigt(S)

    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    values = np.zeros_like(angles)
    for idx, ang in enumerate(angles):
        n = np.cos(ang) * u + np.sin(ang) * v
        if property_name == "youngs_modulus":
            val = directional_youngs_modulus(Sijkl, n)
        elif property_name == "linear_compressibility":
            val = abs(directional_linear_compressibility(Sijkl, n))
        elif property_name == "shear_modulus_max":
            val, _ = directional_shear_modulus_extrema(Sijkl, n)
        elif property_name == "shear_modulus_min":
            _, val = directional_shear_modulus_extrema(Sijkl, n)
        elif property_name == "poisson_max":
            val, _ = directional_poisson_extrema(Sijkl, n)
        elif property_name == "poisson_min":
            _, val = directional_poisson_extrema(Sijkl, n)
        elif is_velocity_prop:
            modes = christoffel_modes(Cijkl, n, density)
            mode = modes[mode_idx]
            val = mode["power_flow_angle_deg"] if is_power_flow else mode["velocity"]
        else:
            val = 0.0
        values[idx] = val

    label, unit = PROPERTY_FUNCS[property_name]
    return {
        "plane": plane,
        "property": property_name,
        "label": label,
        "unit": unit,
        "angle_deg": np.degrees(angles).tolist(),
        "value": values.tolist(),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# --------------------------------------------------------------------------
# Legacy export formats: VRML (.wrl), gnuplot (.gnu), Grace/xmgrace (.agr)
# --------------------------------------------------------------------------
# These mirror the original ElATools tool's wrl_*, gnu_*, and agr_* output
# modules, which all just re-render the same computed property data (mesh
# surfaces or angular curves) in a different legacy plotting format.

def _viridis_approx(t: float) -> tuple:
    """A lightweight 4-stop approximation of the Viridis colormap, t in [0,1]."""
    stops = [
        (0.0, (0.267, 0.005, 0.329)),
        (0.33, (0.229, 0.322, 0.545)),
        (0.66, (0.128, 0.567, 0.551)),
        (1.0, (0.993, 0.906, 0.144)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(c0[i] + frac * (c1[i] - c0[i]) for i in range(3))
    return stops[-1][1]


def export_surface_wrl(mesh: dict) -> str:
    """Export a 3D directional-property surface (from build_directional_surface)
    as a VRML97 (.wrl) IndexedFaceSet -- viewable in legacy VRML viewers or
    CAD software with VRML import."""
    X, Y, Z, V = mesh["x"], mesh["y"], mesh["z"], mesh["value"]
    n_theta = len(X)
    n_phi = len(X[0])
    vmin, vmax = mesh["min"], mesh["max"]
    vrange = (vmax - vmin) or 1.0

    points = []
    colors = []
    for i in range(n_theta):
        for j in range(n_phi):
            points.append((X[i][j], Y[i][j], Z[i][j]))
            t = (V[i][j] - vmin) / vrange
            colors.append(_viridis_approx(t))

    def idx(i, j):
        return i * n_phi + j

    coord_index = []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            coord_index.append(f"{a} {b} {c} {d} -1")

    points_str = ",\n        ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in points)
    colors_str = ",\n        ".join(f"{r:.3f} {g:.3f} {b:.3f}" for r, g, b in colors)
    coord_index_str = ",\n        ".join(coord_index)

    return f"""#VRML V2.0 utf8
# ElATools Web -- {mesh['label']} ({mesh['unit']})
# min {mesh['min']:.4f}, max {mesh['max']:.4f}

Shape {{
  appearance Appearance {{
    material Material {{ diffuseColor 0.8 0.8 0.8 }}
  }}
  geometry IndexedFaceSet {{
    coord Coordinate {{
      point [
        {points_str}
      ]
    }}
    coordIndex [
        {coord_index_str}
    ]
    color Color {{
      color [
        {colors_str}
      ]
    }}
    colorPerVertex TRUE
    solid FALSE
  }}
}}
"""


def export_surface_gnu(mesh: dict) -> str:
    """Export a 3D directional-property surface as a gnuplot script (.gnu)
    with the grid data embedded inline -- run with `gnuplot file.gnu`."""
    X, Y, Z, V = mesh["x"], mesh["y"], mesh["z"], mesh["value"]
    n_theta = len(X)
    n_phi = len(X[0])

    lines = []
    for i in range(n_theta):
        for j in range(n_phi):
            lines.append(f"{X[i][j]:.5f} {Y[i][j]:.5f} {Z[i][j]:.5f} {V[i][j]:.5f}")
        lines.append("")  # blank line between grid rows, required by gnuplot pm3d/splot grid format

    data_block = "\n".join(lines)

    return f"""# ElATools Web -- gnuplot script for {mesh['label']} ({mesh['unit']})
# Run with: gnuplot {mesh['property']}.gnu
set title "{mesh['label']} ({mesh['unit']})"
set xlabel "x"
set ylabel "y"
set zlabel "z"
set hidden3d
set pm3d
set palette rgbformulae 33,13,10
set view equal xyz
splot '-' using 1:2:3:4 with pm3d title "{mesh['label']}"
{data_block}
e
pause -1 "Press enter to close"
"""


def export_curve_agr(angle_deg, value, title: str, xlabel: str = "angle (deg)", ylabel: str = "") -> str:
    """Export a 2D angular property curve as a Grace/xmgrace project file (.agr)."""
    data_lines = "\n".join(f"{a:.4f} {v:.6f}" for a, v in zip(angle_deg, value))
    return f"""# Grace project file
#
@version 50122
@page size 792, 612
@g0 on
@g0 hidden false
@g0 type XY
@with g0
@    title "{title}"
@    xaxis label "{xlabel}"
@    yaxis label "{ylabel}"
@    s0 line color 2
@    s0 line linewidth 2.0
@target G0.S0
@type xy
{data_lines}
&
"""
