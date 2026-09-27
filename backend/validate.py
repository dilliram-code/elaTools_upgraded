"""
validate.py
===========
Sanity-check script: compares this project's computed elastic properties
against the reference values shipped in the original ElATools repository
(example/GaAs/DATA.out), for the GaAs cubic crystal.

Run with:  python3 validate.py

Two known discrepancies vs. the *shipped example output* are expected and
documented (see README.md "Validation notes"):
  - E_H (Hill Young's modulus): shipped example used avg(E_V, E_R) directly;
    this code derives E_H self-consistently from (K_H, G_H), which is the
    formula actually present in the current upstream Fortran source.
  - AL (log-Euclidean anisotropy): the upstream Fortran variable printed for
    "AL" (called `akl`) is declared but never assigned in the current
    source (soc/Eatools_proelast.f90) -- an upstream bug. This code instead
    implements the correctly-cited formula from Kube, AIP Advances 6,
    095209 (2016), as the source code's own comment specifies.
"""
import elastic_core as ec

REFERENCE = dict(
    K_V=75.467, K_R=75.467, K_H=75.467,
    G_V=48.640, G_R=44.626, G_H=46.633,
    E_V=120.114, E_R=111.833,
    nu_V=0.2347, nu_R=0.2530, nu_H=0.2439,
    pugh_V=1.5515, pugh_R=1.6911, pugh_H=1.6183,
    M_V=140.3200, M_R=134.9672, M_H=137.6436,
    AU=0.4498, Ac=0.0430,
)

# From DATA.out's directional Young's / shear modulus extrema (plane 1,0,0):
DIRECTIONAL_REFERENCE = dict(
    youngs_max=141.14, youngs_min=85.26,
    shear_max=59.40, shear_min=32.51,
)


def main():
    C = ec.parse_cij_text(open("../examples/GaAs_Cij.dat").read())
    props = ec.compute_scalar_properties(C)

    print("Scalar properties vs. reference (GaAs, cubic):")
    print(f"{'property':10s} {'computed':>12s} {'reference':>12s} {'match':>8s}")
    all_ok = True
    for k, ref in REFERENCE.items():
        mine = getattr(props, k)
        ok = abs(mine - ref) < 0.01
        all_ok &= ok
        print(f"{k:10s} {mine:12.4f} {ref:12.4f} {'OK' if ok else 'DIFF':>8s}")

    print()
    print("Directional (3D surface) properties vs. reference:")
    S = None
    mesh_y = ec.build_directional_surface(C, "youngs_modulus", n_theta=80, n_phi=160)
    mesh_smax = ec.build_directional_surface(C, "shear_modulus_max", n_theta=80, n_phi=160)
    mesh_smin = ec.build_directional_surface(C, "shear_modulus_min", n_theta=80, n_phi=160)
    computed = dict(
        youngs_max=mesh_y["max"], youngs_min=mesh_y["min"],
        shear_max=mesh_smax["max"], shear_min=mesh_smin["min"],
    )
    for k, ref in DIRECTIONAL_REFERENCE.items():
        mine = computed[k]
        ok = abs(mine - ref) < 0.5
        all_ok &= ok
        print(f"{k:12s} {mine:12.4f} {ref:12.4f} {'OK' if ok else 'DIFF':>8s}")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS DIFFER (see docstring notes)")


if __name__ == "__main__":
    main()
