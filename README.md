# ElATools Web

A from-scratch, deployable web interface for analyzing anisotropic elastic
properties of 3D crystalline materials -- inspired by the terminal-based
[ElATools](https://github.com/shahramyalameha/ElATools) Fortran tool, but
implemented independently in Python + a browser frontend so it can run as
an ordinary web app (no Fortran compiler, no MKL/LAPACK, no interactive
stdin) and be hosted publicly.

Given a 6x6 stiffness tensor (Cij, GPa) for a 3D crystal, a 3x3 in-plane
stiffness matrix (Cij-2D, N/m) for a 2D/monolayer material, or a 2x2
matrix for a 1D nanotube/nanowire, it computes:

**3D materials:**
- Bulk, shear, Young's, P-wave, and Lame's (1st & 2nd) moduli, Poisson's
  ratio, Pugh's ratio, machinability index, Gruneisen constant, and
  thermal expansion estimate -- all via Voigt / Reuss / Hill averaging
- 6 published hardness models (Teter, Tian, Chen, etc.) with guidance on
  which model is most reliable for metals/semiconductors/insulators
- Clarke minimum thermal conductivity (given density and average atomic mass)
- Kleinman parameter (bond-bending vs. bond-stretching character)
- Universal anisotropy index (AU), log-Euclidean anisotropy (AL),
  Chung-Buessem index (Ac), Cauchy pressure
- Mechanical stability (eigenvalue positivity of the stiffness matrix)
- Qualitative ductility/brittleness and bonding-character notes
- Interactive 3D surface, 2D theta/phi heatmap, or a 2D polar plane-slice
  (xy/xz/yz cross-section) of any directional property: Young's modulus,
  linear compressibility, shear modulus max/min, Poisson's ratio max/min
- Wave (phase) velocities -- P, S-fast, S-slow -- and power flow angle
  (deviation between phase and group velocity), via the Christoffel
  equation, given a material density; available as full 3D surfaces,
  heatmaps, plane slices, and single-direction point lookups
- A single-direction lookup by Miller indices (h k l) for any of the
  above properties
- Flexible input: paste/upload a plain Cij.dat-style matrix (also accepts
  INVELC-matrix and ELADAT files transparently, since they're the same
  36-number block with different headers), or enter WIEN2k
  elast.output-style "C11=... C12=..." keyed values for any of 5 crystal
  symmetries
- A real, bundled **offline database of 13,122 Materials-Project-computed
  elastic tensors** (see "The offline database" below) -- search and load
  instantly with no API key
- Live lookup of real DFT-computed elastic tensors from the official
  Materials Project API (with your own free API key), for materials not
  in the bundled snapshot

**2D materials:**
- In-plane Young's moduli (Ex, Ey), shear modulus, Poisson's ratios,
  Voigt/Reuss area & shear moduli, three published 2D anisotropy indices
- Angular (polar-plot) sweeps of Young's modulus, shear modulus,
  Poisson's ratio, and linear compressibility
- In-plane wave velocities (quasi-longitudinal and quasi-transverse),
  given the material's areal (2D) mass density
- Mechanical stability check

**1D materials (nanotubes/nanowires):**
- Axial Young's modulus, shear modulus, and Poisson's ratio from a 2x2
  stiffness matrix, plus a mechanical-stability check

**Downloads:** JSON / CSV reports and a self-contained standalone HTML
report (tables + interactive 3D plot, viewable offline) for 3D materials.

## The offline database

The original ElATools repo bundles a ~16 MB `db/` folder containing
`Cijs.binery` (a custom, completely undocumented text format) and
`All_2ID_cop.csv` (13,122 Materials Project / MVL IDs). Earlier versions
of this project declined to use it, since a wrong parse would silently
ship incorrect reference data for real materials.

That file format has since been fully reverse-engineered and validated:
it's one IEEE-754 single-precision float per line, written as a
right-justified 16-character hex field (Fortran's `Z16` edit descriptor
applied to a `REAL(4)` variable) -- not a fixed-width block as it first
appeared. Every material occupies exactly 36 consecutive lines (a 6x6
matrix, row-major), in the same order as the IDs in `All_2ID_cop.csv`.

Validation performed before trusting this (see `backend/offline_db.py`
for the full writeup):
- All 13,122 decoded matrices are **exactly symmetric** to float
  precision -- real elastic tensors are symmetric by construction, so
  this would be extremely unlikely by chance if the byte alignment were
  even slightly wrong.
- 86.5% are positive-definite (mechanically stable), consistent with
  published literature noting a known fraction of unreliable entries in
  the Materials Project's elasticity dataset from this era.
- Spot checks against well-known Materials Project IDs match published
  values closely -- e.g. `mp-66` (diamond) decodes to C11=1054, C12=126,
  C44=562 GPa against literature values of roughly 1076, 125, and
  562-577 GPa; `mp-13` (BCC iron) decodes to values in the expected range
  for that material.

The pre-extracted, human-readable result ships as
`backend/data/mp_elastic_db.csv` (~1.4 MB) and is loaded into memory at
startup -- no re-parsing of the original binary format at request time.

## Architecture

```
elatools-web/
├── backend/
│   ├── elastic_core.py     # 3D physics: Voigt-Reuss-Hill, hardness, velocities, hkl/slices
│   ├── elastic_core_2d.py  # 2D-materials physics, incl. in-plane wave velocities
│   ├── elastic_core_1d.py  # 1D nanotube/nanowire physics
│   ├── offline_db.py        # bundled 13,122-material real elastic-tensor database
│   ├── mp_client.py         # Materials Project live API client
│   ├── main.py               # FastAPI app: REST API + serves the frontend
│   ├── validate.py           # validation script vs. the original tool's output
│   ├── data/mp_elastic_db.csv  # pre-extracted offline database (~1.4 MB)
│   └── requirements.txt
├── frontend/
│   └── index.html           # single-page app (vanilla JS + Plotly.js via CDN)
├── examples/                 # example Cij / Cij-2D matrices
├── Dockerfile
└── README.md (this file)
```

This is a normal client-server web app: the Python backend does the tensor
math and serves a REST API; the frontend is a static page that calls it and
renders results with Plotly. There is no compiled binary and no subprocess
execution anywhere, which is what makes this safe and simple to host
publicly (see "Notes on public hosting" below).

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000 in a browser. The frontend is served
directly by the same FastAPI app (no separate frontend server needed).

## Running the validation script

```bash
cd backend
python3 validate.py
```

This re-derives every scalar and directional property for the GaAs
example shipped with the original ElATools repo and compares them against
that repo's own recorded output (`example/GaAs/DATA.out`). All checks
pass except two, which are documented next.

## Validation notes (read this if the numbers look "off" vs. the original tool)

While building this, I validated every formula against the real example
output shipped in the original ElATools repository. Nearly everything
matches to 3-4 decimal places. Two values differ slightly, and tracing
them into the *original Fortran source* (not just its example output)
showed the original repo's shipped example was generated by an
older/inconsistent code path:

1. **Hill-averaged Young's modulus (E_H).** The shipped example output
   used the arithmetic mean of E_V and E_R. The current upstream Fortran
   source (`soc/Eatools_proelast.f90`) instead derives E_H from
   `(K_H, G_H)` via the standard `1/E = 1/(3G) + 1/(9K)` relation -- which
   is what this project does, and it is the more standard/self-consistent
   approach (K_H and G_H are themselves derived the same way for Voigt and
   Reuss). The difference is small (~0.03 GPa on a ~116 GPa value).

2. **Log-Euclidean anisotropy parameter (AL).** The upstream source
   declares a variable `akl` for this and prints it, but never assigns it
   anywhere in `Eatools_proelast.f90` -- an uninitialized-variable bug in
   the shipped code. This project instead implements the actual published
   formula the source code cites in its own comments: Kube, *AIP Advances*
   6, 095209 (2016): `AL = sqrt( ln(K_V/K_R)^2 + 5*ln(G_V/G_R)^2 )`.

All Voigt/Reuss scalar properties, the universal anisotropy index (AU),
the Chung-Buessem index (Ac), and all directional (3D surface) extrema for
Young's modulus and shear modulus match the original tool's real output to
within numerical/mesh-resolution precision.

## Deploying publicly

Because this reimplements the physics in pure Python/numpy rather than
shelling out to a compiled binary, hosting it is just hosting a normal
containerized web app -- no sandboxing of a subprocess, no stdin-injection
concerns, no compiler toolchain needed on the host.

**Docker (works on any host):**

```bash
docker build -t elatools-web .
docker run -p 8000:8000 elatools-web
```

**Recommended platforms for a free/cheap public deployment:**

- **Fly.io / Render / Railway** -- push this repo, point them at the
  Dockerfile, done. All have free or very cheap tiers for a small app like
  this.
- **Google Cloud Run** -- deploys arbitrary containers, scales to zero
  when idle (so it's free when no one's using it), has a request timeout
  built in.

Whichever you choose, two small production hardenings are worth adding
before opening it to the public internet:

1. **Rate limiting** on `/api/*` (e.g. via `slowapi` or your platform's
   built-in rate limiting), since the 3D surface computation at "high"
   resolution is the most CPU-intensive endpoint.
2. **A request size/timeout limit**, mostly to bound the "high resolution"
   surface option (160x80 mesh = 12,800 tensor contractions -- fast, but
   worth capping so no single request can be abused).

Neither is required to make it work; they're just good manners for a
public, unauthenticated endpoint.

## Legacy export formats

VRML (`.wrl`), gnuplot (`.gnu`), and Grace/xmgrace (`.agr`) exports are
implemented for the 3D directional properties (and `.agr` for the 2D
angular curves too), matching the original tool's `wrl_*`, `gnu_*`, and
`agr_*` output modules. All three were verified for real, not just
written and assumed correct:
- The `.gnu` script was actually run through real `gnuplot` and rendered
  to a PNG, producing the expected cubic-anisotropy "pillowed cube"
  shape for Young's modulus.
- The `.wrl` file was checked for balanced VRML97 syntax, and every
  `coordIndex` reference was confirmed to point at a valid vertex.
- The `.agr` file follows the standard Grace project-file structure
  (`@version`, `@g0`, `@target G0.S0`, `@type xy`).

## What's still approximated, and why

- **Non-orthogonal hkl-to-Cartesian conversion** -- the (h k l) direction
  lookup is exact for orthogonal crystal systems (cubic, tetragonal,
  orthorhombic) where crystallographic axes align with Cartesian x,y,z.
  For hexagonal, trigonal, monoclinic, or triclinic systems, true
  conversion needs the real-space lattice parameters (a, b, c, alpha,
  beta, gamma), which aren't collected as input; the API response notes
  this limitation explicitly rather than silently giving an approximate
  answer as if it were exact.
- **A documented Fortran bug, fixed rather than copied:** the original
  2D wave-velocity code (`Eatools_wave_cal2d.f90`) builds its stiffness
  tensor in a way that never actually incorporates the in-plane shear
  constant (C33/C66) into the off-diagonal tensor components -- it
  reuses C12 there instead, which looks like a copy-paste bug. This
  project's 2D velocity code uses the standard, physically correct
  tensor construction instead (and the results independently match
  published graphene sound-velocity literature: ~21.3 km/s
  quasi-longitudinal, ~14.1 km/s quasi-transverse).

Everything else from the original tool's scope -- all input formats,
all scalar properties, all directional visualizations for 3D, 2D, and 1D
materials, both the offline and live materials databases, and all three
legacy export formats -- is implemented.
