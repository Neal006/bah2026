# BAH 2026 — Problem Statement 9
# End-to-End Implementation Plan: SH-WFS Pipeline
# Deadline: July 1, 2026 (Idea Submission)

---

## System Constants (locked from synthetic dataset)

```
n_lenslets       = 10×10    (100 subapertures total, 80 active inside circular pupil)
n_slopes         = 160      (2 × 80 — x and y per active subaperture)
n_modes          = 21       (Zernike Z1–Z21, Noll ordering)
G shape          = (160, 21) — interaction matrix
G† shape         = (21, 160) — pre-computed pseudoinverse (offline)
pupil_grid       = 64×64
n_actuators      = 11×11 = 121 (Fried geometry, staggered from lenslet grid)
C_coupling shape = (121, 121)
pix              = 5.5 µm   (pixel pitch)
f_mla            = 18.6 mm  (lenslet focal length)
pitch            = 150 µm   (lenslet pitch)
pps              = 16       (pixels per subaperture side)
wl               = 632.8 nm (wavelength)
fps              = 200 Hz   → 5.0 ms budget per frame (HARD real-time constraint)
stroke_limit     = ±5 µm
coupling_frac    = 0.15
```

---

## Directory Structure

```
shwfs_pipeline/
├── config.py                  # all system constants, loaded from metadata.json
├── 01_centroid.py             # centroiding: CoG, Weighted-CoG, subaperture mask
├── 02_reconstruct.py          # modal (Zernike LS) + zonal (Southwell)
├── 03_turbulence.py           # r0 estimation, tau0, PSD
├── 04_actuator.py             # DM command + coupling deconvolution
├── 05_validate.py             # diff against ground_truth/ npy files
├── 06_benchmark.py            # timing per stage, 200-frame stats
├── run_pipeline.py            # end-to-end runner, produces all outputs
├── precompute.py              # offline: build G, G†, C†, Zernike basis
├── c_ext/
│   ├── cog_centroid.c         # C implementation of CoG inner loop
│   ├── matmul.c               # C matrix-vector multiply (online step)
│   └── Makefile
├── outputs/
│   ├── phase_maps/            # frame_XXXX_phase.npy (64×64, float32)
│   ├── actuator_maps/         # frame_XXXX_act.npy  (11×11, float32, µm)
│   ├── turbulence_report.json # r0, tau0, bias error vs ground truth
│   ├── benchmark_report.json  # latency per stage
│   └── figures/               # PSD plot, residual plot, 3x3 panel
└── idea_submission/
    └── block_diagram.py       # pipeline diagram for the PDF
```

---

## Timeline

```
June 28 (today)  — Phase 0 + Phase 1    [Scaffolding + Centroiding]
June 29          — Phase 2              [Wavefront Reconstruction]
June 30          — Phase 3 + 4 + 5     [Turbulence + Actuator + C ext]
July 1 AM        — Phase 6 + 7         [Integration + Submission]
```

---

---

# PHASE 0 — Project Scaffolding
# Estimated time: 1 hour (June 28, start here)

---

## Step 0.1 — Create project directory and link synthetic data

```bash
mkdir -p shwfs_pipeline/{outputs/phase_maps,outputs/actuator_maps,outputs/figures,c_ext,idea_submission}
cd shwfs_pipeline
ln -s /path/to/shwfs_synthetic data   # symlink to the generated dataset
```

**Verify the link:**
```bash
ls data/frames/ | wc -l     # must print 200
ls data/ground_truth/        # must show all_slopes_x.npy, all_zernike.npy, etc.
```

---

## Step 0.2 — `config.py`

**Purpose:** Single source of truth for all system constants. Every module imports from here — no hardcoded magic numbers anywhere else.

```python
# config.py
import json, numpy as np
from pathlib import Path

DATA_DIR = Path("data")

# ── Load metadata from synthetic dataset ──────────────────────────────────────
with open(DATA_DIR / "metadata.json") as f:
    META = json.load(f)

# ── MLA ───────────────────────────────────────────────────────────────────────
N_LENSLETS_X   = META["mla"]["n_lenslets_x"]          # 10
N_LENSLETS_Y   = META["mla"]["n_lenslets_y"]          # 10
N_SA_ACTIVE    = META["mla"]["n_subapertures_active"]  # 80
PITCH_M        = META["mla"]["pitch_m"]                # 150e-6
F_MLA_M        = META["mla"]["focal_length_m"]         # 18.6e-3

# ── Camera ────────────────────────────────────────────────────────────────────
PIX_M          = META["camera"]["pixel_size_m"]        # 5.5e-6
PPS            = META["camera"]["pixels_per_subaperture"]  # 16
FRAME_W        = META["camera"]["frame_width_px"]      # 160
FRAME_H        = META["camera"]["frame_height_px"]     # 160
FPS            = META["camera"]["frame_rate_hz"]       # 200.0
DT_S           = 1.0 / FPS                            # 0.005

# ── Turbulence / optical ──────────────────────────────────────────────────────
WAVELENGTH_M   = META["turbulence"]["wavelength_m"]    # 632.8e-9
PUPIL_D_M      = META["pupil"]["diameter_m"]           # 1.5e-3

# ── DM ────────────────────────────────────────────────────────────────────────
N_ACT_X        = META["dm"]["n_actuators_x"]           # 11
N_ACT_Y        = META["dm"]["n_actuators_y"]           # 11
N_ACT          = META["dm"]["n_actuators_total"]       # 121
STROKE_M       = META["dm"]["stroke_m"]                # 5e-6
COUPLING_FRAC  = META["dm"]["coupling_fraction"]       # 0.15

# ── Reconstruction ────────────────────────────────────────────────────────────
N_MODES        = 21          # Zernike Z1–Z21 Noll
PUPIL_GRID     = 64          # wavefront map resolution (pixels)
N_SLOPES       = 2 * N_SA_ACTIVE   # 160

# ── Derived: slope → rad/m conversion ────────────────────────────────────────
SLOPE_SCALE    = PIX_M / F_MLA_M   # 0.000296 rad/m per pixel displacement
```

**Sanity check (run immediately):**
```bash
python3 -c "import config; print('N_SLOPES:', config.N_SLOPES, '  N_MODES:', config.N_MODES)"
# Expected: N_SLOPES: 160   N_MODES: 21
```

---

---

# PHASE 1 — Centroiding
# Estimated time: 3 hours (June 28)

---

## Step 1.1 — Subaperture grid definition

Each of the 10×10 subapertures occupies a 16×16 pixel window in the 160×160 frame.
The centre of subaperture (i, j) in pixel coordinates is:
```
cx_ref[i,j] = (j + 0.5) * PPS    →  8, 24, 40, ..., 152
cy_ref[i,j] = (i + 0.5) * PPS    →  8, 24, 40, ..., 152
```

**Active mask** — determined from reference frame intensity. Subapertures whose total
flux is below 10% of the maximum are flagged inactive (outside circular pupil).

```python
# In 01_centroid.py

import numpy as np
from PIL import Image
import config

def build_subaperture_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        sa_cx_ref : (NY, NX) float32 — reference centroid x per subaperture [px]
        sa_cy_ref : (NY, NX) float32 — reference centroid y per subaperture [px]
        active    : (NY, NX) bool    — True where subaperture is inside pupil
    """
    NY, NX = config.N_LENSLETS_Y, config.N_LENSLETS_X
    pps    = config.PPS

    sa_cx_ref = np.zeros((NY, NX), dtype=np.float32)
    sa_cy_ref = np.zeros((NY, NX), dtype=np.float32)

    for i in range(NY):
        for j in range(NX):
            sa_cx_ref[i, j] = (j + 0.5) * pps   # pixel x
            sa_cy_ref[i, j] = (i + 0.5) * pps   # pixel y

    return sa_cx_ref, sa_cy_ref   # active mask built in next step
```

## Step 1.2 — Active subaperture mask from reference frame

```python
def build_active_mask(ref_frame: np.ndarray, threshold_frac: float = 0.1) -> np.ndarray:
    """
    ref_frame : (H, W) uint8 — flat reference frame (sh_flat_ref.bmp)
    Returns   : (NY, NX) bool
    
    Logic: subapertures inside the circular pupil have total flux > threshold_frac × max_flux.
    """
    NY, NX = config.N_LENSLETS_Y, config.N_LENSLETS_X
    pps    = config.PPS
    active = np.zeros((NY, NX), dtype=bool)

    max_flux = 0.0
    fluxes   = np.zeros((NY, NX))

    for i in range(NY):
        for j in range(NX):
            y0, y1 = i * pps, (i+1) * pps
            x0, x1 = j * pps, (j+1) * pps
            patch  = ref_frame[y0:y1, x0:x1].astype(np.float32)
            fluxes[i, j] = patch.sum()

    threshold = threshold_frac * fluxes.max()
    active    = fluxes > threshold

    assert active.sum() == config.N_SA_ACTIVE, \
        f"Expected {config.N_SA_ACTIVE} active SA, got {active.sum()}"
    return active
```

**Why this matters:** With an exact circular pupil, this should return exactly 80 active
subapertures. If it doesn't, adjust `threshold_frac`. This mask is used by every
downstream module — propagate it everywhere as a parameter, never recompute.

## Step 1.3 — Centre-of-Gravity centroiding (numpy baseline)

**Math:**
```
For each subaperture (i, j):
    Extract patch P[16×16]
    Subtract background threshold B (from sh_flat_bg.bmp mean per subaperture)
    Clip negative values to zero
    x_c = Σ_u Σ_v (u · P[v,u]) / Σ_u Σ_v P[v,u]   [pixel, within subaperture]
    y_c = Σ_u Σ_v (v · P[v,u]) / Σ_u Σ_v P[v,u]
    Convert to global frame: x_c_global = j*PPS + x_c
```

```python
def cog_centroid(frame: np.ndarray,
                 bg_frame: np.ndarray,
                 active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    frame    : (H, W) uint8  — turbulated frame
    bg_frame : (H, W) uint8  — background/dark frame
    active   : (NY, NX) bool
    
    Returns:
        cx : (NY, NX) float32  — centroid x [px], 0 for inactive
        cy : (NY, NX) float32  — centroid y [px], 0 for inactive
    """
    NY, NX = config.N_LENSLETS_Y, config.N_LENSLETS_X
    pps    = config.PPS
    cx     = np.zeros((NY, NX), dtype=np.float32)
    cy     = np.zeros((NY, NX), dtype=np.float32)

    # Pre-build pixel coordinate grids within a subaperture [0..PPS-1]
    u_grid, v_grid = np.meshgrid(np.arange(pps, dtype=np.float32),
                                  np.arange(pps, dtype=np.float32))

    for i in range(NY):
        for j in range(NX):
            if not active[i, j]:
                continue
            y0, y1 = i * pps, (i+1) * pps
            x0, x1 = j * pps, (j+1) * pps

            patch  = frame[y0:y1, x0:x1].astype(np.float32)
            bg     = bg_frame[y0:y1, x0:x1].astype(np.float32)
            P      = np.clip(patch - bg.mean(), 0, None)  # threshold subtract

            flux = P.sum()
            if flux < 1e-6:   # degenerate — dark subaperture
                cx[i, j] = j * pps + pps / 2.0
                cy[i, j] = i * pps + pps / 2.0
                continue

            # local centroid within patch
            lx = (u_grid * P).sum() / flux
            ly = (v_grid * P).sum() / flux

            # global pixel coordinate
            cx[i, j] = j * pps + lx
            cy[i, j] = i * pps + ly

    return cx, cy
```

## Step 1.4 — Slope computation

```python
def compute_slopes(cx: np.ndarray, cy: np.ndarray,
                   cx_ref: np.ndarray, cy_ref: np.ndarray,
                   active: np.ndarray) -> np.ndarray:
    """
    Converts pixel displacement to slope in rad/m.
    
    Returns:
        slopes : (N_SLOPES,) float32
            First N_SA_ACTIVE elements  → slope_x for each active subaperture
            Next  N_SA_ACTIVE elements  → slope_y for each active subaperture
    
    Physics:
        Δx [px] → slope_x [rad/m] = Δx * PIX_M / F_MLA_M
    """
    sc      = config.SLOPE_SCALE   # PIX_M / F_MLA_M = 0.000296 rad/m per px
    dx      = (cx - cx_ref) * sc   # (NY, NX) rad/m
    dy      = (cy - cy_ref) * sc

    sx_flat = dx[active].astype(np.float32)   # (80,)
    sy_flat = dy[active].astype(np.float32)   # (80,)

    return np.concatenate([sx_flat, sy_flat])  # (160,)
```

## Step 1.5 — Integration test for centroiding

```python
# Run this immediately after writing the module
if __name__ == "__main__":
    from PIL import Image
    import config

    ref_frame = np.array(Image.open(config.DATA_DIR / "sh_flat_ref.bmp"))
    bg_frame  = np.array(Image.open(config.DATA_DIR / "sh_flat_bg.bmp"))

    active = build_active_mask(ref_frame)
    print(f"Active subapertures: {active.sum()}")   # must be 80

    cx_ref, cy_ref = build_subaperture_grid()

    # On the reference frame itself, centroids should be near reference positions
    cx0, cy0 = cog_centroid(ref_frame, bg_frame, active)
    dx_ref = (cx0 - cx_ref)[active]
    dy_ref = (cy0 - cy_ref)[active]
    print(f"Reference centroid error: {np.abs(dx_ref).mean():.3f} ± {np.abs(dx_ref).std():.3f} px")
    # Expected: < 0.5 px mean error

    # On a turbulated frame
    f0 = np.array(Image.open(config.DATA_DIR / "frames/frame_0000.bmp"))
    cx0t, cy0t = cog_centroid(f0, bg_frame, active)
    slopes = compute_slopes(cx0t, cy0t, cx_ref, cy_ref, active)
    print(f"Slope vector shape:   {slopes.shape}")    # (160,)
    print(f"Slope RMS:            {np.std(slopes):.2f} rad/m")
    # Expected: ~1000 rad/m order (lab turbulence)
```

**Pass criteria:** active=80, centroid error on flat < 0.5 px, slope vector shape (160,).

---

---

# PHASE 2 — Wavefront Reconstruction
# Estimated time: 5 hours (June 28 evening → June 29)

This is the highest-value module. Two methods: modal (primary) and zonal (comparison).

---

## Step 2.1 — Zernike polynomial basis (precompute, offline)

**File:** `precompute.py`

Zernike polynomials on a 64×64 pupil grid, Noll ordering Z1–Z21.
Two things are needed:
(a) The basis itself: Z[k, y, x] — for reconstructing W(x,y) per frame.
(b) The x and y gradient of each mode, integrated over each subaperture — this is the interaction matrix G.

```python
# precompute.py

import numpy as np
import config

def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index j to (n, m) radial/azimuthal orders."""
    n = int(np.ceil((-3 + np.sqrt(9 + 8*(j-1))) / 2))
    j1 = j - n*(n+1)//2 - 1
    m  = (-1)**j * ((n % 2) + 2 * int((np.abs(j1) + (n % 2)) / 2))
    return n, m

def zernike_xy(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate Zernike polynomial Z_n^m at polar coords (rho, theta).
    rho: normalised to [0,1] over unit disk
    """
    from scipy.special import comb
    # Radial polynomial
    R = np.zeros_like(rho)
    for s in range((n - abs(m)) // 2 + 1):
        coeff = ((-1)**s * comb(n-s, s, exact=True) *
                 comb(n-2*s, (n-abs(m))//2 - s, exact=True))
        R += coeff * rho**(n - 2*s)

    # Angular part
    if m == 0:
        Z = np.sqrt(n+1) * R
    elif m > 0:
        Z = np.sqrt(2*(n+1)) * R * np.cos(m * theta)
    else:
        Z = np.sqrt(2*(n+1)) * R * np.sin(abs(m) * theta)

    return Z

def build_zernike_basis(n_modes: int, grid: int) -> np.ndarray:
    """
    Returns Z: (n_modes, grid, grid) float32
    Orthonormal Zernike polynomials on a unit disk.
    Points outside unit disk are set to 0.
    """
    g = grid
    x = np.linspace(-1, 1, g)
    y = np.linspace(-1, 1, g)
    XX, YY = np.meshgrid(x, y)
    rho    = np.sqrt(XX**2 + YY**2)
    theta  = np.arctan2(YY, XX)
    mask   = rho <= 1.0

    Z = np.zeros((n_modes, g, g), dtype=np.float32)
    for j in range(1, n_modes + 1):
        n, m = noll_to_nm(j)
        Zj   = zernike_xy(n, m, rho, theta)
        Zj[~mask] = 0.0
        Z[j-1] = Zj.astype(np.float32)

    return Z   # shape (21, 64, 64)
```

## Step 2.2 — Interaction matrix G (the heart of modal reconstruction)

**Math:** Each column of G corresponds to one Zernike mode. Each row pair corresponds
to one active subaperture (x-slope, y-slope). Element G[2i, k] is the mean x-gradient
of mode k over subaperture i.

```
G[2i,   k] = (1/A_i) ∫∫_{SA_i} ∂Z_k/∂x  dx dy
G[2i+1, k] = (1/A_i) ∫∫_{SA_i} ∂Z_k/∂y  dx dy
```

Numerically: finite-difference gradient on the 64×64 grid, then mean over the
subaperture's footprint on that grid.

```python
def build_interaction_matrix(Z_basis: np.ndarray,
                              active: np.ndarray) -> np.ndarray:
    """
    Z_basis : (N_MODES, GRID, GRID) float32
    active  : (NY, NX)  bool
    
    Returns G : (N_SLOPES, N_MODES) float32
        N_SLOPES = 2 * N_SA_ACTIVE = 160
        N_MODES  = 21
    
    Coordinate mapping:
        Zernike unit disk [-1,1] → pupil plane [0, N_LENSLETS] grid cells
        Each subaperture (i,j) maps to a (GRID/NY) × (GRID/NX) block on the grid.
    """
    N_MODES  = config.N_MODES
    NY, NX   = config.N_LENSLETS_Y, config.N_LENSLETS_X
    grid     = config.PUPIL_GRID         # 64
    n_active = int(active.sum())          # 80
    cell_y   = grid // NY                # 64/10 = 6 (round down)
    cell_x   = grid // NX

    G = np.zeros((2 * n_active, N_MODES), dtype=np.float32)

    # Compute x and y gradients for every mode (finite difference)
    dZdx = np.gradient(Z_basis, axis=2)   # (N_MODES, GRID, GRID)
    dZdy = np.gradient(Z_basis, axis=1)

    sa_idx = 0
    for i in range(NY):
        for j in range(NX):
            if not active[i, j]:
                continue
            # Pixel footprint of subaperture (i,j) on the 64×64 grid
            py0 = i * cell_y
            py1 = min(py0 + cell_y, grid)
            px0 = j * cell_x
            px1 = min(px0 + cell_x, grid)

            for k in range(N_MODES):
                G[sa_idx,            k] = dZdx[k, py0:py1, px0:px1].mean()
                G[n_active + sa_idx, k] = dZdy[k, py0:py1, px0:px1].mean()

            sa_idx += 1

    return G   # (160, 21)
```

## Step 2.3 — Pre-compute and save pseudoinverse G† (offline, once)

```python
def precompute_and_save():
    """Run once. Saves G, G†, Zernike basis, and DM coupling inverse."""
    import numpy as np, scipy.io as sio
    from pathlib import Path
    import config

    print("Building Zernike basis...")
    Z = build_zernike_basis(config.N_MODES, config.PUPIL_GRID)
    np.save("precomputed/Z_basis.npy", Z)

    # Need active mask
    from PIL import Image
    from centroid import build_active_mask, build_subaperture_grid
    ref = np.array(Image.open(config.DATA_DIR / "sh_flat_ref.bmp"))
    active = build_active_mask(ref)

    print("Building interaction matrix G (160, 21)...")
    G = build_interaction_matrix(Z, active)
    np.save("precomputed/G.npy", G)

    print("Computing pseudoinverse G† (21, 160)...")
    # Use SVD-based pinv with regularisation (truncate singular values < 1e-3 × max)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    s_thresh  = 1e-3 * s[0]
    s_inv     = np.where(s > s_thresh, 1.0/s, 0.0)
    G_pinv    = (Vt.T * s_inv) @ U.T    # (21, 160)
    np.save("precomputed/G_pinv.npy", G_pinv)
    print(f"G_pinv shape: {G_pinv.shape}")

    print("Loading DM coupling matrix and computing C†...")
    C      = np.load(config.DATA_DIR / "dm_coupling.npy")   # (121, 121)
    C_pinv = np.linalg.pinv(C)
    np.save("precomputed/C_pinv.npy", C_pinv)

    print("Done. Files in precomputed/:")
    print("  Z_basis.npy  (21, 64, 64)")
    print("  G.npy        (160, 21)")
    print("  G_pinv.npy   (21, 160)")
    print("  C_pinv.npy   (121, 121)")

if __name__ == "__main__":
    import os
    os.makedirs("precomputed", exist_ok=True)
    precompute_and_save()
```

**Run this once before anything else:**
```bash
mkdir precomputed
python3 precompute.py
```
Expected output: four `.npy` files. This is the offline phase — it can take 30–60 seconds.

## Step 2.4 — Online modal reconstruction (per-frame, must be fast)

```python
# In 02_reconstruct.py

import numpy as np
import config

# Load once at module init
_G_pinv  = np.load("precomputed/G_pinv.npy")   # (21, 160)
_Z_basis = np.load("precomputed/Z_basis.npy")   # (21, 64, 64)

def reconstruct_modal(slopes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    slopes : (160,) float32 — slope vector from centroiding
    
    Returns:
        zernike_coeffs : (21,)     float32 — Zernike coefficients [rad]
        phase_map      : (64, 64)  float32 — wavefront phase [rad]
    
    Online cost: one (21,160)×(160,) matmul + one (21, 64×64) weighted sum
    FLOP count: ~3360 + ~87000 ≈ 90K FLOPs (< 0.1 ms on any modern CPU)
    """
    a = _G_pinv @ slopes          # (21,) — Zernike coefficients

    # Wavefront: linear combination of Zernike basis functions
    # _Z_basis shape: (21, 64, 64)
    W = np.einsum('k,kij->ij', a, _Z_basis)   # (64, 64)

    return a.astype(np.float32), W.astype(np.float32)
```

## Step 2.5 — Zonal reconstruction (Southwell method, for comparison)

**Math:** The wavefront is computed on a grid of nodes. Each subaperture slope
links two adjacent nodes. Build a sparse finite-difference matrix D such that:

```
D · w = s   →   w = D† · s    (least-squares zonal solution)
```

The Southwell scheme places wavefront nodes at subaperture centres (not corners),
so it's a square 10×10 grid for a 10×10 lenslet array.

```python
def build_southwell_matrix(active: np.ndarray) -> np.ndarray:
    """
    Builds the finite-difference matrix D for Southwell zonal reconstruction.
    Shape: (N_SLOPES, N_SA_TOTAL) where N_SA_TOTAL = NY*NX = 100
    
    For each active subaperture (i,j):
        slope_x[i,j] ≈ (W[i, j+1] - W[i, j]) / d_sa
        slope_y[i,j] ≈ (W[i+1, j] - W[i, j]) / d_sa
    d_sa = PITCH_M (physical subaperture size)
    """
    NY, NX   = config.N_LENSLETS_Y, config.N_LENSLETS_X
    d_sa     = config.PITCH_M
    n_active = int(active.sum())
    n_nodes  = NY * NX   # 100

    # Two blocks: one for x-slopes, one for y-slopes
    Dx = np.zeros((n_active, n_nodes), dtype=np.float32)
    Dy = np.zeros((n_active, n_nodes), dtype=np.float32)

    sa_idx = 0
    for i in range(NY):
        for j in range(NX):
            if not active[i, j]:
                continue
            node_c = i * NX + j   # current node

            # x-slope: finite difference in j direction
            if j + 1 < NX:
                node_r = i * NX + (j+1)
                Dx[sa_idx, node_r] =  1.0 / d_sa
                Dx[sa_idx, node_c] = -1.0 / d_sa

            # y-slope: finite difference in i direction
            if i + 1 < NY:
                node_d = (i+1) * NX + j
                Dy[sa_idx, node_d] =  1.0 / d_sa
                Dy[sa_idx, node_c] = -1.0 / d_sa

            sa_idx += 1

    D = np.vstack([Dx, Dy])   # (160, 100)
    return D

# Precompute D† and save (add to precompute.py):
# D = build_southwell_matrix(active)
# D_pinv = np.linalg.pinv(D)   # (100, 160)
# np.save("precomputed/D_pinv.npy", D_pinv)

def reconstruct_zonal(slopes: np.ndarray) -> np.ndarray:
    """
    Returns wavefront on a (NY, NX) = (10, 10) node grid [rad].
    """
    D_pinv = np.load("precomputed/D_pinv.npy")   # (100, 160)
    w_flat = D_pinv @ slopes                       # (100,)
    return w_flat.reshape(config.N_LENSLETS_Y, config.N_LENSLETS_X)
```

## Step 2.6 — Integration test for reconstruction

```python
# At bottom of 02_reconstruct.py
if __name__ == "__main__":
    import numpy as np, json
    from PIL import Image
    import config
    from centroid import build_active_mask, build_subaperture_grid, cog_centroid, compute_slopes

    ref_frame = np.array(Image.open(config.DATA_DIR / "sh_flat_ref.bmp"))
    bg_frame  = np.array(Image.open(config.DATA_DIR / "sh_flat_bg.bmp"))
    active    = build_active_mask(ref_frame)
    cx_ref, cy_ref = build_subaperture_grid()

    # Load ground truth for frame 0
    gt = json.load(open(config.DATA_DIR / "ground_truth/per_frame.json"))
    gt_zk_0 = np.array(gt[0]["zernike_noll"])   # (21,)

    f0 = np.array(Image.open(config.DATA_DIR / "frames/frame_0000.bmp"))
    cx, cy = cog_centroid(f0, bg_frame, active)
    slopes = compute_slopes(cx, cy, cx_ref, cy_ref, active)

    a, W = reconstruct_modal(slopes)

    print(f"Zernike coeffs [rad]: {a[:5]}")
    print(f"Phase map  shape:     {W.shape}   min:{W.min():.3f}  max:{W.max():.3f}")
    print(f"Phase RMS [rad]:      {np.std(W[W!=0]):.4f}")
    # Expected: similar order to gt[0]['phase_rms_rad'] ≈ 0.25 rad

    # RMSE vs ground truth
    rmse = np.sqrt(np.mean((a - gt_zk_0)**2))
    print(f"Zernike RMSE vs GT:   {rmse:.4f} rad")
```

**Pass criteria:** Phase map shape (64,64), phase RMS in 0.1–1.0 rad range,
Zernike RMSE < 2 rad for frame 0 (note: GT Zernike includes piston/tilt from
full phase screen; reconstruction only recovers what slopes encode).

---

---

# PHASE 3 — Turbulence Characterisation
# Estimated time: 2.5 hours (June 29)

---

## Step 3.1 — r0 estimation from slope variance

**File:** `03_turbulence.py`

```python
import numpy as np
import config

def estimate_r0(all_slopes_x: np.ndarray,
                all_slopes_y: np.ndarray,
                active: np.ndarray) -> float:
    """
    all_slopes_x : (N_FRAMES, NY, NX) float32  — or use all_slopes_x.npy from GT
    all_slopes_y : (N_FRAMES, NY, NX) float32
    active       : (NY, NX) bool
    
    Returns r0 in metres.
    
    Formula (Roddier 1981):
        σ²_slope = 6.88 · r0^{-5/3} · d^{-1/3}
        r0 = ( σ² / (6.88 · d^{-1/3}) )^{-3/5}
    
    σ²_slope is the variance of slopes (rad/m) across all active subapertures
    across all frames. Average x and y.
    """
    d_sa = config.PITCH_M   # 150e-6 m

    sx = all_slopes_x[:, active]   # (N_FRAMES, 80)
    sy = all_slopes_y[:, active]

    sigma2 = (np.var(sx) + np.var(sy)) / 2.0

    r0 = (sigma2 / (6.88 * d_sa**(-1/3)))**(-3/5)
    return float(r0)
```

## Step 3.2 — τ0 estimation from temporal autocorrelation

```python
def estimate_tau0(zernike_time_series: np.ndarray,
                  dt: float,
                  mode_idx: int = 1) -> float:
    """
    zernike_time_series : (N_FRAMES, N_MODES) float32
    dt                  : frame interval [s]  = 1/FPS = 0.005 s
    mode_idx            : which Zernike mode to use (default 1 = Z2 tip)
    
    Method: normalised autocorrelation of tip Zernike coefficient.
    τ0 = lag at which autocorrelation drops to 1/e.
    
    Returns tau0 in seconds.
    """
    z = zernike_time_series[:, mode_idx]   # (N_FRAMES,)
    z = z - z.mean()

    # Full autocorrelation via FFT (circular, then take positive lags)
    N    = len(z)
    Z    = np.fft.rfft(z, n=2*N)
    acf  = np.fft.irfft(Z * np.conj(Z))[:N]
    acf /= acf[0]   # normalise to 1 at lag 0

    # Find first crossing of 1/e
    target = 1.0 / np.e
    idx    = np.where(acf < target)[0]
    if len(idx) == 0:
        return float('nan')   # didn't decay within observation window

    # Linear interpolation for sub-frame accuracy
    i0   = idx[0]
    t0   = (target - acf[i0-1]) / (acf[i0] - acf[i0-1]) + (i0 - 1)
    tau0 = t0 * dt

    return float(tau0)
```

## Step 3.3 — Kolmogorov PSD (for the plot that wins over judges)

```python
def plot_zernike_psd(zernike_time_series: np.ndarray,
                     dt: float,
                     save_path: str = "outputs/figures/zernike_psd.png"):
    """
    Plots temporal PSD of Z2 (tip) and overlays f^{-11/3} Kolmogorov slope.
    This is the physics validation figure.
    """
    import matplotlib.pyplot as plt

    z2 = zernike_time_series[:, 1]   # tip
    N  = len(z2)
    freqs = np.fft.rfftfreq(N, d=dt)   # Hz
    psd   = np.abs(np.fft.rfft(z2))**2 / N

    # Kolmogorov slope: PSD ∝ f^{-11/3}
    f_fit  = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), 100)
    # Normalise to match PSD at mid-frequency
    mid_idx = len(freqs) // 4
    A       = psd[mid_idx] / (freqs[mid_idx]**(-11/3))
    psd_fit = A * f_fit**(-11/3)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(freqs[1:], psd[1:], label="Z2 (tip) PSD", alpha=0.8)
    ax.loglog(f_fit, psd_fit, 'r--', label=r"Kolmogorov $f^{-11/3}$", lw=2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [rad² / Hz]")
    ax.set_title("Temporal PSD of Zernike Tip — Kolmogorov turbulence validation")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PSD figure saved → {save_path}")
```

## Step 3.4 — Integration test for turbulence module

```python
if __name__ == "__main__":
    import numpy as np, config

    # Load ground truth slopes
    sx_gt = np.load(config.DATA_DIR / "ground_truth/all_slopes_x.npy")  # (200,10,10)
    sy_gt = np.load(config.DATA_DIR / "ground_truth/all_slopes_y.npy")

    from PIL import Image
    from centroid import build_active_mask, build_subaperture_grid
    ref = np.array(Image.open(config.DATA_DIR / "sh_flat_ref.bmp"))
    active = build_active_mask(ref)

    r0 = estimate_r0(sx_gt, sy_gt, active)
    print(f"r0 estimated: {r0*1000:.2f} mm  (ground truth: 4.00 mm)")
    # Expected: ~3.1 mm (23% bias is physically expected)
```

**Pass criteria:** r0 within factor 2 of 4.0 mm, τ0 between 10–40 ms.

---

---

# PHASE 4 — Actuator Mapping
# Estimated time: 2 hours (June 29)

---

## Step 4.1 — Resample wavefront onto actuator grid (Fried geometry)

**Key physics:** In Fried geometry, actuator nodes are at corners of lenslet cells.
The 11×11 actuator grid spans the same physical area as the 10×10 lenslet grid,
but the actuator positions are offset by half a lenslet pitch in both axes.

```
Lenslet centres:   (j + 0.5) * pitch   for j = 0..9
Actuator positions: j * pitch           for j = 0..10
```

So actuator (0,0) is at the top-left corner, actuator (5,5) is at the centre of lenslet (4,4).

```python
# In 04_actuator.py

import numpy as np
import scipy.ndimage as ndi
import config

_C_pinv = np.load("precomputed/C_pinv.npy")   # (121, 121), loaded once

def wavefront_to_actuator_grid(W: np.ndarray) -> np.ndarray:
    """
    W : (PUPIL_GRID, PUPIL_GRID) = (64, 64) float32 — phase in rad
    
    Returns W_act : (N_ACT_X, N_ACT_Y) = (11, 11) float32 — phase in rad at actuator positions
    
    Method: bilinear interpolation from 64×64 pupil grid to 11×11 actuator positions.
    
    Fried geometry:
        Lenslet centre j maps to pupil coordinate (j+0.5)/N_LENSLETS within [0,1]
        Actuator node  j maps to pupil coordinate  j/N_ACT               within [0,1]
    """
    grid     = config.PUPIL_GRID      # 64
    n_act_x  = config.N_ACT_X        # 11
    n_act_y  = config.N_ACT_Y        # 11

    # Actuator positions in units of [0, PUPIL_GRID-1]
    act_x = np.linspace(0, grid-1, n_act_x)
    act_y = np.linspace(0, grid-1, n_act_y)
    AX, AY = np.meshgrid(act_x, act_y)

    # scipy map_coordinates: input is (row=y, col=x)
    W_act = ndi.map_coordinates(W, [AY.ravel(), AX.ravel()],
                                  order=1, mode='nearest')
    return W_act.reshape(n_act_y, n_act_x).astype(np.float32)
```

## Step 4.2 — DM command with coupling correction

```python
def compute_actuator_map(W: np.ndarray,
                         wavelength_m: float = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Full DM actuator map computation with coupling deconvolution.
    
    W            : (64, 64) float32 — reconstructed wavefront phase [rad]
    wavelength_m : laser wavelength (default from config)
    
    Returns:
        u_raw  : (11, 11) float32 — raw command [µm] (before coupling correction)
        u_corr : (11, 11) float32 — coupling-corrected command [µm]
        info   : dict with saturation stats
    
    Pipeline:
        1. Resample W onto 11×11 actuator grid
        2. Convert rad → metres: W_m = W * λ / (4π)   [OPD = λ/2, stroke = OPD/2]
        3. Conjugate (DM corrects by adding -W)
        4. Apply coupling deconvolution: u = C† · (-W_flat)
        5. Convert to µm and clip to ±stroke_limit
    
    Phase → OPD → stroke:
        OPD [m] = W [rad] * wavelength / (2π)
        DM stroke = OPD / 2    (double-pass; for single-pass AO: OPD/1)
        Here we use: u_m = -W * wavelength / (4π)
    """
    wl    = wavelength_m or config.WAVELENGTH_M
    lim   = config.STROKE_M   # 5e-6 m

    W_act  = wavefront_to_actuator_grid(W)               # (11, 11) rad
    u_m    = -W_act * wl / (4 * np.pi)                   # (11, 11) metres, conjugate
    u_flat = u_m.ravel()                                  # (121,)

    # Inter-actuator coupling deconvolution
    u_corr_flat = _C_pinv @ u_flat                        # (121,)
    u_corr      = u_corr_flat.reshape(11, 11)

    # Convert to µm for output
    u_raw_um  = (u_m   * 1e6).astype(np.float32)
    u_corr_um = (u_corr * 1e6).astype(np.float32)

    # Clip to stroke limits
    n_sat  = int(np.sum(np.abs(u_corr_um) > config.STROKE_M * 1e6))
    u_corr_um = np.clip(u_corr_um, -config.STROKE_M * 1e6, config.STROKE_M * 1e6)

    info = {
        "n_saturated": n_sat,
        "max_stroke_um": float(np.abs(u_corr_um).max()),
        "rms_stroke_um": float(np.std(u_corr_um))
    }
    return u_raw_um, u_corr_um, info
```

## Step 4.3 — Residual wavefront (the "closed loop" demonstration)

```python
def compute_residual(W: np.ndarray,
                     u_corr_um: np.ndarray,
                     wavelength_m: float = None) -> np.ndarray:
    """
    Compute the residual wavefront after DM correction.
    
    In an ideal AO loop:
        W_residual = W + W_DM    (DM cancels the aberration)
    
    Here we approximate W_DM from the actuator commands by bilinear interpolation
    back onto the pupil grid.
    
    Returns W_residual : (64, 64) float32 [rad]
    """
    from scipy import ndimage
    wl  = wavelength_m or config.WAVELENGTH_M

    # Convert actuator commands (µm) back to phase (rad)
    u_m    = u_corr_um * 1e-6                            # (11, 11) metres
    W_dm   = u_m * (4 * np.pi) / wl                     # (11, 11) rad (negative of W_act)

    # Upsample 11×11 → 64×64
    zoom   = config.PUPIL_GRID / config.N_ACT_X         # 64/11 ≈ 5.82
    W_dm64 = ndimage.zoom(W_dm, zoom, order=1)           # (64, 64) approx

    W_residual = W + W_dm64   # should be near zero for well-corrected wavefront
    return W_residual.astype(np.float32)
```

---

---

# PHASE 5 — C Extension (Speed Differentiator)
# Estimated time: 3 hours (June 30)

The problem statement explicitly recommends C. Implement CoG centroiding in C —
it's the most loop-heavy part of the pipeline and the easiest to port.

---

## Step 5.1 — `c_ext/cog_centroid.c`

```c
/* c_ext/cog_centroid.c
 * CoG centroiding for a single subaperture window.
 * Callable from Python via ctypes.
 *
 * compile: gcc -O3 -march=native -shared -fPIC -o cog_centroid.so cog_centroid.c
 */
#include <stdint.h>

/*
 * cog_single: compute centroid of one PPS×PPS uint8 patch after background subtract.
 *
 * patch   : row-major uint8 array, size pps*pps
 * bg_mean : background mean to subtract
 * pps     : pixels per subaperture (16)
 * cx_out  : output centroid x [0..pps-1]
 * cy_out  : output centroid y [0..pps-1]
 */
void cog_single(const uint8_t *patch, float bg_mean, int pps,
                float *cx_out, float *cy_out) {
    float flux = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
    for (int v = 0; v < pps; v++) {
        for (int u = 0; u < pps; u++) {
            float I = (float)patch[v * pps + u] - bg_mean;
            if (I < 0.0f) I = 0.0f;
            flux  += I;
            sum_x += u * I;
            sum_y += v * I;
        }
    }
    if (flux < 1e-6f) {
        *cx_out = (float)pps * 0.5f;
        *cy_out = (float)pps * 0.5f;
    } else {
        *cx_out = sum_x / flux;
        *cy_out = sum_y / flux;
    }
}

/*
 * cog_frame: process all NY*NX subapertures in one frame.
 *
 * frame    : (H, W) uint8 row-major, H = W = NY*NPS
 * bg_frame : (H, W) uint8 background frame
 * active   : (NY*NX) int32, 1 = active, 0 = inactive
 * cx_out   : (NY*NX) float32 output centroid x [global px]
 * cy_out   : (NY*NX) float32 output centroid y [global px]
 * NY, NX, PPS, W : grid and frame dimensions
 */
void cog_frame(const uint8_t *frame, const uint8_t *bg_frame,
               const int32_t *active,
               float *cx_out, float *cy_out,
               int NY, int NX, int pps, int W) {
    for (int i = 0; i < NY; i++) {
        for (int j = 0; j < NX; j++) {
            int idx = i * NX + j;
            if (!active[idx]) {
                cx_out[idx] = (j + 0.5f) * pps;
                cy_out[idx] = (i + 0.5f) * pps;
                continue;
            }
            /* Compute background mean for this subaperture */
            float bg_sum = 0.0f;
            int y0 = i * pps, x0 = j * pps;
            for (int v = 0; v < pps; v++)
                for (int u = 0; u < pps; u++)
                    bg_sum += (float)bg_frame[(y0+v)*W + (x0+u)];
            float bg_mean = bg_sum / (pps * pps);

            /* Extract patch (copy to local buffer for contiguous access) */
            uint8_t patch[pps * pps];   /* VLA — fine for pps=16 */
            for (int v = 0; v < pps; v++)
                for (int u = 0; u < pps; u++)
                    patch[v*pps + u] = frame[(y0+v)*W + (x0+u)];

            float lx, ly;
            cog_single(patch, bg_mean, pps, &lx, &ly);
            cx_out[idx] = x0 + lx;
            cy_out[idx] = y0 + ly;
        }
    }
}
```

## Step 5.2 — `c_ext/Makefile`

```makefile
CC      = gcc
CFLAGS  = -O3 -march=native -Wall -fPIC -shared

all: cog_centroid.so

cog_centroid.so: cog_centroid.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.so
```

```bash
cd c_ext && make && cd ..
```

## Step 5.3 — Python ctypes wrapper

```python
# In 01_centroid.py (add after the numpy CoG functions)

import ctypes, os, numpy as np

_lib_path = os.path.join(os.path.dirname(__file__), "c_ext", "cog_centroid.so")

def _load_c_lib():
    try:
        lib = ctypes.CDLL(_lib_path)
        lib.cog_frame.restype  = None
        lib.cog_frame.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # frame
            ctypes.POINTER(ctypes.c_uint8),   # bg_frame
            ctypes.POINTER(ctypes.c_int32),   # active (flat)
            ctypes.POINTER(ctypes.c_float),   # cx_out
            ctypes.POINTER(ctypes.c_float),   # cy_out
            ctypes.c_int,  # NY
            ctypes.c_int,  # NX
            ctypes.c_int,  # pps
            ctypes.c_int,  # W (frame width)
        ]
        return lib
    except OSError:
        return None

_c_lib = _load_c_lib()   # None if not compiled yet

def cog_centroid_c(frame: np.ndarray,
                   bg_frame: np.ndarray,
                   active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    C-backed CoG centroiding. Falls back to numpy version if .so not found.
    active : (NY, NX) bool
    """
    if _c_lib is None:
        return cog_centroid(frame, bg_frame, active)   # numpy fallback

    NY, NX = active.shape
    pps    = config.PPS
    W_frame = frame.shape[1]

    frame_c   = np.ascontiguousarray(frame,     dtype=np.uint8)
    bg_c      = np.ascontiguousarray(bg_frame,  dtype=np.uint8)
    active_c  = np.ascontiguousarray(active.ravel().astype(np.int32))
    cx_out    = np.zeros(NY * NX, dtype=np.float32)
    cy_out    = np.zeros(NY * NX, dtype=np.float32)

    _c_lib.cog_frame(
        frame_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        bg_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        active_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cy_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(NY), ctypes.c_int(NX),
        ctypes.c_int(pps), ctypes.c_int(W_frame)
    )
    return cx_out.reshape(NY, NX), cy_out.reshape(NY, NX)
```

---

---

# PHASE 6 — Integration: run_pipeline.py
# Estimated time: 2 hours (June 30)

---

## Step 6.1 — Full pipeline runner

```python
# run_pipeline.py

import numpy as np, json, time
from pathlib import Path
from PIL import Image
import config

from centroid   import (build_active_mask, build_subaperture_grid,
                        cog_centroid_c, compute_slopes)
from reconstruct import reconstruct_modal
from turbulence  import estimate_r0, estimate_tau0, plot_zernike_psd
from actuator    import compute_actuator_map, compute_residual

def run(data_dir: str = "data", n_frames: int = None):
    DATA = Path(data_dir)
    out  = Path("outputs")
    (out / "phase_maps").mkdir(parents=True, exist_ok=True)
    (out / "actuator_maps").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # ── Offline precomputed assets ────────────────────────────────────────────
    print("[1/7] Loading precomputed matrices...")
    # G_pinv, Z_basis, C_pinv loaded in reconstruct.py / actuator.py at import time

    # ── Reference and background frames ──────────────────────────────────────
    print("[2/7] Loading reference and background frames...")
    ref_frame = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    bg_frame  = np.array(Image.open(DATA / "sh_flat_bg.bmp"))
    active    = build_active_mask(ref_frame)
    cx_ref, cy_ref = build_subaperture_grid()
    print(f"      Active subapertures: {active.sum()}")

    # ── Frame list ────────────────────────────────────────────────────────────
    frame_paths = sorted((DATA / "frames").glob("frame_????.bmp"))
    if n_frames:
        frame_paths = frame_paths[:n_frames]
    N = len(frame_paths)
    print(f"[3/7] Processing {N} frames...")

    # ── Per-frame storage ─────────────────────────────────────────────────────
    all_slopes_x  = np.zeros((N, config.N_LENSLETS_Y, config.N_LENSLETS_X), np.float32)
    all_slopes_y  = np.zeros_like(all_slopes_x)
    all_zernike   = np.zeros((N, config.N_MODES),                            np.float32)
    all_phase_rms = np.zeros(N,                                               np.float32)
    all_residual_rms = np.zeros(N,                                            np.float32)

    # ── Timing ────────────────────────────────────────────────────────────────
    t_centroid = t_reconstruct = t_actuator = 0.0

    for idx, fpath in enumerate(frame_paths):
        frame = np.array(Image.open(fpath))

        # Stage 1: Centroiding
        t0 = time.perf_counter()
        cx, cy  = cog_centroid_c(frame, bg_frame, active)
        slopes  = compute_slopes(cx, cy, cx_ref, cy_ref, active)
        t_centroid += time.perf_counter() - t0

        # Unpack for storage
        sx_2d = np.zeros((config.N_LENSLETS_Y, config.N_LENSLETS_X), np.float32)
        sy_2d = np.zeros_like(sx_2d)
        sx_2d[active] = slopes[:config.N_SA_ACTIVE]
        sy_2d[active] = slopes[config.N_SA_ACTIVE:]
        all_slopes_x[idx] = sx_2d
        all_slopes_y[idx] = sy_2d

        # Stage 2: Reconstruction
        t0 = time.perf_counter()
        a, W = reconstruct_modal(slopes)
        t_reconstruct += time.perf_counter() - t0
        all_zernike[idx]   = a
        all_phase_rms[idx] = float(np.std(W[W != 0]))

        # Save phase map
        np.save(out / "phase_maps" / f"frame_{idx:04d}_phase.npy", W)

        # Stage 3: Actuator map
        t0 = time.perf_counter()
        u_raw, u_corr, info = compute_actuator_map(W)
        t_actuator += time.perf_counter() - t0

        np.save(out / "actuator_maps" / f"frame_{idx:04d}_act.npy", u_corr)

        # Residual
        W_res = compute_residual(W, u_corr)
        all_residual_rms[idx] = float(np.std(W_res[W_res != 0]))

        if idx % 50 == 0:
            print(f"      Frame {idx:3d}/{N}  |  phase_rms={all_phase_rms[idx]:.3f} rad"
                  f"  |  max_stroke={info['max_stroke_um']:.3f} µm")

    # ── Turbulence statistics ─────────────────────────────────────────────────
    print("[4/7] Computing turbulence statistics...")
    r0   = estimate_r0(all_slopes_x, all_slopes_y, active)
    tau0 = estimate_tau0(all_zernike, config.DT_S, mode_idx=1)

    # ── PSD figure ────────────────────────────────────────────────────────────
    print("[5/7] Generating figures...")
    plot_zernike_psd(all_zernike, config.DT_S,
                     save_path=str(out / "figures/zernike_psd.png"))

    # ── Benchmark report ──────────────────────────────────────────────────────
    print("[6/7] Reporting timing...")
    total_ms   = (t_centroid + t_reconstruct + t_actuator) * 1000 / N
    benchmark  = {
        "n_frames":          N,
        "centroid_ms_mean":  round(t_centroid   * 1000 / N, 3),
        "reconstruct_ms_mean": round(t_reconstruct * 1000 / N, 3),
        "actuator_ms_mean":  round(t_actuator   * 1000 / N, 3),
        "total_pipeline_ms_mean": round(total_ms, 3),
        "real_time_budget_ms": 1000 / config.FPS,
        "meets_real_time":   total_ms < (1000 / config.FPS)
    }

    # ── Turbulence report ─────────────────────────────────────────────────────
    turb_report = {
        "r0_estimated_mm":      round(r0 * 1000, 3),
        "r0_ground_truth_mm":   4.000,
        "r0_error_pct":         round(abs(r0*1000 - 4.000) / 4.000 * 100, 1),
        "tau0_estimated_ms":    round(tau0 * 1000, 3),
        "tau0_ground_truth_ms": 19.385,
        "mean_phase_rms_rad":   round(float(all_phase_rms.mean()), 4),
        "mean_residual_rms_rad": round(float(all_residual_rms.mean()), 4),
    }

    json.dump(benchmark,   open(out / "benchmark_report.json",   "w"), indent=2)
    json.dump(turb_report, open(out / "turbulence_report.json",  "w"), indent=2)

    print("[7/7] Done.")
    print(f"\n{'='*60}")
    print(f"  r0 estimated     : {turb_report['r0_estimated_mm']:.2f} mm  (GT: 4.00 mm, err: {turb_report['r0_error_pct']}%)")
    print(f"  tau0 estimated   : {turb_report['tau0_estimated_ms']:.2f} ms (GT: 19.39 ms)")
    print(f"  Phase RMS mean   : {turb_report['mean_phase_rms_rad']:.4f} rad")
    print(f"  Residual RMS mean: {turb_report['mean_residual_rms_rad']:.4f} rad")
    print(f"  Pipeline latency : {benchmark['total_pipeline_ms_mean']:.2f} ms/frame  (budget: {benchmark['real_time_budget_ms']} ms)")
    print(f"  Real-time capable: {benchmark['meets_real_time']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run()
```

## Step 6.2 — Run and collect results

```bash
python3 run_pipeline.py
```

**Expected output table:**
```
r0 estimated     : 3.09 mm  (GT: 4.00 mm, err: 22.8%)
tau0 estimated   : ~18–22 ms (GT: 19.39 ms)
Phase RMS mean   : 0.40 rad
Residual RMS mean: < 0.15 rad  (60%+ reduction after DM correction)
Pipeline latency : < 5.0 ms/frame
Real-time capable: True
```

---

---

# PHASE 7 — Validation Against Ground Truth
# Estimated time: 1 hour (June 30)

---

## Step 7.1 — `05_validate.py`

```python
# 05_validate.py

import numpy as np, json
from pathlib import Path
import config

def validate_against_ground_truth(outputs_dir: str = "outputs",
                                   data_dir: str    = "data"):
    OUT  = Path(outputs_dir)
    DATA = Path(data_dir) / "ground_truth"

    # Load ground truth
    gt_sx   = np.load(DATA / "all_slopes_x.npy")   # (200, 10, 10)
    gt_sy   = np.load(DATA / "all_slopes_y.npy")
    gt_zk   = np.load(DATA / "all_zernike.npy")     # (200, 21)

    # Load pipeline output Zernike coefficients
    # (collected from per-frame .npy — or from run_pipeline.py output)
    # Here we re-collect from saved phase maps (for comparison of phase, not Zernike)
    n_frames = len(list((OUT / "phase_maps").glob("*.npy")))
    pred_rms = np.array([
        np.std(np.load(OUT / "phase_maps" / f"frame_{i:04d}_phase.npy"))
        for i in range(n_frames)
    ])
    gt_rms   = np.array([json.load(open(DATA / "per_frame.json"))[i]["phase_rms_rad"]
                          for i in range(n_frames)])

    rms_error = np.abs(pred_rms - gt_rms).mean()
    rms_corr  = np.corrcoef(pred_rms, gt_rms)[0,1]

    print(f"Phase RMS RMSE (predicted vs GT): {rms_error:.4f} rad")
    print(f"Phase RMS correlation:            {rms_corr:.4f}")
    print(f"Mean predicted phase RMS:         {pred_rms.mean():.4f} rad")
    print(f"Mean GT phase RMS:                {gt_rms.mean():.4f} rad")

    # Actuator stats
    act_maps = [np.load(OUT / "actuator_maps" / f"frame_{i:04d}_act.npy")
                for i in range(n_frames)]
    all_act  = np.stack(act_maps)
    print(f"\nActuator map stats (µm):")
    print(f"  Mean |u| : {np.abs(all_act).mean():.3f}")
    print(f"  Max  |u| : {np.abs(all_act).max():.3f}")
    print(f"  Saturated: {(np.abs(all_act) > config.STROKE_M*1e6).sum()} actuator-frames")

if __name__ == "__main__":
    validate_against_ground_truth()
```

---

---

# PHASE 8 — Submission Prep (July 1 AM)
# Estimated time: 2 hours

---

## Step 8.1 — Block diagram figure for PDF

```python
# idea_submission/block_diagram.py
# Run: python3 block_diagram.py → saves pipeline_block_diagram.png

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

BLOCKS = [
    ("SH-WFS\nFrame\n(BMP)",         "#4A90D9"),
    ("Background\nSubtract &\nThreshold",  "#5BA05B"),
    ("CoG\nCentroiding\n(C kernel)",   "#E8A838"),
    ("Slope\nComputation\n(Δx,Δy→rad/m)", "#E8A838"),
    ("Modal\nReconstruction\n(G† · s)",    "#9B59B6"),
    ("Wavefront\nPhase Map\nW(x,y) [rad]", "#E74C3C"),
    ("Turbulence\nStats\nr₀, τ₀",         "#1ABC9C"),
    ("DM Actuator\nMap (C†·(-W))\n[µm]",  "#E74C3C"),
]

fig, ax = plt.subplots(figsize=(16, 3.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 3.5); ax.axis('off')

xpos = [0.6 + i*2.0 for i in range(len(BLOCKS))]

for x, (label, color) in zip(xpos, BLOCKS):
    ax.add_patch(mpatches.FancyBboxPatch((x-0.7, 0.8), 1.4, 1.9,
                 boxstyle="round,pad=0.1", fc=color, ec='white', alpha=0.85))
    ax.text(x, 1.75, label, ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold', multialignment='center')

for i in range(len(BLOCKS)-1):
    ax.annotate('', xy=(xpos[i+1]-0.72, 1.75), xytext=(xpos[i]+0.72, 1.75),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Side branch: turbulence stats
ax.annotate('', xy=(xpos[6], 0.8), xytext=(xpos[5], 0.8),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                            connectionstyle="arc3,rad=-0.3"))

ax.set_title("BAH 2026 PS-9 — SH-WFS Reconstruction Pipeline  |  Real-time budget: 5.0 ms/frame",
             fontsize=10, pad=10)
plt.tight_layout()
plt.savefig("idea_submission/pipeline_block_diagram.png", dpi=200, bbox_inches='tight')
print("Saved → idea_submission/pipeline_block_diagram.png")
```

## Step 8.2 — Idea submission checklist (July 1)

```
[ ] run_pipeline.py executes end-to-end with zero errors
[ ] benchmark_report.json shows total_pipeline_ms_mean < 5.0
[ ] turbulence_report.json shows r0 within 30% of 4.0 mm
[ ] outputs/figures/zernike_psd.png shows f^{-11/3} slope
[ ] pipeline_block_diagram.png generated
[ ] 1-page PDF contains:
      - Problem statement paraphrase (2 sentences)
      - Block diagram (the figure above)
      - Key novelty: "validated against synthetic dataset with known ground truth;
        quantitative RMSE and r0 error budget reported"
      - Benchmark table: latency per stage, real-time capable = True
      - Next steps after shortlisting: swap synthetic BMPs for real ISRO frames
[ ] GitHub repo pushed with: config.py, all modules, precompute.py, run_pipeline.py,
    c_ext/cog_centroid.c, README with install and run instructions
```

---

---

# APPENDIX — Quick Debug Cheatsheet

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| active_mask returns ≠80 subapertures | threshold_frac wrong | Try 0.05–0.20 |
| Slopes all near zero | slope_scale wrong | Check PIX_M / F_MLA_M |
| Phase RMS >> 1 rad | PSD → phase conversion bug | Check 2π/λ factor (already baked into generator) |
| G_pinv gives nonsense | G has linearly dependent rows | Add Tikhonov reg: `np.linalg.pinv(G, rcond=1e-3)` |
| r0 way off | Wrong d_sa unit | Must be in metres (0.00015, not 150) |
| tau0 = nan | Series too short to decay | Use tau0 formula via r0 and wind speed instead |
| C† overcorrects | C is near-singular | Use pinv with rcond=0.01 |
| Latency >> 5ms | Python CoG loop is bottleneck | Enable C extension (`make` in c_ext/) |
| Residual RMS > W RMS | Actuator map sign wrong | Flip sign in `u_m = -W_act * wl/(4π)` |