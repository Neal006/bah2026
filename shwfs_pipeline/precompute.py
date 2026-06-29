"""
Precompute and cache: G matrix, G_pinv, D_pinv (Southwell), Zernike basis.
Run once before the pipeline: python precompute.py
"""
import numpy as np
from pathlib import Path
from PIL import Image
import config
from centroid import build_active_mask

_PRECOMP = Path(__file__).parent / "precomputed"
_PRECOMP.mkdir(exist_ok=True)

N_MODES = config.N_MODES
NY, NX  = config.N_LENSLETS_Y, config.N_LENSLETS_X
PITCH   = config.PITCH_M
R       = config.R_PUPIL_M
N_SA    = config.N_SA_ACTIVE


# ── Zernike basis on unit disk ─────────────────────────────────────────────

def _build_zernike(ng: int) -> np.ndarray:
    """Returns Z (21, ng, ng) Noll-normalized on [-1,1]^2 grid."""
    xi = np.linspace(-1, 1, ng)
    XX, YY = np.meshgrid(xi, xi)
    r, th = np.sqrt(XX**2 + YY**2), np.arctan2(YY, XX)
    Z = np.zeros((N_MODES, ng, ng), dtype=np.float64)
    Z[0]  = 1.0
    Z[1]  = 2*r*np.cos(th)
    Z[2]  = 2*r*np.sin(th)
    Z[3]  = np.sqrt(3)*(2*r**2 - 1)
    Z[4]  = np.sqrt(6)*r**2*np.sin(2*th)
    Z[5]  = np.sqrt(6)*r**2*np.cos(2*th)
    Z[6]  = np.sqrt(8)*(3*r**3 - 2*r)*np.sin(th)
    Z[7]  = np.sqrt(8)*(3*r**3 - 2*r)*np.cos(th)
    Z[8]  = np.sqrt(8)*r**3*np.sin(3*th)
    Z[9]  = np.sqrt(8)*r**3*np.cos(3*th)
    Z[10] = np.sqrt(5)*(6*r**4 - 6*r**2 + 1)
    Z[11] = np.sqrt(10)*(4*r**4 - 3*r**2)*np.cos(2*th)
    Z[12] = np.sqrt(10)*(4*r**4 - 3*r**2)*np.sin(2*th)
    Z[13] = np.sqrt(10)*r**4*np.cos(4*th)
    Z[14] = np.sqrt(10)*r**4*np.sin(4*th)
    Z[15] = np.sqrt(12)*(10*r**5 - 12*r**3 + 3*r)*np.cos(th)
    Z[16] = np.sqrt(12)*(10*r**5 - 12*r**3 + 3*r)*np.sin(th)
    Z[17] = np.sqrt(12)*(5*r**5 - 4*r**3)*np.cos(3*th)
    Z[18] = np.sqrt(12)*(5*r**5 - 4*r**3)*np.sin(3*th)
    Z[19] = np.sqrt(12)*r**5*np.cos(5*th)
    Z[20] = np.sqrt(12)*r**5*np.sin(5*th)
    return Z


def build_interaction_matrix(active: np.ndarray, ng: int = 128) -> np.ndarray:
    """
    Builds G (N_SLOPES, N_MODES) using analytical Zernike gradients.

    G[sa, k] = mean(dZ_k/dx_norm / R_pupil) over subaperture k.
    Slopes are in rad/m; Zernike coefficients are in radians.

    Subaperture centre in normalised pupil coords:
        x_norm[j] = (j - (NX-1)/2) * PITCH/R
    """
    Z   = _build_zernike(ng)
    xi  = np.linspace(-1, 1, ng)
    dx  = xi[1] - xi[0]
    dZx = np.gradient(Z, dx, axis=2) / R   # dZ/dx_phys  [rad/m per rad Zernike]
    dZy = np.gradient(Z, dx, axis=1) / R

    sa_norm = PITCH / R   # subaperture width in normalised units

    Gx = np.zeros((N_SA, N_MODES), dtype=np.float64)
    Gy = np.zeros((N_SA, N_MODES), dtype=np.float64)
    sa_idx = 0

    def n2g(n):
        return int(round((n + 1) / 2 * (ng - 1)))

    for i in range(NY):
        for j in range(NX):
            if not active[i, j]:
                continue
            xn = (j - (NX - 1) / 2) * sa_norm
            yn = (i - (NY - 1) / 2) * sa_norm

            jlo = max(0, n2g(xn - sa_norm / 2))
            jhi = min(ng, n2g(xn + sa_norm / 2) + 1)
            ilo = max(0, n2g(yn - sa_norm / 2))
            ihi = min(ng, n2g(yn + sa_norm / 2) + 1)

            for k in range(N_MODES):
                Gx[sa_idx, k] = dZx[k, ilo:ihi, jlo:jhi].mean()
                Gy[sa_idx, k] = dZy[k, ilo:ihi, jlo:jhi].mean()
            sa_idx += 1

    return np.vstack([Gx, Gy]).astype(np.float32)   # (160, 21)


def build_southwell_matrix(active: np.ndarray) -> np.ndarray:
    """
    Builds Southwell zonal reconstruction matrix D (N_SLOPES, N_VALID_NODES).

    Maps slope measurements at subaperture centres to phase at the surrounding
    Fried-geometry actuator nodes. Uses finite-difference stencil from Southwell (1980).

    Returns D (N_SLOPES, N_ACT_X*N_ACT_Y) float32.
    """
    n_act_x = config.N_ACT_X
    n_act_y = config.N_ACT_Y
    N_act   = n_act_x * n_act_y   # 121

    N_sx = int(active.sum())  # 80 active x-slopes
    N_slopes = 2 * N_sx       # 160 total

    D = np.zeros((N_slopes, N_act), dtype=np.float32)

    sa_idx = 0
    for i in range(NY):
        for j in range(NX):
            if not active[i, j]:
                continue
            # x-slope row: neighbouring actuator nodes at (i,j) and (i,j+1)
            row_x = sa_idx
            node_lo = i * n_act_x + j        # left actuator
            node_hi = i * n_act_x + (j + 1)  # right actuator
            if j + 1 < n_act_x:
                D[row_x, node_lo] = -1.0 / PITCH
                D[row_x, node_hi] =  1.0 / PITCH

            # y-slope row: neighbouring actuator nodes at (i,j) and (i+1,j)
            row_y = N_sx + sa_idx
            node_lo_y = i       * n_act_x + j
            node_hi_y = (i + 1) * n_act_x + j
            if i + 1 < n_act_y:
                D[row_y, node_lo_y] = -1.0 / PITCH
                D[row_y, node_hi_y] =  1.0 / PITCH

            sa_idx += 1

    return D


def precompute_and_save():
    ref_frame = np.array(Image.open(config.DATA_DIR / "sh_flat_ref.bmp"))
    active    = build_active_mask(ref_frame)

    print("Building interaction matrix G …")
    G = build_interaction_matrix(active, ng=128)
    np.save(_PRECOMP / "G.npy",       G)
    np.save(_PRECOMP / "active.npy",  active)

    print("Computing G_pinv …")
    G_pinv = np.linalg.pinv(G.astype(np.float64)).astype(np.float32)
    np.save(_PRECOMP / "G_pinv.npy",  G_pinv)

    print("Building Southwell matrix D …")
    D = build_southwell_matrix(active)
    np.save(_PRECOMP / "D.npy", D)

    print("Computing D_pinv …")
    D_pinv = np.linalg.pinv(D.astype(np.float64)).astype(np.float32)
    np.save(_PRECOMP / "D_pinv.npy", D_pinv)

    print("Saving Zernike basis …")
    Z = _build_zernike(ng=config.PUPIL_GRID)
    np.save(_PRECOMP / "Z_basis.npy", Z.astype(np.float32))

    print(f"Done. Files written to {_PRECOMP}/")
    print(f"  G:      {G.shape}   G_pinv:  {G_pinv.shape}")
    print(f"  D:      {D.shape}   D_pinv:  {D_pinv.shape}")
    print(f"  Z_basis: {Z.shape}")


if __name__ == "__main__":
    precompute_and_save()
