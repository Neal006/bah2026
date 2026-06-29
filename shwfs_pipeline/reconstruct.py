"""
Wavefront reconstruction from SH-WFS slopes.

Modal: least-squares Zernike fit via G_pinv.
Zonal: Southwell integration via D_pinv.

GT Zernike are stored 50× their radian value; actual rad = stored / 50.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import config

_PRECOMP = Path(__file__).parent / "precomputed"

_G_pinv  = np.load(_PRECOMP / "G_pinv.npy")   # (21, 160) float32
_D_pinv  = np.load(_PRECOMP / "D_pinv.npy")   # (121, 160) float32
_Z_basis = np.load(_PRECOMP / "Z_basis.npy")  # (21, 64, 64) float32
_active  = np.load(_PRECOMP / "active.npy")   # (10, 10) bool


def reconstruct_modal(slopes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Modal LS Zernike reconstruction.

    slopes : (160,) float32  — [sx_0..sx_79, sy_0..sy_79] in rad/m
    Returns
        a     : (21,) float32  — Zernike coefficients in rad (Z1..Z21 Noll)
        phase : (64, 64) float32 — phase map in rad on pupil grid
    """
    a     = _G_pinv @ slopes.astype(np.float32)          # (21,)
    phase = (_Z_basis * a[:, None, None]).sum(axis=0)    # (64, 64)
    return a, phase


def reconstruct_zonal(slopes: np.ndarray) -> np.ndarray:
    """
    Zonal Southwell reconstruction.

    slopes : (160,) float32
    Returns phi : (11, 11) float32 — phase at actuator nodes [rad]
    """
    phi_flat = _D_pinv @ slopes.astype(np.float32)  # (121,)
    return phi_flat.reshape(config.N_ACT_Y, config.N_ACT_X)


def phase_rms(phase: np.ndarray, pupil_mask: np.ndarray | None = None) -> float:
    """RMS of phase over the pupil mask (or full array if mask is None)."""
    if pupil_mask is None:
        xi = np.linspace(-1, 1, phase.shape[0])
        XX, YY = np.meshgrid(xi, xi)
        pupil_mask = (XX**2 + YY**2) <= 1.0
    vals = phase[pupil_mask]
    return float(np.sqrt(np.mean((vals - vals.mean())**2)))


if __name__ == "__main__":
    from PIL import Image
    import config
    from centroid import build_active_mask, cog_centroid, compute_slopes, build_subaperture_grid

    DATA = config.DATA_DIR
    ref = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    bg  = np.array(Image.open(DATA / "sh_flat_bg.bmp"))

    active   = build_active_mask(ref)
    cx_ref_m, cy_ref_m = cog_centroid(ref, bg, active)  # measured reference centroids

    # --- Test on GT slopes ---
    sx_gt = np.load(DATA / "ground_truth/all_slopes_x.npy")
    sy_gt = np.load(DATA / "ground_truth/all_slopes_y.npy")
    zk_gt = np.load(DATA / "ground_truth/all_zernike.npy")

    GT_UNIT = 1.0 / 50.0   # stored = actual_rad * 50

    print("=== Modal reconstruction from GT slopes ===")
    errs = []
    for t in range(min(10, 200)):
        slopes_t = np.concatenate([
            sx_gt[t][active].astype(np.float32),
            sy_gt[t][active].astype(np.float32),
        ])
        a, phase = reconstruct_modal(slopes_t)
        a_gt_rad = zk_gt[t] * GT_UNIT
        rms_err  = np.sqrt(np.mean((a[1:] - a_gt_rad[1:])**2))  # exclude piston
        errs.append(rms_err)

    print(f"  Zernike RMSE (Z2..Z21) over 10 frames: {np.mean(errs):.5f} rad")
    print(f"  Mean GT phase RMS: {np.mean([zk_gt[t, 1:] * GT_UNIT for t in range(10)]):.4f} rad")

    # Phase map from frame 0
    slopes_0 = np.concatenate([sx_gt[0][active], sy_gt[0][active]])
    a0, phase0 = reconstruct_modal(slopes_0)
    xi = np.linspace(-1, 1, config.PUPIL_GRID)
    XX, YY = np.meshgrid(xi, xi)
    pupil = (XX**2 + YY**2) <= 1.0
    print(f"  Frame 0 reconstructed phase RMS: {phase_rms(phase0, pupil):.4f} rad  (GT: 0.2544 rad)")
