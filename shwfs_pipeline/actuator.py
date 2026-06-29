"""
DM actuator command computation.

Given reconstructed phase at actuator nodes (zonal) and the DM coupling
matrix C (121×121), compute the actuator voltage commands that minimise
the residual wavefront:

    phi_act = D_pinv @ slopes          (zonal reconstruction)
    v_cmd   = C_pinv @ (-phi_act)      (conjugate correction + decouple)

Stroke limit applied per actuator: |v_cmd[i]| <= STROKE_M / 2.
"""
import numpy as np
from pathlib import Path
import config

_STROKE     = config.STROKE_M          # ±5e-6 m peak-to-valley
_STROKE_HALF = _STROKE / 2            # per-actuator limit

_C     = np.load(config.DATA_DIR / "dm_coupling.npy").astype(np.float64)  # (121,121)
_C_pinv = np.linalg.pinv(_C).astype(np.float32)                           # (121,121)


_RAD_TO_M = config.WAVELENGTH_M / (4 * np.pi)   # reflection DM: OPD = 2×displacement


def compute_commands(phi_act: np.ndarray) -> np.ndarray:
    """
    phi_act : (121,) float32 — zonal phase at actuator nodes [rad]
    Returns v_cmd : (121,) float32 — actuator commands [m], stroke-limited.
    """
    phi_m = phi_act.astype(np.float32) * _RAD_TO_M   # rad -> physical DM stroke [m]
    v = _C_pinv @ (-phi_m)
    return np.clip(v, -_STROKE_HALF, _STROKE_HALF)


def saturated_count(v_cmd: np.ndarray) -> int:
    return int(np.sum(np.abs(v_cmd) >= _STROKE_HALF))


if __name__ == "__main__":
    from pathlib import Path
    from reconstruct import reconstruct_zonal
    import numpy as np

    DATA = config.DATA_DIR
    sx_gt = np.load(DATA / "ground_truth/all_slopes_x.npy")
    sy_gt = np.load(DATA / "ground_truth/all_slopes_y.npy")

    from centroid import build_active_mask
    from PIL import Image
    ref    = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    active = build_active_mask(ref)

    sat_counts = []
    v_norms    = []
    for t in range(200):
        slopes = np.concatenate([
            sx_gt[t][active].astype(np.float32),
            sy_gt[t][active].astype(np.float32),
        ])
        phi = reconstruct_zonal(slopes).ravel()    # (121,)
        v   = compute_commands(phi)
        sat_counts.append(saturated_count(v))
        v_norms.append(float(np.linalg.norm(v)))

    print(f"DM command stats over 200 GT frames:")
    print(f"  mean |v| norm : {np.mean(v_norms):.4e} m")
    print(f"  max  |v| norm : {np.max(v_norms):.4e} m")
    print(f"  saturated acts: {np.mean(sat_counts):.1f} / 121  (mean per frame)")
    print(f"  max saturation: {np.max(sat_counts)} / 121")
    print(f"  C_pinv shape  : {_C_pinv.shape}")
