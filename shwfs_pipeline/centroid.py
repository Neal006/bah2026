import numpy as np
from PIL import Image
import config


def build_subaperture_grid():
    """Returns (sa_cx_ref, sa_cy_ref): (NY, NX) float32 reference centroid positions [px]."""
    NY, NX = config.N_LENSLETS_Y, config.N_LENSLETS_X
    pps    = config.PPS
    cx_ref = np.zeros((NY, NX), dtype=np.float32)
    cy_ref = np.zeros((NY, NX), dtype=np.float32)
    for i in range(NY):
        for j in range(NX):
            cx_ref[i, j] = (j + 0.5) * pps
            cy_ref[i, j] = (i + 0.5) * pps
    return cx_ref, cy_ref


def build_active_mask(ref_frame: np.ndarray, threshold_frac: float = 0.3) -> np.ndarray:
    """
    ref_frame : (H, W) uint8 — flat reference frame
    Returns   : (NY, NX) bool — True inside circular pupil

    Subapertures with flux > threshold_frac * max_flux are active.
    """
    NY, NX = config.N_LENSLETS_Y, config.N_LENSLETS_X
    pps    = config.PPS
    fluxes = np.zeros((NY, NX))
    for i in range(NY):
        for j in range(NX):
            y0, y1 = i * pps, (i + 1) * pps
            x0, x1 = j * pps, (j + 1) * pps
            fluxes[i, j] = ref_frame[y0:y1, x0:x1].astype(np.float32).sum()
    active = fluxes > threshold_frac * fluxes.max()
    assert active.sum() == config.N_SA_ACTIVE, \
        f"Expected {config.N_SA_ACTIVE} active SA, got {active.sum()} — adjust threshold_frac"
    return active


def cog_centroid(frame: np.ndarray,
                 bg_frame: np.ndarray,
                 active: np.ndarray):
    """
    Centre-of-Gravity centroiding (numpy).

    frame    : (H, W) uint8
    bg_frame : (H, W) uint8
    active   : (NY, NX) bool

    Returns cx, cy: (NY, NX) float32 — global centroid positions [px]
    """
    NY, NX = config.N_LENSLETS_Y, config.N_LENSLETS_X
    pps    = config.PPS
    cx     = np.zeros((NY, NX), dtype=np.float32)
    cy     = np.zeros((NY, NX), dtype=np.float32)

    u_grid, v_grid = np.meshgrid(np.arange(pps, dtype=np.float32),
                                  np.arange(pps, dtype=np.float32))

    for i in range(NY):
        for j in range(NX):
            if not active[i, j]:
                cx[i, j] = (j + 0.5) * pps
                cy[i, j] = (i + 0.5) * pps
                continue

            y0, x0 = i * pps, j * pps
            patch  = frame[y0:y0+pps, x0:x0+pps].astype(np.float32)
            bg     = bg_frame[y0:y0+pps, x0:x0+pps].astype(np.float32)
            P      = np.clip(patch - bg.mean(), 0.0, None)

            flux = P.sum()
            if flux < 1e-6:
                cx[i, j] = (j + 0.5) * pps
                cy[i, j] = (i + 0.5) * pps
                continue

            cx[i, j] = x0 + (u_grid * P).sum() / flux
            cy[i, j] = y0 + (v_grid * P).sum() / flux

    return cx, cy


def compute_slopes(cx: np.ndarray, cy: np.ndarray,
                   cx_ref: np.ndarray, cy_ref: np.ndarray,
                   active: np.ndarray) -> np.ndarray:
    """
    Converts pixel displacement to tilt angle θ [rad].

    Returns slopes: (N_SLOPES,) float32
        [0:N_SA_ACTIVE]      → θ_x per active subaperture
        [N_SA_ACTIVE:160]    → θ_y per active subaperture
    """
    sc = config.SLOPE_SCALE          # PIX_M / F_MLA_M = 2.957e-4 rad/px
    sx = ((cx - cx_ref) * sc)[active].astype(np.float32)
    sy = ((cy - cy_ref) * sc)[active].astype(np.float32)
    return np.concatenate([sx, sy])  # (160,)


# ── C extension (optional, falls back to numpy) ────────────────────────────────

import ctypes, os

_lib_candidates = [
    os.path.join(os.path.dirname(__file__), "c_ext", "cog_centroid.dll"),  # Windows
    os.path.join(os.path.dirname(__file__), "c_ext", "cog_centroid.so"),   # Linux/WSL
]

def _load_c_lib():
    for path in _lib_candidates:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.cog_frame.restype  = None
            lib.cog_frame.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
            ]
            return lib
        except OSError:
            pass
    return None

_c_lib = _load_c_lib()


def cog_centroid_c(frame: np.ndarray, bg_frame: np.ndarray, active: np.ndarray):
    """C-backed CoG; falls back to numpy if .dll/.so not found."""
    if _c_lib is None:
        return cog_centroid(frame, bg_frame, active)

    NY, NX  = config.N_LENSLETS_Y, config.N_LENSLETS_X
    W_frame = frame.shape[1]

    frame_c  = np.ascontiguousarray(frame,    dtype=np.uint8)
    bg_c     = np.ascontiguousarray(bg_frame, dtype=np.uint8)
    active_c = np.ascontiguousarray(active.ravel(), dtype=np.int32)
    cx_out   = np.zeros(NY * NX, dtype=np.float32)
    cy_out   = np.zeros(NY * NX, dtype=np.float32)

    _c_lib.cog_frame(
        frame_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        bg_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        active_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cy_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(NY), ctypes.c_int(NX),
        ctypes.c_int(config.PPS), ctypes.c_int(W_frame),
    )
    return cx_out.reshape(NY, NX), cy_out.reshape(NY, NX)


if __name__ == "__main__":
    DATA = config.DATA_DIR

    ref_frame = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    bg_frame  = np.array(Image.open(DATA / "sh_flat_bg.bmp"))

    active = build_active_mask(ref_frame)
    print(f"Active subapertures : {active.sum()}")   # must be 80

    cx_ref, cy_ref = build_subaperture_grid()

    cx0, cy0 = cog_centroid(ref_frame, bg_frame, active)
    dx = (cx0 - cx_ref)[active]
    dy = (cy0 - cy_ref)[active]
    print(f"Ref centroid error  : {np.abs(dx).mean():.4f} ± {np.abs(dx).std():.4f} px (x)")
    print(f"                      {np.abs(dy).mean():.4f} ± {np.abs(dy).std():.4f} px (y)")

    f0     = np.array(Image.open(DATA / "frames/frame_0000.bmp"))
    cx0t, cy0t = cog_centroid(f0, bg_frame, active)
    slopes = compute_slopes(cx0t, cy0t, cx_ref, cy_ref, active)
    print(f"Slope shape         : {slopes.shape}")       # (160,)
    print(f"Slope std           : {np.std(slopes):.5f} rad")
    print(f"C lib loaded        : {_c_lib is not None}")
