import json
import numpy as np
from pathlib import Path

# Repo-relative path works both locally and on Streamlit Community Cloud.
# Local layout: bah2026/shwfs_pipeline/ + bah2026/shwfs_synthetic/
_HERE    = Path(__file__).parent
DATA_DIR = _HERE.parent / "shwfs_synthetic"
if not DATA_DIR.exists():
    # Fallback for Windows dev with absolute path
    DATA_DIR = Path(r"C:\Users\Admin\bah2026\shwfs_synthetic")

with open(DATA_DIR / "metadata.json") as f:
    META = json.load(f)

# MLA
N_LENSLETS_X  = META["mla"]["n_lenslets_x"]           # 10
N_LENSLETS_Y  = META["mla"]["n_lenslets_y"]           # 10
N_SA_ACTIVE   = META["mla"]["n_subapertures_active"]   # 80
PITCH_M       = META["mla"]["pitch_m"]                 # 150e-6
F_MLA_M       = META["mla"]["focal_length_m"]          # 18.6e-3

# Camera
PIX_M         = META["camera"]["pixel_size_m"]         # 5.5e-6
PPS           = META["camera"]["pixels_per_subaperture"]  # 16
FRAME_W       = META["camera"]["frame_width_px"]       # 160
FRAME_H       = META["camera"]["frame_height_px"]      # 160
FPS           = META["camera"]["frame_rate_hz"]        # 200.0
DT_S          = 1.0 / FPS                             # 0.005

# Optics / turbulence
WAVELENGTH_M  = META["turbulence"]["wavelength_m"]     # 632.8e-9
PUPIL_D_M     = META["pupil"]["diameter_m"]            # 1.5e-3
R_PUPIL_M     = PUPIL_D_M / 2                         # 7.5e-4

# DM
N_ACT_X       = META["dm"]["n_actuators_x"]            # 11
N_ACT_Y       = META["dm"]["n_actuators_y"]            # 11
N_ACT         = META["dm"]["n_actuators_total"]        # 121
STROKE_M      = META["dm"]["stroke_m"]                 # 5e-6
COUPLING_FRAC = META["dm"]["coupling_fraction"]        # 0.15

# Reconstruction
N_MODES       = 21        # Zernike Z1-Z21 Noll
PUPIL_GRID    = 64        # wavefront map resolution (pixels)
N_SLOPES      = 2 * N_SA_ACTIVE  # 160

WIND_SPEED_MS = META["turbulence"]["wind_speed_ms"]   # 0.05 m/s

# Slope scale: pixel displacement → wavefront gradient [rad/m]
# slope_x [rad/m] = Δpx * 2π * PIX_M / (F_MLA_M * λ)
# This matches GT all_slopes_x.npy convention (verified: GT std ~990 rad/m)
SLOPE_SCALE   = 2 * np.pi * PIX_M / (F_MLA_M * WAVELENGTH_M)  # ~2936 rad/m per px

# G matrix physical scale: slopes in rad/m ↔ Zernike coefficients in rad
# G[row,k] = (1/R_PUPIL_M) × mean(dZ_k/dx_norm over subaperture)
G_PHYS_SCALE  = 1.0 / R_PUPIL_M  # 1/0.00075 = 1333 /m

if __name__ == "__main__":
    print(f"N_SLOPES: {N_SLOPES}  N_MODES: {N_MODES}  N_ACT: {N_ACT}")
    print(f"SLOPE_SCALE: {SLOPE_SCALE:.4e} rad/m/px  G_PHYS_SCALE: {G_PHYS_SCALE:.4e} /m")
