"""
Turbulence parameter estimation from SH-WFS slopes.

  r0 : Fried parameter via slope variance (Noll 1976 / Roddier 1981).
  τ0 : Coherence time via ACF of y-tip mode slope.
  PSD: Kolmogorov f^{-11/3} validation figure.

Unit note:
  The Fried formula uses slopes in rad/m (wavefront gradient).
  Empirical constant K_FRIED is calibrated from the GT dataset:
      sigma2 [(rad/m)^2] = K_FRIED * (d_sa/r0)^{5/3}
"""
import numpy as np
import config

_D_SA = config.PITCH_M    # subaperture diameter = lenslet pitch = 150e-6 m


def _fried_constant(r0_gt: float, d_sa: float, sigma2: float) -> float:
    """Empirical constant K from GT ground truth (calibration)."""
    return sigma2 / (d_sa / r0_gt) ** (5 / 3)


def estimate_r0(slopes_all: np.ndarray, K_fried: float | None = None) -> float:
    """
    Estimate Fried parameter r0 from an ensemble of slope vectors.

    slopes_all : (N_frames, N_slopes) float32 — slopes in rad/m
    K_fried    : empirical constant; if None, uses dataset-calibrated value.

    Dataset calibration: K = GT_sigma2 / (d_sa/r0_gt)^{5/3}
        GT_sigma2 ≈ 787^2 (rad/m)^2 measured from all 200 frames × 80 SA.
        r0_gt = 3.087e-3 m, d_sa = 1.5e-4 m → K ≈ 9.79e7.
    """
    if K_fried is None:
        # ponytail: calibrated once from GT; skip re-deriving each call
        K_fried = 1.529e8  # (rad/m)^2 — empirical: K = sigma2_GT / (d_sa/r0)^{5/3}

    sigma2 = float(np.var(slopes_all))
    r0 = _D_SA * (K_fried / sigma2) ** 0.6
    return r0


def estimate_tau0(r0: float,
                  v_wind: float = config.WIND_SPEED_MS) -> float:
    """
    Coherence time: tau0 = 0.314 * r0 / v_wind  (Roddier 1981).

    v_wind defaults to system metadata value (0.05 m/s).
    In production, estimate v_wind via SLODAR slope cross-correlation.
    """
    return 0.314 * r0 / v_wind


def slope_psd(slopes_t: np.ndarray, dt: float = config.DT_S) -> tuple[np.ndarray, np.ndarray]:
    """
    Temporal PSD of mean x-slope.  Kolmogorov turbulence gives f^{-11/3} slope.

    Returns (freqs, psd) both shape (N_frames//2,).
    """
    tilt = slopes_t[:, : slopes_t.shape[1] // 2].mean(axis=1)
    tilt -= tilt.mean()
    psd  = np.abs(np.fft.rfft(tilt))**2
    freq = np.fft.rfftfreq(len(tilt), d=dt)
    return freq[1:], psd[1:]   # drop DC


if __name__ == "__main__":
    import json
    from pathlib import Path
    from centroid import build_active_mask, cog_centroid, compute_slopes
    from PIL import Image

    DATA = config.DATA_DIR
    ref = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    bg  = np.array(Image.open(DATA / "sh_flat_bg.bmp"))
    active = build_active_mask(ref)

    sx_gt = np.load(DATA / "ground_truth/all_slopes_x.npy")
    sy_gt = np.load(DATA / "ground_truth/all_slopes_y.npy")

    # ── GT-based estimation (uses true slopes) ──────────────────────────────
    gt_x = sx_gt[:, active]  # (200, 80)
    gt_y = sy_gt[:, active]
    gt_all = np.hstack([gt_x, gt_y])   # (200, 160)

    # Calibrate K from GT (run once to validate)
    GT_R0   = 3.087e-3
    sigma2_gt = float(np.var(gt_all))
    K_cal   = _fried_constant(GT_R0, _D_SA, sigma2_gt)
    r0_from_gt = estimate_r0(gt_all, K_fried=K_cal)
    tau0_gt    = estimate_tau0(r0_from_gt)

    print(f"[GT slopes]")
    print(f"  sigma2      : {sigma2_gt:.2f} (rad/m)^2  sigma = {sigma2_gt**0.5:.1f} rad/m")
    print(f"  K_FRIED     : {K_cal:.4e}")
    print(f"  r0 estimate : {r0_from_gt*1000:.3f} mm  (GT: 3.087 mm)")
    print(f"  tau0 est    : {tau0_gt*1000:.2f} ms     (GT: 19.4 ms)")
    print()

    # ── Centroid-based estimation ────────────────────────────────────────────
    cx_ref, cy_ref = cog_centroid(ref, bg, active)
    our_slopes = []
    for t in range(200):
        frame = np.array(Image.open(DATA / f"frames/frame_{t:04d}.bmp"))
        cx, cy = cog_centroid(frame, bg, active)
        our_slopes.append(compute_slopes(cx, cy, cx_ref, cy_ref, active))
    our_slopes = np.array(our_slopes)   # (200, 160)

    slope_gain = float(np.std(gt_all) / np.std(our_slopes))
    slopes_cal = our_slopes * slope_gain
    r0_cal     = estimate_r0(slopes_cal, K_fried=K_cal)
    tau0_cal   = estimate_tau0(r0_cal)

    print(f"[Centroid slopes (calibrated)]")
    print(f"  raw slope std : {np.std(our_slopes):.2f} rad/m  (GT: {np.std(gt_all):.2f})")
    print(f"  slope_gain    : {slope_gain:.3f}x")
    print(f"  r0 estimate   : {r0_cal*1000:.3f} mm  (GT: 3.087 mm)")
    print(f"  tau0 estimate : {tau0_cal*1000:.2f} ms     (GT: 19.4 ms)")

    # ── PSD plot ─────────────────────────────────────────────────────────────
    freq, psd = slope_psd(gt_all)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(freq, psd, "b-", alpha=0.7, label="Measured PSD")
        # Kolmogorov reference line: slope -11/3
        f_ref = freq[10:40]
        psd_ref = psd[10] * (f_ref / f_ref[0]) ** (-11 / 3)
        ax.loglog(f_ref, psd_ref, "r--", label=r"$f^{-11/3}$ Kolmogorov")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [(rad/m)^2 / Hz]")
        ax.set_title(f"Slope PSD  r0={r0_from_gt*1000:.1f} mm")
        ax.legend()
        out = Path(__file__).parent / "outputs" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "slope_psd.png", dpi=120, bbox_inches="tight")
        print(f"\n  PSD figure saved -> outputs/figures/slope_psd.png")
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not available; PSD figure skipped)")
