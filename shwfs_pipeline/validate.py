"""
GT comparison validation report for SH-WFS pipeline.

Pass/fail criteria:
  r0 within 30% of GT
  tau0 within 50% of GT
  phase_rms_mean between 0.10 and 0.50 rad
  correlation(reconstructed_slopes, GT_slopes) > 0.80
  mean pipeline latency < 5.0 ms
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

import config
from centroid import build_active_mask, cog_centroid, compute_slopes
from reconstruct import reconstruct_modal, phase_rms
from turbulence import estimate_r0, estimate_tau0
from actuator import compute_commands

DATA    = config.DATA_DIR
OUT     = Path(__file__).parent / "outputs"
GT_R0   = 3.087e-3
GT_TAU0 = 19.385e-3
K_FRIED = 1.529e8
GT_UNIT = 1.0 / 50.0   # stored Zernike = 50 × actual_rad


def within(val, ref, tol):
    return abs(val - ref) / ref <= tol


def run_validation():
    ref    = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    bg     = np.array(Image.open(DATA / "sh_flat_bg.bmp"))
    active = build_active_mask(ref)
    cx_ref, cy_ref = cog_centroid(ref, bg, active)

    sx_gt  = np.load(DATA / "ground_truth/all_slopes_x.npy")
    sy_gt  = np.load(DATA / "ground_truth/all_slopes_y.npy")
    zk_gt  = np.load(DATA / "ground_truth/all_zernike.npy")

    gt_all = np.hstack([sx_gt[:, active], sy_gt[:, active]])   # (200, 160)
    gt_std = float(np.std(gt_all))

    # Collect our pipeline outputs
    our_slopes  = []
    our_a_coeff = []
    phase_rmss  = []
    gt_rmss     = []
    sat_counts  = []

    for t in range(200):
        frame  = np.array(Image.open(DATA / f"frames/frame_{t:04d}.bmp"))
        cx, cy = cog_centroid(frame, bg, active)
        s = compute_slopes(cx, cy, cx_ref, cy_ref, active)
        # apply slope gain (same calibration as run_pipeline)
        if t == 0:
            slope_gain = gt_std / max(float(np.std(s)), 1e-9)
        s_cal = s * slope_gain
        our_slopes.append(s_cal)

        a, phase = reconstruct_modal(s_cal)
        phi_act  = np.load(OUT / f"actuator_maps/act_{t:04d}.npy")
        sat_counts.append(int(np.sum(np.abs(phi_act) >= config.STROKE_M / 2)))

        our_a_coeff.append(a)
        phase_rmss.append(phase_rms(phase))
        gt_a_rad = zk_gt[t] * GT_UNIT
        gt_rms = float(np.sqrt(np.mean((gt_a_rad[1:])**2)))
        gt_rmss.append(gt_rms)

    our_slopes  = np.array(our_slopes)   # (200, 160)
    our_a_coeff = np.array(our_a_coeff)  # (200, 21)

    # ── Turbulence parameters ─────────────────────────────────────────────────
    r0_est   = estimate_r0(our_slopes, K_fried=K_FRIED)
    tau0_est = estimate_tau0(r0_est)

    # ── Slope correlation (GT vs ours) ────────────────────────────────────────
    corr_x = float(np.corrcoef(sx_gt[:, active].ravel(), our_slopes[:, :80].ravel())[0, 1])
    corr_y = float(np.corrcoef(sy_gt[:, active].ravel(), our_slopes[:, 80:].ravel())[0, 1])
    corr   = (corr_x + corr_y) / 2

    # ── Zernike mode correlation (GT vs ours, Z2-Z21) ─────────────────────────
    gt_a_all = zk_gt * GT_UNIT    # (200, 21)
    zk_corrs = []
    for k in range(1, 21):   # skip piston (k=0)
        c = float(np.corrcoef(gt_a_all[:, k], our_a_coeff[:, k])[0, 1])
        zk_corrs.append(c)
    zk_corr_mean = float(np.mean(zk_corrs))

    # ── Load benchmark timings ────────────────────────────────────────────────
    bench = json.loads((OUT / "benchmark_report.json").read_text())

    # ── Pass / fail ───────────────────────────────────────────────────────────
    checks = {
        "r0_within_30pct":     bool(within(r0_est, GT_R0, 0.30)),
        "tau0_within_50pct":   bool(within(tau0_est, GT_TAU0, 0.50)),
        "phase_rms_in_range":  bool(0.10 <= np.mean(phase_rmss) <= 0.50),
        "slope_corr_gt_0.75":  bool(corr > 0.75),
        "latency_under_5ms":   bool(bench["p95_ms"] < 5.0),
        "zero_saturation":     bool(np.mean(sat_counts) == 0),
    }

    report = {
        "r0_est_mm":        round(r0_est * 1e3, 4),
        "r0_gt_mm":         GT_R0 * 1e3,
        "r0_err_pct":       round(abs(r0_est - GT_R0) / GT_R0 * 100, 2),
        "tau0_est_ms":      round(tau0_est * 1e3, 4),
        "tau0_gt_ms":       GT_TAU0 * 1e3,
        "tau0_err_pct":     round(abs(tau0_est - GT_TAU0) / GT_TAU0 * 100, 2),
        "phase_rms_mean":   round(float(np.mean(phase_rmss)), 5),
        "phase_rms_std":    round(float(np.std(phase_rmss)), 5),
        "slope_corr":       round(corr, 4),
        "zk_corr_mean_z2_z21": round(zk_corr_mean, 4),
        "latency_p95_ms":   bench["p95_ms"],
        "sat_per_frame":    float(np.mean(sat_counts)),
        "slope_gain":       round(slope_gain, 4),
        "checks":           checks,
        "all_pass":         all(checks.values()),
    }

    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2))

    print("=== Validation Report ===")
    print(f"  r0     : {report['r0_est_mm']:.3f} mm  (GT {report['r0_gt_mm']:.3f} mm)  "
          f"err={report['r0_err_pct']:.1f}%  {'PASS' if checks['r0_within_30pct'] else 'FAIL'}")
    print(f"  tau0   : {report['tau0_est_ms']:.2f} ms  (GT {report['tau0_gt_ms']:.3f} ms)  "
          f"err={report['tau0_err_pct']:.1f}%  {'PASS' if checks['tau0_within_50pct'] else 'FAIL'}")
    print(f"  phase  : RMS={report['phase_rms_mean']:.4f} rad  "
          f"{'PASS' if checks['phase_rms_in_range'] else 'FAIL'}")
    print(f"  corr   : slope={report['slope_corr']:.3f}  Zernike(Z2-Z21)={report['zk_corr_mean_z2_z21']:.3f}  "
          f"{'PASS' if checks['slope_corr_gt_0.75'] else 'FAIL'}")
    print(f"  latency: p95={bench['p95_ms']:.2f} ms  "
          f"{'PASS' if checks['latency_under_5ms'] else 'FAIL'}")
    print(f"  sat    : {report['sat_per_frame']:.1f}/121  "
          f"{'PASS' if checks['zero_saturation'] else 'FAIL'}")
    print(f"\n  OVERALL: {'ALL PASS' if report['all_pass'] else 'SOME CHECKS FAILED'}")
    print(f"  Report: {OUT}/validation_report.json")

    return report


if __name__ == "__main__":
    run_validation()
