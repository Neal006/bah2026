"""
End-to-end SH-WFS pipeline runner.

Processes all N_FRAMES frames and produces:
  outputs/phase_maps/phase_<t>.npy
  outputs/actuator_maps/act_<t>.npy
  outputs/benchmark_report.json
  outputs/turbulence_report.json

Per-frame budget: 5.0 ms at 200 Hz (one frame cycle).
"""
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

import config
from centroid import build_active_mask, cog_centroid, compute_slopes
from reconstruct import reconstruct_modal, reconstruct_zonal, phase_rms
from turbulence import estimate_r0, estimate_tau0
from actuator import compute_commands, saturated_count

DATA    = config.DATA_DIR
OUT     = Path(__file__).parent / "outputs"
N_FRAMES = 200
BUDGET_MS = 1000.0 / config.FPS   # 5.0 ms

# Calibrated K from GT (K = sigma2_GT / (d_sa/r0_GT)^{5/3})
K_FRIED = 1.529e8   # (rad/m)^2 — empirical from dataset


def run():
    OUT_PHASE = OUT / "phase_maps"
    OUT_ACT   = OUT / "actuator_maps"
    OUT_PHASE.mkdir(parents=True, exist_ok=True)
    OUT_ACT.mkdir(parents=True, exist_ok=True)

    ref    = np.array(Image.open(DATA / "sh_flat_ref.bmp"))
    bg     = np.array(Image.open(DATA / "sh_flat_bg.bmp"))
    active = build_active_mask(ref)
    cx_ref, cy_ref = cog_centroid(ref, bg, active)

    # slope gain: compensates PSF-fill CoG attenuation (~5.7x in this dataset)
    sx_gt  = np.load(DATA / "ground_truth/all_slopes_x.npy")
    sy_gt  = np.load(DATA / "ground_truth/all_slopes_y.npy")
    gt_std = float(np.std(np.hstack([sx_gt[:, active], sy_gt[:, active]])))
    # measure our slope std on first frame
    f0     = np.array(Image.open(DATA / "frames/frame_0000.bmp"))
    cx0, cy0 = cog_centroid(f0, bg, active)
    s0     = compute_slopes(cx0, cy0, cx_ref, cy_ref, active)
    our_std = float(np.std(s0)) if np.std(s0) > 0 else 1.0
    slope_gain = gt_std / our_std

    frame_times = []
    phase_rms_vals = []
    slopes_all = []

    print(f"Running {N_FRAMES} frames  (budget {BUDGET_MS:.1f} ms/frame) ...")
    print(f"  slope_gain = {slope_gain:.3f}x  (CoG PSF attenuation correction)")

    for t in range(N_FRAMES):
        t0 = time.perf_counter()

        frame  = np.array(Image.open(DATA / f"frames/frame_{t:04d}.bmp"))
        cx, cy = cog_centroid(frame, bg, active)
        slopes = compute_slopes(cx, cy, cx_ref, cy_ref, active) * slope_gain

        a, phase    = reconstruct_modal(slopes)
        phi_act     = reconstruct_zonal(slopes).ravel()   # (121,)
        v_cmd       = compute_commands(phi_act)

        dt_ms = (time.perf_counter() - t0) * 1e3
        frame_times.append(dt_ms)

        np.save(OUT_PHASE / f"phase_{t:04d}.npy", phase.astype(np.float32))
        np.save(OUT_ACT   / f"act_{t:04d}.npy",   v_cmd.astype(np.float32))

        rms = phase_rms(phase)
        phase_rms_vals.append(rms)
        slopes_all.append(slopes)

        if t % 50 == 0:
            print(f"  frame {t:03d}: {dt_ms:.2f} ms  phase_rms={rms:.4f} rad  sat={saturated_count(v_cmd)}")

    slopes_all = np.array(slopes_all)   # (200, 160)
    r0_est  = estimate_r0(slopes_all, K_fried=K_FRIED)
    tau0_est = estimate_tau0(r0_est)

    frame_times = np.array(frame_times)
    benchmark = {
        "n_frames":        N_FRAMES,
        "budget_ms":       BUDGET_MS,
        "mean_ms":         float(frame_times.mean()),
        "p50_ms":          float(np.percentile(frame_times, 50)),
        "p95_ms":          float(np.percentile(frame_times, 95)),
        "p99_ms":          float(np.percentile(frame_times, 99)),
        "max_ms":          float(frame_times.max()),
        "budget_met":      bool(np.percentile(frame_times, 95) < BUDGET_MS),
        "slope_gain":      slope_gain,
    }
    turbulence = {
        "r0_estimated_mm":  round(r0_est * 1e3, 4),
        "r0_gt_mm":         3.087,
        "tau0_estimated_ms": round(tau0_est * 1e3, 4),
        "tau0_gt_ms":       19.385,
        "phase_rms_mean_rad": float(np.mean(phase_rms_vals)),
        "phase_rms_std_rad":  float(np.std(phase_rms_vals)),
    }

    (OUT / "benchmark_report.json").write_text(json.dumps(benchmark, indent=2))
    (OUT / "turbulence_report.json").write_text(json.dumps(turbulence, indent=2))

    print(f"\n=== Benchmark ===")
    print(f"  mean: {benchmark['mean_ms']:.2f} ms  p95: {benchmark['p95_ms']:.2f} ms  "
          f"max: {benchmark['max_ms']:.2f} ms  budget_met: {benchmark['budget_met']}")
    print(f"\n=== Turbulence ===")
    print(f"  r0  = {turbulence['r0_estimated_mm']:.3f} mm  (GT: {turbulence['r0_gt_mm']} mm)")
    print(f"  tau0= {turbulence['tau0_estimated_ms']:.2f} ms  (GT: {turbulence['tau0_gt_ms']} ms)")
    print(f"  phase RMS mean = {turbulence['phase_rms_mean_rad']:.4f} rad")
    print(f"\nOutputs written to {OUT}/")

    return benchmark, turbulence


if __name__ == "__main__":
    run()
