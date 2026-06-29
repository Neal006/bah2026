"""
SH-WFS Digital Twin — Standalone Matplotlib Animation

Plays back 200 frames in a loop as a live sensor digital twin.
Displays all 10 pipeline stages in a 3×4 grid at ~30 fps.

Run from shwfs_pipeline/ directory:
    python dashboard/digital_twin.py
"""
import sys
import itertools
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

import config
from centroid import build_active_mask, cog_centroid, compute_slopes
from reconstruct import reconstruct_modal, reconstruct_zonal, phase_rms
from turbulence import estimate_r0, estimate_tau0
from actuator import compute_commands, saturated_count

D        = config.DATA_DIR
N_FRAMES = 200

# ── Startup loading ───────────────────────────────────────────────────────────
print("Loading frames…")
frames = np.stack([
    np.array(Image.open(D / f"frames/frame_{t:04d}.bmp"))
    for t in range(N_FRAMES)
])  # (200, 160, 160) uint8

ref    = np.array(Image.open(D / "sh_flat_ref.bmp"))
bg     = np.array(Image.open(D / "sh_flat_bg.bmp"))
active = build_active_mask(ref)                         # (10,10) bool
cx_ref, cy_ref = cog_centroid(ref, bg, active)          # measured reference centroids

gt_sx = np.load(D / "ground_truth/all_slopes_x.npy")   # (200,10,10)
gt_sy = np.load(D / "ground_truth/all_slopes_y.npy")
gt_zk = np.load(D / "ground_truth/all_zernike.npy")    # (200,21), stored 50x rad

gt_std     = float(np.std(np.hstack([gt_sx[:, active], gt_sy[:, active]])))
s0         = compute_slopes(*cog_centroid(frames[0], bg, active), cx_ref, cy_ref, active)
slope_gain = gt_std / max(float(np.std(s0)), 1e-9)

active_ij = np.argwhere(active)   # (80,2)

GT_R0_MM     = 3.087
GT_TAU0_MS   = 19.385
GT_PHASE_RMS = float(np.mean([
    float(np.sqrt(np.sum((gt_zk[t, 1:] / 50.0) ** 2)))
    for t in range(N_FRAMES)
]))

# Rolling buffers
rms_buf    = deque(maxlen=50)
lat_buf    = deque(maxlen=50)
slopes_buf = deque(maxlen=50)
r0_buf     = deque(maxlen=50)

print(f"slope_gain={slope_gain:.3f}x  GT_PHASE_RMS={GT_PHASE_RMS:.4f} rad")
print("Initialising figure…")


# ── Process one frame ─────────────────────────────────────────────────────────
def process_frame(t):
    t0       = time.perf_counter()
    cx, cy   = cog_centroid(frames[t], bg, active)
    slopes   = compute_slopes(cx, cy, cx_ref, cy_ref, active) * slope_gain
    a, phase = reconstruct_modal(slopes)
    phi_act  = reconstruct_zonal(slopes).ravel()
    v_cmd    = compute_commands(phi_act)
    rms      = phase_rms(phase)
    sat      = saturated_count(v_cmd)
    lat_ms   = (time.perf_counter() - t0) * 1e3

    slopes_buf.append(slopes); rms_buf.append(rms); lat_buf.append(lat_ms)
    r0_mm  = estimate_r0(np.array(slopes_buf)) * 1e3 if len(slopes_buf) >= 2 else GT_R0_MM
    r0_buf.append(r0_mm)
    tau0_ms = estimate_tau0(r0_mm * 1e-3) * 1e3

    sx_map = np.zeros((10, 10), dtype=np.float32)
    sy_map = np.zeros((10, 10), dtype=np.float32)
    sx_map[active] = slopes[:80]
    sy_map[active] = slopes[80:]

    gt_slopes_t = np.concatenate([gt_sx[t][active], gt_sy[t][active]])
    corr = float(np.corrcoef(slopes, gt_slopes_t)[0, 1]) if np.std(slopes) > 0 else 0.0

    return dict(
        frame=frames[t], cx=cx, cy=cy, slopes=slopes,
        a=a, phase=phase, v_cmd=v_cmd, rms=rms, sat=sat,
        lat_ms=lat_ms, r0_mm=r0_mm, tau0_ms=tau0_ms, corr=corr,
        gt_a=gt_zk[t] / 50.0,
        sx_map=sx_map, sy_map=sy_map,
    )


# ── Figure & axes ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 12), facecolor="#0e1117")
fig.suptitle("SH-WFS Digital Twin", fontsize=13, fontweight="bold", color="white")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
                        top=0.93, bottom=0.06, left=0.04, right=0.97)

ax1  = fig.add_subplot(gs[0, 0])
ax2  = fig.add_subplot(gs[0, 1])
ax3  = fig.add_subplot(gs[0, 2:])
ax4  = fig.add_subplot(gs[1, 0])
ax5  = fig.add_subplot(gs[1, 1:3])
ax6  = fig.add_subplot(gs[1, 3])
ax7  = fig.add_subplot(gs[2, 0])
ax8  = fig.add_subplot(gs[2, 1])
ax9  = fig.add_subplot(gs[2, 2])
ax10 = fig.add_subplot(gs[2, 3])

_LABEL_KW  = dict(color="white", fontsize=8, fontweight="bold")
_TICK_KW   = dict(colors="#aaaaaa", labelsize=7)
_BG        = "#1a1d24"

for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]:
    ax.set_facecolor(_BG)
    ax.tick_params(color="#aaaaaa", labelcolor="#aaaaaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")


def _title(ax, txt):
    ax.set_title(txt, color="white", fontsize=8.5, fontweight="bold", pad=4)


# P1 — raw camera
im1 = ax1.imshow(np.zeros((160, 160)), cmap="gray", vmin=0, vmax=255,
                 origin="upper", animated=True)
for k in range(1, 10):
    ax1.axhline(k * 16 - 0.5, color="#00ff88", lw=0.4, alpha=0.5)
    ax1.axvline(k * 16 - 0.5, color="#00ff88", lw=0.4, alpha=0.5)
ax1.axis("off"); _title(ax1, "P1  Camera + SA Grid")

# P2 — centroids overlay
im2     = ax2.imshow(np.zeros((160, 160)), cmap="gray", vmin=0, vmax=255,
                     origin="upper", animated=True)
sc_meas = ax2.scatter([], [], c="#ff4444", s=8, label="meas", zorder=3)
sc_ref  = ax2.scatter([], [], c="#4488ff", s=8, label="ref",  zorder=2)
ax2.legend(fontsize=6, loc="upper right",
           facecolor="#333", labelcolor="white", framealpha=0.7)
ax2.axis("off"); ax2.set_xlim(0, 160); ax2.set_ylim(160, 0)
_title(ax2, "P2  Centroids")

# P3 — slope quiver
yi, xi = np.mgrid[0:10, 0:10]
X_q = xi[active].ravel(); Y_q = yi[active].ravel()
quiv = ax3.quiver(X_q, Y_q, np.zeros(80), np.zeros(80), np.zeros(80),
                  cmap="viridis", scale_units="xy", scale=6000,
                  width=0.012, pivot="mid", clim=(0, 2000))
cb3 = fig.colorbar(quiv, ax=ax3, fraction=0.025, pad=0.02)
cb3.set_label("rad/m", color="white", fontsize=7)
cb3.ax.yaxis.set_tick_params(color="#aaaaaa", labelcolor="#aaaaaa")
ax3.set_xlim(-0.7, 9.7); ax3.set_ylim(9.7, -0.7)
ax3.set_aspect("equal"); ax3.set_xlabel("SA col", **_LABEL_KW)
ax3.set_ylabel("SA row", **_LABEL_KW)
_title(ax3, "P3  Slope Vector Field")

# P4 — phase map
im4 = ax4.imshow(np.zeros((64, 64)), cmap="coolwarm", vmin=-0.5, vmax=0.5,
                 origin="upper", animated=True)
cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cb4.set_label("rad", color="white", fontsize=7)
cb4.ax.yaxis.set_tick_params(color="#aaaaaa", labelcolor="#aaaaaa")
_title(ax4, "P4  Wavefront Phase Map")

# P5 — Zernike bars
z_idx    = np.arange(1, 22)
bars_gt  = ax5.bar(z_idx - 0.22, np.zeros(21), width=0.38,
                   color="#ff9944", alpha=0.85, label="GT")
bars_est = ax5.bar(z_idx + 0.22, np.zeros(21), width=0.38,
                   color="#4488ff", alpha=0.85, label="Est")
ax5.axhline(0, color="#888", lw=0.5)
ax5.set_xlim(0.3, 21.7); ax5.set_ylim(-0.5, 0.5)
ax5.set_xlabel("Noll index", color="#aaaaaa", fontsize=7)
ax5.set_ylabel("rad", color="#aaaaaa", fontsize=7)
ax5.legend(fontsize=7, facecolor="#333", labelcolor="white", framealpha=0.7)
ax5.set_xticks(z_idx[::2]); ax5.tick_params(labelcolor="#aaaaaa")
_title(ax5, "P5  Zernike Coefficients Z1-Z21")

# P6 — DM actuator map
im6 = ax6.imshow(np.zeros((11, 11)), cmap="RdBu_r", animated=True)
cb6 = fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cb6.set_label("nm", color="white", fontsize=7)
cb6.ax.yaxis.set_tick_params(color="#aaaaaa", labelcolor="#aaaaaa")
_title(ax6, "P6  DM Actuators (nm)")

# P7 — phase RMS
line7,  = ax7.plot([], [], color="#4488ff", lw=1.8, label="RMS")
ax7.axhline(GT_PHASE_RMS, color="#ffff44", ls="--", lw=1.2,
            label=f"GT {GT_PHASE_RMS:.3f}")
ax7.set_xlim(0, 50); ax7.set_ylim(0, 0.7)
ax7.set_xlabel("last 50 frames", color="#aaaaaa", fontsize=7)
ax7.set_ylabel("rad", color="#aaaaaa", fontsize=7)
ax7.legend(fontsize=7, facecolor="#333", labelcolor="white", framealpha=0.7)
_title(ax7, "P7  Phase RMS")

# P8 — r0 rolling
line8,  = ax8.plot([], [], color="#44ff88", lw=1.8, label="r0 est")
ax8.axhline(GT_R0_MM, color="#ffff44", ls="--", lw=1.2, label=f"GT {GT_R0_MM}")
ax8.set_xlim(0, 50); ax8.set_ylim(0, 8)
ax8.set_xlabel("last 50 frames", color="#aaaaaa", fontsize=7)
ax8.set_ylabel("mm", color="#aaaaaa", fontsize=7)
ax8.legend(fontsize=7, facecolor="#333", labelcolor="white", framealpha=0.7)
_title(ax8, "P8  r₀ Estimate (rolling)")

# P9 — latency
bars9  = ax9.bar(range(50), np.zeros(50), color="#4488ff", width=0.85, alpha=0.75)
ax9.axhline(5.0, color="red", lw=1.5, label="5ms budget")
ax9.set_xlim(0, 50); ax9.set_ylim(0, 10)
ax9.set_xlabel("last 50 frames", color="#aaaaaa", fontsize=7)
ax9.set_ylabel("ms", color="#aaaaaa", fontsize=7)
ax9.legend(fontsize=7, facecolor="#333", labelcolor="white", framealpha=0.7)
_title(ax9, "P9  Pipeline Latency")

# P10 — text metrics
ax10.axis("off"); _title(ax10, "P10  Live Metrics")
txt10 = ax10.text(0.08, 0.88, "", transform=ax10.transAxes,
                  va="top", ha="left", fontsize=9.5,
                  fontfamily="monospace", color="white",
                  bbox=dict(boxstyle="round", fc="#2a2d34", ec="#444", alpha=0.9))


# ── Update function ───────────────────────────────────────────────────────────
def update(t):
    d = process_frame(t)

    # P1
    im1.set_data(d["frame"])

    # P2
    meas_xy = np.column_stack([d["cx"][active_ij[:, 0], active_ij[:, 1]],
                                d["cy"][active_ij[:, 0], active_ij[:, 1]]])
    ref_xy  = np.column_stack([cx_ref[active_ij[:, 0], active_ij[:, 1]],
                                cy_ref[active_ij[:, 0], active_ij[:, 1]]])
    im2.set_data(d["frame"])
    sc_meas.set_offsets(meas_xy)
    sc_ref.set_offsets(ref_xy)

    # P3
    sx_a = d["sx_map"][active]; sy_a = d["sy_map"][active]
    mag  = np.sqrt(sx_a**2 + sy_a**2)
    quiv.set_UVC(sx_a, sy_a, mag)

    # P4
    vmax = max(float(np.abs(d["phase"]).max()), 0.01)
    im4.set_data(d["phase"]); im4.set_clim(-vmax, vmax)

    # P5
    gt_clip = np.clip(d["gt_a"], -1, 1)
    est_clip = np.clip(d["a"], -1, 1)
    for rect, h in zip(bars_gt,  gt_clip):  rect.set_height(h)
    for rect, h in zip(bars_est, est_clip): rect.set_height(h)
    ylim = max(float(np.abs(est_clip).max()), float(np.abs(gt_clip).max()), 0.05) * 1.4
    ax5.set_ylim(-ylim, ylim)

    # P6
    dm_nm = d["v_cmd"].reshape(11, 11) * 1e9
    im6.set_data(dm_nm); im6.set_clim(dm_nm.min() - 0.01, dm_nm.max() + 0.01)

    # P7
    rms_l = list(rms_buf)
    line7.set_data(range(len(rms_l)), rms_l)

    # P8
    r0_l = list(r0_buf)
    line8.set_data(range(len(r0_l)), r0_l)

    # P9
    lat_l = list(lat_buf)
    for i, rect in enumerate(bars9):
        rect.set_height(lat_l[i] if i < len(lat_l) else 0)
        rect.set_color("#ff4444" if (i < len(lat_l) and lat_l[i] > 5.0) else "#4488ff")

    # P10
    txt10.set_text(
        f" Frame  : {t:03d} / 199\n"
        f" τ0     : {d['tau0_ms']:6.1f} ms  (GT {GT_TAU0_MS:.1f})\n"
        f" r₀     : {d['r0_mm']:6.3f} mm  (GT {GT_R0_MM:.3f})\n"
        f" Sat    :   {d['sat']:3d} / 121\n"
        f" Latency: {d['lat_ms']:5.2f} ms\n"
        f" Corr   : {d['corr']:+.3f}"
    )

    fig.suptitle(
        f"SH-WFS Digital Twin  ·  Frame {t:03d}/199  "
        f"·  r₀={d['r0_mm']:.2f} mm  "
        f"·  RMS={d['rms']:.3f} rad  "
        f"·  {d['lat_ms']:.1f} ms/frame",
        fontsize=12, fontweight="bold", color="white"
    )


anim = FuncAnimation(
    fig, update,
    frames=itertools.cycle(range(N_FRAMES)),
    interval=50,            # 50ms = ~20fps target; actual rate CPU-limited
    blit=False,             # blit=True breaks ylim + clim updates
    cache_frame_data=False, # update() is stateful (mutates deques)
    repeat=False,           # cycle() handles looping
)

plt.show()
