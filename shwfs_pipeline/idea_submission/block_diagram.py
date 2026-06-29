"""
Generates the SH-WFS pipeline block diagram figure for the BAH 2026 idea submission.
Saves to idea_submission/block_diagram.png
"""
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")

OUT = Path(__file__).parent / "block_diagram.png"

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

# Colour palette
C_INPUT  = "#4A90D9"
C_PROC   = "#2E7D32"
C_OUT    = "#C62828"
C_TURB   = "#E65100"
C_DM     = "#6A1B9A"
C_ARROW  = "#37474F"

def box(ax, x, y, w, h, label, sublabel=None, color=C_PROC, fs=10):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08", linewidth=1.5,
        edgecolor=color, facecolor=color + "30"
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color)
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                fontsize=7.5, color="#555555")

def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                fontsize=7.5, color=C_ARROW, style="italic")

# ── Row 1: main pipeline (y=5.8) ────────────────────────────────────────────
y0 = 5.8
boxes_top = [
    (1.2,  y0, "SH-WFS Frame",    "160x160 px / 5ms", C_INPUT),
    (3.6,  y0, "Centroiding",      "CoG (C ext, SIMD)", C_PROC),
    (6.0,  y0, "Slope Computation","[160] rad/m",       C_PROC),
    (8.4,  y0, "Modal Recon.",     "G_pinv (21x160)",   C_PROC),
    (10.8, y0, "Wavefront Phase",  "64x64 Zernike sum", C_OUT),
]
for (x, y, lab, sub, c) in boxes_top:
    box(ax, x, y, 2.0, 0.9, lab, sub, c)

# arrows top row
arrow(ax, 2.2,  y0, 2.6,  y0)
arrow(ax, 4.6,  y0, 5.0,  y0)
arrow(ax, 7.0,  y0, 7.4,  y0, "s [rad/m]")
arrow(ax, 9.4,  y0, 9.8,  y0, "a_k [rad]")

# ── Row 2: zonal + DM path (y=3.5) ─────────────────────────────────────────
y1 = 3.5
boxes_bot = [
    (6.0,  y1, "Zonal Recon.",     "D_pinv (121x160)",  C_PROC),
    (8.4,  y1, "DM Commands",      "C_pinv (121x121)",  C_DM),
    (10.8, y1, "Actuator Map",     "11x11 strokes [m]", C_OUT),
]
for (x, y, lab, sub, c) in boxes_bot:
    box(ax, x, y, 2.0, 0.9, lab, sub, c)

arrow(ax, 7.0,  y1, 7.4,  y1, "phi_act [121]")
arrow(ax, 9.4,  y1, 9.8,  y1)

# slope -> zonal
ax.annotate("", xy=(6.0, y1+0.45), xytext=(6.0, y0-0.45),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
ax.text(6.3, (y0+y1)/2, "slopes", ha="left", va="center",
        fontsize=7.5, color=C_ARROW, style="italic")

# ── Turbulence branch (y=1.4) ────────────────────────────────────────────────
y2 = 1.4
box(ax, 6.0, y2, 2.0, 0.9, "Turbulence Est.", "r0, tau0", C_TURB)
box(ax, 8.4, y2, 2.8, 0.9, "Turbulence Params",
    "r0=2.90mm / tau0=18.2ms", C_TURB)
ax.annotate("", xy=(6.0, y1-0.45), xytext=(6.0, y2+0.45),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
arrow(ax, 7.0, y2, 7.0, y2)
ax.annotate("", xy=(7.0, y2), xytext=(7.0, y2),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
arrow(ax, 7.1, y2, 7.9, y2)

# ── Precompute box ─────────────────────────────────────────────────────────
box(ax, 13.4, 4.7, 2.2, 2.5,
    "Precomputed\nMatrices",
    "G_pinv, D_pinv\nC_pinv, Z_basis",
    "#37474F", fs=9)

# arrows from precompute to modal/zonal/dm
for y_tgt in [y0, y1, y2]:
    ax.annotate("", xy=(9.4, y_tgt), xytext=(12.3, 4.7),
                arrowprops=dict(arrowstyle="-|>", color="#37474F",
                                lw=1.2, linestyle="dashed"))

# ── Performance badge ────────────────────────────────────────────────────────
perf_box = mpatches.FancyBboxPatch((0.3, 0.3), 4.5, 1.8,
    boxstyle="round,pad=0.1", linewidth=2,
    edgecolor="#1565C0", facecolor="#E3F2FD")
ax.add_patch(perf_box)
ax.text(2.55, 1.65, "Pipeline Performance", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#1565C0")
ax.text(2.55, 1.2,  "Latency: 2.2 ms mean / 2.6 ms p95", ha="center",
        va="center", fontsize=8.5, color="#1565C0")
ax.text(2.55, 0.8,  "Budget: 5.0 ms @ 200 Hz  [PASS]", ha="center",
        va="center", fontsize=8.5, color="#1565C0")
ax.text(2.55, 0.45, "r0 err: 6.2%   slope corr: 0.783", ha="center",
        va="center", fontsize=8.5, color="#1565C0")

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(8.0, 7.6, "SH-WFS Real-Time Pipeline — BAH 2026 PS-9",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#212121")
ax.text(8.0, 7.15,
        "10x10 lenslet array | 80 active SA | 160x160 px camera | 200 Hz | 11x11 DM",
        ha="center", va="center", fontsize=9, color="#555555")

fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Block diagram saved -> {OUT}")
