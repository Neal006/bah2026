"""
SH-WFS Digital Twin — Streamlit Web Dashboard

Real-time simulation of the SH-WFS pipeline with live visualization.

Local:
    streamlit run dashboard/shwfs_dashboard.py

Streamlit Community Cloud:
    Push repo with shwfs_synthetic/ committed.
    Set entry file to shwfs_pipeline/dashboard/shwfs_dashboard.py.
    requirements.txt: numpy, Pillow, matplotlib, plotly, opencv-python-headless, scipy, streamlit
"""
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

import config
from centroid import build_active_mask, cog_centroid, compute_slopes
from reconstruct import reconstruct_modal, reconstruct_zonal, phase_rms
from turbulence import estimate_r0, estimate_tau0
from actuator import compute_commands, saturated_count

N_FRAMES  = 200
GT_R0_MM  = 3.087
GT_TAU0_MS = 19.385


# ── Cached data loader (shared across all Streamlit sessions) ─────────────────
@st.cache_resource(show_spinner="Loading SH-WFS data…")
def load_data():
    D = config.DATA_DIR

    frames = np.stack([
        np.array(Image.open(D / f"frames/frame_{t:04d}.bmp"))
        for t in range(N_FRAMES)
    ])  # (200, 160, 160) uint8

    ref    = np.array(Image.open(D / "sh_flat_ref.bmp"))
    bg     = np.array(Image.open(D / "sh_flat_bg.bmp"))
    active = build_active_mask(ref)
    cx_ref, cy_ref = cog_centroid(ref, bg, active)

    gt_sx = np.load(D / "ground_truth/all_slopes_x.npy")
    gt_sy = np.load(D / "ground_truth/all_slopes_y.npy")
    gt_zk = np.load(D / "ground_truth/all_zernike.npy")

    gt_std     = float(np.std(np.hstack([gt_sx[:, active], gt_sy[:, active]])))
    s0         = compute_slopes(*cog_centroid(frames[0], bg, active), cx_ref, cy_ref, active)
    slope_gain = gt_std / max(float(np.std(s0)), 1e-9)

    active_ij = np.argwhere(active)  # (80,2)

    gt_phase_rms = float(np.mean([
        float(np.sqrt(np.sum((gt_zk[t, 1:] / 50.0) ** 2)))
        for t in range(N_FRAMES)
    ]))

    return dict(
        frames=frames, ref=ref, bg=bg, active=active,
        cx_ref=cx_ref, cy_ref=cy_ref,
        gt_sx=gt_sx, gt_sy=gt_sy, gt_zk=gt_zk,
        slope_gain=slope_gain, active_ij=active_ij,
        gt_phase_rms=gt_phase_rms,
    )


def process_frame(t, data, state):
    frames     = data["frames"]
    bg         = data["bg"]
    active     = data["active"]
    cx_ref     = data["cx_ref"]
    cy_ref     = data["cy_ref"]
    gt_sx      = data["gt_sx"]
    gt_sy      = data["gt_sy"]
    gt_zk      = data["gt_zk"]
    slope_gain = data["slope_gain"]
    active_ij  = data["active_ij"]

    t0       = time.perf_counter()
    cx, cy   = cog_centroid(frames[t], bg, active)
    slopes   = compute_slopes(cx, cy, cx_ref, cy_ref, active) * slope_gain
    a, phase = reconstruct_modal(slopes)
    phi_act  = reconstruct_zonal(slopes).ravel()
    v_cmd    = compute_commands(phi_act)
    rms      = phase_rms(phase)
    sat      = saturated_count(v_cmd)
    lat_ms   = (time.perf_counter() - t0) * 1e3

    state["slopes_buf"].append(slopes)
    state["rms_buf"].append(rms)
    state["lat_buf"].append(lat_ms)

    slopes_win = np.array(state["slopes_buf"])
    r0_mm = estimate_r0(slopes_win) * 1e3 if len(state["slopes_buf"]) >= 2 else GT_R0_MM
    state["r0_buf"].append(r0_mm)
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
        sx_map=sx_map, sy_map=sy_map, active_ij=active_ij,
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="SH-WFS Digital Twin",
    page_icon="🔭",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 0rem; }
  h1 { font-size: 1.4rem !important; }
  h3 { font-size: 0.95rem !important; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "frame_idx"   not in st.session_state: st.session_state.frame_idx   = 0
if "playing"     not in st.session_state: st.session_state.playing     = False
if "speed"       not in st.session_state: st.session_state.speed       = 2.0
if "rms_buf"     not in st.session_state: st.session_state.rms_buf     = deque(maxlen=50)
if "lat_buf"     not in st.session_state: st.session_state.lat_buf     = deque(maxlen=50)
if "slopes_buf"  not in st.session_state: st.session_state.slopes_buf  = deque(maxlen=50)
if "r0_buf"      not in st.session_state: st.session_state.r0_buf      = deque(maxlen=50)

data  = load_data()
state = {
    "rms_buf":    st.session_state.rms_buf,
    "lat_buf":    st.session_state.lat_buf,
    "slopes_buf": st.session_state.slopes_buf,
    "r0_buf":     st.session_state.r0_buf,
}

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 SH-WFS Controls")
    st.markdown("---")

    ca, cb, cc = st.columns(3)
    with ca:
        if st.button("▶" if not st.session_state.playing else "⏸", use_container_width=True):
            st.session_state.playing = not st.session_state.playing
    with cb:
        if st.button("⏭", use_container_width=True, help="Step one frame"):
            st.session_state.frame_idx = (st.session_state.frame_idx + 1) % N_FRAMES
            st.session_state.playing = False
    with cc:
        if st.button("↺", use_container_width=True, help="Reset to frame 0"):
            st.session_state.frame_idx = 0
            st.session_state.playing   = False
            for buf in state.values():
                buf.clear()

    st.session_state.speed = st.slider(
        "Playback Speed", 0.1, 20.0, st.session_state.speed, 0.1, format="%.1f×"
    )

    st.markdown("---")
    st.metric("Frame", f"{st.session_state.frame_idx:03d} / 199")
    st.caption(f"Slope gain: {data['slope_gain']:.2f}×")
    st.caption(f"GT r₀: {GT_R0_MM} mm  |  GT τ₀: {GT_TAU0_MS} ms")

    st.markdown("---")
    st.markdown("**Hardware**")
    st.caption(f"10×10 lenslet | 80 active SA")
    st.caption(f"160×160 px @ 200 Hz")
    st.caption(f"11×11 DM | 121 actuators")


# ── Process current frame ─────────────────────────────────────────────────────
t = st.session_state.frame_idx
d = process_frame(t, data, state)

gt_phase_rms = data["gt_phase_rms"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"### 🔭 SH-WFS Digital Twin  ·  Frame **{t:03d}/199**  "
    f"·  r₀ = **{d['r0_mm']:.2f} mm**  "
    f"·  RMS = **{d['rms']:.3f} rad**  "
    f"·  **{d['lat_ms']:.1f} ms**/frame"
)

st.markdown("---")

# ── Row 1: Sensor view ────────────────────────────────────────────────────────
st.markdown("#### Row 1 — Sensor View")
col1, col2, col3 = st.columns(3)

with col1:
    overlay1 = cv2.cvtColor(data["frames"][t], cv2.COLOR_GRAY2RGB)
    for k in range(1, 10):
        cv2.line(overlay1, (k * 16, 0),   (k * 16, 159), (0, 255, 136), 1)
        cv2.line(overlay1, (0, k * 16),   (159, k * 16), (0, 255, 136), 1)
    st.image(overlay1, caption="P1  Camera + Subaperture Grid", use_container_width=True)

with col2:
    overlay2 = cv2.cvtColor(data["frames"][t], cv2.COLOR_GRAY2RGB)
    aij = data["active_ij"]
    for (row, col) in aij:
        mx = int(round(float(d["cx"][row, col])))
        my = int(round(float(d["cy"][row, col])))
        rx = int(round(float(data["cx_ref"][row, col])))
        ry = int(round(float(data["cy_ref"][row, col])))
        cv2.circle(overlay2, (mx, my), 2, (255, 68,  68),  -1)   # red = measured
        cv2.circle(overlay2, (rx, ry), 2, (68,  136, 255), -1)   # blue = reference
    st.image(overlay2, caption="P2  Centroids (red=measured, blue=reference)", use_container_width=True)

with col3:
    fig_q, ax_q = plt.subplots(figsize=(4, 3.5), facecolor="#1a1d24")
    ax_q.set_facecolor("#1a1d24")
    sx_a = d["sx_map"][data["active"]]
    sy_a = d["sy_map"][data["active"]]
    mag  = np.sqrt(sx_a**2 + sy_a**2)
    yi, xi = np.mgrid[0:10, 0:10]
    q = ax_q.quiver(
        xi[data["active"]], yi[data["active"]],
        sx_a / max(mag.max(), 1e-6),
        sy_a / max(mag.max(), 1e-6),
        mag, cmap="viridis", scale=8, width=0.018, pivot="mid",
        clim=(0, mag.max() if mag.max() > 0 else 1),
    )
    cb = fig_q.colorbar(q, ax=ax_q, fraction=0.04, pad=0.02)
    cb.set_label("rad/m", color="#aaaaaa", fontsize=7)
    cb.ax.tick_params(colors="#aaaaaa", labelsize=6)
    ax_q.set_xlim(-0.7, 9.7); ax_q.set_ylim(9.7, -0.7)
    ax_q.set_aspect("equal")
    ax_q.set_title("P3  Slope Vectors", color="white", fontsize=9)
    ax_q.set_xlabel("Lenslet col", color="#aaaaaa", fontsize=7)
    ax_q.set_ylabel("Lenslet row", color="#aaaaaa", fontsize=7)
    ax_q.tick_params(colors="#aaaaaa", labelcolor="#aaaaaa", labelsize=7)
    for sp in ax_q.spines.values(): sp.set_edgecolor("#444")
    st.pyplot(fig_q, use_container_width=True)
    plt.close(fig_q)

st.markdown("---")

# ── Row 2: Reconstruction ─────────────────────────────────────────────────────
st.markdown("#### Row 2 — Reconstruction")
col4, col5, col6 = st.columns(3)

with col4:
    vmax = max(float(np.abs(d["phase"]).max()), 0.01)
    fig_p4 = go.Figure(go.Heatmap(
        z=d["phase"].tolist(), colorscale="RdBu", zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title=dict(text="rad", font=dict(color="white")),
                      thickness=12, len=0.8, tickfont=dict(color="white"))
    ))
    fig_p4.update_layout(
        title=dict(text="P4  Wavefront Phase Map", font=dict(color="white"), x=0.5),
        height=260, margin=dict(l=0, r=0, t=35, b=0),
        paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
        font=dict(color="white"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   autorange="reversed"),
    )
    st.plotly_chart(fig_p4, use_container_width=True, key="p4")

with col5:
    z_idx = list(range(1, 22))
    fig_p5 = go.Figure([
        go.Bar(x=z_idx, y=d["gt_a"].tolist(), name="GT",  marker_color="#ff9944", opacity=0.85),
        go.Bar(x=z_idx, y=d["a"].tolist(),    name="Est", marker_color="#4488ff", opacity=0.85),
    ])
    ymax5 = max(float(np.abs(d["gt_a"]).max()), float(np.abs(d["a"]).max()), 0.05)
    fig_p5.update_layout(
        title=dict(text="P5  Zernike Z1–Z21", font=dict(color="white"), x=0.5),
        height=260, barmode="group", margin=dict(l=0, r=0, t=35, b=30),
        paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
        font=dict(color="white"),
        legend=dict(x=0.75, y=1, bgcolor="#2a2d34", font=dict(color="white")),
        yaxis=dict(title="rad", range=[-ymax5 * 1.15, ymax5 * 1.15], gridcolor="#333"),
        xaxis=dict(title="Noll index", dtick=2, gridcolor="#333"),
        bargap=0.15, bargroupgap=0.05,
    )
    st.plotly_chart(fig_p5, use_container_width=True, key="p5")

with col6:
    dm_arr = d["v_cmd"].reshape(11, 11) * 1e9
    dm_sym = max(float(np.abs(dm_arr).max()), 0.1)
    fig_p6 = go.Figure(go.Heatmap(
        z=dm_arr.tolist(), colorscale="RdBu", zmid=0,
        zmin=-dm_sym, zmax=dm_sym,
        colorbar=dict(title=dict(text="nm", font=dict(color="white")),
                      thickness=12, len=0.8, tickfont=dict(color="white"))
    ))
    fig_p6.update_layout(
        title=dict(text="P6  DM Actuators (nm)", font=dict(color="white"), x=0.5),
        height=260, margin=dict(l=0, r=0, t=35, b=0),
        paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
        font=dict(color="white"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   autorange="reversed"),
    )
    st.plotly_chart(fig_p6, use_container_width=True, key="p6")

st.markdown("---")

# ── Row 3: Performance metrics ────────────────────────────────────────────────
st.markdown("#### Row 3 — Performance & Turbulence")
col7, col8, col9 = st.columns(3)

_LAYOUT_BASE = dict(
    height=220, margin=dict(l=0, r=0, t=35, b=30),
    paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
    font=dict(color="white"),
    xaxis=dict(gridcolor="#333"),
    yaxis=dict(gridcolor="#333"),
)

with col7:
    rms_l = list(state["rms_buf"])
    fig_p7 = go.Figure()
    fig_p7.add_trace(go.Scatter(y=rms_l, mode="lines", name="RMS",
                                line=dict(color="#4488ff", width=2)))
    fig_p7.add_hline(y=gt_phase_rms, line_dash="dash", line_color="#ffff44",
                     annotation_text=f"GT {gt_phase_rms:.3f}",
                     annotation_font_color="white")
    fig_p7.update_layout(
        title=dict(text="P7  Phase RMS", font=dict(color="white"), x=0.5),
        showlegend=False,
        yaxis=dict(title="rad", range=[0, max(gt_phase_rms * 1.3, 0.1)], gridcolor="#333"),
        xaxis=dict(title="last 50 frames", gridcolor="#333"),
        **{k: v for k, v in _LAYOUT_BASE.items() if k not in ("yaxis", "xaxis")},
    )
    st.plotly_chart(fig_p7, use_container_width=True, key="p7")

with col8:
    r0_l = list(state["r0_buf"])
    fig_p8 = go.Figure()
    fig_p8.add_trace(go.Scatter(y=r0_l, mode="lines", name="r0",
                                line=dict(color="#44ff88", width=2)))
    fig_p8.add_hline(y=GT_R0_MM, line_dash="dash", line_color="#ffff44",
                     annotation_text=f"GT {GT_R0_MM}",
                     annotation_font_color="white")
    fig_p8.update_layout(
        title=dict(text="P8  r₀ Estimate", font=dict(color="white"), x=0.5),
        showlegend=False,
        yaxis=dict(title="mm", range=[0, GT_R0_MM * 2.0], gridcolor="#333"),
        xaxis=dict(title="last 50 frames", gridcolor="#333"),
        **{k: v for k, v in _LAYOUT_BASE.items() if k not in ("yaxis", "xaxis")},
    )
    st.plotly_chart(fig_p8, use_container_width=True, key="p8")

with col9:
    lat_l = list(state["lat_buf"])
    colors_lat = ["#ff4444" if v > 5.0 else "#4488ff" for v in lat_l]
    fig_p9 = go.Figure()
    fig_p9.add_trace(go.Bar(y=lat_l, name="ms",
                            marker_color=colors_lat, opacity=0.8))
    fig_p9.add_hline(y=5.0, line_color="red", line_width=2,
                     annotation_text="5ms budget",
                     annotation_font_color="white")
    fig_p9.update_layout(
        title=dict(text="P9  Pipeline Latency", font=dict(color="white"), x=0.5),
        showlegend=False,
        yaxis=dict(title="ms", range=[0, 7], gridcolor="#333"),
        xaxis=dict(title="last 50 frames", gridcolor="#333"),
        **{k: v for k, v in _LAYOUT_BASE.items() if k not in ("yaxis", "xaxis")},
    )
    st.plotly_chart(fig_p9, use_container_width=True, key="p9")


st.markdown("---")

# ── Row 4: Live scalar metrics (horizontal) ───────────────────────────────────
st.markdown("#### Live Metrics")
r0_delta   = d["r0_mm"]  - GT_R0_MM
tau0_delta = d["tau0_ms"] - GT_TAU0_MS
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Coherence Time τ₀", f"{d['tau0_ms']:.1f} ms",
           delta=f"{tau0_delta:+.1f} ms vs GT", delta_color="off")
mc2.metric("Fried Parameter r₀", f"{d['r0_mm']:.3f} mm",
           delta=f"{r0_delta:+.3f} mm vs GT", delta_color="off")
mc3.metric("DM Saturation", f"{d['sat']} / 121 actuators")
mc4.metric("Pipeline Latency", f"{d['lat_ms']:.2f} ms",
           delta="under budget" if d["lat_ms"] < 5.0 else "OVER BUDGET",
           delta_color="normal" if d["lat_ms"] < 5.0 else "inverse")
mc5.metric("Slope Correlation", f"{d['corr']:.3f}")

# ── Animation loop ────────────────────────────────────────────────────────────
if st.session_state.playing:
    st.session_state.frame_idx = (t + 1) % N_FRAMES
    sleep_s = max(0.0, config.DT_S / st.session_state.speed - d["lat_ms"] / 1000.0)
    if sleep_s > 0:
        time.sleep(sleep_s)
    st.rerun()
