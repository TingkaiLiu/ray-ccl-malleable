import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIGURE_HEIGHT_IN = 7

FS_AXIS = 16
FS_TICK = 18
FS_INLINE = 16
FS_LEGEND = 18
FS_ANNOT = 16

# Rank-edge labels (ax1): default vs. last segment (e.g. revert 96->80)
INLINE_BBOX_FC = "cyan"
INLINE_BBOX_ALPHA = 0.3
LAST_INLINE_BBOX_FC = "orange"
# Former +0.03 s vertical nudge for rank labels on ax1 (draw_tune historically used seconds).
INLINE_LABEL_OFFSET_MS = 30.0

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_PATH = _SCRIPT_DIR / "phase1-final.json"

with open(_DATA_PATH, encoding="utf-8") as f:
    _data = json.load(f)

profiles = _data["phase1"]["phase1_tuning_poll_profiles"]
threshold = float(_data["metadata"]["cost_threshold_per_iter"])

avg_iter_s = [p["avg_iter_s"] for p in profiles]
cost_per_iter = [p["cost_per_iteration"] for p in profiles]
ranks = [p["world_size"] for p in profiles]

# Synthetic step: revert last scale-out, reuse metrics from the last-but-one poll
if len(profiles) >= 2:
    prev = profiles[-2]
    avg_iter_s.append(prev["avg_iter_s"])
    cost_per_iter.append(prev["cost_per_iteration"])
    ranks.append(prev["world_size"])

avg_iter_ms = [t * 1000.0 for t in avg_iter_s]
steps = list(range(len(avg_iter_ms)))
_OUTPUT_PATH = _SCRIPT_DIR / "phase1.png"

# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, FIGURE_HEIGHT_IN), sharex=True)

# ---------------------------------------------------------
# GRAPH 1: Iteration Time Performance
# ---------------------------------------------------------
color_time = "#2c7fb8"
ax1.set_ylabel("Avg Iteration Time (ms)", color=color_time, fontweight="bold", fontsize=FS_AXIS)
line1, = ax1.plot(
    steps,
    avg_iter_ms,
    marker="o",
    color=color_time,
    linewidth=2.5,
    label="Iteration Time (ms)",
)
ax1.tick_params(axis="y", labelcolor=color_time, labelsize=FS_TICK)
ax1.grid(True, linestyle=":", alpha=0.6)
# ax1.set_title("Tuning Performance: Iteration Time", fontsize=14, pad=15)

# Adding rank change labels to Graph 1
for i in range(len(steps) - 1):
    x_mid = (steps[i] + steps[i + 1]) / 2
    y_pos = (avg_iter_ms[i] + avg_iter_ms[i + 1]) / 2
    label = f"{ranks[i]}->{ranks[i + 1]}"
    is_last_segment = i == len(steps) - 2
    bbox_fc = LAST_INLINE_BBOX_FC if is_last_segment else INLINE_BBOX_FC
    ax1.text(
        x_mid,
        y_pos + INLINE_LABEL_OFFSET_MS,
        label,
        ha="center",
        va="bottom",
        fontsize=FS_INLINE,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc=bbox_fc, alpha=INLINE_BBOX_ALPHA, ec="none"),
    )

# ---------------------------------------------------------
# GRAPH 2: Cost per Iteration
# ---------------------------------------------------------
color_cost = "#d95f02"
ax2.set_xlabel("Phase 1 Tuning Step", fontsize=FS_AXIS)
ax2.set_ylabel("Cost per Iteration", color=color_cost, fontweight="bold", fontsize=FS_AXIS)
line2, = ax2.plot(
    steps,
    cost_per_iter,
    marker="s",
    linestyle="-",
    color=color_cost,
    linewidth=2.5,
    label="Cost per Iteration",
)
ax2.tick_params(axis="x", labelsize=FS_TICK)
ax2.tick_params(axis="y", labelcolor=color_cost, labelsize=FS_TICK)
ax2.set_xticks(steps)
ax2.grid(True, linestyle=":", alpha=0.6)
# ax2.set_title("Tuning Efficiency: Cost vs. Budget", fontsize=14, pad=15)

# Add Threshold Line
line3 = ax2.axhline(
    y=threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Cost Threshold",
)

_cost_span = max(cost_per_iter) - min(cost_per_iter) + 1e-6

# Annotate the rightmost step that still exceeds the budget (pre-revert "current" violation)
violating_idx = [i for i, c in enumerate(cost_per_iter) if c > threshold]
if violating_idx:
    vi = violating_idx[-1]
    vc = cost_per_iter[vi]
    ax2.annotate(
        "Budget Exceeded\nRevert Node Addition",
        xy=(vi, vc),
        xytext=(vi, vc + 0.2 * _cost_span),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
        fontweight="bold",
        color="red",
        ha="center",
        fontsize=FS_ANNOT,
    )

mi = min(range(len(cost_per_iter)), key=lambda i: cost_per_iter[i])
mc = cost_per_iter[mi]
ax2.annotate(
    "Minimal Cost",
    xy=(mi, mc),
    xytext=(mi, mc + 0.35 * _cost_span),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=7),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.9),
    fontweight="bold",
    color="#1a5c1a",
    ha="center",
    fontsize=FS_ANNOT,
)

# ---------------------------------------------------------
# Final layout and legend (top-right of upper subplot)
# ---------------------------------------------------------

_rank_legend_entry = Line2D(
    [],
    [],
    linestyle="none",
    color="black",
    marker=r"$\rightarrow$",
    markersize=max(10, FS_LEGEND + 2),
    label="Rank Number Change",
)
_legend_handles = [line1, line2, line3, _rank_legend_entry]
ax1.legend(
    handles=_legend_handles,
    loc="upper right",
    ncol=1,
    frameon=True,
    shadow=True,
    fontsize=FS_LEGEND,
)

plt.tight_layout()
fig.savefig(_OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
