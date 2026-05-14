import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt

# --- meta: input and tick window on phase2.tick_timeline (see phase2-20.json ~L2385) ---
DATA_FILENAME = "phase2-10.json"
# Inclusive bounds on each entry's "tick"; None means unbounded on that side.
TICK_SEGMENT: Tuple[Optional[int], Optional[int]] = (2, 11)


FIGURE_HEIGHT_IN = 7
OUTPUT_FILENAME = "phase2-trace.png"

FS_AXIS = 16
FS_TICK = 18
FS_LEGEND = 18
FS_INLINE = 15

# Rank-edge labels (ax1): same palette as phase1 draw_tune — scale-up vs scale-down.    
INCREASE_BBOX_FC = "cyan"
INLINE_BBOX_ALPHA = 0.3
DECREASE_BBOX_FC = "orange"


def _slice_timeline(
    timeline: list[dict],
    tick_lo: Optional[int],
    tick_hi: Optional[int],
) -> list[dict]:
    out = timeline
    if tick_lo is not None:
        out = [e for e in out if e["tick"] >= tick_lo]
    if tick_hi is not None:
        out = [e for e in out if e["tick"] <= tick_hi]
    return out


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / DATA_FILENAME
    output_path = script_dir / OUTPUT_FILENAME
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    full_timeline = data["phase2"]["tick_timeline"]
    tick_lo, tick_hi = TICK_SEGMENT
    timeline = _slice_timeline(full_timeline, tick_lo, tick_hi)
    if not timeline:
        raise ValueError(
            f"No tick_timeline entries in range TICK_SEGMENT={TICK_SEGMENT!r} "
            f"(file has {len(full_timeline)} ticks, "
            f"tick range {full_timeline[0]['tick']}..{full_timeline[-1]['tick']})."
        )

    coord_ticks = [entry["tick"] for entry in timeline]
    # X axis: 0-based index into this segment (not coordinator tick numbers).
    x_step = list(range(len(timeline)))
    avg_iter_ms = [entry["avg_iter_s"] * 1000.0 for entry in timeline]
    cost_per_iter = [entry["cost_per_iteration"] for entry in timeline]
    world_sizes = [entry["world_size"] for entry in timeline]

    # Detect rank changes between consecutive segment rows.
    rank_change_events = []
    for i in range(len(timeline) - 1):
        curr_world = world_sizes[i]
        next_world = world_sizes[i + 1]
        if next_world != curr_world:
            rank_change_events.append(
                {
                    "index": i,
                    "next_index": i + 1,
                    "x_mid": (x_step[i] + x_step[i + 1]) / 2,
                    "from_world": curr_world,
                    "to_world": next_world,
                }
            )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, FIGURE_HEIGHT_IN), sharex=True)

    color_time = "#2c7fb8"
    ax1.set_ylabel(
        "Avg Iteration Time (ms)", color=color_time, fontweight="bold", fontsize=FS_AXIS
    )
    (line_time,) = ax1.plot(
        x_step,
        avg_iter_ms,
        marker="o",
        color=color_time,
        linewidth=2.5,
        label="Iteration Time",
    )
    ax1.tick_params(axis="y", labelcolor=color_time, labelsize=FS_TICK)
    ax1.grid(True, linestyle=":", alpha=0.6)

    color_cost = "#d95f02"
    ax2.set_xlabel("Phase 2 Execution Step", fontsize=FS_AXIS)
    ax2.set_ylabel(
        "Cost per Iteration", color=color_cost, fontweight="bold", fontsize=FS_AXIS
    )
    (line_cost,) = ax2.plot(
        x_step,
        cost_per_iter,
        marker="s",
        linestyle="-",
        color=color_cost,
        linewidth=2.5,
        label="Cost per Iteration",
    )
    ax2.tick_params(axis="y", labelcolor=color_cost, labelsize=FS_TICK)
    ax2.tick_params(axis="x", labelsize=FS_TICK)
    ax2.set_xticks(x_step)
    ax2.grid(True, linestyle=":", alpha=0.6)

    for event in rank_change_events:
        i = event["index"]
        j = event["next_index"]
        x_mid = event["x_mid"]
        rank_label = f"{event['from_world']}->{event['to_world']}"
        time_mid = (avg_iter_ms[i] + avg_iter_ms[j]) / 2

        if event["to_world"] > event["from_world"]:
            bbox_fc = INCREASE_BBOX_FC
        else:
            bbox_fc = DECREASE_BBOX_FC
        ax1.text(
            x_mid,
            time_mid,
            rank_label,
            ha="center",
            va="bottom",
            fontsize=FS_INLINE,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc=bbox_fc,
                alpha=INLINE_BBOX_ALPHA,
                ec="none",
            ),
        )

    seg = f"coord. ticks {coord_ticks[0]}..{coord_ticks[-1]}"
    if tick_lo is not None or tick_hi is not None:
        seg = f"TICK_SEGMENT={TICK_SEGMENT!r} ({seg})"
    # fig.suptitle(
    #     f"Phase 2 execution trace — {DATA_FILENAME} ({seg})",
    #     fontsize=18,
    # )

    ax1.legend(
        [line_time, line_cost],
        [line_time.get_label(), line_cost.get_label()],
        loc="upper left",
        frameon=True,
        shadow=True,
        fontsize=FS_LEGEND,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
