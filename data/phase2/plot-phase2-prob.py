#!/usr/bin/env python3
"""Plot avg iteration execution time vs failure prob + average cost over [start,end] iterations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DIR = Path(__file__).resolve().parent
NOTE = DIR / "note.txt"
MPI_PROFILE = DIR / "profile-mpi.json"
ON_DEMAND_COST = 1.0
SPOT_COST = 0.5
MPI_COST = ON_DEMAND_COST*5
TIME_BAR_COLOR = "#1f77b4"
COST_BAR_COLOR = "#ff7f0e"
BAR_ALPHA = 0.5
BAR_WIDTH = 0.5
FIGURE_HEIGHT_IN = 7.6
BASE_FONT_SIZE = 16
TITLE_FONT_SIZE = 15
ANNOTATION_FONT_SIZE = 12
LIVE_NODE_LABEL_FONT_SIZE = 13
LIVE_NODE_LABEL_COLOR = "#ffffff"
LIVE_NODE_LABEL_Y = 100
LIVE_NODE_BBOX_ALPHA = 0.8
MS_PER_SECOND = 1000.0
FAILURE_PROBS = (0, 2, 5, 10, 15, 20)

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 1,
        "ytick.labelsize": BASE_FONT_SIZE - 1,
        "legend.fontsize": BASE_FONT_SIZE - 1,
    }
)


def parse_iteration_ranges(note_path: Path) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for raw in note_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "old":
            break
        parts = line.split()
        if len(parts) == 3 and parts[0] == "FailProb":
            continue
        if len(parts) != 3:
            continue
        prob = int(parts[0])
        start, end = int(parts[1]), int(parts[2])
        ranges[prob] = (start, end)
    return ranges

def load_timeline_timestamps(phase2: dict) -> list[dict]:
    """Events only from nested tick_timeline ``events`` lists."""
    tl = phase2["phase2"]["tick_timeline"]
    rows: list[dict] = []
    for tick in tl:
        for ev in tick.get("events") or []:
            rows.append(
                {
                    "timestamp_s": float(ev["timestamp_s"]),
                    "kind": ev["kind"],
                }
            )
    rows.sort(key=lambda r: r["timestamp_s"])
    return rows


def profile_times_in_range(exec_map: dict[str, float], start_i: int, end_i: int, label: str) -> list[float]:
    times = [float(exec_map[str(i)]) for i in range(start_i, end_i + 1) if str(i) in exec_map]
    missing = (end_i - start_i + 1) - len(times)
    if missing:
        raise ValueError(f"{label}: missing {missing} execution_time keys for [{start_i},{end_i}]")
    return times


def n_at_elapsed(events_sorted: list[dict], elapsed_s: float) -> int:
    """Cluster node count follows initial 5, then +/-1 at each draining/relaunch event."""
    n = 5
    for ev in events_sorted:
        if ev["elapsed_s"] > elapsed_s:
            break
        if ev["kind"] == "node_drained":
            n -= 1
        elif ev["kind"] == "node_relaunched_and_expanded":
            n += 1
    return max(0, n)


def cost_integrated(start_elapsed_s: float, end_elapsed_s: float, events_sorted: list[dict]) -> float:
    """Integral of rate(n)=1+(n-1)*0.5 over [start_elapsed_s, end_elapsed_s]."""
    lo = min(start_elapsed_s, end_elapsed_s)
    hi = max(start_elapsed_s, end_elapsed_s)
    if hi <= lo:
        return 0.0

    total = 0.0
    seg_left = lo
    n_curr = n_at_elapsed(events_sorted, lo)

    for ev in events_sorted:
        ts = ev["elapsed_s"]
        if ts <= seg_left:
            continue
        if ts >= hi:
            break
        dur = ts - seg_left
        rate = ON_DEMAND_COST + (n_curr - 1) * SPOT_COST
        total += rate * dur
        if ev["kind"] == "node_drained":
            n_curr -= 1
        elif ev["kind"] == "node_relaunched_and_expanded":
            n_curr += 1
        seg_left = ts

    rate = ON_DEMAND_COST + (n_curr - 1) * SPOT_COST
    total += rate * max(0.0, hi - seg_left)
    return total


def average_live_nodes(start_elapsed_s: float, end_elapsed_s: float, events_sorted: list[dict]) -> float:
    """Time-weighted average live node count over [start_elapsed_s, end_elapsed_s]."""
    lo = min(start_elapsed_s, end_elapsed_s)
    hi = max(start_elapsed_s, end_elapsed_s)
    duration = hi - lo
    if duration <= 0:
        return 0.0

    weighted_nodes = 0.0
    seg_left = lo
    n_curr = n_at_elapsed(events_sorted, lo)

    for ev in events_sorted:
        ts = ev["elapsed_s"]
        if ts <= seg_left:
            continue
        if ts >= hi:
            break
        dur = ts - seg_left
        weighted_nodes += n_curr * dur
        if ev["kind"] == "node_drained":
            n_curr -= 1
        elif ev["kind"] == "node_relaunched_and_expanded":
            n_curr += 1
        seg_left = ts

    weighted_nodes += n_curr * max(0.0, hi - seg_left)
    return weighted_nodes / duration


def main() -> None:
    ranges_by_prob = parse_iteration_ranges(NOTE)
    probs: list[float] = []
    avg_exec: list[float] = []
    avg_cost_per_iter: list[float] = []
    avg_live_nodes: list[float] = []
    latency_increase_pct: list[float] = []
    cost_saving_pct: list[float] = []

    with MPI_PROFILE.open() as fh:
        mpi_profile = json.load(fh)
    mpi_start_i, mpi_end_i = ranges_by_prob[0]
    mpi_times = profile_times_in_range(mpi_profile["execution_time"], mpi_start_i, mpi_end_i, "profile-mpi")
    mpi_avg_exec = sum(mpi_times) / len(mpi_times)
    mpi_cost_per_iter = mpi_avg_exec * MPI_COST

    for prob in FAILURE_PROBS:
        start_i, end_i = ranges_by_prob[prob]

        phase2_path = DIR / f"phase2-{prob}.json"
        profile_path = DIR / f"profile-{prob}.json"
        with phase2_path.open() as fh:
            phase2 = json.load(fh)
        with profile_path.open() as fh:
            profile = json.load(fh)

        exec_map = profile["execution_time"]

        times_in_range = profile_times_in_range(exec_map, start_i, end_i, f"profile-{prob}")
        num_iters = len(times_in_range)
        exec_sum = sum(times_in_range)

        tl = phase2["phase2"]["tick_timeline"]
        t_wall0 = float(tl[0]["timestamp_s"])
        events_sorted = load_timeline_timestamps(phase2)
        for ev in events_sorted:
            ev["elapsed_s"] = ev["timestamp_s"] - t_wall0

        # phase2-{prob}.json is used only for node up/down events. The duration
        # and iteration count come from the selected profile-{prob}.json slice.
        total_cost = cost_integrated(0.0, exec_sum, events_sorted)
        avg_nodes = average_live_nodes(0.0, exec_sum, events_sorted)

        cost_per_iter = total_cost / num_iters
        avg_exec_per_iter = exec_sum / num_iters
        latency_inc = (avg_exec_per_iter - mpi_avg_exec) / mpi_avg_exec * 100.0
        cost_save = (mpi_cost_per_iter - cost_per_iter) / mpi_cost_per_iter * 100.0

        probs.append(prob)
        avg_exec.append(avg_exec_per_iter)
        avg_cost_per_iter.append(cost_per_iter)
        avg_live_nodes.append(avg_nodes)
        latency_increase_pct.append(latency_inc)
        cost_saving_pct.append(cost_save)

    avg_exec_ms = [v * MS_PER_SECOND for v in avg_exec]
    mpi_avg_exec_ms = mpi_avg_exec * MS_PER_SECOND

    x = range(len(probs))
    labels = [str(prob) for prob in probs]

    fig, (ax_time, ax_cost) = plt.subplots(2, 1, figsize=(8.0, FIGURE_HEIGHT_IN), sharex=True)
    bars_time = ax_time.bar(
        x, avg_exec_ms, width=BAR_WIDTH, color=TIME_BAR_COLOR, alpha=BAR_ALPHA, 
        # label="Average Iteration Time"
    )
    mpi_time_line = ax_time.axhline(
        mpi_avg_exec_ms,
        color="green",
        linestyle="--",
        linewidth=2.0,
        label=f"Static MPI Baseline Iteration Time",
    )
    ax_time.set_ylabel("Average Iteration Time (ms)", color=TIME_BAR_COLOR, fontweight="bold")
    ax_time.tick_params(axis="y", labelcolor=TIME_BAR_COLOR)
    # ax_time.set_title("Phase-2 average iteration time (ms) vs failure probability")
    ax_time.grid(True, axis="y", alpha=0.3)
    ax_time.set_ylim(bottom=0)
    for bar, pct, avg_nodes in zip(bars_time, latency_increase_pct, avg_live_nodes):
        x_center = bar.get_x() + bar.get_width() / 2.0
        y_bar = bar.get_height()
        y_ref = mpi_avg_exec_ms
        live_nodes_text = ax_time.text(
            x_center,
            LIVE_NODE_LABEL_Y,
            f"n={avg_nodes:.2f}",
            ha="center",
            va="center",
            fontsize=LIVE_NODE_LABEL_FONT_SIZE,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc=LIVE_NODE_LABEL_COLOR,
                ec="#000000",
                lw=1.0,
                alpha=LIVE_NODE_BBOX_ALPHA,
            ),
        )
        # live_nodes_text.set_path_effects([pe.Stroke(linewidth=1.6, foreground="#000000"), pe.Normal()])
        ax_time.annotate(
            "",
            xy=(x_center, y_bar),
            xytext=(x_center, y_ref),
            arrowprops=dict(
                arrowstyle="<->",
                color="#ffd400",
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        time_text = ax_time.text(
            x_center,
            (y_bar + y_ref) / 2.0,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="bold",
            color="#ffd400",
        )
        time_text.set_path_effects([pe.Stroke(linewidth=2.0, foreground="#111111"), pe.Normal()])
    live_nodes_legend = Patch(
        facecolor=LIVE_NODE_LABEL_COLOR,
        edgecolor="#000000",
        linewidth=1.2,
        label="Average Live Node Counts",
    )
    ax_time.legend(
        handles=[bars_time[0], mpi_time_line, live_nodes_legend],
        loc="best",
        frameon=True,
        edgecolor="#bbbbbb",
        framealpha=0.8,
    )

    bars_cost = ax_cost.bar(
        x, avg_cost_per_iter, width=BAR_WIDTH, color=COST_BAR_COLOR, alpha=BAR_ALPHA, 
        # label="Average Cost per Iteration"
    )
    ax_cost.axhline(
        mpi_cost_per_iter,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Static MPI Baseline Cost per Iteration",
    )
    ax_cost.set_xlabel("Spot Instances Interruption Probability (%)")
    ax_cost.set_ylabel("Average Cost per Iteration", color=COST_BAR_COLOR, fontweight="bold")
    ax_cost.tick_params(axis="y", labelcolor=COST_BAR_COLOR)
    # ax_cost.set_title("Phase-2 average cost per iteration vs failure probability")
    ax_cost.grid(True, axis="y", alpha=0.3)
    ax_cost.set_ylim(bottom=0)
    ax_cost.set_xticks(list(x))
    ax_cost.set_xticklabels(labels)
    for bar, pct in zip(bars_cost, cost_saving_pct):
        x_center = bar.get_x() + bar.get_width() / 2.0
        y_bar = bar.get_height()
        y_ref = mpi_cost_per_iter
        ax_cost.annotate(
            "",
            xy=(x_center, y_bar),
            xytext=(x_center, y_ref),
            arrowprops=dict(
                arrowstyle="<->",
                color="gray",
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax_cost.text(
            x_center,
            (y_bar + y_ref) / 2.0,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="bold",
            color="#111111",
        )
    ax_cost.legend(
        loc="best",
        frameon=True,
        edgecolor="#bbbbbb",
        framealpha=0.8,
    )
    fig.tight_layout()

    outp = DIR / "phase2-prob.png"
    fig.savefig(outp, dpi=150)
    print(f"wrote {outp}")
    print("\nMPI baseline summary:")
    print(f"  avg iteration time = {mpi_avg_exec_ms:.3f} ms")
    print(f"  cost per iteration = {mpi_cost_per_iter:.6f}")
    print("\nPhase-2 vs MPI:")
    print("prob(%) | avg_iter_ms | cost_per_iter | avg_live_nodes | latency_increase(%) | cost_saving(%)")
    for prob, exec_ms, cost_iter, avg_nodes, lat_inc, save in zip(
        probs, avg_exec_ms, avg_cost_per_iter, avg_live_nodes, latency_increase_pct, cost_saving_pct
    ):
        print(
            f"{prob:>7} | {exec_ms:>11.3f} | {cost_iter:>13.6f} | {avg_nodes:>14.3f} | {lat_inc:>19.2f} | {save:>14.2f}"
        )


if __name__ == "__main__":
    main()
