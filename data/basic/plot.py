import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).resolve().parent
NODES = [1, 2, 4, 8]
# Index 0/1/2/3 corresponds to node counts 1/2/4/8 in NODES.
FILE_MPI = ["mpi-1.json", "mpi-2.json", "mpi-4.json", "mpi-8.json"]
FILE_GLOO = ["gloo-1.json", "gloo-2.json", "gloo-4.json", "gloo-8.json"]
FILE_MALLEABLE = ["malle-1.json", "malle-2.json", "malle-4.json", "malle-8.json"]

RUNTIMES = {
    "OpenMPI Baseline": FILE_MPI,
    "Ray-GLOO": FILE_GLOO,
    "Malleable": FILE_MALLEABLE,
}
IQR_FACTOR = 3.0 # The larger the factor, the less outliers are removed.
COMPONENTS = ["comp_time", "comm_time"]
DISPLAY_NAMES = {
    "comm_time": "Communication",
    "comp_time": "Computation",
}
COLORS = {
    "OpenMPI Baseline": "#1F77B4",
    "Ray-GLOO": "#FF7F0E",
    "Malleable": "#2CA02C",
}
COMPONENT_ALPHA = {
    "comp_time": 1.0,
    "comm_time": 0.45,
}
HATCHES = {
    "OpenMPI Baseline": {"comp_time": "//", "comm_time": "\\\\"},
    "Ray-GLOO": {"comp_time": "xx", "comm_time": ".."},
    "Malleable": {"comp_time": "++", "comm_time": "oo"},
}
# TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 17


def iqr_filter(values):
    """Remove outliers using the IQR_FACTOR * IQR rule."""
    if len(values) < 4:
        return values
    arr = np.array(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return values
    low = q1 - IQR_FACTOR * iqr
    high = q3 + IQR_FACTOR * iqr
    return arr[(arr >= low) & (arr <= high)].tolist()


def resolve_series_files(file_name):
    """
    Expand a base file to include repeated runs.
    Example: gloo-8.json -> [gloo-8.json, gloo-8-2.json, gloo-8-3.json, ...]
    """
    base = Path(file_name)
    stem = base.stem
    candidates = sorted(DATA_DIR.glob(f"{stem}*.json"))
    if candidates:
        return candidates
    return [DATA_DIR / file_name]


def load_iteration_values(file_path, key):
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    values = data.get(key, {})
    if not values:
        return []
    return [float(v) for v in values.values()]


def average_component(file_name, key):
    all_values = []
    for file_path in resolve_series_files(file_name):
        all_values.extend(load_iteration_values(file_path, key))
    if not all_values:
        return 0.0
    filtered = iqr_filter(all_values)
    if not filtered:
        return 0.0
    return float(np.mean(filtered))


def build_summary():
    summary = {runtime: {comp: [] for comp in COMPONENTS} for runtime in RUNTIMES}
    for runtime, file_list in RUNTIMES.items():
        for idx, _node in enumerate(NODES):
            file_name = file_list[idx]
            comp_avg = average_component(file_name, "comp_time")
            comm_avg = average_component(file_name, "comm_time")
            reconfig_avg = average_component(file_name, "reconfig_time")

            summary[runtime]["comp_time"].append(comp_avg)
            # Merge reconfiguration overhead into communication for plotting.
            summary[runtime]["comm_time"].append(comm_avg + reconfig_avg)
    return summary


def plot_breakdown(summary):
    x = np.arange(len(NODES))
    width = 0.24
    offsets = [-width, 0.0, width]
    runtime_names = list(RUNTIMES.keys())
    ms_scale = 1000.0

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, runtime in enumerate(runtime_names):
        bottoms = np.zeros(len(NODES))
        for comp in COMPONENTS:
            vals = np.array(summary[runtime][comp], dtype=float) * ms_scale
            ax.bar(
                x + offsets[idx],
                vals,
                width=width,
                bottom=bottoms,
                color=COLORS[runtime],
                alpha=COMPONENT_ALPHA[comp],
                edgecolor="black",
                linewidth=0.6,
                hatch=HATCHES[runtime][comp],
            )
            bottoms += vals

        # for j, total_ms in enumerate(bottoms):
        #     ax.text(
        #         x[j] + offsets[idx],
        #         total_ms + 8.0,
        #         f"{total_ms:.1f}",
        #         ha="center",
        #         va="bottom",
        #         fontsize=TICK_FONTSIZE - 1,
        #     )

    # ax.set_title("Average Execution Time Breakdown", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Number of Nodes", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Time (ms)", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NODES], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    legend_handles = []
    for runtime in runtime_names:
        for comp in COMPONENTS:
            legend_handles.append(
                Patch(
                    facecolor=COLORS[runtime],
                    alpha=COMPONENT_ALPHA[comp],
                    edgecolor="black",
                    hatch=HATCHES[runtime][comp],
                    label=f"{runtime} - {DISPLAY_NAMES[comp]}",
                )
            )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    output = DATA_DIR / "basic.png"
    plt.savefig(output, dpi=200)
    # plt.show()
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    summary_data = build_summary()
    plot_breakdown(summary_data)
