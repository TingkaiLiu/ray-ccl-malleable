import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRANSITIONS = [
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 64),
    (64, 32),
    (32, 16),
]
STEP_ITERS = 20
# Fill this list with file names under data/change, e.g. ["change.json", "change-2.json"].
# Leave empty to auto-read all change*.json files.
FILE_LIST = ["change-1.json", "change-2.json"]
# Bottom-to-top stack order in the bar chart.
# Allowed values: "data", "rank", "comm", "meta"
BAR_ORDER = ["comm", "meta", "data", "rank"]
# Base font size for title, labels, ticks, and legend.
FONT_SIZE = 18
BAR_WIDTH = 0.6
TIME_SCALE = 1000.0  # seconds -> milliseconds
TIME_UNIT = "ms"
SERIES_LABELS = {
    "data": "New State Generation",
    "rank": "New Rank Launching",
    "comm": "Communicator Recreation",
    "meta": "Meta Processing",
}
SERIES_COLORS = {
    "data": "#ff7f27",
    "rank": "#74f466",
    "comm": "#4e79a7",
    "meta": "#4cd2e2",
}
SERIES_HATCHES = {
    "data": "xx",
    "rank": "\\\\",
    "comm": "//",
    "meta": "..",
}


def load_change_files(change_dir: Path):
    if FILE_LIST:
        paths = [change_dir / name for name in FILE_LIST]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "These files in FILE_LIST do not exist:\n" + "\n".join(missing)
            )
        return paths

    return sorted([p for p in change_dir.glob("change*.json") if p.is_file()], key=lambda p: p.name)


def dict_int_keys(d):
    return {int(k): v for k, v in d.items()}


def collect_transition_times(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    reconfig_time = dict_int_keys(raw.get("reconfig_time", {}))
    reconfig_record = dict_int_keys(raw.get("reconfig_record", {}))

    per_transition = {t: [] for t in TRANSITIONS}

    # Rank changes happen every 20 iterations. Use that iteration index
    # (20, 40, 60, ...) to locate the corresponding total reconfig_time.
    for event_idx in sorted(reconfig_record):
        transition = tuple(reconfig_record[event_idx])
        if transition not in per_transition:
            continue
        iter_idx = event_idx * STEP_ITERS
        if iter_idx in reconfig_time:
            per_transition[transition].append(float(reconfig_time[iter_idx]))

    return per_transition


def collect_transition_breakdown(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    reconfig_time = dict_int_keys(raw.get("reconfig_time", {}))
    reconfig_record = dict_int_keys(raw.get("reconfig_record", {}))
    reconfig_data_time = dict_int_keys(raw.get("reconfig_data_time", {}))
    reconfig_rank_time = dict_int_keys(raw.get("reconfig_rank_time", {}))
    reconfig_comm_time = dict_int_keys(raw.get("reconfig_comm_time", {}))

    per_transition = {t: [] for t in TRANSITIONS}

    for event_idx in sorted(reconfig_record):
        transition = tuple(reconfig_record[event_idx])
        if transition not in per_transition:
            continue

        iter_idx = event_idx * STEP_ITERS
        if iter_idx not in reconfig_time:
            continue

        total = float(reconfig_time[iter_idx])
        data_t = float(reconfig_data_time.get(event_idx, 0.0))
        rank_t = float(reconfig_rank_time.get(event_idx, 0.0))
        comm_t = float(reconfig_comm_time.get(event_idx, 0.0))
        meta_t = max(0.0, total - data_t - rank_t - comm_t)

        per_transition[transition].append(
            {
                "event_idx": event_idx,
                "total": total,
                "data": data_t,
                "rank": rank_t,
                "comm": comm_t,
                "meta": meta_t,
            }
        )

    return per_transition


def remove_extremes(values):
    if len(values) <= 2:
        return values[:]

    arr = np.array(values, dtype=float)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [v for v in values if lower <= v <= upper]
    if not filtered:
        filtered = values[:]

    # Explicitly drop one largest point when we still have enough samples.
    # This helps remove cases like the first [16, 32] reconfiguration spike.
    if len(filtered) >= 4:
        filtered = sorted(filtered)[:-1]

    return filtered


def remove_extreme_samples(samples):
    if len(samples) <= 2:
        return samples[:]

    totals = np.array([s["total"] for s in samples], dtype=float)
    q1 = np.percentile(totals, 25)
    q3 = np.percentile(totals, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [s for s in samples if lower <= s["total"] <= upper]
    if not filtered:
        filtered = samples[:]

    if len(filtered) >= 4:
        largest = max(filtered, key=lambda s: s["total"])
        filtered.remove(largest)

    return filtered


def main():
    change_dir = Path(__file__).resolve().parent
    files = load_change_files(change_dir)
    if not files:
        raise FileNotFoundError(f"No change*.json files found in {change_dir}")

    all_samples = {t: [] for t in TRANSITIONS}
    for path in files:
        data = collect_transition_breakdown(path)
        for t in TRANSITIONS:
            all_samples[t].extend(data[t])

    # For 16->32, always drop the first change in each file (reconfig_record == 1).
    all_samples[(16, 32)] = [s for s in all_samples[(16, 32)] if s["event_idx"] != 1]

    cleaned = {t: remove_extreme_samples(all_samples[t]) for t in TRANSITIONS}
    data_means = [float(np.mean([s["data"] for s in cleaned[t]])) if cleaned[t] else np.nan for t in TRANSITIONS]
    rank_means = [float(np.mean([s["rank"] for s in cleaned[t]])) if cleaned[t] else np.nan for t in TRANSITIONS]
    comm_means = [float(np.mean([s["comm"] for s in cleaned[t]])) if cleaned[t] else np.nan for t in TRANSITIONS]
    meta_means = [float(np.mean([s["meta"] for s in cleaned[t]])) if cleaned[t] else np.nan for t in TRANSITIONS]
    total_means = [
        (data_means[i] + rank_means[i] + comm_means[i] + meta_means[i])
        if np.isfinite(data_means[i])
        else np.nan
        for i in range(len(TRANSITIONS))
    ]
    data_means = [v * TIME_SCALE if np.isfinite(v) else v for v in data_means]
    rank_means = [v * TIME_SCALE if np.isfinite(v) else v for v in rank_means]
    comm_means = [v * TIME_SCALE if np.isfinite(v) else v for v in comm_means]
    meta_means = [v * TIME_SCALE if np.isfinite(v) else v for v in meta_means]
    total_means = [v * TIME_SCALE if np.isfinite(v) else v for v in total_means]

    labels = [f"{a}->{b}" for a, b in TRANSITIONS]
    x = np.arange(len(labels))

    series = {
        "data": np.array(data_means),
        "rank": np.array(rank_means),
        "comm": np.array(comm_means),
        "meta": np.array(meta_means),
    }
    invalid = [k for k in BAR_ORDER if k not in series]
    if invalid:
        raise ValueError(f"Invalid entries in BAR_ORDER: {invalid}")
    if len(BAR_ORDER) != len(set(BAR_ORDER)):
        raise ValueError("BAR_ORDER has duplicated entries.")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 2,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE - 2,
        }
    )
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(labels), dtype=float)
    top_bars = None
    for key in BAR_ORDER:
        top_bars = plt.bar(
            x,
            series[key],
            width=BAR_WIDTH,
            bottom=bottom,
            label=SERIES_LABELS[key],
            color=SERIES_COLORS[key],
            hatch=SERIES_HATCHES.get(key, ""),
            edgecolor="black",
            linewidth=0.6,
        )
        bottom = bottom + series[key]

    plt.xticks(x, labels)
    plt.ylabel(f"Time ({TIME_UNIT})")
    plt.xlabel("Rank Change")
    # plt.title("Reconfiguration breakdown by rank change")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    output = change_dir / "change.png"
    plt.savefig(output, dpi=200)
    print(f"Saved plot: {output}")

    print(f"\nAverages ({TIME_UNIT}):")
    for t in TRANSITIONS:
        used = len(cleaned[t])
        total = len(all_samples[t])
        if cleaned[t]:
            data_mean = float(np.mean([s["data"] for s in cleaned[t]]))
            rank_mean = float(np.mean([s["rank"] for s in cleaned[t]]))
            comm_mean = float(np.mean([s["comm"] for s in cleaned[t]]))
            meta_mean = float(np.mean([s["meta"] for s in cleaned[t]]))
            total_mean = data_mean + rank_mean + comm_mean + meta_mean
            data_mean *= TIME_SCALE
            rank_mean *= TIME_SCALE
            comm_mean *= TIME_SCALE
            meta_mean *= TIME_SCALE
            total_mean *= TIME_SCALE
        else:
            data_mean = rank_mean = comm_mean = meta_mean = total_mean = float("nan")

        print(
            f"{t[0]}->{t[1]}: total={total_mean:.6f}, data={data_mean:.6f}, "
            f"rank={rank_mean:.6f}, comm={comm_mean:.6f}, meta={meta_mean:.6f} "
            f"(used {used}/{total} samples)"
        )


if __name__ == "__main__":
    main()
