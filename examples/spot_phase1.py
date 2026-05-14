#!/usr/bin/env python3
"""Phase 1 driver for simulated node-granularity tuning.

Current implementation
======================
This script performs a node-by-node search for a good application size using
the malleable coordinator. It assumes:

1. A Ray head node and a malleable application are already running.
2. Additional worker machines are available by IP address.
3. Adding one worker node to the Ray cluster is simulated by:
   - sleeping for a configurable spin-up delay,
   - SSHing to that machine,
   - running `ray start --address=...`
4. After the worker joins Ray, the script expands the malleable job by one
   node's worth of ranks.

Two-stage search
----------------
Phase 1 currently uses a two-stage search strategy.

Stage 1: find the minimum-price point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The script keeps adding nodes until the weighted cost per iteration stops
decreasing. The weighted cost is:

    filtered_average_iteration_time * total_price_rate

where `total_price_rate` includes:

- the expensive head node price,
- plus the cheaper worker-node prices.

This stage is needed because the head-only configuration can start with a high
cost, and adding the first few worker nodes may reduce cost by amortizing the
head-node price.

If the latest node addition makes weighted cost increase:

- in `lowest_price` mode:
  the latest node is removed immediately and the previous node count is the
  chosen result.
- in `fastest_under_budget` mode:
  the previous node count is treated as the minimum-price point. If that point
  is still above budget, the latest node is removed and tuning stops there.
  Otherwise, tuning enters stage 2.

Stage 2: fastest configuration under budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Only used when `PHASE1_STOPPING_MODE == "fastest_under_budget"`.

After the minimum-price point is found, the script keeps adding nodes as long
as the latest accepted configuration is still under the budget threshold.
Intermediate under-budget points are allowed to remain, even if their weighted
cost is higher than the minimum-price point, because they may enable faster
computation at larger node counts.

If the latest added node pushes weighted cost above budget:

- only that latest node is removed,
- and the previous node count becomes the chosen result.

Sampling and averaging
----------------------
At each tuning step, the script polls the coordinator's tuner profiling API and
collects:

- iteration durations,
- reconfiguration durations.

Iteration time averages use the same IQR-based outlier filtering style as
`tuner.py`.

Outputs
-------
At the end of phase 1 the script writes:

- the chosen target node count,
- the final application state snapshot,
- per-iteration weighted cost samples across the tuning process,
- the final accepted cost per iteration,
- the average weighted cost per iteration over the full tuning run,
- and recorded reconfiguration durations.
- ``phase1_tuning_poll_profiles``: one entry per tuning-loop iteration that
  produced a decision snapshot, each with ``node_count``, ``world_size``,
  ``avg_iter_s``, and ``cost_per_iteration`` (same fields as the
  ``phase1: node_count=...`` log line), in chronological order.
"""

from __future__ import annotations

from spot_common import (
    COORDINATOR_NAME,
    COST_THRESHOLD_PER_ITER,
    PHASE1_OUTPUT_PATH,
    PHASE1_STOPPING_MODE,
    POLL_S,
    allocate_node_into_ray_cluster,
    average_execution_time,
    collect_iteration_costs,
    disconnect_node_from_ray_cluster,
    expand_job_for_node,
    initial_available_ips,
    log,
    metadata_dict,
    price_rate_for_node_count,
    seed_app_state,
    seed_random,
    shrink_job_by_removing_node,
    summarize_cost_samples,
    wait_for_coordinator,
    wait_for_ray,
)

import time


def _revert_latest_node(coord, app_state, available_ips, reason: str):
    """Undo the most recent node addition and stop phase 1."""
    removed_node_id = str(app_state["ordered_node_ids"][-1])
    removed_node_ip = str(app_state["ordered_node_ips"][-1])
    shrink_job_by_removing_node(coord, app_state, removed_node_id)
    disconnect_node_from_ray_cluster(removed_node_ip)
    available_ips.insert(0, removed_node_ip)
    log(reason % (removed_node_id, removed_node_ip))
    return (
        int(app_state["node_count"]),
        available_ips,
    )


def phase1_find_target_node_count(coord, app_state, available_ips):
    """Expand by node according to the configured phase-1 stopping mode."""
    phase_iter_samples = []
    phase_cost_samples = []
    phase_reconfig_samples = []
    phase_reconfig_cost_samples = []
    phase1_tuning_poll_profiles = []
    pending_expansion_baseline_cost = None
    stop_mode = str(PHASE1_STOPPING_MODE).strip().lower()
    if stop_mode not in {"lowest_price", "fastest_under_budget"}:
        raise RuntimeError(
            "PHASE1_STOPPING_MODE must be 'lowest_price' or 'fastest_under_budget'. "
            f"Got '{PHASE1_STOPPING_MODE}'."
        )
    stage = "find_min_price"

    while True:

        # Sample the current runtime cost before deciding whether to add a node.
        log(f"phase1: sleeping for {float(POLL_S)} seconds")
        time.sleep(float(POLL_S))
        log(f"phase1: collecting iteration costs")

        iter_durations_s, reconfig_durations_s = collect_iteration_costs(
            coord,
            int(app_state["node_count"]),
            phase_cost_samples,
            phase_reconfig_samples,
            phase_reconfig_cost_samples,
        )
        phase_iter_samples.extend(float(x) for x in iter_durations_s)
        # print(f"iter_durations_s: {iter_durations_s}", flush=True)
        avg_iter_s = average_execution_time(iter_durations_s)
        if reconfig_durations_s:
            for dt in reconfig_durations_s:
                log(f"phase1 reconfig_time_s={dt:.6f}")
        if avg_iter_s is None:
            print(f"Cannot get runtime iteration time. Make sure profiling is enabled.", flush=True)
            continue

        cost_per_iteration = float(avg_iter_s) * float(
            price_rate_for_node_count(int(app_state["node_count"]))
        )
        log(
            "phase1: node_count=%d world_size=%d avg_iter_s=%.6f cost_per_iter=%.6f"
            % (
                int(app_state["node_count"]),
                int(app_state["world_size"]),
                float(avg_iter_s),
                float(cost_per_iteration),
            )
        )
        phase1_tuning_poll_profiles.append(
            {
                "node_count": int(app_state["node_count"]),
                "world_size": int(app_state["world_size"]),
                "avg_iter_s": float(avg_iter_s),
                "cost_per_iteration": float(cost_per_iteration),
            }
        )

        # Evaluate the latest node addition before deciding whether to scale further.
        if pending_expansion_baseline_cost is not None and int(app_state["node_count"]) > 1:
            if stage == "find_min_price":
                # Stage 1 for both modes: keep adding nodes until the weighted
                # price per iteration stops decreasing. This prevents the
                # expensive head-only point from causing an early stop.
                if float(cost_per_iteration) > float(pending_expansion_baseline_cost):
                    if stop_mode == "lowest_price":
                        target_node_count, available_ips = _revert_latest_node(
                            coord,
                            app_state,
                            available_ips,
                            (
                                "phase1: weighted cost regressed after adding a node "
                                "(%.6f -> %.6f); reverted node_id=%%s node_ip=%%s"
                                % (
                                    float(pending_expansion_baseline_cost),
                                    float(cost_per_iteration),
                                )
                            ),
                        )
                        return (
                            target_node_count,
                            float(pending_expansion_baseline_cost),
                            phase_iter_samples,
                            phase_cost_samples,
                            phase_reconfig_samples,
                            phase_reconfig_cost_samples,
                            phase1_tuning_poll_profiles,
                            available_ips,
                        )
                    stage = "fastest_under_budget"
                    min_price_cost = float(pending_expansion_baseline_cost)
                    pending_expansion_baseline_cost = None
                    if min_price_cost >= float(COST_THRESHOLD_PER_ITER):
                        target_node_count, available_ips = _revert_latest_node(
                            coord,
                            app_state,
                            available_ips,
                            (
                                "phase1: minimum-price point is above budget "
                                "(%.6f); reverted node_id=%%s node_ip=%%s"
                                % float(cost_per_iteration)
                            ),
                        )
                        log(
                            "phase1: minimum-price point is still above budget; "
                            "stopping at node_count=%d"
                            % int(target_node_count)
                        )
                        return (
                            target_node_count,
                            float(min_price_cost),
                            phase_iter_samples,
                            phase_cost_samples,
                            phase_reconfig_samples,
                            phase_reconfig_cost_samples,
                            phase1_tuning_poll_profiles,
                            available_ips,
                        )
                    # Treat the current (just-added) node as the first stage-2
                    # candidate and enforce the same budget rule used below.
                    if float(cost_per_iteration) >= float(COST_THRESHOLD_PER_ITER):
                        target_node_count, available_ips = _revert_latest_node(
                            coord,
                            app_state,
                            available_ips,
                            (
                                "phase1: first stage-2 candidate exceeds budget "
                                "(%.6f >= %.6f); reverted node_id=%%s node_ip=%%s"
                                % (
                                    float(cost_per_iteration),
                                    float(COST_THRESHOLD_PER_ITER),
                                )
                            ),
                        )
                        return (
                            target_node_count,
                            float(min_price_cost),
                            phase_iter_samples,
                            phase_cost_samples,
                            phase_reconfig_samples,
                            phase_reconfig_cost_samples,
                            phase1_tuning_poll_profiles,
                            available_ips,
                        )
                    log(
                        "phase1: minimum-price point found at previous node count; "
                        "continuing stage 2 from current under-budget node_count=%d"
                        % int(app_state["node_count"])
                    )
            elif stage == "fastest_under_budget":
                # Stage 2 for fastest_under_budget: every under-budget point can
                # remain as a stepping stone; only revert the latest node if it
                # pushes weighted cost above the budget.
                if float(cost_per_iteration) >= float(COST_THRESHOLD_PER_ITER):
                    target_node_count, available_ips = _revert_latest_node(
                        coord,
                        app_state,
                        available_ips,
                        "phase1: budget exceeded after latest node add; reverted node_id=%s node_ip=%s",
                    )
                    return (
                        target_node_count,
                        float(pending_expansion_baseline_cost),
                        phase_iter_samples,
                        phase_cost_samples,
                        phase_reconfig_samples,
                        phase_reconfig_cost_samples,
                        phase1_tuning_poll_profiles,
                        available_ips,
                    )
            else:
                raise RuntimeError(f"Unknown phase1 stage '{stage}'")

        if not available_ips:
            log("phase1: no more available worker IPs; using current node count as target")
            return (
                int(app_state["node_count"]),
                float(cost_per_iteration),
                phase_iter_samples,
                phase_cost_samples,
                phase_reconfig_samples,
                phase_reconfig_cost_samples,
                phase1_tuning_poll_profiles,
                available_ips,
            )

        # Scale up by one node at a time: first join the worker to Ray, then
        # expand the malleable job so it can consume the new capacity.
        if (
            stage == "fastest_under_budget"
            and float(cost_per_iteration) >= float(COST_THRESHOLD_PER_ITER)
        ):
            log(
                "phase1: current accepted point is not under budget anymore; "
                "stopping at node_count=%d"
                % int(app_state["node_count"])
            )
            return (
                int(app_state["node_count"]),
                float(cost_per_iteration),
                phase_iter_samples,
                phase_cost_samples,
                phase_reconfig_samples,
                phase_reconfig_cost_samples,
                phase1_tuning_poll_profiles,
                available_ips,
            )
        node_ip = available_ips.pop(0)
        previous_node_count = int(app_state["node_count"])
        pending_expansion_baseline_cost = float(cost_per_iteration)
        node_id = allocate_node_into_ray_cluster(node_ip)
        expand_job_for_node(coord, app_state, node_ip, node_id)
        # Immediately consume the mixed old/new profiling window created by the
        # resize so the next poll sees only steady-state samples for the new
        # node count. Charge iteration time to the old count and reconfig time
        # to the new count for this one post-add collection.
        post_add_iter_durations_s, _ = collect_iteration_costs(
            coord,
            previous_node_count,
            phase_cost_samples,
            phase_reconfig_samples,
            phase_reconfig_cost_samples,
            reconfig_node_count=int(app_state["node_count"]),
        )
        phase_iter_samples.extend(float(x) for x in post_add_iter_durations_s)


def main() -> int:
    """Run the phase-1 node-granularity expansion search."""
    wait_for_ray(address="auto", namespace="default", timeout_s=60.0)
    coord = wait_for_coordinator(COORDINATOR_NAME, timeout_s=30.0)

    # Start from the current app snapshot and the list of unused candidate IPs.
    app_state = seed_app_state(coord)
    available_ips = initial_available_ips(app_state)
    log(
        "phase1 initial app node_count=%d world_size=%d active_node_ips=%s"
        % (
            int(app_state["node_count"]),
            int(app_state["world_size"]),
            app_state["ordered_node_ips"],
        )
    )

    (
        target_node_count,
        final_cost_per_iteration,
        phase1_iter_samples,
        phase1_costs,
        phase1_reconfigs,
        phase1_reconfig_costs,
        phase1_tuning_poll_profiles,
        remaining_ips,
    ) = phase1_find_target_node_count(
        coord, app_state, available_ips
    )

    from spot_common import write_json
    # Unfiltered average over all iterations, including total reconfiguration cost.
    avg_cost_over_tuning = (
        (sum(float(x) for x in phase1_costs) + sum(float(x) for x in phase1_reconfig_costs))
        / float(len(phase1_costs))
        if phase1_costs
        else None
    )

    # Persist the chosen target node count so phase 2 can reuse it directly.
    write_json(
        PHASE1_OUTPUT_PATH,
        {
            "metadata": metadata_dict(),
            "phase1": {
                "target_node_count": int(target_node_count),
                "remaining_available_ips": list(remaining_ips),
                "app_state": dict(app_state),
                "iter_durations_s": phase1_iter_samples,
                "cost_samples": phase1_costs,
                "reconfig_durations_s": phase1_reconfigs,
                "reconfig_cost": sum(float(x) for x in phase1_reconfig_costs),
                "reconfig_cost_samples": phase1_reconfig_costs,
                "final_cost_per_iteration": final_cost_per_iteration,
                "average_cost_per_iteration_over_tuning": avg_cost_over_tuning,
                "phase1_tuning_poll_profiles": phase1_tuning_poll_profiles,
                **summarize_cost_samples(phase1_costs),
            },
        },
    )
    return 0


if __name__ == "__main__":
    seed_random(time.time())
    raise SystemExit(main())
