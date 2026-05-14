#!/usr/bin/env python3
"""Phase 2: simulate spot failure, drain, relaunch, and recover target size.

Current implementation
======================
This file models the steady-state behavior after phase 1 has already selected a
target node count. It assumes:

1. The Ray head node and malleable job are already running.
2. `spot_phase1.py` has already produced:
   - the target node count,
   - the remaining candidate worker IPs,
   - and the current app-state snapshot.
3. Phase 2 should maintain that target node count in the presence of simulated
   worker-node failures and later recoveries.

Two-layer design
----------------
The implementation is intentionally split into two layers.

### 1. Simulation layer

The simulation layer decides when nodes fail and when they become available
again. It does **not** call any malleable control APIs directly.

This is implemented by:

- `build_node_simulation_state()`
- `advance_node_simulation(...)`

The simulation layer supports two modes:

- `random`
  - worker failures are generated using `FAILURE_PROBABILITY_PER_TICK`
  - recovered nodes become available again using
    `RECOVERY_PROBABILITY_PER_TICK`
- `manual`
  - the user supplies explicit events in `PHASE2_MANUAL_EVENTS`
  - each event defines:
    - `node_ip`
    - `tick`
    - `action` (`drain` or `recover`)

The simulation layer emits only abstract events such as:

- node failure,
- drain request,
- node available again.

### 2. Real control layer

The control layer consumes simulation events and performs the actual runtime
actions needed to keep the application at the target node count.

This is implemented by:

- `phase2_control_loop(...)`

The control loop:

1. polls coordinator profiling data each tick,
2. asks the simulation layer whether any nodes should fail or recover,
3. drains nodes from the malleable job when required,
4. disconnects drained workers from the Ray cluster with `ray stop`,
5. rejoins recovered workers to Ray with `ray start`,
6. expands the malleable job back toward the target node count.

The control code uses the same two-step workflow as the other examples:

- shrink path:
  1. `request_remove_nodes(...)`
  2. disconnect worker from the Ray cluster

- expand path:
  1. join worker into the Ray cluster
  2. `request_resize(...)` to add ranks

Per-tick accounting
-------------------
Each phase-2 tick:

1. sleeps for `PHASE2_POLL_S`,
2. polls iteration and reconfiguration profiling from the coordinator,
3. computes weighted cost per iteration from the current node count,
4. advances the simulation state,
5. runs the corresponding control actions,
6. records emitted events for the final JSON output.

Outputs
-------
At the end of phase 2 the script writes:

- the final app-state snapshot,
- remaining available worker IPs,
- node-weighted cost samples,
- reconfiguration durations,
- and a detailed event log describing failures, drains, recoveries, and relaunches.
"""

from __future__ import annotations

import time

from spot_common import (
    COORDINATOR_NAME,
    FAILURE_PROBABILITY_PER_TICK,
    PHASE1_OUTPUT_PATH,
    PHASE2_OUTPUT_PATH,
    PHASE2_FAILURE_MODE,
    PHASE2_MANUAL_EVENTS,
    PHASE2_TICKS,
    PHASE2_POLL_S,
    RECOVERY_PROBABILITY_PER_TICK,
    allocate_node_into_ray_cluster,
    collect_iteration_costs,
    disconnect_node_from_ray_cluster,
    expand_job_for_node,
    log,
    metadata_dict,
    price_rate_for_node_count,
    shrink_job_by_removing_node,
    seed_random,
    summarize_cost_samples,
    wait_for_coordinator,
    wait_for_ray,
    write_json,
)


def load_phase1_target() -> tuple[int, list[str], dict]:
    """Load the phase-1 target node count and remaining candidate IPs."""
    import json

    with open(PHASE1_OUTPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    phase1 = dict(payload.get("phase1", {}))
    target_node_count = int(phase1.get("target_node_count", 0))
    remaining_available_ips = [str(ip) for ip in phase1.get("remaining_available_ips", [])]
    app_state = dict(phase1.get("app_state", {}))
    if target_node_count <= 0:
        raise RuntimeError(
            f"phase1 result at {PHASE1_OUTPUT_PATH} does not contain a valid target_node_count"
        )
    if not app_state:
        raise RuntimeError(
            f"phase1 result at {PHASE1_OUTPUT_PATH} does not contain an app_state snapshot"
        )

    print(f"loaded phase1 target_node_count={target_node_count}")
    print(f"loaded phase1 remaining_available_ips={remaining_available_ips}")
    print(f"loaded phase1 app_state={app_state}")
    
    return target_node_count, remaining_available_ips, app_state


def _normalize_manual_events() -> list[dict]:
    """Normalize manual failure events into a sorted per-tick schedule."""
    out = []
    for idx, event in enumerate(PHASE2_MANUAL_EVENTS):
        if not isinstance(event, dict):
            raise RuntimeError(f"manual phase2 event #{idx} must be a dict")
        node_ip = str(event.get("node_ip", "") or "").strip()
        tick = event.get("tick", None)
        action = str(event.get("action", "") or "").strip().lower()
        if not node_ip:
            raise RuntimeError(f"manual phase2 event #{idx} is missing node_ip")
        if tick is None or not action:
            raise RuntimeError(
                f"manual phase2 event for node_ip={node_ip} must include tick and action"
            )
        tick = int(tick)
        if action not in {"drain", "recover"}:
            raise RuntimeError(
                f"manual phase2 event for node_ip={node_ip} has invalid action={action!r}; "
                "expected 'drain' or 'recover'"
            )
        out.append(
            {
                "node_ip": node_ip,
                "tick": tick,
                "action": action,
            }
        )
    out.sort(key=lambda event: (int(event["tick"]), str(event["node_ip"]), str(event["action"])))
    return out


def _worker_candidate_pairs(app_state: dict) -> list[tuple[str, str]]:
    """Return the currently active worker nodes tracked by the application."""
    head_node_id = str(app_state.get("rank0_node_id", "") or "")
    head_node_ip = str(app_state.get("rank0_node_ip", "") or "")
    return [
        (str(node_id), str(node_ip))
        for node_id, node_ip in zip(app_state["ordered_node_ids"], app_state["ordered_node_ips"])
        if str(node_id) != head_node_id and str(node_ip) != head_node_ip
    ]


def build_node_simulation_state(app_state: dict) -> dict:
    """Build the phase-2 node-failure simulation state machine."""
    failure_mode = str(PHASE2_FAILURE_MODE).strip().lower()
    if failure_mode not in {"random", "manual"}:
        raise RuntimeError(
            "PHASE2_FAILURE_MODE must be 'random' or 'manual'. "
            f"Got '{PHASE2_FAILURE_MODE}'."
        )
    manual_events = _normalize_manual_events() if failure_mode == "manual" else []
    head_node_ip = str(app_state.get("rank0_node_ip", "") or "")
    if failure_mode == "manual":
        for event in manual_events:
            if str(event["node_ip"]) == head_node_ip:
                raise RuntimeError(
                    f"manual phase2 event for node_ip={head_node_ip} targets the Ray head node, "
                    "but phase2 only simulates worker-node failures"
                )
    if failure_mode == "random":
        return {
            "failure_mode": failure_mode,
            "pending_relaunch_node_ips": set(),
        }
    return {
        "failure_mode": failure_mode,
        "manual_events": manual_events,
        "manual_event_emitted": set(),
    }


def advance_node_simulation(tick: int, sim_state: dict, app_state: dict) -> tuple[list[dict], list[str]]:
    """Advance simulated node status and return drain/recovery events.

    Returns:
        - drain_requests: list of {"node_id", "node_ip"} that should now be drained
        - recovered_node_ips: list of node_ip that are now available again
    """
    import random

    failure_mode = sim_state["failure_mode"]
    drain_requests: list[dict] = []
    recovered_node_ips: list[str] = []

    candidate_pairs = _worker_candidate_pairs(app_state)
    node_id_by_ip = {str(node_ip): str(node_id) for node_id, node_ip in candidate_pairs}

    if failure_mode == "random":
        pending_relaunch_node_ips = sim_state["pending_relaunch_node_ips"]

        # Randomly recover already-failed workers back into the available pool.
        log(f"phase2: pending recovery of {len(pending_relaunch_node_ips)} nodes")
        for node_ip in list(pending_relaunch_node_ips):
            if random.random() < float(RECOVERY_PROBABILITY_PER_TICK):
                recovered_node_ips.append(str(node_ip))
                log(f"phase2: node_ip={node_ip} became available again for relaunch")
                pending_relaunch_node_ips.remove(node_ip)

        # Randomly fail active workers.
        for node_id, node_ip in candidate_pairs:
            if random.random() < float(FAILURE_PROBABILITY_PER_TICK):
                pending_relaunch_node_ips.add(str(node_ip))
                drain_requests.append(
                    {
                        "node_id": str(node_id),
                        "node_ip": str(node_ip),
                    }
                )
                log(f"phase2: simulated failure for node_id={node_id} node_ip={node_ip}")
    else:
        manual_events = sim_state["manual_events"]
        manual_event_emitted = sim_state["manual_event_emitted"]

        # Emit scheduled manual drain/recovery events.
        for event in manual_events:
            event_key = (str(event["node_ip"]), int(event["tick"]), str(event["action"]))
            if event_key in manual_event_emitted:
                continue
            if int(event["tick"]) != int(tick):
                continue
            node_ip = str(event["node_ip"])
            action = str(event["action"])
            if action == "recover":
                recovered_node_ips.append(node_ip)
                log(f"phase2: manual recovery for node_ip={node_ip}")
                manual_event_emitted.add(event_key)
                continue
            node_id = node_id_by_ip.get(node_ip, "")
            if not node_id:
                log(
                    f"phase2: manual drain for node_ip={node_ip} skipped because "
                    "the node is not currently in the application"
                )
                manual_event_emitted.add(event_key)
                continue
            drain_requests.append(
                {
                    "node_id": str(node_id),
                    "node_ip": str(node_ip),
                }
            )
            manual_event_emitted.add(event_key)
            log(f"phase2: manual drain for node_id={node_id} node_ip={node_ip}")

    return drain_requests, recovered_node_ips


def phase2_control_loop(coord, target_node_count: int, available_ips: list[str], app_state: dict):
    """Main control loop that maintains the target node count explored by phase 1."""
    phase_cost_samples = []
    phase_reconfig_samples = []
    event_log = []
    tick_timeline = []
    sim_state = build_node_simulation_state(app_state)

    for tick in range(int(PHASE2_TICKS)):
        # Sample current application cost on each simulation tick.
        log(f"phase2: tick={tick}")

        log(f"phase2: sleeping for {PHASE2_POLL_S} seconds")
        time.sleep(float(PHASE2_POLL_S))
        # print(f"phase2 woke up after {PHASE2_POLL_S} seconds")

        iter_durations_s, reconfig_durations_s = collect_iteration_costs(
            coord,
            app_state["node_count"],
            phase_cost_samples,
            phase_reconfig_samples,
        )
        avg_iter_s = (
            None
            if not iter_durations_s
            else sum(iter_durations_s) / float(len(iter_durations_s))
        )
        cost_per_iteration = (
            None
            if avg_iter_s is None
            else float(avg_iter_s) * float(price_rate_for_node_count(int(app_state["node_count"])))
        )
        tick_record = {
            "tick": int(tick),
            "timestamp_s": float(time.time()),
            "node_count": int(app_state["node_count"]),
            "world_size": int(app_state["world_size"]),
            "avg_iter_s": None if avg_iter_s is None else float(avg_iter_s),
            "cost_per_iteration": cost_per_iteration,
            "reconfig_durations_s": [float(dt) for dt in reconfig_durations_s],
            "events": [],
        }
        if reconfig_durations_s:
            for dt in reconfig_durations_s:
                log(f"phase2 reconfig_time_s={dt:.6f}")
        if avg_iter_s is not None:
            log(
                "phase2 tick=%d node_count=%d world_size=%d avg_iter_s=%.6f cost_per_iter=%.6f"
                % (
                    int(tick),
                    int(app_state["node_count"]),
                    int(app_state["world_size"]),
                    float(avg_iter_s),
                    float(cost_per_iteration),
                )
            )

        # Query the simulation layer for current failure/recovery events, then
        # let the control layer react to them using the real malleable APIs.
        drain_requests, recovered_node_ips = advance_node_simulation(tick, sim_state, app_state)
        post_change_previous_node_count = None

        for req in drain_requests:
            node_id = str(req["node_id"])
            node_ip = str(req["node_ip"])
            if post_change_previous_node_count is None:
                post_change_previous_node_count = int(app_state["node_count"])
            shrink_job_by_removing_node(coord, app_state, node_id)
            disconnect_node_from_ray_cluster(node_ip)
            event = {
                "tick": int(tick),
                "kind": "node_drained",
                "node_id": node_id,
                "node_ip": node_ip,
                "timestamp_s": float(time.time()),
            }
            event_log.append(event)
            tick_record["events"].append(dict(event))

        for node_ip in recovered_node_ips:
            if node_ip not in available_ips:
                available_ips.append(node_ip)
            # event = {
            #     "tick": int(tick),
            #     "kind": "node_available_again",
            #     "node_ip": str(node_ip),
            # }
            # event_log.append(event)
            # tick_record["events"].append(dict(event))

        # If the app is below target size and workers are available again,
        # rejoin as many as needed to reach the target node count.
        while int(app_state["node_count"]) < int(target_node_count) and available_ips:
            node_ip = available_ips.pop(0)
            if post_change_previous_node_count is None:
                post_change_previous_node_count = int(app_state["node_count"])
            node_id = allocate_node_into_ray_cluster(node_ip)
            expand_job_for_node(coord, app_state, node_ip, node_id)
            event = {
                "tick": int(tick),
                "kind": "node_relaunched_and_expanded",
                "node_ip": str(node_ip),
                "timestamp_s": float(time.time()),
            }
            event_log.append(event)
            tick_record["events"].append(dict(event))

        if post_change_previous_node_count is not None:
            collect_iteration_costs(
                coord,
                post_change_previous_node_count,
                phase_cost_samples,
                phase_reconfig_samples,
                reconfig_node_count=int(app_state["node_count"]),
            )

        tick_timeline.append(tick_record)

        

    return phase_cost_samples, phase_reconfig_samples, event_log, tick_timeline


def main() -> int:
    """Run the phase-2 failure/recovery simulation."""
    wait_for_ray(address="auto", namespace="default", timeout_s=60.0)
    coord = wait_for_coordinator(COORDINATOR_NAME, timeout_s=30.0)
    target_node_count, available_ips, app_state = load_phase1_target()

    # Reuse the phase-1 target node count as the steady-state capacity goal.
    log(f"phase2 target_node_count={target_node_count}")
    phase2_costs, phase2_reconfigs, event_log, tick_timeline = phase2_control_loop(
        coord, target_node_count, available_ips, app_state
    )

    # Persist phase-2 cost samples and event history for later analysis.
    write_json(
        PHASE2_OUTPUT_PATH,
        {
            "metadata": metadata_dict(),
            "phase2": {
                "target_node_count": int(target_node_count),
                "remaining_available_ips": list(available_ips),
                "app_state": dict(app_state),
                "cost_samples": phase2_costs,
                "reconfig_durations_s": phase2_reconfigs,
                "tick_timeline": tick_timeline,
                "events": event_log,
                **summarize_cost_samples(phase2_costs),
            },
        },
    )
    return 0


if __name__ == "__main__":
    seed_random(66)
    raise SystemExit(main())
