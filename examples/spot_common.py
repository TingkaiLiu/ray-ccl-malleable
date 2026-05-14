#!/usr/bin/env python3
"""
Shared utilities for the simulated spot-node workflow examples.

This module keeps all tunable metadata at the top of the file so users can fill
in the environment-specific values once and reuse them in both phase drivers.

The simulation model is:
- Assume head node is already running on on-demand instance.
- all candidate worker IPs already exist,
- they are treated like spot nodes for workflow purposes,
- adding a node sleeps for a tunable "spin-up" delay,
- then SSHes to the node and runs `ray start --address=...`.
"""

from __future__ import annotations

import json
import math
import random
import shlex
import statistics
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
from ray.util.state import list_nodes


# ---------------------------------------------------------------------------
# Fill these metadata values for your environment before running.
# ---------------------------------------------------------------------------

COORDINATOR_NAME = "malleable_coordinator_stencil"  # Running app coordinator.
RAY_ADDRESS = "auto"  # Ray bootstrap address for this controller process.
RAY_NAMESPACE = "default"  # Ray namespace used by the running job.

# Head node address used by remote `ray start --address=...`.
RAY_HEAD_ADDRESS = "172.31.79.110:6379"  # Address passed to remote `ray start`.

# Candidate worker addresses for SSH (hostnames or IPs). The workflow treats
# them like spot nodes that may disappear and later return.
WORKER_NODE_IPS = [
    # "cn094",
    # "cn095",
    # "cn096",
    # "cn064",
    # "cn003",
    "172.31.73.58",
    "172.31.65.98",
    "172.31.71.111",
    "172.31.74.91",
    # "172.31.76.233",
    # "172.31.68.181",
    # "172.31.65.19",
]  # SSH targets for simulated join/rejoin (must match keys below when mapping).

# CPU capacity per worker, keyed by the same strings as WORKER_NODE_IPS.
# Omit a key to let Ray auto-detect CPU count on that host.
WORKER_NODE_CPUS = {
    # "cn094": 16,
    # "cn095": 16,
    # "cn096": 16,
    # "cn064": 16,
    # "cn003": 16,
    "172.31.73.58": 16,
    "172.31.65.98": 16,
    "172.31.71.111": 16,
    "172.31.74.91": 16,
    # "172.31.76.233": 16,
    # "172.31.68.181": 16,
    # "172.31.65.19": 16,
}  # Per-node CPU counts passed via `ray start --num-cpus=...`.

# When Ray's State API reports a different ``node_ip`` than your SSH hostname,
# map each WORKER_NODE_IPS entry to that address. SSH always uses the dict key
# (and WORKER_NODE_IPS / WORKER_NODE_CPUS); Ray join/resolve/wait uses the value.
WORKER_RAY_NODE_IP_BY_HOSTNAME: Dict[str, str] = {
    # "cn094": "141.142.253.157",
    # "cn095": "141.142.253.158",
    # "cn096": "141.142.253.159",
    # "cn064": "141.142.253.127"
    # "cn003": "141.142.253.66"
    "172.31.73.58": "172.31.73.58",
    "172.31.65.98": "172.31.65.98",
    "172.31.71.111": "172.31.71.111",
    "172.31.74.91": "172.31.74.91",
    # "172.31.76.233": "172.31.76.233",
    # "172.31.68.181": "172.31.68.181",
    # "172.31.65.19": "172.31.65.19",
}

# SSH settings for joining a worker node to the Ray cluster.
SSH_USER = "ec2-user"  # SSH username for worker machines.
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=no",
]  # SSH options used for remote Ray worker control.
SSH_TIMEOUT_S = 60.0  # Timeout for one SSH command.
DRY_RUN_SSH = False  # If true, only print SSH commands without executing them.
RAY_START_EXTRA_ARGS = ""  # Extra flags appended to `ray start`.
RAY_STOP_ON_FAILURE = True  # Stop Ray on a failed node during phase 2.

# Application/node-granularity model.
RANKS_PER_NODE = 16  # Application ranks added per joined worker node.

# Price model. The head node is assumed to be a more expensive on-demand node,
# while added worker nodes are treated like cheaper spot instances.
HEAD_NODE_PRICE_PER_S = 1.0
WORKER_NODE_PRICE_PER_S = 0.5

# Phase 1: cost-based node expansion.
POLL_S = 20.0  # Poll interval for tuner loops.
PHASE1_STOPPING_MODE = "fastest_under_budget"  # "fastest_under_budget" Or "lowest_price".
COST_THRESHOLD_PER_ITER = 2.0  # Budget used once the minimum-price point is found.
NODE_JOIN_DELAY_S = 0.0  # Simulated cloud instance spin-up delay.
PHASE1_WAIT_TIMEOUT_S = 300.0  # Timeout for cluster/app convergence waits.
PHASE1_OUTPUT_PATH = "spot_phase1.json"  # Phase 1 JSON summary path.

# Phase 2: simulated failures and relaunch.
PHASE2_FAILURE_MODE = "random"  # Or "manual".
FAILURE_PROBABILITY_PER_TICK = 0.05  # Per-tick failure probability for each node.
RECOVERY_PROBABILITY_PER_TICK = 0.20  # Per-tick recovery probability in random mode.
PHASE2_TICKS = 50  # Number of phase-2 polling iterations to simulate.
PHASE2_POLL_S = 20.0  # Poll interval for the phase-2 control loop.
PHASE2_OUTPUT_PATH = "spot_phase2.json"  # Phase 2 JSON summary path.

# Manual phase-2 event stream. Each event is keyed by node_ip and uses
# tick indices from the phase-2 loop:
# - tick: when the action should happen
# - action: either "drain" or "recover"
#
# Example:
# PHASE2_MANUAL_EVENTS = [
#     {"node_ip": "10.0.0.2", "tick": 5, "action": "drain"},
#     {"node_ip": "10.0.0.2", "tick": 20, "action": "recover"},
# ]
PHASE2_MANUAL_EVENTS: List[Dict[str, Any]] = []

def log(msg: str) -> None:
    """Print a prefixed log line for the simulated spot workflow."""
    print(f"[sim-spot] {msg}", flush=True)


def wait_for_ray(address: str, namespace: str, timeout_s: float) -> None:
    """Connect this controller process to the running Ray cluster."""
    deadline = time.time() + float(timeout_s)
    while True:
        try:
            ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
            return
        except Exception as e:
            if time.time() >= deadline:
                raise RuntimeError(f"timed out connecting to Ray: {e}") from e
            time.sleep(2.0)


def wait_for_coordinator(name: str, timeout_s: float):
    """Resolve the detached malleable coordinator actor by name."""
    deadline = time.time() + float(timeout_s)
    while True:
        try:
            return ray.get_actor(name)
        except ValueError:
            if time.time() >= deadline:
                raise RuntimeError(f"timed out waiting for coordinator '{name}'")
            time.sleep(2.0)


def _ssh_target(ip: str) -> str:
    """Format the SSH target string from configured user and IP."""
    user = str(SSH_USER or "").strip()
    return f"{user}@{ip}" if user else str(ip)


def _run_ssh(ip: str, remote_cmd: str, quiet_stdout: bool = False) -> None:
    """Execute one SSH command against a worker node."""
    target = _ssh_target(ip)
    cmd = ["ssh", *SSH_OPTIONS, target, remote_cmd]
    log(f"ssh {target}: {remote_cmd}")
    if DRY_RUN_SSH:
        return
    subprocess.run(
        cmd,
        check=True,
        timeout=float(SSH_TIMEOUT_S),
        stdout=subprocess.DEVNULL if quiet_stdout else None,
    )


def _ray_start_command(node_ip: str) -> str:
    """Build the remote `ray start` command for one worker IP."""
    cmd = f"ray start --address={shlex.quote(str(RAY_HEAD_ADDRESS))}"
    node_cpus = WORKER_NODE_CPUS.get(str(node_ip), None)
    if node_cpus is not None:
        cmd = f"{cmd} --num-cpus={int(node_cpus)}"
    if str(RAY_START_EXTRA_ARGS or "").strip():
        cmd = f"{cmd} {str(RAY_START_EXTRA_ARGS).strip()}"
    return cmd


def _ray_stop_command() -> str:
    """Build the remote command used to stop Ray on a worker."""
    return "ray stop"


def _alive_nodes() -> List[Any]:
    """Return the current alive Ray nodes visible through the State API."""
    return list_nodes(
        filters=[("state", "=", "ALIVE")],
        detail=False,
        raise_on_missing_output=False,
    )


def ray_state_node_ip(worker_label: str) -> str:
    """Ray State API ``node_ip`` to use for this worker (see WORKER_RAY_NODE_IP_BY_HOSTNAME)."""
    label = str(worker_label).strip()
    mapped = WORKER_RAY_NODE_IP_BY_HOSTNAME.get(label)
    if mapped is not None and str(mapped).strip():
        return str(mapped).strip()
    return label


def node_ip_is_alive(node_ip: str) -> bool:
    """Check whether this worker (SSH label) is joined to Ray, using mapped Ray ``node_ip``."""
    ray_ip = ray_state_node_ip(node_ip)
    return any(
        str(getattr(node, "node_ip", "") or "") == str(ray_ip) for node in _alive_nodes()
    )


def wait_for_node_ip_alive(node_ip: str, timeout_s: float) -> None:
    """Wait until this worker (SSH label, mapped Ray IP) appears as an alive Ray node."""
    deadline = time.time() + float(timeout_s)
    while True:
        if node_ip_is_alive(node_ip):
            return
        if time.time() >= deadline:
            raise RuntimeError(f"timed out waiting for Ray node_ip={node_ip} to join")
        time.sleep(2.0)


def wait_for_node_ip_dead(node_ip: str, timeout_s: float) -> None:
    """Wait until this worker (SSH label, mapped Ray IP) disappears from the Ray cluster."""
    deadline = time.time() + float(timeout_s)
    while True:
        if not node_ip_is_alive(node_ip):
            return
        if time.time() >= deadline:
            raise RuntimeError(f"timed out waiting for Ray node_ip={node_ip} to leave")
        time.sleep(2.0)


def average_execution_time(durations_s: List[float]) -> Optional[float]:
    """Compute an outlier-filtered mean of non-negative iteration durations."""
    durations = [float(x) for x in durations_s if float(x) >= 0.0]
    if not durations:
        return None
    values = np.asarray(durations, dtype=np.float64)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = [float(v) for v in values if lower_bound <= v <= upper_bound]
    if not filtered:
        return None
    return float(statistics.fmean(filtered))


def _alive_node_ip_by_id() -> Dict[str, str]:
    """Return alive node-id to node-ip mapping from the State API."""
    return {
        str(getattr(node, "node_id", "") or ""): str(getattr(node, "node_ip", "") or "")
        for node in _alive_nodes()
        if str(getattr(node, "node_id", "") or "")
    }


def get_current_world_size(coord) -> int:
    """Return the application's current world size from the coordinator."""
    return int(ray.get(coord.get_current_world_size.remote()))


def resolve_alive_node_id_by_ip(node_ip: str) -> str:
    """Resolve a joined Ray worker's ``node_id`` from its SSH worker label (mapped Ray IP)."""
    ray_ip = ray_state_node_ip(node_ip)
    for node in _alive_nodes():
        if str(getattr(node, "node_ip", "") or "") == str(ray_ip):
            return str(getattr(node, "node_id", "") or "")
    raise RuntimeError(
        f"could not resolve alive Ray node_id for worker={node_ip!r} (ray node_ip={ray_ip!r})"
    )


def get_head_node_snapshot() -> Tuple[str, str]:
    """Return the current head node's (node_id, node_ip)."""
    for node in _alive_nodes():
        if bool(getattr(node, "is_head_node", False)):
            node_id = str(getattr(node, "node_id", "") or "")
            node_ip = str(getattr(node, "node_ip", "") or "")
            if node_id and node_ip:
                return node_id, node_ip
    raise RuntimeError("could not resolve the current Ray head node")


def seed_app_state(coord) -> Dict[str, Any]:
    """Seed local app metadata once from the current coordinator/head state."""
    head_node_id, head_node_ip = get_head_node_snapshot()
    return {
        "world_size": get_current_world_size(coord),
        "ordered_node_ids": [head_node_id],
        "ordered_node_ips": [head_node_ip],
        "node_count": 1,
        "rank0_node_id": head_node_id,
        "rank0_node_ip": head_node_ip,
    }


def wait_until_no_pending_reconfig(coord, timeout_s: float) -> None:
    """Wait until the coordinator reports no reconfiguration in flight."""
    deadline = time.time() + float(timeout_s)
    while True:
        if ray.get(coord.get_reconfig_target.remote()) is None:
            return
        if time.time() >= deadline:
            raise RuntimeError("timed out waiting for pending reconfiguration to clear")
        time.sleep(2.0)


def wait_for_world_size(coord, expected_world_size: int, timeout_s: float) -> None:
    """Wait until the application reaches the expected world size."""
    deadline = time.time() + float(timeout_s)
    expected_world_size = int(expected_world_size)
    while True:
        current = int(ray.get(coord.get_current_world_size.remote()))
        if current == expected_world_size:
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"timed out waiting for world_size={expected_world_size}, current={current}"
            )
        time.sleep(2.0)


def collect_iteration_costs(
    coord,
    node_count: int,
    phase_cost_samples: List[float],
    phase_reconfig_samples: List[float],
    phase_reconfig_cost_samples: Optional[List[float]] = None,
    reconfig_node_count: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """Poll coordinator profiling and accumulate node-weighted iteration cost."""
    iter_durations_s, reconfig_durations_s = ray.get(coord.get_tuner_durations.remote())
    # print(f"iter_durations_s: {iter_durations_s}", flush=True)
    # print(f"reconfig_durations_s: {reconfig_durations_s}", flush=True)
    iter_durations_s = [float(x) for x in iter_durations_s]
    reconfig_durations_s = [float(x) for x in reconfig_durations_s]
    iter_price_rate = price_rate_for_node_count(int(node_count))
    reconfig_price_rate = price_rate_for_node_count(
        int(reconfig_node_count) if reconfig_node_count is not None else int(node_count)
    )
    phase_cost_samples.extend(float(dt) * float(iter_price_rate) for dt in iter_durations_s)
    phase_reconfig_samples.extend(reconfig_durations_s)
    if phase_reconfig_cost_samples is not None:
        phase_reconfig_cost_samples.extend(
            float(dt) * float(reconfig_price_rate) for dt in reconfig_durations_s
        )
    return iter_durations_s, reconfig_durations_s


def allocate_node_into_ray_cluster(node_ip: str) -> str:
    """Simulate node spin-up, then join the worker into the Ray cluster."""
    # This models cloud allocation latency before the worker can be started.
    # log(f"simulating node acquisition delay for {node_ip} ({NODE_JOIN_DELAY_S}s)")
    # time.sleep(float(NODE_JOIN_DELAY_S)) # TODO: remove for phase 2

    # Join the existing Ray cluster only if the worker is not already present.
    if not node_ip_is_alive(node_ip):
        _run_ssh(node_ip, _ray_start_command(node_ip))
        print(f"Ray worker node on {node_ip} started", flush=True)
        # wait_for_node_ip_alive(node_ip, float(PHASE1_WAIT_TIMEOUT_S))
    else:
        log(f"node {node_ip} is already alive in the Ray cluster")
    return resolve_alive_node_id_by_ip(node_ip)


def expand_job_for_node(coord, app_state: Dict[str, Any], node_ip: str, node_id: str) -> None:
    """Expand the malleable job after the worker node has joined Ray."""
    current_world_size = int(app_state["world_size"])
    # First wait until any previous resize/drain operation has completed.
    wait_until_no_pending_reconfig(coord, float(PHASE1_WAIT_TIMEOUT_S))
    log(
        f"requesting expand for node_ip={node_ip} from world_size={current_world_size} "
        f"to {current_world_size + int(RANKS_PER_NODE)}"
    )
    # Then expand the application by one node's worth of ranks.
    ray.get(coord.request_resize.remote(int(current_world_size + int(RANKS_PER_NODE))))
    wait_for_world_size(
        coord,
        int(current_world_size + int(RANKS_PER_NODE)),
        float(PHASE1_WAIT_TIMEOUT_S),
    )
    app_state["world_size"] = int(current_world_size + int(RANKS_PER_NODE))
    app_state["ordered_node_ids"].append(str(node_id))
    app_state["ordered_node_ips"].append(str(node_ip))
    app_state["node_count"] = len(app_state["ordered_node_ids"])


def disconnect_node_from_ray_cluster(node_ip: str) -> None:
    """Optionally stop Ray on a worker after it has been removed from the app."""
    if not RAY_STOP_ON_FAILURE:
        return
    _run_ssh(node_ip, _ray_stop_command(), quiet_stdout=True)
    print(f"Ray worker node on {node_ip} stopped", flush=True)
    if not DRY_RUN_SSH:
        try:
            wait_for_node_ip_dead(node_ip, float(PHASE1_WAIT_TIMEOUT_S))
        except Exception:
            log(f"warning: node_ip={node_ip} did not leave Ray before timeout")


def shrink_job_by_removing_node(
    coord,
    app_state: Dict[str, Any],
    node_id: str,
) -> None:
    """Drain one node from the app and update local membership metadata."""
    current_world_size = int(app_state["world_size"])
    wait_until_no_pending_reconfig(coord, float(PHASE1_WAIT_TIMEOUT_S))
    log(f"draining node_id={node_id}")
    ray.get(coord.request_remove_nodes.remote([str(node_id)]))
    wait_for_world_size(
        coord,
        int(current_world_size - int(RANKS_PER_NODE)),
        float(PHASE1_WAIT_TIMEOUT_S),
    )

    if str(node_id) in app_state["ordered_node_ids"]:
        idx = app_state["ordered_node_ids"].index(str(node_id))
        app_state["ordered_node_ids"].pop(idx)
        app_state["ordered_node_ips"].pop(idx)
    app_state["world_size"] = int(current_world_size - int(RANKS_PER_NODE))
    app_state["node_count"] = len(app_state["ordered_node_ids"])


def initial_available_ips(app_state: Dict[str, Any]) -> List[str]:
    """Return candidate worker IPs that are not already used by the app."""
    available_ips = [str(ip).strip() for ip in WORKER_NODE_IPS if str(ip).strip()]
    return [ip for ip in available_ips if ip not in set(app_state["ordered_node_ips"])]


def price_rate_for_node_count(node_count: int) -> float:
    """Return total price rate for the current application node count."""
    node_count = max(1, int(node_count))
    worker_count = max(0, node_count - 1)
    return float(HEAD_NODE_PRICE_PER_S) + float(worker_count) * float(WORKER_NODE_PRICE_PER_S)


def summarize_cost_samples(samples: List[float]) -> Dict[str, Any]:
    """Summarize per-iteration node-weighted cost samples with filtered mean."""
    values = [float(x) for x in samples]
    avg_cost = average_execution_time(values)
    return {
        "num_samples": len(values),
        "avg_cost_per_iteration": avg_cost,
        "min_cost_per_iteration": min(values) if values else None,
        "max_cost_per_iteration": max(values) if values else None,
    }


def metadata_dict() -> Dict[str, Any]:
    """Return a JSON-serializable copy of the simulation metadata."""
    return {
        "coordinator_name": COORDINATOR_NAME,
        "ray_head_address": RAY_HEAD_ADDRESS,
        "worker_node_ips": list(WORKER_NODE_IPS),
        "worker_node_cpus": dict(WORKER_NODE_CPUS),
        "worker_ray_node_ip_by_hostname": dict(WORKER_RAY_NODE_IP_BY_HOSTNAME),
        "ranks_per_node": int(RANKS_PER_NODE),
        "head_node_price_per_s": float(HEAD_NODE_PRICE_PER_S),
        "worker_node_price_per_s": float(WORKER_NODE_PRICE_PER_S),
        "phase1_stopping_mode": str(PHASE1_STOPPING_MODE),
        "cost_threshold_per_iter": float(COST_THRESHOLD_PER_ITER),
        "node_join_delay_s": float(NODE_JOIN_DELAY_S),
        "phase2_failure_mode": str(PHASE2_FAILURE_MODE),
        "failure_probability_per_tick": float(FAILURE_PROBABILITY_PER_TICK),
        "recovery_probability_per_tick": float(RECOVERY_PROBABILITY_PER_TICK),
        "phase2_manual_events": list(PHASE2_MANUAL_EVENTS),
        "phase2_ticks": int(PHASE2_TICKS),
        "poll_s": float(POLL_S),
        "phase2_poll_s": float(PHASE2_POLL_S),
        "dry_run_ssh": bool(DRY_RUN_SSH),
        "ray_stop_on_failure": bool(RAY_STOP_ON_FAILURE),
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write one simulation result payload to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote simulation summary to {path}")


def seed_random(seed: int = 0) -> None:
    """Seed the random generator for repeatable failure simulation."""
    random.seed(seed)
