from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import ray

from . import types


def _raise_with_log(
    message: str,
    exc_type=RuntimeError,
    *,
    cause: Optional[BaseException] = None,
) -> None:
    print(f"[malleable] {message}", flush=True)
    if cause is not None:
        raise exc_type(message) from cause
    raise exc_type(message)


def normalize_actor_id(actor_id: Any) -> str:
    """Normalize Ray actor IDs into a stable string identifier.

    We use ActorID.hex() when available because it is stable across processes
    and survives serialization. Fallbacks are best-effort.
    """
    if actor_id is None:
        _raise_with_log("actor_id is None", ValueError)
    hex_fn = getattr(actor_id, "hex", None)
    if callable(hex_fn):
        return hex_fn()
    if isinstance(actor_id, (bytes, bytearray, memoryview)):
        return bytes(actor_id).hex()
    binary = getattr(actor_id, "binary", None)
    if callable(binary):
        return binary().hex()
    return str(actor_id)


# Per-process handler registries (used inside rank actors).
_checkpoint_handlers: Dict[str, Callable[[], Any]] = {}
_restore_handlers: Dict[str, Callable[[Any], None]] = {}
_redistribute_handlers: Dict[str, Callable[[List[Any], int, int], List[Any]]] = {}


def register_checkpoint_handler(group_name: str, fn: Callable[[], Any]) -> None:
    _checkpoint_handlers[group_name] = fn


def register_restore_handler(group_name: str, fn: Callable[[Any], None]) -> None:
    _restore_handlers[group_name] = fn


def register_redistribute_handler(
    group_name: str, fn: Callable[[List[Any], int, int], List[Any]]
) -> None:
    _redistribute_handlers[group_name] = fn


def _get_checkpoint_handler(group_name: str) -> Optional[Callable[[], Any]]:
    return _checkpoint_handlers.get(group_name)


def _get_restore_handler(group_name: str) -> Optional[Callable[[Any], None]]:
    return _restore_handlers.get(group_name)


def _get_redistribute_handler(
    group_name: str,
) -> Optional[Callable[[List[Any], int, int], List[Any]]]:
    return _redistribute_handlers.get(group_name)


class _AsyncBarrier:
    """Simple reusable async barrier for Ray actors."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._count = 0

    async def wait(self, target: int) -> None:
        # Wait for previous cycle to clear.
        while self._event.is_set():
            await asyncio.sleep(0)

        self._count += 1
        if self._count < target:
            await self._event.wait()
            return

        # Release this cycle.
        self._event.set()
        self._count = 0
        self._event.clear()


@ray.remote(num_cpus=0, max_concurrency=1024, resources={"head_node": 0.001})
class MalleableCoordinator:
    """Coordinator for malleable Ray CCL groups (CPU/Gloo first).

    Coordinates:
    - reconfigure_test() decisions at safe points
    - checkpoint collection
    - epoch commit (planned resize)
    """

    def __init__(
        self,
        base_group_name: str,
        actor_ids_by_rank: List[Any],
        checkpoint_threshold: int = 0,
        controller_name: str = "",
        tuner_profile_enabled: bool = False,
        tuner_profile_maxlen: int = 10000,
        reconfig_profile_enabled: bool = False,
    ):
        self.base_group_name = base_group_name
        self.epoch = 0
        self.actor_ids_by_rank = [normalize_actor_id(a) for a in actor_ids_by_rank]
        self.world_size = len(self.actor_ids_by_rank)
        self.controller_name = str(controller_name or "")

        self.checkpoint_threshold = int(checkpoint_threshold)
        self._checkpoint_counter = 0
        self._should_checkpoint = False
        self._checkpoint_round_id = 0
        self._checkpoint_round_done = asyncio.Event()

        # Resize request (planned).
        self._pending_target_actor_ids_by_rank: Optional[List[Any]] = None
        self._pending_target_world_size: Optional[int] = None
        self._pending_next_epoch: Optional[int] = None
        self._pending_epoch_from: Optional[int] = None
        self._pending_reconfig_kind: Optional[str] = None
        self._pending_removed_node_ids: Optional[List[str]] = None

        # Checkpoint collection for a reconfig.
        self._ckpt_barrier = _AsyncBarrier()
        self._ckpt_by_rank: Dict[int, Any] = {}
        self._latest_checkpoint_by_rank: Dict[int, Any] = {}

        # Redistribution for next epoch.
        self._states_ready = asyncio.Event()
        self._next_epoch_states_by_rank: Optional[List[Any]] = None
        self._prepared_epoch: Optional[int] = None
        self._redistribute_lock = asyncio.Lock()
        self._redistribute_handler: Optional[
            Callable[[List[Any], int, int], List[Any]]
        ] = None

        # Commit gate for next epoch (meta updated once).
        self._commit_applied = asyncio.Event()

        # Optional: lightweight runtime profile for external tuner.
        # Records durations computed inside the coordinator (rank0 only):
        # - pure iteration durations (excluding reconfig/checkpoint/resize time)
        # - reconfig durations (time spent in reconfig path)
        self._tuner_profile_enabled = bool(tuner_profile_enabled)
        self._tuner_profile_maxlen = int(tuner_profile_maxlen)
        self._tuner_iter_durations_s: List[float] = []
        self._tuner_reconfig_durations_s: List[float] = []
        self._tuner_last_get_iter_idx: int = 0
        self._tuner_last_get_reconfig_idx: int = 0
        self._tuner_last_iter_ts_s: Optional[float] = None
        self._tuner_reconfig_start_ts_s: Optional[float] = None
        self._tuner_reconfig_inflight: bool = False

        # Reconfiguration profiling (compatible with old Ray-MPI style records).
        # Keys are epoch ids (next_epoch of each resize).
        self._reconfig_profile_enabled = bool(reconfig_profile_enabled)
        self._reconfig_record: Dict[int, Tuple[int, int]] = {}
        self._reconfig_data_time_s: Dict[int, float] = {}
        self._reconfig_rank_time_s: Dict[int, float] = {}
        self._reconfig_comm_time_s: Dict[int, float] = {}
        # Cached placement state:
        # - _actor_node_ids is the low-level actor_id -> node_id lookup
        # - _ordered_nodes records the current node block order in logical-rank space
        # - _node_rank_counts records how many logical ranks each node owns
        #
        # Startup placement is seeded lazily from Ray state once. After that,
        # the coordinator updates these caches itself whenever membership changes.
        self._actor_node_ids: Dict[str, str] = {}
        self._ordered_nodes: List[str] = []
        self._node_rank_counts: Dict[str, int] = {}

    def get_tuner_profile_enabled(self) -> bool:
        return bool(self._tuner_profile_enabled)

    def get_reconfig_profile_enabled(self) -> bool:
        return bool(self._reconfig_profile_enabled)

    def reset_tuner_timestamps(self) -> None:
        """Reset tuner profiling buffers.

        Kept for backward compatibility with earlier prototypes.
        """
        self._tuner_iter_durations_s = []
        self._tuner_reconfig_durations_s = []
        self._tuner_last_get_iter_idx = 0
        self._tuner_last_get_reconfig_idx = 0
        self._tuner_last_iter_ts_s = None
        self._tuner_reconfig_start_ts_s = None
        self._tuner_reconfig_inflight = False

    def get_tuner_durations(self) -> Tuple[List[float], List[float]]:
        """Return (iter_durations_s, reconfig_durations_s) since last call (consuming)."""
        i0 = int(self._tuner_last_get_iter_idx)
        r0 = int(self._tuner_last_get_reconfig_idx)
        self._tuner_last_get_iter_idx = len(self._tuner_iter_durations_s)
        self._tuner_last_get_reconfig_idx = len(self._tuner_reconfig_durations_s)
        return (
            list(self._tuner_iter_durations_s[i0:]),
            list(self._tuner_reconfig_durations_s[r0:]),
        )

    def get_reconfig_record(
        self,
    ) -> Tuple[
        Dict[int, Tuple[int, int]],
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
    ]:
        """Return resize profiling maps for external analysis.

        Returns:
            - reconfig_record: epoch -> (old_world_size, new_world_size)
            - reconfig_data_time: epoch -> redistribution handler time (seconds)
            - reconfig_rank_time: epoch -> rank creation time (seconds)
            - reconfig_comm_time: epoch -> rank0 communicator rebuild time (seconds)
        """
        if not self._reconfig_profile_enabled:
            return {}, {}, {}, {}
        return (
            dict(self._reconfig_record),
            dict(self._reconfig_data_time_s),
            dict(self._reconfig_rank_time_s),
            dict(self._reconfig_comm_time_s),
        )

    def report_comm_rebuild_time(self, epoch: int, duration_s: float) -> None:
        if not self._reconfig_profile_enabled:
            return
        self._reconfig_comm_time_s[int(epoch)] = float(duration_s)

    def _plan_resize(
        self,
        new_world_size: int,
        target_actor_ids_by_rank: Optional[List[Any]] = None,
        reconfig_kind: Optional[str] = None,
    ) -> None:
        """Store the next planned standard membership change."""
        
        # print("plan_resize", int(new_world_size), reconfig_kind, flush=True)
        
        new_world_size = int(new_world_size)
        if new_world_size <= 0:
            _raise_with_log(
                f"new_world_size must be > 0, got {new_world_size}", ValueError
            )

        if new_world_size == self.world_size:
            return

        if new_world_size > self.world_size and not target_actor_ids_by_rank:
            # Managed expansion: we *do not* create actors here yet, because
            # new ranks must be created with a redistributed per-rank state.
            # Actor creation happens during redistribute_next_epoch().
            if not self.controller_name:
                _raise_with_log(
                    "Expanding world size requires target_actor_ids_by_rank "
                    "unless managed ranks controller is configured.",
                    ValueError,
                )

        if target_actor_ids_by_rank is not None:
            if len(target_actor_ids_by_rank) != new_world_size:
                _raise_with_log(
                    "target_actor_ids_by_rank length must match new_world_size. "
                    f"Got {len(target_actor_ids_by_rank)} != {new_world_size}",
                    ValueError,
                )
            # MVP constraint: keep current rank 0 in the group so we always have
            # a stable "control rank" during reconfiguration.
            if self.actor_ids_by_rank and (
                self.actor_ids_by_rank[0]
                not in [normalize_actor_id(a) for a in target_actor_ids_by_rank]
            ):
                _raise_with_log(
                    "target_actor_ids_by_rank must include current rank 0 actor.",
                    ValueError,
                )
            self._pending_target_actor_ids_by_rank = list(target_actor_ids_by_rank)
        else:
            if new_world_size < self.world_size:
                # Shrink: keep the first new_world_size ranks.
                self._pending_target_actor_ids_by_rank = self.actor_ids_by_rank[
                    :new_world_size
                ]
            else:
                # Expand: membership will be completed during redistribution.
                self._pending_target_actor_ids_by_rank = None

        # Record the standard transition kind at planning time so commit can
        # update cached placement/layout state without re-inferring the shape.
        if reconfig_kind is not None:
            self._pending_reconfig_kind = str(reconfig_kind)
        elif target_actor_ids_by_rank is None:
            self._pending_reconfig_kind = (
                "append_only_expand"
                if new_world_size > self.world_size
                else "prefix_shrink"
            )
        else:
            normalized_target = [normalize_actor_id(a) for a in target_actor_ids_by_rank]
            if (
                new_world_size < self.world_size
                and normalized_target == self.actor_ids_by_rank[:new_world_size]
            ):
                self._pending_reconfig_kind = "prefix_shrink"
            else:
                _raise_with_log(
                    "Unsupported request_resize planning shape. Only append-only expand "
                    "and prefix shrink are allowed through request_resize(). "
                    "Use request_remove_nodes() for remove-node shrink."
                )

        self._pending_target_world_size = new_world_size
        # Fix next_epoch for this resize cycle to avoid races between callers.
        self._pending_next_epoch = self.epoch + 1
        self._pending_epoch_from = self.epoch
        # Only existing ranks participate in the reconfig path; on expand, new
        # ranks will join after the epoch is committed.
        # Reset per-resize state. This ensures keep-ranks will wait on a fresh
        # `_states_ready` event for the upcoming redistribution.
        self._states_ready = asyncio.Event()
        self._next_epoch_states_by_rank = None
        self._prepared_epoch = None
        # Fresh commit gate for this resize cycle.
        self._commit_applied = asyncio.Event()

    def request_resize(
        self,
        new_world_size: int,
        target_actor_ids_by_rank: Optional[List[Any]] = None,
    ) -> None:
        self._plan_resize(new_world_size, target_actor_ids_by_rank)

    def request_remove_nodes(self, node_ids: List[str]) -> None:
        """Shrink membership by removing ranks placed on specific node ids.

        The surviving ranks keep their relative order, so current rank 0 stays
        rank 0 as long as its node is not requested for removal.
        """
        remove_node_ids = {str(n).strip() for n in (node_ids or []) if str(n).strip()}
        if not remove_node_ids:
            return

        current_actor_ids = list(self.actor_ids_by_rank)
        if not current_actor_ids:
            return

        self._ensure_actor_node_ids(current_actor_ids)

        # print("removeNodes: ensure actor ids", flush=True)

        
        remove_actor_ids = {
            actor_id
            for actor_id in current_actor_ids
            if str(self._actor_node_ids.get(normalize_actor_id(actor_id), "") or "")
            in remove_node_ids
        }

        # print("remove_nodes: got actor ids", flush=True)

        rank0_actor_id = str(current_actor_ids[0])
        if rank0_actor_id in remove_actor_ids:
            print(
                "request_remove_nodes cannot remove the node containing current rank 0. "
                f"rank0_actor_id={rank0_actor_id}", flush=True
            )
            return

        survivors = [
            actor_id
            for actor_id in current_actor_ids
            if str(actor_id) not in remove_actor_ids
        ]
        if len(survivors) == len(current_actor_ids):
            return

        self._plan_resize(
            int(len(survivors)),
            target_actor_ids_by_rank=list(survivors),
            reconfig_kind="remove_node_shrink",
        )
        self._pending_removed_node_ids = sorted(
            {
                str(self._actor_node_ids.get(normalize_actor_id(actor_id), "") or "")
                for actor_id in remove_actor_ids
                if str(self._actor_node_ids.get(normalize_actor_id(actor_id), "") or "")
            }
        )

    def _get_alive_node_cpu_capacity(self) -> Dict[str, int]:
        """Return a snapshot of alive Ray nodes and their integer CPU capacity."""
        from ray.util.state import list_nodes as ray_list_nodes

        capacities: Dict[str, int] = {}
        nodes = ray_list_nodes(
            filters=[("state", "=", "ALIVE")],
            detail=False,
            raise_on_missing_output=False,
        )
        for node_state in nodes:
            node_id = str(getattr(node_state, "node_id", "") or "")
            if not node_id:
                continue
            resources_total = getattr(node_state, "resources_total", {}) or {}
            cpu_total = int(
                max(0, float(resources_total.get("CPU", 0.0) or 0.0))
            )
            if cpu_total > 0:
                capacities[node_id] = cpu_total
        return capacities

    def _ensure_actor_node_ids(self, actor_ids: List[str]) -> None:
        """Populate cached actor->node placement for any ids not seen before."""
        from ray.util.state import list_actors as ray_list_actors

        missing_ids = [actor_id for actor_id in actor_ids if actor_id not in self._actor_node_ids]
        if not missing_ids:
            return

        missing_id_set = set(missing_ids) # for better search performance
        actor_states = ray_list_actors(
            filters=[("state", "=", "ALIVE")],
            detail=False,
        )
        for actor_state in actor_states:
            actor_id = str(getattr(actor_state, "actor_id", "") or "")
            node_id = str(getattr(actor_state, "node_id", "") or "")
            if actor_id in missing_id_set and node_id:
                self._actor_node_ids[actor_id] = node_id
        missing = [aid for aid in missing_ids if aid not in self._actor_node_ids]
        if missing:
            _raise_with_log(
                "Missing Ray node ids for current rank actors: "
                + ", ".join(str(a) for a in missing)
            )

    def _ensure_cached_node_layout(self) -> None:
        """Seed the cached packed node layout once from current membership/state.

        After startup, only standard membership transitions are allowed, so the
        coordinator maintains this cache incrementally. This helper is only for
        the initial step.
        """
        if self._ordered_nodes:
            return
        
        current_actor_ids = [normalize_actor_id(a) for a in self.actor_ids_by_rank]
        if not current_actor_ids:
            _raise_with_log("Cannot inspect node layout without current actors.")

        self._ensure_actor_node_ids(current_actor_ids)
        node_rank_counts: Dict[str, int] = {}
        ordered_nodes: List[str] = []
        seen_nodes: set[str] = set()
        last_node_id: Optional[str] = None
        for actor_id in current_actor_ids:
            node_id = str(self._actor_node_ids.get(actor_id, "") or "")
            if not node_id:
                _raise_with_log(f"Actor {actor_id} has no Ray node assignment.")
            if node_id != last_node_id:
                if node_id in seen_nodes:
                    _raise_with_log(
                        "Packed placement invariant broken at startup: node "
                        f"{node_id} appears in multiple non-contiguous rank blocks."
                    )
                ordered_nodes.append(node_id)
                node_rank_counts[node_id] = 0
                seen_nodes.add(node_id)
                last_node_id = node_id
            node_rank_counts[node_id] += 1
        self._ordered_nodes = ordered_nodes
        self._node_rank_counts = node_rank_counts

    def _apply_append_only_expand_to_layout(self, new_rank_node_ids: List[str]) -> None:
        """Incrementally append new ranks to the cached packed node layout."""
        self._ensure_cached_node_layout()
        for node_id in new_rank_node_ids:
            node_id = str(node_id)
            if not self._ordered_nodes or node_id != self._ordered_nodes[-1]:
                if node_id in self._node_rank_counts:
                    _raise_with_log(
                        "Append-only expansion would reopen a non-tail node block "
                        f"for node {node_id}."
                    )
                self._ordered_nodes.append(node_id)
                self._node_rank_counts[node_id] = 0
            self._node_rank_counts[node_id] += 1

    def _apply_prefix_shrink_to_layout(self, target_world_size: int) -> None:
        """Incrementally shrink cached layout when keeping a simple prefix."""
        self._ensure_cached_node_layout()
        remaining = int(target_world_size)
        new_ordered_nodes: List[str] = []
        new_counts: Dict[str, int] = {}
        for node_id in self._ordered_nodes:
            if remaining <= 0:
                break
            keep = min(int(self._node_rank_counts[node_id]), remaining)
            if keep > 0:
                new_ordered_nodes.append(node_id)
                new_counts[node_id] = int(keep)
                remaining -= int(keep)
        self._ordered_nodes = new_ordered_nodes
        self._node_rank_counts = new_counts

    def _apply_remove_nodes_to_layout(self, removed_node_ids: List[str]) -> None:
        """Drop whole node blocks after a remove-node shrink request."""
        self._ensure_cached_node_layout()
        removed = {str(node_id) for node_id in (removed_node_ids or []) if str(node_id)}
        if not removed:
            return
        self._ordered_nodes = [
            node_id for node_id in self._ordered_nodes if node_id not in removed
        ]
        self._node_rank_counts = {
            node_id: count
            for node_id, count in self._node_rank_counts.items()
            if node_id not in removed
        }

    def _plan_expand_new_rank_nodes(self, target_world_size: int) -> List[str]:
        """Plan only the placement of new ranks for an append-only expansion.

        Existing logical ranks stay fixed. New ranks first fill any spare
        capacity on the current tail node; once that node is full, additional
        ranks spill onto entirely new nodes.

        Return a list of node ids, with length equal to the number of new ranks to create. Each entry is the node id to create a new rank on.
        """
        self._ensure_cached_node_layout()
        ordered_nodes = self._ordered_nodes
        node_rank_counts = self._node_rank_counts
        node_capacity = self._get_alive_node_cpu_capacity()
        for node_id, count in node_rank_counts.items():
            node_capacity[node_id] = max(int(node_capacity.get(node_id, 0)), int(count))
        
        remaining = int(target_world_size) - int(self.world_size)
        if remaining <= 0:
            return []
        create_node_ids: List[str] = []
        
        # Fill the tail node first
        tail_node_id = ordered_nodes[-1]
        tail_capacity = int(node_capacity.get(tail_node_id, 0))
        tail_used = int(node_rank_counts[tail_node_id])
        tail_free = max(0, tail_capacity - tail_used)
        if tail_free > 0:
            add = min(remaining, tail_free)
            create_node_ids.extend([tail_node_id] * add)
            remaining -= add

        # Then fill the extra nodes
        extra_nodes = [node_id for node_id in node_capacity if node_id not in node_rank_counts]
        extra_nodes.sort(key=lambda node_id: (-int(node_capacity[node_id]), node_id)) # sort new nodes by capacity, preserving for future use case
        for node_id in extra_nodes:
            if remaining <= 0:
                break
            capacity = int(node_capacity.get(node_id, 0))
            if capacity <= 0:
                continue
            add = min(remaining, capacity)
            create_node_ids.extend([node_id] * add)
            remaining -= add

        if remaining > 0:
            print(
                f"Not enough Ray CPU capacity to place {target_world_size} ranks. "
                f"Unplaced ranks={remaining}.", 
                flush=True
            )
        return create_node_ids

    def get_current_epoch(self) -> int:
        return self.epoch

    def get_current_membership(self) -> Tuple[int, List[Any]]:
        return self.world_size, list(self.actor_ids_by_rank)

    def get_current_world_size(self) -> int:
        return self.world_size

    def get_rank_for_actor(self, actor_id: Any) -> int:
        actor_id = normalize_actor_id(actor_id)
        try:
            return self.actor_ids_by_rank.index(actor_id)
        except ValueError:
            return -1

    async def reconfigure_test(self, rank: int, epoch: int) -> bool:
        """Check whether a resize/checkpoint should run at this safe point.

        `epoch` is the caller's current epoch. This prevents newly-added ranks
        (after an expand) from blocking on the old-epoch safe-point barrier.
        """
        if epoch != self.epoch:
            return False

        should_resize = (
            self._pending_target_world_size is not None
            and self._pending_epoch_from == self.epoch
        )

        # _should_checkpoint is a latched flag and becomes effective at this
        # iteration only if it was set in a previous safe point (or externally).
        should_ckpt = self._should_checkpoint
        should = should_resize or should_ckpt

        # Counter/threshold updates happen after deciding `should` so a newly
        # triggered threshold checkpoint takes effect on the next iteration.
        # If checkpoint is already pending, skip threshold checks entirely.
        if not should and rank == 0 and self.checkpoint_threshold > 0:
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= self.checkpoint_threshold:
                # Start a new checkpoint-only round if not already in one.
                if not self._should_checkpoint:
                    self._checkpoint_round_id += 1
                    self._checkpoint_round_done = asyncio.Event()
                self._should_checkpoint = True

        # Optional tuner profiling: compute durations inside coordinator (rank0 only).
        if self._tuner_profile_enabled and int(rank) == 0:
            now_s = float(time.time())
            if self._tuner_last_iter_ts_s is not None:
                dt = now_s - float(self._tuner_last_iter_ts_s)
                if dt >= 0:
                    self._tuner_iter_durations_s.append(float(dt))
            
            if should:
                # Trigger safe-point: include this iteration above, then start
                # reconfig timing from the same safe-point timestamp.
                self._tuner_reconfig_inflight = True
                self._tuner_reconfig_start_ts_s = now_s
            else:
                # Normal steady-state safe point: set baseline for next iteration duration.
                self._tuner_last_iter_ts_s = now_s

            # Clean up
            if self._tuner_profile_maxlen > 0:
                # Keep amortized O(1): drop oldest half when over limit.
                if len(self._tuner_iter_durations_s) > self._tuner_profile_maxlen:
                    drop = len(self._tuner_iter_durations_s) // 2
                    self._tuner_iter_durations_s = self._tuner_iter_durations_s[drop:]
                    self._tuner_last_get_iter_idx = max(
                        0, self._tuner_last_get_iter_idx - drop
                    )
                if len(self._tuner_reconfig_durations_s) > self._tuner_profile_maxlen:
                    drop = len(self._tuner_reconfig_durations_s) // 2
                    self._tuner_reconfig_durations_s = self._tuner_reconfig_durations_s[
                        drop:
                    ]
                    self._tuner_last_get_reconfig_idx = max(
                        0, self._tuner_last_get_reconfig_idx - drop
                    )

        return should

    async def wait_until_member(
        self, actor_id: Any, timeout_s: float = 10.0
    ) -> Tuple[int, int, int]:
        """Block until actor becomes a member; return (epoch, world_size, rank)."""
        actor_id = normalize_actor_id(actor_id)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + float(timeout_s)
        while True:
            r = self.get_rank_for_actor(actor_id)
            if r >= 0:
                return self.epoch, self.world_size, r
            if loop.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for actor {actor_id} to join group {self.base_group_name}."
                )
            await asyncio.sleep(0.05)

    def get_reconfig_target(self) -> Optional[Tuple[int, int, List[Any]]]:
        if self._pending_target_world_size is None:
            return None
        # For managed expansion, membership may be populated later during
        # redistribute_next_epoch(). Until then, return an empty list.
        return (
            int(self._pending_next_epoch or (self.epoch + 1)),
            self._pending_target_world_size,
            list(self._pending_target_actor_ids_by_rank or []),
        )

    def reset_checkpoint_counter(self) -> None:
        self._checkpoint_counter = 0
        self._should_checkpoint = False

    def request_checkpoint(self) -> int:
        """Request a checkpoint-only round at the next safe point.

        Returns the checkpoint round id for observability.
        """
        if not self._should_checkpoint:
            self._checkpoint_round_id += 1
            self._checkpoint_round_done = asyncio.Event()
        self._should_checkpoint = True
        return self._checkpoint_round_id

    async def save_checkpoint_round(self, round_id: int) -> None:
        """Finalize a checkpoint-only round (no resize).

        Called once by rank 0 after all ranks have submitted checkpoints and
        synchronized at a safe point.
        """
        if round_id != self._checkpoint_round_id:
            # Stale call; ignore.
            return
        # Persist this round as the latest checkpoint snapshot.
        for r, ckpt in self._ckpt_by_rank.items():
            self._latest_checkpoint_by_rank[r] = ckpt
        # Clear staging buffer now that the snapshot is committed.
        self._ckpt_by_rank = {}
        self.reset_checkpoint_counter()
        self._checkpoint_round_done.set()

    async def wait_checkpoint_round_done(self, round_id: int) -> None:
        if round_id != self._checkpoint_round_id:
            return
        await self._checkpoint_round_done.wait()

    def get_latest_checkpoints(self) -> Tuple[int, int, List[Any]]:
        """Return (epoch, world_size, checkpoints_by_rank)."""
        ckpts = [self._latest_checkpoint_by_rank.get(i) for i in range(self.world_size)]
        return self.epoch, self.world_size, ckpts

    async def redistribute_next_epoch_with_handler(
        self, fn: Optional[Callable[[List[Any], int, int], List[Any]]]
    ) -> Optional[Tuple[int, int, List[Any]]]:
        """Set redistribute handler and run redistribution in one RPC."""
        self._redistribute_handler = fn
        return await self.redistribute_next_epoch()

    def submit_checkpoint(self, rank: int, checkpoint: Any) -> None:
        if rank < 0 or rank >= self.world_size:
            _raise_with_log(
                f"submit_checkpoint received invalid rank={rank} for world_size={self.world_size}",
                ValueError,
            )
        self._ckpt_by_rank[rank] = checkpoint

    async def submit_checkpoint_and_get_action(
        self, rank: int, epoch: int, checkpoint: Any
    ) -> Tuple[str, Optional[Tuple[int, int, List[Any]]]]:
        """Submit a checkpoint and return the reconfig action in one RPC.

        This collapses the per-rank RPC sequence:
          submit_checkpoint -> checkpoint barrier -> get_reconfig_target
        and also handles checkpoint-only rounds internally.

        Returns:
            ("resize", target_tuple) or ("checkpoint_only", None)
        """
        if epoch != self.epoch:
            # Stale caller (e.g., newly-added ranks after expand). Treat as no-op.
            return "checkpoint_only", None

        self.submit_checkpoint(int(rank), checkpoint)
        await self._ckpt_barrier.wait(self.world_size)

        target = self.get_reconfig_target()
        if target is not None:
            return "resize", target

        # Checkpoint-only round (threshold reached), no resize.
        round_id = int(self._checkpoint_round_id)
        if int(rank) == 0:
            await self.save_checkpoint_round(round_id)
        await self.wait_checkpoint_round_done(round_id)

        # Profiling: checkpoint-only is still a reconfig pause; close it out on rank0.
        if (
            self._tuner_profile_enabled
            and int(rank) == 0
            and self._tuner_reconfig_inflight
            and self._tuner_reconfig_start_ts_s is not None
        ):
            end_s = float(time.time())
            dt = end_s - float(self._tuner_reconfig_start_ts_s)
            if dt >= 0:
                self._tuner_reconfig_durations_s.append(float(dt))
            self._tuner_last_iter_ts_s = end_s
            self._tuner_reconfig_start_ts_s = None
            self._tuner_reconfig_inflight = False

        return "checkpoint_only", None

    async def redistribute_next_epoch(self) -> Optional[Tuple[int, int, List[Any]]]:
        """Run redistribution inside the coordinator. Called by only one of the ranks.

        The coordinator stores the resulting states in `_next_epoch_states_by_rank`
        and also treats them as the latest checkpoint for the new epoch.
        """
        target = self.get_reconfig_target()
        if target is None:
            return None

        next_epoch, new_world_size, target_actor_ids = target
        old_world_size = int(self.world_size)

        print(f"Changing from {old_world_size} ranks to {new_world_size} ranks", flush=True)

        async with self._redistribute_lock:

            if self._prepared_epoch == next_epoch:
                return target

            # Order checkpoints by old rank.
            missing = [i for i in range(self.world_size) if i not in self._ckpt_by_rank]
            if missing:
                _raise_with_log(
                    "redistribute_next_epoch called before all checkpoints are submitted. "
                    f"Missing ranks={missing}, present={sorted(self._ckpt_by_rank.keys())}"
                )
            checkpoints = [self._ckpt_by_rank[i] for i in range(self.world_size)]

            # print(f"Calling redistribute_handler", flush=True)
            redist_t0 = time.perf_counter() if self._reconfig_profile_enabled else 0.0
            
            if self._redistribute_handler is None:
                # Default behavior: keep first M checkpoints; pad by repeating last.
                states = list(checkpoints)[:new_world_size]
                if new_world_size > len(states):
                    pad = checkpoints[-1] if checkpoints else None
                    while len(states) < new_world_size:
                        states.append(pad)
            else:
                states = self._redistribute_handler(
                    list(checkpoints), len(checkpoints), new_world_size
                )

            if self._reconfig_profile_enabled:
                redist_dt = float(time.perf_counter() - redist_t0)
                self._reconfig_record[int(next_epoch)] = (
                    int(old_world_size),
                    int(new_world_size),
                )
                self._reconfig_data_time_s[int(next_epoch)] = redist_dt
                # Non-expand resize has no new-rank creation on the controller path.
                self._reconfig_rank_time_s[int(next_epoch)] = 0.0
            # print(f"Redistribute_handler returned {len(states)} states", flush=True)

            if len(states) != new_world_size:
                _raise_with_log(
                    "redistribute_handler must return a list of length new_world_size."
                )

            # Managed expand: create new ranks after we have their assigned state.
            if new_world_size > old_world_size and not target_actor_ids:
                if not self.controller_name:
                    _raise_with_log(
                        "Expansion requested but controller_name is not configured."
                    )
                
                create_t0 = time.perf_counter() if self._reconfig_profile_enabled else 0.0
                # print(f"Getting controller", flush=True)
                ctrl = ray.get_actor(self.controller_name)
                # Ask the external lifecycle controller to create and start each new rank
                # with its assigned state. State is treated as an opaque object.
                # print(f"Creating new ranks", flush=True)
                # Expansion is append-only: keep the existing membership order
                # untouched and only create actors for the newly added ranks.
                target_actor_ids = [normalize_actor_id(a) for a in self.actor_ids_by_rank]
                create_ranks: List[int] = []
                create_state_ref_ls: List[List[object]] = []
                create_node_ids: List[str] = []
                new_rank_node_ids = self._plan_expand_new_rank_nodes(new_world_size)
                
                for rank_idx, node_id in zip(
                    range(old_world_size, new_world_size), new_rank_node_ids
                ):
                    state_ref = ray.put(states[rank_idx])
                    create_ranks.append(int(rank_idx))
                    create_state_ref_ls.append([state_ref])
                    create_node_ids.append(node_id)
                
                created_actor_ids = ray.get(
                    ctrl.create_and_run_ranks.remote(
                        create_ranks, create_state_ref_ls, create_node_ids
                    )
                )
                if len(created_actor_ids) != len(create_ranks):
                    _raise_with_log(
                        "Controller returned unexpected number of created actor ids. "
                        f"expected={len(create_ranks)} got={len(created_actor_ids)}"
                    )

                if self._reconfig_profile_enabled:
                    self._reconfig_rank_time_s[int(next_epoch)] = float(
                        time.perf_counter() - create_t0
                    )
                
                for actor_id, node_id in zip(created_actor_ids, create_node_ids):
                    norm_actor_id = normalize_actor_id(actor_id)
                    target_actor_ids.append(norm_actor_id)
                    # Record placement for newly created actors immediately so
                    # commit-time layout updates can derive node assignment
                    # without another state query.
                    self._actor_node_ids[norm_actor_id] = str(node_id)
                
                # print(f"New ranks created", flush=True)
                # Publish membership for commit phase.
                self._pending_target_actor_ids_by_rank = list(target_actor_ids)

            # Publish next-epoch states and treat them as the new checkpoint.
            self._next_epoch_states_by_rank = list(states)
            self._states_ready.set()

            # Keep the latest checkpoint snapshot aligned with new ranks.
            # (After commit, `world_size`/membership will update.)
            self._latest_checkpoint_by_rank = {i: s for i, s in enumerate(states)}

            # Mark prepared; do NOT clear _ckpt_by_rank here so it remains available
            # for debugging until finalize_resize() resets state.
            self._prepared_epoch = next_epoch

        # Return possibly-updated target tuple (with membership if expanded).
        return (next_epoch, new_world_size, list(target_actor_ids))

    async def commit_next_epoch(self, actor_id: Any) -> Tuple[bool, int, int, str, Any]:
        """Commit the new epoch (meta update done by new rank 0).

        Returns:
            (keep, new_rank, new_world_size, epoch_group_name)
        """
        actor_id = normalize_actor_id(actor_id)
        if self._pending_target_world_size is None:
            # No resize; keep current.
            return (
                True,
                self.get_rank_for_actor(actor_id),
                self.world_size,
                self._epoch_name(self.epoch),
                None,
            )

        next_epoch = int(self._pending_next_epoch or (self.epoch + 1))
        target_world_size = int(self._pending_target_world_size)
        target_actor_ids = [
            normalize_actor_id(a)
            for a in (self._pending_target_actor_ids_by_rank or [])
        ]
        if target_world_size > self.world_size and not target_actor_ids:
            # Managed expansion populates membership during redistribution.
            # Block until redistribution finishes, then re-read membership.
            await self._states_ready.wait()
            target_actor_ids = [
                normalize_actor_id(a)
                for a in (self._pending_target_actor_ids_by_rank or [])
            ]
            if not target_actor_ids:
                _raise_with_log(
                    "Expansion commit attempted before membership was populated."
                )

        # Determine membership for this actor.
        try:
            new_rank = target_actor_ids.index(actor_id)
            keep = True
        except ValueError:
            new_rank = -1
            keep = False

        # Apply new membership/epoch exactly once by the new rank 0 actor,
        # then release everyone (including removed ranks).
        new_rank0 = target_actor_ids[0]
        if actor_id == new_rank0:
            old_world_size = self.world_size
            self.epoch = next_epoch
            self.actor_ids_by_rank = target_actor_ids
            self.world_size = target_world_size
            # Update the cached packed layout using the transition kind that was
            # chosen at planning time.
            if self._pending_reconfig_kind == "append_only_expand":
                new_rank_node_ids = [
                    str(self._actor_node_ids.get(actor_id, "") or "")
                    for actor_id in target_actor_ids[old_world_size:]
                ]
                if any(not node_id for node_id in new_rank_node_ids):
                    _raise_with_log(
                        "Missing cached node placement for newly created ranks during commit."
                    )
                self._apply_append_only_expand_to_layout(new_rank_node_ids)
            elif self._pending_reconfig_kind == "prefix_shrink":
                self._apply_prefix_shrink_to_layout(target_world_size)
            elif self._pending_reconfig_kind == "remove_node_shrink":
                removed_node_ids = list(self._pending_removed_node_ids or [])
                self._apply_remove_nodes_to_layout(removed_node_ids)
            else:
                _raise_with_log(
                    "Unsupported or missing planned reconfiguration kind during commit."
                )
            # Drop placement cache entries for removed actors. Newly created
            # actor ids are inserted when expansion completes above.
            self._actor_node_ids = {
                aid: self._actor_node_ids[aid]
                for aid in target_actor_ids
                if aid in self._actor_node_ids
            }
            self._commit_applied.set()

        await self._commit_applied.wait()

        assigned_state = None
        if keep:
            # Block until redistributed states have been published.
            await self._states_ready.wait()
            assert self._next_epoch_states_by_rank is not None
            assigned_state = self._next_epoch_states_by_rank[new_rank]

        return keep, new_rank, target_world_size, self._epoch_name(next_epoch), assigned_state

    async def finalize_resize(self, actor_id: Any) -> None:
        """Finalize a resize after post-restore sync.

        Called by new rank 0 only (after all kept ranks pass the post-restore
        communicator barrier in reconfig_barrier()).
        """

        # Note: removed ranks self-terminate via `ray.actor.exit_actor()` in the
        # reconfig_barrier path after commit, so we must not double-delete here.
        # Clear pending request and per-resize state.
        self._pending_target_world_size = None
        self._pending_target_actor_ids_by_rank = None
        self._pending_next_epoch = None
        self._pending_epoch_from = None
        self._pending_reconfig_kind = None
        self._pending_removed_node_ids = None
        self._states_ready = asyncio.Event()
        self._next_epoch_states_by_rank = None
        self._prepared_epoch = None
        self._ckpt_by_rank = {}
        self.reset_checkpoint_counter()

        # Profiling: close out resize reconfig pause on new rank 0 (same actor by design).
        if (
            self._tuner_profile_enabled
            and self._tuner_reconfig_inflight
            and self._tuner_reconfig_start_ts_s is not None
        ):
            end_s = float(time.time())
            dt = end_s - float(self._tuner_reconfig_start_ts_s)
            if dt >= 0:
                self._tuner_reconfig_durations_s.append(float(dt))
            self._tuner_last_iter_ts_s = end_s
            self._tuner_reconfig_start_ts_s = None
            self._tuner_reconfig_inflight = False
        
        return

    def _epoch_name(self, epoch: int) -> str:
        return f"{self.base_group_name}#e{epoch}"


def coordinator_actor_name_for_group(group_name: str) -> str:
    return f"malleable_coordinator_{group_name}"


def _resolve_coordinator_for_group(group_name: str):
    """Resolve the MalleableCoordinator actor for `group_name`.

    Coordinator name is tightly coupled with group name:
    `malleable_coordinator_<group_name>`.
    """
    coordinator_name = coordinator_actor_name_for_group(group_name)
    try:
        return ray.get_actor(coordinator_name)
    except ValueError as e:
        _raise_with_log(
            "Failed to resolve malleable coordinator for group "
            f"'{group_name}'. Expected a live detached actor named "
            f"'{coordinator_name}'.",
            RuntimeError,
            cause=e,
        )


def request_resize(
    group_name: str,
    new_world_size: int,
    target_actor_ids_by_rank: Optional[List[Any]] = None,
) -> None:
    coord = _resolve_coordinator_for_group(group_name)
    ray.get(
        coord.request_resize.remote(
            new_world_size, target_actor_ids_by_rank=target_actor_ids_by_rank
        )
    )


def request_checkpoint(group_name: str) -> int:
    coord = _resolve_coordinator_for_group(group_name)
    return ray.get(coord.request_checkpoint.remote())


class MalleableGroupProxy:
    """Proxy implementing the BaseGroup interface, backed by epoch Gloo groups."""

    def __init__(
        self,
        base_group_name: str,
        coordinator,
        backend: str,
        gloo_timeout: int,
    ):
        self._base_group_name = base_group_name
        self._coord = coordinator
        self._backend = types.Backend(backend)
        self._gloo_timeout = gloo_timeout
        try:
            self._reconfig_profile_enabled = bool(
                ray.get(self._coord.get_reconfig_profile_enabled.remote())
            )
        except Exception:
            self._reconfig_profile_enabled = False

        self._epoch: Optional[int] = None
        self._epoch_group_name: Optional[str] = None
        self._joined_epoch: Optional[int] = None
        # Cached local epoch group handle for fast-path collectives.
        self._local_group = None
        # Cached membership info (no per-op coordinator calls once ready).
        self._cached_rank: Optional[int] = None
        self._cached_world_size: Optional[int] = None

    @property
    def rank(self) -> int:
        # Fast path: avoid coordinator RPCs on hot collectives (e.g. send/recv
        # rank-validity checks). Membership only changes at `reconfig_barrier()`.
        if self._cached_rank is not None:
            return int(self._cached_rank)
        # User code must not observe an undefined rank; block until admitted.
        self._ensure_epoch_group()
        assert self._cached_rank is not None
        return int(self._cached_rank)

    @property
    def world_size(self) -> int:
        # Fast path: avoid coordinator RPCs on hot collectives.
        if self._cached_world_size is not None:
            return int(self._cached_world_size)
        # User code must not observe an undefined world_size; block until admitted.
        self._ensure_epoch_group()
        assert self._cached_world_size is not None
        return int(self._cached_world_size)

    @property
    def group_name(self) -> str:
        return self._base_group_name

    def destroy_group(self) -> None:
        # Nothing to do here; epoch groups are destroyed on swap.
        return

    @classmethod
    def backend(cls):
        return types.Backend.GLOO

    # ---- Internal helpers ----
    def _ensure_epoch_group(self) -> Tuple[int, str, bool]:
        # Hot-path: once we've initialized/joined the current epoch group, do not
        # ping the coordinator again for every collective. Epoch/membership only
        # changes at the explicit safe point `reconfig_barrier()`.
        if (
            self._local_group is not None
            and self._epoch is not None
            and self._epoch_group_name is not None
            and self._cached_rank is not None
            and self._cached_world_size is not None
        ):
            return int(self._epoch), str(self._epoch_group_name), False

        my_actor_id = _get_my_actor_id()
        # For expansion, new ranks may start before they are admitted. Block
        # until the coordinator assigns a rank.
        my_rank = ray.get(self._coord.get_rank_for_actor.remote(my_actor_id))
        world_size: Optional[int] = None
        if my_rank < 0:
            epoch, world_size, my_rank = ray.get(
                self._coord.wait_until_member.remote(my_actor_id)
            )
        else:
            epoch = ray.get(self._coord.get_current_epoch.remote())

        epoch_group_name = f"{self._base_group_name}#e{epoch}"
        joined_now = False
        if self._epoch_group_name != epoch_group_name:
            self._epoch = epoch
            self._epoch_group_name = epoch_group_name
            self._local_group = None
            # Lazy-init happens when op is first called.
            if not _is_group_initialized(epoch_group_name):
                if world_size is None:
                    world_size = ray.get(self._coord.get_current_world_size.remote())
                _init_epoch_group(
                    epoch_group_name,
                    rank=int(my_rank),
                    world_size=int(world_size),
                    backend=self._backend,
                    gloo_timeout=self._gloo_timeout,
                )
            # Ensure all ranks have joined this epoch group before any other
            # collectives run (especially important for expansion).
            if self._joined_epoch != epoch:
                g = _get_group_handle_local(epoch_group_name)
                if g is None:
                    _raise_with_log(
                        f"Epoch group '{epoch_group_name}' was not initialized locally."
                    )
                g.barrier(types.BarrierOptions()) # This barrier also releases the old existing ranks from waiting for new ranks to join
                self._joined_epoch = epoch
                self._local_group = g
                joined_now = True

            # State recovery is a part of the rank creation
            # If we just joined a new epoch (e.g., via expand), restore from the
            # latest coordinator-published checkpoint for our rank.
            # if self._restored_epoch != epoch and epoch > 0:
            #     restore_fn = _get_restore_handler(self._base_group_name)
            #     if restore_fn is not None:
            #         _, _, ckpts = ray.get(self._coord.get_latest_checkpoints.remote())
            #         if 0 <= my_rank < len(ckpts):
            #             restore_fn(ckpts[my_rank])
            #             self._restored_epoch = epoch
                        
        # Cache membership info for cheap access by frequent collectives.
        if world_size is None:
            world_size = ray.get(self._coord.get_current_world_size.remote())
        self._cached_rank = int(my_rank)
        self._cached_world_size = int(world_size)
        return epoch, epoch_group_name, joined_now

    def _get_epoch_handle(self):
        # Fast path: after first ensure/join, no coordinator calls.
        if self._local_group is not None:
            return self._local_group
        _, epoch_group_name, _ = self._ensure_epoch_group()
        g = _get_group_handle_local(epoch_group_name)
        if g is None:
            _raise_with_log(
                f"Epoch group '{epoch_group_name}' was not initialized locally."
            )
        self._local_group = g
        return g

    # ---- Collectives ----
    def allreduce(self, tensor, allreduce_options=types.AllReduceOptions()):
        g = self._get_epoch_handle()
        return g.allreduce(tensor, allreduce_options)

    def barrier(self, barrier_options=types.BarrierOptions()):
        g = self._get_epoch_handle()
        return g.barrier(barrier_options)

    def reconfig_barrier(self, barrier_options=types.BarrierOptions()):
        """Safe-point barrier that may trigger checkpoint/resize."""
        # Step 1: reach a consistent safe point like a traditional barrier.
        # If we just performed the epoch-join barrier inside `_ensure_epoch_group`,
        # don't issue a second barrier here.
        epoch, _, joined_now = self._ensure_epoch_group()
        g = self._get_epoch_handle()

        if not joined_now:
            g.barrier(barrier_options)
        else:
            return

        # Step 2: check whether to reconfigure at this safe point.
        my_rank = int(self.rank)
        should_reconfig = ray.get(self._coord.reconfigure_test.remote(my_rank, epoch))
        if not should_reconfig:
            return

        # Always checkpoint on a resize, and also on threshold-triggered checkpoint.
        # print(f"[rank{self.rank}] checkpointing", flush=True)
        ckpt_fn = _get_checkpoint_handler(self._base_group_name)
        checkpoint = ckpt_fn() if ckpt_fn else None
        # Collapse checkpoint submission + checkpoint barrier + target query
        # into a single coordinator RPC.
        # print(f"[rank{self.rank}] submitting checkpoint", flush=True)
        kind, target = ray.get(
            self._coord.submit_checkpoint_and_get_action.remote(my_rank, epoch, checkpoint)
        )
        if kind == "checkpoint_only":
            return
        if kind != "resize" or target is None:
            _raise_with_log(
                f"Unexpected safe-point action: kind={kind}, target={target}"
            )

        next_epoch, new_world_size, _ = target

        # Rank 0 computes redistributed states and stores them in the coordinator.
        if my_rank == 0:
            redist = _get_redistribute_handler(self._base_group_name)
            ray.get(self._coord.redistribute_next_epoch_with_handler.remote(redist))

        # Commit epoch and decide membership for this actor.
        keep, new_rank, new_world_size, new_epoch_group_name, assigned_state = ray.get(
            self._coord.commit_next_epoch.remote(_get_my_actor_id())
        )

        comm_t0 = (
            time.perf_counter()
            if (my_rank == 0 and self._reconfig_profile_enabled)
            else 0.0
        )
        # Tear down old epoch group.
        if self._epoch_group_name and _is_group_initialized(self._epoch_group_name):
            _destroy_collective_group(self._epoch_group_name)

        if not keep:
            ray.actor.exit_actor()
            return

        # Initialize new epoch group with updated rank/world_size.
        # Will wait until all ranks have joined the new epoch group, or until the timeout is reached.
        _init_collective_group(
            world_size=new_world_size,
            rank=new_rank,
            backend=self._backend,
            group_name=new_epoch_group_name,
            gloo_timeout=self._gloo_timeout,
        )
        self._epoch = next_epoch
        self._epoch_group_name = new_epoch_group_name
        # Epoch-join synchronization: ensure all ranks (including newly-added ones)
        # have initialized and joined the new epoch group before proceeding to any
        # other collectives. This avoids existing ranks running ahead on the new
        # communicator while new ranks are still starting.
        g2 = _get_group_handle_local(new_epoch_group_name)
        if g2 is None:
            _raise_with_log(
                f"Epoch group '{new_epoch_group_name}' was not initialized locally."
            )

        if my_rank == 0 and self._reconfig_profile_enabled:
            ray.get(
                self._coord.report_comm_rebuild_time.remote(
                    int(next_epoch), float(time.perf_counter() - comm_t0)
                )
            )

        self._joined_epoch = next_epoch
        self._local_group = g2
        self._cached_rank = int(new_rank)
        self._cached_world_size = int(new_world_size)
        # Restore state if provided.
        state = assigned_state
        restore_fn = _get_restore_handler(self._base_group_name)
        if restore_fn is not None:
            restore_fn(state)

        # Wait for all ranks, including the newly created ranks, to join the new epoch group
        g2.barrier(types.BarrierOptions())

        # Rank0-only finalize. Post-restore barrier above ensures all kept ranks
        # finished transition before coordinator cleanup.
        if my_rank == 0:
            ray.get(self._coord.finalize_resize.remote(_get_my_actor_id()))

    def reduce(self, tensor, reduce_options=types.ReduceOptions()):
        g = self._get_epoch_handle()
        return g.reduce(tensor, reduce_options)

    def allgather(self, tensor_list, tensor, allgather_options=types.AllGatherOptions()):
        g = self._get_epoch_handle()
        return g.allgather(tensor_list, tensor, allgather_options)

    def broadcast(self, tensor, broadcast_options=types.BroadcastOptions()):
        g = self._get_epoch_handle()
        return g.broadcast(tensor, broadcast_options)

    def reducescatter(
        self, tensor, tensor_list, reducescatter_options=types.ReduceScatterOptions()
    ):
        g = self._get_epoch_handle()
        return g.reducescatter(tensor, tensor_list, reducescatter_options)

    def send(self, tensor, send_options: types.SendOptions):
        g = self._get_epoch_handle()
        return g.send(tensor, send_options)

    def recv(self, tensor, recv_options: types.RecvOptions):
        g = self._get_epoch_handle()
        return g.recv(tensor, recv_options)


# ---- Late imports to avoid circular dependencies ----


def _get_my_actor_id():
    worker = ray._private.worker.global_worker
    return normalize_actor_id(worker.core_worker.get_actor_id())


def _init_collective_group(*, world_size, rank, backend, group_name, gloo_timeout):
    from . import collective as _col

    return _col.init_collective_group(
        world_size=world_size,
        rank=rank,
        backend=backend,
        group_name=group_name,
        gloo_timeout=gloo_timeout,
    )


def _destroy_collective_group(group_name: str):
    from . import collective as _col

    return _col.destroy_collective_group(group_name)


def _get_group_handle_local(group_name: str):
    from . import collective as _col

    return _col._get_group_handle_local(group_name)


def _is_group_initialized(group_name: str) -> bool:
    from . import collective as _col

    return _col.is_group_initialized(group_name)


def _init_epoch_group(
    epoch_group_name: str,
    *,
    rank: int,
    world_size: int,
    backend,
    gloo_timeout: int,
):
    if int(rank) < 0:
        _raise_with_log(
            f"Cannot init epoch group '{epoch_group_name}' with rank={rank}."
        )
    _init_collective_group(
        world_size=int(world_size),
        rank=int(rank),
        backend=backend,
        group_name=epoch_group_name,
        gloo_timeout=gloo_timeout,
    )

