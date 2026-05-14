import os
import time
import sys
import argparse
import json
import math
import copy
import asyncio

from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime

import ray
import numpy
import ray.util.collective as col
from ray.util.collective.malleable import normalize_actor_id
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes

import kernels

COMM_TIME, COMP_TIME, ITER_TIME = range(3)


def _raise_with_log(message: str, exc_type=RuntimeError) -> None:
    print(f"[stencil] {message}", flush=True)
    raise exc_type(message)


def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = int(r/fac1) 
            break
    return fac1, fac2

@ray.remote(num_cpus=1)
class StencilActor:
    # def __init__(self, world_size, MPIRuntime, n, m, iterations, warmup, output_filepath):  
    def __init__(self, rank: int, group_name: str, state_ref_l):
        if state_ref_l is None:
            print("You must provide init arguments for mg actor", flush=True)
            exit(-1)

        allowed_cpus = sorted(os.sched_getaffinity(0))
        cpu_id = allowed_cpus[rank % len(allowed_cpus)]
        os.sched_setaffinity(0, {cpu_id})

        self.state = ray.get(state_ref_l[0])

        self.group_name = group_name
        self.rank = int(rank)
        self.world_size = self.state["world_size"]
        self.x, self.y = factor(self.world_size)
        self.coords = self.get_cart_coords(rank)
        self.X = self.coords[0]
        self.Y = self.coords[1]

        self.n = self.state["n"]
        self.m = self.state["m"]

        self.iterations = self.state["iterations"]
        self.warmup = self.state["warmup"]
        self.output_filepath = self.state["output_filepath"]
        
        # Calculate grid dimensions
        # print(f"[Init] Actor created with world_size={world_size}, n={n}, m={m}, x = {self.x}, y = {self.y}")
        
        self.is_initialized = self.state["init"]
        self.current_iteration = self.state["citn"]
        self.nodetype = self.state["nodetype"]
        self.malleable = bool(self.state.get("malleable", True))

        if self.malleable:
            # Register malleability handlers (used by reconfig_barrier).
            col.register_checkpoint_handler(self.group_name, self.checkpoint)
            col.register_restore_handler(self.group_name, self.restore)
            col.register_redistribute_handler(self.group_name, self.reconfig_handler)

    def run_iteration(self):
        # Per-iteration timing split: comm/sync vs compute.
        comm_time_s = 0.0
        comp_time_s = 0.0

        t0 = time.perf_counter()
        # col.barrier(group_name=self.group_name) 
        # comm_time_s += time.perf_counter() - t0
        
        # Handle communication
        # start = time.perf_counter_ns()
        if self.Y % 2 == 0:
            if self.Y < self.y-1:
                # print(f"[Rank {self.rank}] Preparing to send top buffer to rank {self.top_nbr}")
                t0 = time.perf_counter()
                kernels.pack_top(self.T, self.top_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()  # Start timing this comm operation
                # ray.get(self.MPIRuntime.send.remote(self.top_buf_out, self.rank, self.top_nbr))
                t0 = time.perf_counter()
                self.send(self.top_buf_out, self.rank, self.top_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start  # Accumulate time
                # print(f"[Rank {self.rank}] Finished sending top buffer to {self.top_nbr}")
            if self.Y > 0:
                # print(f"[Rank {self.rank}] Preparing to send bottom buffer to rank {self.bot_nbr}")
                t0 = time.perf_counter()
                kernels.pack_bottom(self.T, self.bot_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.bot_buf_out, self.rank, self.bot_nbr))
                t0 = time.perf_counter()
                self.send(self.bot_buf_out, self.rank, self.bot_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                # print(f"[Rank {self.rank}] Finished sending bottom buffer to {self.bot_nbr}")
            
            if self.Y < self.y-1:
                # print(f"[Rank {self.rank}] Receiving top buffer from rank {self.top_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.top_buf_in = ray.get(self.MPIRuntime.recv.remote(self.top_nbr, self.rank))
                t0 = time.perf_counter()
                self.top_buf_in = self.recv(self.top_nbr, self.rank, self.top_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_top(self.T, self.top_buf_in)
                comm_time_s += time.perf_counter() - t0
            if self.Y > 0:
                # print(f"[Rank {self.rank}] Receiving bottom buffer from rank {self.bot_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.bot_buf_in = ray.get(self.MPIRuntime.recv.remote(self.bot_nbr, self.rank))
                t0 = time.perf_counter()
                self.bot_buf_in = self.recv(self.bot_nbr, self.rank, self.bot_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_bottom(self.T, self.bot_buf_in)
                comm_time_s += time.perf_counter() - t0
        else:
            if self.Y < self.y-1:
                # print(f"[Rank {self.rank}] Receiving top buffer from rank {self.top_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.top_buf_in = ray.get(self.MPIRuntime.recv.remote(self.top_nbr, self.rank))
                t0 = time.perf_counter()
                self.top_buf_in = self.recv(self.top_nbr, self.rank, self.top_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_top(self.T, self.top_buf_in)
                comm_time_s += time.perf_counter() - t0
            if self.Y > 0:
                # print(f"[Rank {self.rank}] Receiving bottom buffer from rank {self.bot_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.bot_buf_in = ray.get(self.MPIRuntime.recv.remote(self.bot_nbr, self.rank))
                t0 = time.perf_counter()
                self.bot_buf_in = self.recv(self.bot_nbr, self.rank, self.bot_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_bottom(self.T, self.bot_buf_in)
                comm_time_s += time.perf_counter() - t0
                
            if self.Y < self.y-1:
                # print(f"[Rank {self.rank}] Preparing to send top buffer to rank {self.top_nbr}")
                t0 = time.perf_counter()
                kernels.pack_top(self.T, self.top_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.top_buf_out, self.rank, self.top_nbr))
                t0 = time.perf_counter()
                self.send(self.top_buf_out, self.rank, self.top_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
            if self.Y > 0:
                # print(f"[Rank {self.rank}] Preparing to send bottom buffer to rank {self.bot_nbr}")
                t0 = time.perf_counter()
                kernels.pack_bottom(self.T, self.bot_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.bot_buf_out, self.rank, self.bot_nbr))
                t0 = time.perf_counter()
                self.send(self.bot_buf_out, self.rank, self.bot_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start

        if self.X % 2 == 0:
            if self.X < self.x-1:
                # print(f"[Rank {self.rank}] Preparing to send right buffer to rank {self.right_nbr}")
                t0 = time.perf_counter()
                kernels.pack_right(self.T, self.right_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.right_buf_out, self.rank, self.right_nbr))
                t0 = time.perf_counter()
                self.send(self.right_buf_out, self.rank, self.right_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                # print(f"[Rank {self.rank}] Finished sending right buffer to {self.right_nbr}")
            if self.X > 0:
                # print(f"[Rank {self.rank}] Preparing to send left buffer to rank {self.left_nbr}")
                t0 = time.perf_counter()
                kernels.pack_left(self.T, self.left_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.left_buf_out, self.rank, self.left_nbr))
                t0 = time.perf_counter()
                self.send(self.left_buf_out, self.rank, self.left_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                # print(f"[Rank {self.rank}] Finished sending left buffer to {self.left_nbr}")
                
            if self.X < self.x-1:
                # print(f"[Rank {self.rank}] Receiving right buffer from rank {self.right_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.right_buf_in = ray.get(self.MPIRuntime.recv.remote(self.right_nbr, self.rank))
                t0 = time.perf_counter()
                self.right_buf_in = self.recv(self.right_nbr, self.rank, self.right_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_right(self.T, self.right_buf_in)
                comm_time_s += time.perf_counter() - t0
            if self.X > 0:
                # print(f"[Rank {self.rank}] Receiving left buffer from rank {self.left_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.left_buf_in = ray.get(self.MPIRuntime.recv.remote(self.left_nbr, self.rank))
                t0 = time.perf_counter()
                self.left_buf_in = self.recv(self.left_nbr, self.rank, self.left_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_left(self.T, self.left_buf_in)
                comm_time_s += time.perf_counter() - t0
        else:
            if self.X < self.x-1:
                # print(f"[Rank {self.rank}] Receiving right buffer from rank {self.right_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.right_buf_in = ray.get(self.MPIRuntime.recv.remote(self.right_nbr, self.rank))
                t0 = time.perf_counter()
                self.right_buf_in = self.recv(self.right_nbr, self.rank, self.right_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_right(self.T, self.right_buf_in)
                comm_time_s += time.perf_counter() - t0
            if self.X > 0:
                # print(f"[Rank {self.rank}] Receiving left buffer from rank {self.left_nbr}")
                # comm_start = time.perf_counter_ns()
                # self.left_buf_in = ray.get(self.MPIRuntime.recv.remote(self.left_nbr, self.rank))
                t0 = time.perf_counter()
                self.left_buf_in = self.recv(self.left_nbr, self.rank, self.left_buf_in)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
                t0 = time.perf_counter()
                kernels.unpack_left(self.T, self.left_buf_in)
                comm_time_s += time.perf_counter() - t0
                
            if self.X < self.x-1:
                # print(f"[Rank {self.rank}] Preparing to send right buffer to rank {self.right_nbr}")
                t0 = time.perf_counter()
                kernels.pack_right(self.T, self.right_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.right_buf_out, self.rank, self.right_nbr))
                t0 = time.perf_counter()
                self.send(self.right_buf_out, self.rank, self.right_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
            if self.X > 0:
                # print(f"[Rank {self.rank}] Preparing to send left buffer to rank {self.left_nbr}")
                t0 = time.perf_counter()
                kernels.pack_left(self.T, self.left_buf_out)
                comm_time_s += time.perf_counter() - t0
                # comm_start = time.perf_counter_ns()
                # ray.get(self.MPIRuntime.send.remote(self.left_buf_out, self.rank, self.left_nbr))
                t0 = time.perf_counter()
                self.send(self.left_buf_out, self.rank, self.left_nbr)
                comm_time_s += time.perf_counter() - t0
                # comm_time += time.perf_counter_ns() - comm_start
        # end = time.perf_counter_ns()
        # print(end-start, flush=True)
        # print(f"[Rank {self.rank}] Communication phase completed", flush=True)
        # comm_start = time.perf_counter_ns()
        # Pure barrier after halo exchange.
        t0 = time.perf_counter()
        col.barrier(group_name=self.group_name)
        comm_time_s += time.perf_counter() - t0
        # print(
        #     f"[rank{self.rank}] after barrier-2 it={self.current_iteration}",
        #     flush=True,
        # )
        # comm_time += time.perf_counter_ns() - comm_start
        # Computation
        t0 = time.perf_counter()
        kernels.compute(self.newT, self.T)
        self.newT, self.T = self.T, self.newT
        kernels.enforce_BC(self.T)
        comp_time_s += time.perf_counter() - t0
        # end = time.perf_counter()
        # print(end-start)  
        
        # iter_end = time.perf_counter_ns()
        
        # self.timing_info[self.current_iteration][COMM_TIME] = comm_time
        # self.timing_info[self.current_iteration][ITER_TIME] = iter_end - iter_start
        # self.timing_info[self.current_iteration][COMP_TIME] = self.timing_info[self.current_iteration][ITER_TIME] - self.timing_info[self.current_iteration][COMM_TIME]

        return comm_time_s, comp_time_s
    
    
    def checkpoint(self):
        self.update_state()
        return dict(self.state)

    def restore(self, state):
        if state is None:
            return
        self.state = dict(state)
        # Redistribution assigns a full per-rank post-reconfiguration state,
        # including logical rank and world size, so restore can be purely
        # state-driven with no extra collective-runtime queries.
        self.update_actor_from_state()

    def send(self, data, src_rank: int, dest_rank: int):
        # src_rank is implicit (this actor); kept for compatibility with the old workload code.
        return col.send(data, dst_rank=int(dest_rank), group_name=self.group_name)
    
    def recv(self, src_rank: int, dest_rank: int, buf):
        # Receive into pre-allocated numpy buffer (in-place).
        col.recv(buf, src_rank=int(src_rank), group_name=self.group_name)
        return buf


    def update_state(self):
        self.state["rank"] = self.rank
        self.state["world_size"] = self.world_size
        self.state["n"] = self.n
        self.state["m"] = self.m
        self.state["iterations"] = self.iterations
        self.state["warmup"] = self.warmup
        self.state["output_filepath"] = self.output_filepath
        self.state["citn"] = self.current_iteration
        self.state["init"] = self.is_initialized
        self.state["nodetype"] = self.nodetype
        self.state["malleable"] = self.malleable

        self.state["width"] = self.width
        self.state["height"] = self.height
        self.state["T"] = self.T

    def _rebuild_rank_dependent_runtime(self):
        """Rebuild buffers and neighbor topology from current rank/state."""
        self.x, self.y = factor(self.world_size)
        self.coords = self.get_cart_coords(self.rank)
        self.X = self.coords[0]
        self.Y = self.coords[1]

        kernels.set_block_params(self.width, self.height)
        self.newT = numpy.zeros_like(self.T)
        self.top_buf_out = numpy.zeros(self.width)
        self.top_buf_in = numpy.zeros(self.width)
        self.bot_buf_out = numpy.zeros(self.width)
        self.bot_buf_in = numpy.zeros(self.width)
        self.right_buf_out = numpy.zeros(self.height)
        self.right_buf_in = numpy.zeros(self.height)
        self.left_buf_out = numpy.zeros(self.height)
        self.left_buf_in = numpy.zeros(self.height)

        # Clear old topology first so a boundary move after reconfiguration
        # cannot leave stale neighbor ids behind.
        self.top_nbr = None
        self.bot_nbr = None
        self.left_nbr = None
        self.right_nbr = None

        if self.Y < self.y - 1:
            self.top_nbr = self.get_cart_rank([self.X, self.Y + 1])
        if self.Y > 0:
            self.bot_nbr = self.get_cart_rank([self.X, self.Y - 1])
        if self.X > 0:
            self.left_nbr = self.get_cart_rank([self.X - 1, self.Y])
        if self.X < self.x - 1:
            self.right_nbr = self.get_cart_rank([self.X + 1, self.Y])

        kernels.enforce_BC(self.T)

    def update_actor_from_state(self):
        self.rank = int(self.state.get("rank", self.rank))
        self.world_size = self.state["world_size"]
        self.n = self.state["n"]
        self.m = self.state["m"]
        self.iterations = self.state["iterations"]
        self.warmup = self.state["warmup"]
        self.output_filepath = self.state["output_filepath"]
        self.current_iteration = self.state["citn"]
        self.is_initialized = self.state["init"]
        self.nodetype = self.state["nodetype"]
        self.malleable = bool(self.state.get("malleable", True))
        
        self.width = self.state["width"]
        self.height = self.state["height"]
        self.T = self.state["T"].copy()
        self._rebuild_rank_dependent_runtime()
        self.timing_info = numpy.zeros((self.warmup + self.iterations, 3), dtype=numpy.float64)

    def get_cart_coords(self, rank):
        return [rank // self.y, rank % self.y]
    
    def get_cart_rank(self, coords):
        return coords[0] * self.y + coords[1]
    
    def reconfig_handler(self, source_states, N, M):
        # `source_states` are checkpoint objects (not Ray object refs).
        #
        # IMPORTANT: rank->(X,Y) mapping must match get_cart_coords():
        #   X = rank // y, Y = rank % y  where (x,y)=factor(world_size)
        template_state = dict(source_states[0])

        old_x, old_y = factor(N)
        new_x, new_y = factor(M)

        # print(
        #     f"Reconfig handler: old_x {old_x}, old_y {old_y}, new_x {new_x}, new_y {new_y}",
        #     flush=True,
        # )

        old_block_width = int(template_state["width"])
        old_block_height = int(template_state["height"])

        # Reconstruct the global interior grid from old shards using the same
        # cartesian rank mapping as the runtime.
        #
        # NOTE: this stencil implementation assumes a uniform block size across ranks.
        # When the global domain is not divisible by (x,y), the original workload
        # effectively truncates via integer division. We preserve that behavior here:
        # global_h/global_w reflect what the old decomposition actually covered.
        global_h = int(old_x) * int(old_block_height)
        global_w = int(old_y) * int(old_block_width)
        global_data = numpy.zeros((global_h, global_w), dtype=numpy.float64)

        for r in range(int(N)):
            X = r // old_y
            Y = r % old_y

            source_state = dict(source_states[r])
            data_T = numpy.asarray(source_state["T"])
            data_T_2D = data_T.reshape(old_block_height + 2, old_block_width + 2)
            global_data[
                X * old_block_height : (X + 1) * old_block_height,
                Y * old_block_width : (Y + 1) * old_block_width,
            ] = data_T_2D[1:-1, 1:-1]

        # New uniform block sizes (same rule as setup(): integer division).
        new_block_height = int(global_h) // int(new_x)
        new_block_width = int(global_w) // int(new_y)
        if new_block_height <= 0 or new_block_width <= 0:
            raise ValueError(
                f"Invalid new block size ({new_block_height},{new_block_width}) for "
                f"global ({global_h},{global_w}) and grid ({new_x},{new_y})."
            )

        # If not divisible, we truncate the remainder to keep uniform block sizes
        # (required by the current halo exchange + kernels implementation).
        cropped_h = int(new_x) * int(new_block_height)
        cropped_w = int(new_y) * int(new_block_width)
        if cropped_h != global_h or cropped_w != global_w:
            print(
                f"[reconfig_handler] WARNING: truncating global interior from "
                f"({global_h},{global_w}) to ({cropped_h},{cropped_w}) for grid ({new_x},{new_y})",
                flush=True,
            )
            global_data = global_data[:cropped_h, :cropped_w]

        # Produce new per-rank states. Avoid deepcopy of large arrays; `T` is replaced.
        base_state = dict(template_state)
        return_val = [None] * int(M)
        for r in range(int(M)):
            X = r // new_y
            Y = r % new_y
            new_state = dict(base_state)
            new_state["rank"] = int(r)
            new_state["world_size"] = int(M)
            new_state["width"] = int(new_block_width)
            new_state["height"] = int(new_block_height)

            new_data = numpy.zeros(
                (new_block_height + 2, new_block_width + 2), dtype=numpy.float64
            )
            new_data[1:-1, 1:-1] = global_data[
                X * new_block_height : (X + 1) * new_block_height,
                Y * new_block_width : (Y + 1) * new_block_width,
            ]
            new_state["T"] = new_data.flatten()
            return_val[r] = new_state
        return return_val

    def setup(self):
        # print(f"[Rank {self.rank}] Setting up worker...")
        width = self.m // self.y
        height = self.n // self.x
        
        self.width = width
        self.height = height

        my_blocksize = (self.n//self.x)*(self.m//self.y)
        ghost_size = ((self.n//self.x) * 2) + ((self.m//self.y) * 2) + 4
        kernels.set_block_params(self.m//self.y, self.n//self.x)
        self.T = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)
        self._rebuild_rank_dependent_runtime()
        
        self.timing_info = numpy.zeros((self.warmup + self.iterations, 3), dtype=numpy.float64)
        # print(f"[Rank {self.rank}] Setup complete. Block size: {my_blocksize}, Ghost size: {ghost_size}")
        
    
    def main(self):
        if self.is_initialized == False:
            self.setup()
            self.is_initialized = True
        else:
            self.update_actor_from_state()

        if self.rank == 0:
            self.iteration_time = {}
            self.reconfig_time = {}
            self.comm_time = {}
            self.comp_time = {}
        
        while self.current_iteration < self.warmup + self.iterations:
            if self.rank == 0:
                print("Iter: ", self.current_iteration, flush=True)
                exe_start_time = time.time()
            
            if self.malleable:
                # Safe point at the beginning of the iteration.
                if self.rank == 0:
                    t0 = time.perf_counter()
                    col.reconfig_barrier(group_name=self.group_name)
                    self.reconfig_time[self.current_iteration] = time.perf_counter() - t0
                else:
                    col.reconfig_barrier(group_name=self.group_name)

            comm_s, comp_s = self.run_iteration()
            self.current_iteration += 1

            if self.rank == 0:
                exe_time = time.time() - exe_start_time
                self.iteration_time[self.current_iteration] = exe_time
                self.comm_time[self.current_iteration] = float(comm_s)
                self.comp_time[self.current_iteration] = float(comp_s)
            
            # Trigger resizes (rank0 only) when malleability is enabled.
            # if self.malleable and self.rank == 0:
            #     if self.current_iteration in (10, 20, 30):
            #         col.request_resize(self.group_name, new_world_size=self.world_size * 2)
            #     elif self.current_iteration in (40, 50, 60):
            #         col.request_resize(self.group_name, new_world_size=max(1, self.world_size / 2))

            # if self.malleable and self.rank == 0:
            #     if self.current_iteration in (20, 40, 60, 140, 160, 180, 260, 280, 300, 380, 400, 420):
            #         col.request_resize(self.group_name, new_world_size=int(self.world_size * 2))
            #     elif self.current_iteration in (80, 100, 120, 200, 220, 240, 320, 340, 360, 440, 460, 480):
            #         col.request_resize(self.group_name, new_world_size=max(1, int(self.world_size / 2)))



        if self.rank == 0:
            # Store timing info to file (no Ray-MPI profiling here).
            reconfig_record = {}
            reconfig_data_time = {}
            reconfig_rank_time = {}
            reconfig_comm_time = {}
            if self.malleable:
                try:
                    coord = ray.get_actor(f"malleable_coordinator_{self.group_name}")
                    (
                        reconfig_record,
                        reconfig_data_time,
                        reconfig_rank_time,
                        reconfig_comm_time,
                    ) = ray.get(coord.get_reconfig_record.remote())
                except Exception:
                    # Keep output generation robust even if coordinator profiling
                    # is disabled or unavailable.
                    print("Failed to get reconfig record", flush=True)
            with open("profile.json", "w") as f:
                json.dump(
                    {
                        "execution_time": self.iteration_time,
                        "comm_time": self.comm_time,
                        "comp_time": self.comp_time,
                        "reconfig_time": self.reconfig_time,
                        "reconfig_record": reconfig_record,
                        "reconfig_data_time": reconfig_data_time,
                        "reconfig_rank_time": reconfig_rank_time,
                        "reconfig_comm_time": reconfig_comm_time,
                    },
                    f,
                )

    def gather_timing_info(self):
        # Old Ray-MPI gather path removed. Keep as a no-op for compatibility.
        return

    def output_timing(self, data, n, m, np, iterations, warmup, nodetype):
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        job_id = os.environ.get('SLURM_JOB_ID', 'nojob')
        output_filename = f"ray_mpi_sync_results/{dt_string}_job{job_id}_n{n}_m{m}_np{np}_i{iterations}_w{warmup}_ray_ncpu{int(ray.cluster_resources()['CPU'])}_{nodetype}.csv"
        header = "Total Time,Communication Time,Computation Time"
    
        combined_timing = numpy.zeros((iterations, 3))
    
        for process_data in data:
            combined_timing += process_data[warmup:]

        final_avg = numpy.sum(combined_timing, axis=0)
        final_avg = final_avg / 1e9
        with open(output_filename, 'w') as output_file:
            output_file.write(f'#{" ".join(sys.argv)}\n')
            output_file.write(header + '\n')
            output_file.write(','.join(map(str, [
                final_avg[ITER_TIME],
                final_avg[COMM_TIME],
                final_avg[COMP_TIME]
            ])) + '\n')

    
    def get_timing_info(self):
        return self.timing_info

@ray.remote(num_cpus=0)
class Controller:
    """Controller that only manages rank lifecycles (same as send_recv.py)."""

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.ranks_handle: Dict[str, Any] = {}
        self._startup_actor_names: List[str] = []
        self._startup_open = True
        self.finished_count = 0
        self._actor_seq = 0

        self.ranks_condition = asyncio.Condition()

    def _allocate_actor_name(self) -> str:
        """Return a unique opaque detached Ray actor name for one physical actor."""
        self._actor_seq += 1
        return f"{self.group_name}-actor-{self._actor_seq}"

    def _create_and_run_rank_impl(
        self,
        rank: int,
        state_ref_l: List[object],
        node_id: Optional[str] = None,
    ) -> str:
        """Create one detached rank actor and schedule its main loop."""
        rank = int(rank)
        actor_name = self._allocate_actor_name()
        actor_options: Dict[str, Any] = {
            "name": actor_name,
            "lifetime": "detached",
            "get_if_exists": True,
        }
        if node_id:
            actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                str(node_id), soft=False
            )
        # The controller is responsible only for physical actor creation; the
        # coordinator decides which logical rank and which node this actor serves.
        rank_handle = StencilActor.options(**actor_options).remote(
            rank, self.group_name, state_ref_l
        )
        self.ranks_handle[actor_name] = rank_handle
        if self._startup_open:
            # Startup creates initial ranks in logical-rank order, so creation
            # order is sufficient for building the initial collective group.
            self._startup_actor_names.append(actor_name)
        asyncio.create_task(self._run_rank(actor_name))
        return normalize_actor_id(getattr(rank_handle, "_ray_actor_id"))

    async def create_and_run_rank(
        self,
        rank: int,
        state_ref_l: List[object],
        node_id: Optional[str] = None,
    ) -> str:
        return self._create_and_run_rank_impl(rank, state_ref_l, node_id)

    async def create_and_run_ranks(
        self,
        ranks: List[int],
        state_ref_ls: List[List[object]],
        node_ids: List[Optional[str]],
    ) -> List[str]:
        """Create a batch of new rank actors and return actor ids in input order."""
        if len(ranks) != len(state_ref_ls) or len(ranks) != len(node_ids):
            _raise_with_log(
                "create_and_run_ranks expects equal-length ranks/state_ref_ls/node_ids.",
                ValueError,
            )

        actor_ids: List[str] = [""] * len(ranks)
        for idx in range(len(ranks)):
            # Preserve input order so the returned actor ids line up with the
            # coordinator's logical-rank slots.
            actor_ids[idx] = self._create_and_run_rank_impl(
                int(ranks[idx]),
                state_ref_ls[idx],
                node_ids[idx],
            )
        return actor_ids

    async def get_startup_handles(self, expected_count: int) -> List[Any]:
        """Return startup actor handles in creation order and seal startup."""
        expected_count = int(expected_count)
        if len(self._startup_actor_names) != expected_count:
            _raise_with_log(
                "Startup actor handle count mismatch. "
                f"expected={expected_count} got={len(self._startup_actor_names)}",
                ValueError,
            )
        self._startup_open = False
        handles: List[Any] = []
        for actor_name in self._startup_actor_names:
            handle = self.ranks_handle.get(actor_name)
            if handle is None:
                _raise_with_log(
                    f"Missing startup actor handle for actor {actor_name}.",
                    ValueError,
                )
            handles.append(handle)
        return handles

    async def wait_all_ranks(self):
        async with self.ranks_condition:
            while True:
                if self.finished_count == len(self.ranks_handle):
                    print(f"All {self.finished_count} ranks finished", flush=True)
                    break
                await self.ranks_condition.wait()

    async def delete_all_ranks(self):
        for actor_name in list(self.ranks_handle.keys()):
            print(f"Delete actor {actor_name}", flush=True)
            ray.kill(self.ranks_handle[actor_name])
        self.ranks_handle = {}
        self._startup_actor_names = []
        self._startup_open = True

    async def _run_rank(self, actor_name: str):
        try:
            await self.ranks_handle[actor_name].main.remote()
            async with self.ranks_condition:
                self.finished_count += 1
                self.ranks_condition.notify_all()
        except Exception:
            self.ranks_handle.pop(actor_name, None)
            return


def _plan_startup_node_ids(world_size: int) -> List[Optional[str]]:
    """Pack initial ranks onto as few alive Ray nodes as possible."""
    alive_nodes = list_nodes(
        filters=[("state", "=", "ALIVE")],
        detail=False,
        raise_on_missing_output=False,
    )
    candidates: List[Tuple[str, bool, int]] = []
    for node_state in alive_nodes:
        node_id = str(getattr(node_state, "node_id", "") or "")
        if not node_id:
            continue
        resources_total = getattr(node_state, "resources_total", {}) or {}
        cpu_total = int(max(0, math.floor(float(resources_total.get("CPU", 0.0) or 0.0))))
        if cpu_total <= 0:
            continue
        is_head = bool(getattr(node_state, "is_head_node", False))
        candidates.append((node_id, is_head, cpu_total))

    if not candidates:
        return [None] * int(world_size)

    head_nodes = [info for info in candidates if info[1]]
    worker_nodes = [info for info in candidates if not info[1]]
    worker_nodes.sort(key=lambda info: (-int(info[2]), info[0]))
    ordered = head_nodes[:1] + worker_nodes
    if not ordered:
        ordered = sorted(candidates, key=lambda info: (-int(info[2]), info[0]))

    # Use as few nodes as possible at startup so the initial epoch already
    # satisfies the packed-placement invariant expected by later resizes.
    node_ids_by_rank: List[Optional[str]] = []
    remaining = int(world_size)
    for node_id, _is_head, cpu_total in ordered:
        if remaining <= 0:
            break
        assign = min(int(cpu_total), remaining)
        for _ in range(assign):
            node_ids_by_rank.append(node_id)
        remaining -= assign

    if remaining > 0:
        _raise_with_log(
            f"Not enough alive Ray CPU capacity for startup world_size={world_size}. "
            f"unplaced_ranks={remaining}"
        )
    return node_ids_by_rank

def main():
    parser = argparse.ArgumentParser(description='Python Ray-MPI Stencil execution on 2D grid')
    parser.add_argument('iterations', type=int, help='Number of iterations')
    parser.add_argument('n', type=int, help='Array dimension X')
    parser.add_argument('m', type=int, nargs='?', help='Array dimension Y (if not provided, uses n)')
    parser.add_argument('--nodetype', '-nt', type=str, choices=['c', 'i'], required=True, help='Node type (c or i)')
    parser.add_argument('--warmup', '-w', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument(
        "--malleable",
        action="store_true",
        help="Enable malleable Ray CCL (reconfig_barrier + resize). If unset, runs pure Ray CCL with zero reconfig checks.",
    )
    args = parser.parse_args()
    
    iterations = args.iterations
    n = args.n
    m = args.m if args.m is not None else n
    nodetype = args.nodetype
    
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    try:
        ray.init(address="auto", namespace="default")
    except Exception:
        ray.init(namespace="default")
    print("[driver] ray.init done", flush=True)

    world_size = 16 # TODO:
    x, y = factor(world_size)
    
    # print(f"X, y: {x} {y}", flush=True)
    print('Number of processors        = ', world_size)
    print('Grid shape (x,y)            = ', x, y)
    print('Problem Domain (x,y)        = ', n, m)
    print('Number of warmup iterations = ', args.warmup)
    print('Number of iterations        = ', iterations)
    
    # print("[Main] Starting Ray-MPI Stencil application...")

    # Fixed names so external tools can attach deterministically.
    group_name = "stencil"
    coordinator_name = "malleable_coordinator_stencil"
    print(
        f"[driver] group_name={group_name} coordinator_name={coordinator_name}",
        flush=True,
    )
    controller = None
    try:
        controller = Controller.options(
            name=f"controller_{group_name}", lifetime="detached", get_if_exists=True
        ).remote(group_name)

        warmup = args.warmup
        startup_node_ids = _plan_startup_node_ids(world_size)
        for i in range(world_size):
            init_state = {
                "world_size": world_size,
                "n": n,
                "m": m,
                "iterations": iterations,
                "warmup": warmup,
                "output_filepath": ".",
                "citn": 0,
                "init": False,
                "nodetype": nodetype,
                "malleable": bool(args.malleable),
            }
            state_ref = ray.put(init_state)
            ray.get(
                controller.create_and_run_rank.remote(
                    i,
                    [state_ref],
                    startup_node_ids[i],
                )
            )

        print("[driver] ranks created", flush=True)

        # Get startup handles from the controller rather than reconstructing
        # them from rank-encoded actor names.
        actors = ray.get(controller.get_startup_handles.remote(world_size))
        print("[driver] creating collective group", flush=True)
        if args.malleable:
            col.create_collective_group(
                actors=actors,
                world_size=world_size,
                ranks=list(range(world_size)),
                backend="gloo",
                group_name=group_name,
                gloo_timeout=120000,
                malleable_options={
                    "enabled": True,
                    "controller_name": f"controller_{group_name}",
                    "checkpoint_threshold": 0,
                    "coordinator_name": coordinator_name,
                    # "reconfig_profile_enabled": True, # TODO:
                    "tuner_profile_enabled": True,
                    "tuner_profile_maxlen": 20000,
                },
            )
        else:
            # Pure Ray CCL group: no coordinator, no proxy, no reconfig_barrier checks.
            col.create_collective_group(
                actors=actors,
                world_size=world_size,
                ranks=list(range(world_size)),
                backend="gloo",
                group_name=group_name,
                gloo_timeout=120000,
            )
        print("[driver] collective group created", flush=True)

        ray.get(controller.wait_all_ranks.remote())
    finally:
        # Best-effort cleanup even on Ctrl-C.
        try:
            if controller is not None:
                ray.get(controller.delete_all_ranks.remote())
        except Exception:
            pass
        try:
            if controller is not None:
                ray.kill(controller)
        except Exception:
            pass
        if args.malleable:
            try:
                ray.kill(ray.get_actor(coordinator_name))
            except Exception:
                pass
            try:
                ray.kill(ray.get_actor("info_" + group_name))
            except Exception:
                pass

if __name__ == '__main__':
    main()
