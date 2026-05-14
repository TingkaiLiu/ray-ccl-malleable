# Ray-CCL-Malleable

This repo includes the source code of enabling malleable MPI execution, extending Ray-CCL, with cloud elasticity manager, which guides the malleable MPI program to execute on spot instances in cost-effective way. 

For detail, refer to the paper "Enabling Cost-Effective MPI on Cloud Spot Instances via Malleable Execution using Ray"

Notice, when installing ray with pip, 'ray[default]', which includes the dashboard, is needed, as the coordinator rely on it to fetch live node information. Follow the instruction so basic GLOO communicator can be used: https://docs.ray.io/en/latest/ray-more-libs/ray-collective.html


## Layout
- `bundle/`: patched sources of malleable MPI runtime
- `scripts/`: installation scripts
- `examples/`: workloads and cloud elasticity manager
- `data/`: the raw experiment results presented in the paper

## Installing the malleable runtime

As the implementation is an extension to Ray-CCL. The installation uses scripts to copy source files under `bundle/` to correct locations into Ray source library.

From this folder:

If ray is installed under conda environment:
```bash
python3 scripts/sync_ray_collective_patch.py --env ray
```

If ray is installed under system library:
```bash
python3 scripts/sync_ray_local.py
```

This finds the `ray.util.collective` site-packages directory and copies patched files over.

## Starting a Ray cluster

As the first step, start a Ray cluster following Ray official document. Notice, the Ray head node should be started with customize resource "head_node", as the coordinator rely on it for placement. For example:

```bash
ray start --head --resources='{"head_node": 1}' --num-cpus 16
```

## Running the example workload

The example stencil workload is given under `examples/`, where:
- Baseline static MPI program: `examples/stencil-mpi-sync.py` with `examples/kernel.py`
- Malleable MPI adaption: `examples/stencil-ray-mpi-sync.py`

Specifically, when enabling the the malleable communicator, original Ray-CCL is used with an additional arguments "malleable_options"
```python
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
```

To execute the program, simply use
```bash
python3 stencil-ray-mpi-sync.py 100 3360 3360 -nt c -w 1 --malleable
```

## Cloud elasticity manager

The cloud elasticity manager includes the files:
- `examples/spot_common.py`
- `examples/spot_phase1.py`
- `examples/spot_phase2.py`

Refer to the paper for detail design. 

When using cloud elasticity manager to guide malleable MPI execution. First, launch the workload. Then, use another terminal to start the cloud elasticity manager, guiding the execution include two phases:

```bash
python3 phase1.py
```

This tries to find the suitable node count for a given workload, and produces a json file for the configuration. Then, 

```bash
python3 phase2.py
```

This consumes the json file produced by phase 1, and start phase 2 execution. The file includes the simulation of different spot instance interruption and recovery rate at each tick. 

## Experiment results

`data/` includes the raw experiment results presented in the paper, with plotting scripts. 


# Thank you!
