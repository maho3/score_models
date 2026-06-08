# Adapting Existing Score Models to New Features

Use this quick checklist to update an existing training script to support:
- DistributedDataParallel (DDP) training in `ScoreModel.fit`
- Periodic U-net padding (`padding_mode="circular"`) in `NCSNpp`

## 1) Update NCSNpp for periodic domains (3D only)
If your data is periodic (e.g. voxels/cosmology), build `NCSNpp` with:
- `dimensions=3`
- `padding_mode="circular"`
- `fir=False` (required with circular padding)

Example:
```python
net = NCSNpp(
    channels=1,
    dimensions=3,
    nf=128,
    ch_mult=(2, 2, 2, 2),
    attention=False,
    fir=False,
    padding_mode="circular",
)
```

## 2) Keep ScoreModel construction unchanged
Create `ScoreModel` as usual (VESDE or VPSDE settings unchanged):
```python
model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=50, device="cuda")
```

## 3) Enable DDP in training
Launch your script with `torchrun`:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py
```

Then call `fit(...)` either:
- with auto-detection (if process group already initialized by `torchrun`), or
- explicitly with `distributed=True`

```python
model.fit(dataset, epochs=100, batch_size=8, learning_rate=1e-4, distributed=True)
```

### SLURM (DeltaAI-style) recommendations
- Request one task per GPU (`--ntasks-per-node == --gpus-per-node`) and bind CPUs accordingly.
- Use `srun` to launch `torchrun` so ranks inherit SLURM networking/env settings.
- Set rendezvous parameters explicitly (`MASTER_ADDR`, `MASTER_PORT`, `--rdzv_backend=c10d`) for multi-node jobs.
- Write checkpoints/logs to fast shared storage (`$SCRATCH`/project storage), not local `/tmp`.

Minimal job-launch pattern:
```bash
srun torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --node_rank=${SLURM_NODEID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py
```

## 4) Common pitfalls
- `padding_mode="circular"` is only supported for `dimensions=3`.
- `padding_mode="circular"` is not supported with `fir=True`.
- `distributed=True` requires an initialized `torch.distributed` process group.

## 5) Quick validation
Run tests:
```bash
pip install -e .
pytest -q
```
