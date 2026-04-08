# Data directory

This repository does not ship generated checkpoints, benchmark outputs, or paper tables.
The public CIFAR-10 dataset can be downloaded into this directory with:

```bash
python scripts/download_cifar10.py
```

The download script stores CIFAR-10 under:

```text
data/raw/cifar10
```

These directories are expected to be created locally when running the pipeline:

- `data/raw/cifar10/`
- `data/generated/`
- `data/checkpoints/`

If you want to reproduce the real-board workflow, the device directory will be created under:

```text
data/generated/synthetic_cifar10/devices/real/stm32f405rgt6_000
```
