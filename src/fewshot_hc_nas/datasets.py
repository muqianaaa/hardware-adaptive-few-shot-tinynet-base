from __future__ import annotations

import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import BudgetSpec, DeviceRecord

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


class CIFAR10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.augment:
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1]
            pad = 4
            padded = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
            top = np.random.randint(0, pad * 2 + 1)
            left = np.random.randint(0, pad * 2 + 1)
            image = padded[:, top : top + 32, left : left + 32]
        return torch.from_numpy(image.copy()), int(self.labels[index])


class ArchitectureAccuracyDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class FewShotTaskDataset:
    def __init__(
        self,
        devices: list[DeviceRecord],
        measurements_by_device: dict[str, list[dict[str, Any]]],
        support_size: int = 8,
        query_size: int = 24,
        seed: int = 0,
    ):
        self.devices = devices
        self.measurements_by_device = measurements_by_device
        self.support_size = support_size
        self.query_size = query_size
        self.rng = np.random.default_rng(seed)

    def sample_task(self) -> dict[str, Any]:
        device = self.devices[int(self.rng.integers(0, len(self.devices)))]
        rows = self.measurements_by_device[device.static.name]
        order = self.rng.permutation(len(rows))
        support_rows = [rows[i] for i in order[: self.support_size]]
        query_rows = [rows[i] for i in order[self.support_size : self.support_size + self.query_size]]
        latency_values = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
        sram_values = np.asarray([float(row["peak_sram_bytes"]) for row in rows], dtype=np.float64)
        flash_values = np.asarray([float(row["flash_bytes"]) for row in rows], dtype=np.float64)
        latency_quantile = float(self.rng.uniform(0.35, 0.55))
        memory_quantile = float(self.rng.uniform(0.45, 0.70))
        flash_quantile = float(self.rng.uniform(0.45, 0.70))
        budget = BudgetSpec(
            t_max_ms=float(np.quantile(latency_values, latency_quantile) * self.rng.uniform(1.02, 1.12)),
            m_max_bytes=float(np.quantile(sram_values, memory_quantile) * self.rng.uniform(1.02, 1.12)),
            f_max_bytes=float(np.quantile(flash_values, flash_quantile) * self.rng.uniform(1.02, 1.12)),
        )
        return {
            "device": device,
            "budget": budget,
            "support_rows": support_rows,
            "query_rows": query_rows,
        }


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == 0:
        target.unlink()
    urllib.request.urlretrieve(url, target)


def ensure_cifar10(root: str | Path, force_redownload: bool = False) -> Path:
    root = Path(root)
    extracted = root / "cifar-10-batches-py"
    if extracted.exists():
        return extracted
    archive_name = "cifar-10-python-redownload.tar.gz" if force_redownload else "cifar-10-python.tar.gz"
    archive = root / "downloads" / archive_name
    if force_redownload or not archive.exists():
        _download(CIFAR10_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def _invalidate_cifar10_artifacts(root: str | Path) -> None:
    root = Path(root)
    extracted = root / "cifar-10-batches-py"
    archive = root / "downloads" / "cifar-10-python.tar.gz"
    retry_archive = root / "downloads" / "cifar-10-python-redownload.tar.gz"
    if extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)
    for path in (archive, retry_archive):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def ensure_cifar100(root: str | Path, local_fallback_path: str | Path | None = None) -> Path:
    root = Path(root)
    extracted = root / "cifar-100-python"
    if extracted.exists():
        return extracted
    if local_fallback_path is not None:
        local_path = Path(local_fallback_path)
        if local_path.exists():
            return local_path
    archive = root / "downloads" / "cifar-100-python.tar.gz"
    if not archive.exists():
        _download(CIFAR100_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def synthetic_cifar10_arrays(train_size: int = 12000, test_size: int = 3000, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    train_x = rng.integers(0, 256, size=(train_size, 3, 32, 32), dtype=np.uint8)
    test_x = rng.integers(0, 256, size=(test_size, 3, 32, 32), dtype=np.uint8)
    train_y = rng.integers(0, 10, size=(train_size,), dtype=np.int64)
    test_y = rng.integers(0, 10, size=(test_size,), dtype=np.int64)
    return {"train": (train_x, train_y), "test": (test_x, test_y)}


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = payload[b"data"].reshape(-1, 3, 32, 32)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    return data, labels


def _load_cifar100_batch(path: Path, label_mode: str = "fine") -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = payload[b"data"].reshape(-1, 3, 32, 32)
    label_key = b"fine_labels" if label_mode == "fine" else b"coarse_labels"
    labels = np.asarray(payload[label_key], dtype=np.int64)
    return data, labels


def load_cifar10_arrays(root: str | Path, allow_synthetic_fallback: bool = True, synthetic_seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            extracted = ensure_cifar10(root, force_redownload=bool(attempt))
            train_images: list[np.ndarray] = []
            train_labels: list[np.ndarray] = []
            for idx in range(1, 6):
                data, labels = _load_batch(extracted / f"data_batch_{idx}")
                train_images.append(data)
                train_labels.append(labels)
            test_data, test_labels = _load_batch(extracted / "test_batch")
            train_x = np.concatenate(train_images, axis=0)
            train_y = np.concatenate(train_labels, axis=0)
            return {"train": (train_x, train_y), "test": (test_data, test_labels)}
        except Exception as exc:
            last_error = exc
            _invalidate_cifar10_artifacts(root)
    if allow_synthetic_fallback:
        return synthetic_cifar10_arrays(seed=synthetic_seed)
    assert last_error is not None
    raise last_error


def load_cifar100_arrays(root: str | Path, local_fallback_path: str | Path | None = None, label_mode: str = "fine") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    extracted = ensure_cifar100(root, local_fallback_path=local_fallback_path)
    train_x, train_y = _load_cifar100_batch(extracted / "train", label_mode=label_mode)
    test_x, test_y = _load_cifar100_batch(extracted / "test", label_mode=label_mode)
    return {"train": (train_x, train_y), "test": (test_x, test_y)}


def build_cifar10_datasets(root: str | Path, val_size: int = 5_000, seed: int = 0, allow_synthetic_fallback: bool = True, force_synthetic: bool = False) -> dict[str, Dataset]:
    arrays = synthetic_cifar10_arrays(seed=seed) if force_synthetic else load_cifar10_arrays(root, allow_synthetic_fallback=allow_synthetic_fallback, synthetic_seed=seed)
    train_x, train_y = arrays["train"]
    test_x, test_y = arrays["test"]
    rng = np.random.default_rng(seed)
    effective_val_size = min(int(val_size), max(int(train_y.shape[0]) - 1, 1))
    order = rng.permutation(train_y.shape[0])
    val_idx = order[:effective_val_size]
    train_idx = order[effective_val_size:]
    return {
        "train": CIFAR10Dataset(train_x[train_idx], train_y[train_idx], augment=True),
        "val": CIFAR10Dataset(train_x[val_idx], train_y[val_idx], augment=False),
        "test": CIFAR10Dataset(test_x, test_y, augment=False),
    }


def build_cifar100_datasets(root: str | Path, val_size: int = 5_000, seed: int = 0, local_fallback_path: str | Path | None = None, label_mode: str = "fine") -> dict[str, Dataset]:
    arrays = load_cifar100_arrays(root, local_fallback_path=local_fallback_path, label_mode=label_mode)
    train_x, train_y = arrays["train"]
    test_x, test_y = arrays["test"]
    rng = np.random.default_rng(seed)
    effective_val_size = min(int(val_size), max(int(train_y.shape[0]) - 1, 1))
    order = rng.permutation(train_y.shape[0])
    val_idx = order[:effective_val_size]
    train_idx = order[effective_val_size:]
    return {
        "train": CIFAR10Dataset(train_x[train_idx], train_y[train_idx], augment=True),
        "val": CIFAR10Dataset(train_x[val_idx], train_y[val_idx], augment=False),
        "test": CIFAR10Dataset(test_x, test_y, augment=False),
    }


def dataset_num_classes(name: str, label_mode: str = "fine") -> int:
    if name in {"cifar10", "synthetic_cifar10"}:
        return 10
    if name == "cifar100":
        return 100 if label_mode == "fine" else 20
    raise ValueError(f"Unsupported dataset name: {name}")


def build_image_datasets(
    name: str,
    root: str | Path,
    val_size: int = 5_000,
    seed: int = 0,
    allow_synthetic_fallback: bool = True,
    force_synthetic: bool = False,
    cifar100_local_path: str | Path | None = None,
    cifar100_label_mode: str = "fine",
) -> dict[str, Dataset]:
    if name == "cifar10":
        return build_cifar10_datasets(root, val_size=val_size, seed=seed, allow_synthetic_fallback=allow_synthetic_fallback, force_synthetic=force_synthetic)
    if name == "synthetic_cifar10":
        return build_cifar10_datasets(root, val_size=val_size, seed=seed, allow_synthetic_fallback=True, force_synthetic=True)
    if name == "cifar100":
        return build_cifar100_datasets(root, val_size=val_size, seed=seed, local_fallback_path=cifar100_local_path, label_mode=cifar100_label_mode)
    raise ValueError(f"Unsupported dataset name: {name}")
