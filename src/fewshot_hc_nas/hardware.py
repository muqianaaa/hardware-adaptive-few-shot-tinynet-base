from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import ensure_dir, write_json, write_jsonl
from .search_space import compute_stage_metrics, default_architecture
from .types import (
    ArchitectureSpec,
    BudgetSpec,
    DeviceRecord,
    HardwareResponseCoefficients,
    HardwareStaticSpec,
    ProbeMeasurement,
    ReferenceMeasurement,
    StageSpec,
)


@dataclass(frozen=True)
class DeviceFamilyConfig:
    name: str
    static_mean: dict[str, Any]
    static_std: dict[str, float]
    gamma_mean: dict[str, dict[int, float]]
    gamma_std: float
    beta_mem: tuple[float, float]
    rho_launch: tuple[float, float]
    rho_copy: tuple[float, float]


DEVICE_FAMILIES: dict[str, DeviceFamilyConfig] = {
    "low_memory_mcu": DeviceFamilyConfig(
        name="low_memory_mcu",
        static_mean=dict(
            sram_bytes=96 * 1024,
            flash_bytes=768 * 1024,
            freq_mhz=144,
            dsp=0.38,
            simd=0.28,
            cache_kb=4.0,
            bus_width=16.0,
            kernel_int8=0.78,
            kernel_int4=0.34,
            kernel_int2=0.12,
            runtime_type="tflm",
            ccm_bytes=0,
            fpu=0.0,
            dma=0.34,
            art_accelerator=0.0,
            ram_bank_count=1.0,
            unaligned_access_efficiency=0.30,
        ),
        static_std=dict(
            sram_bytes=12_000,
            flash_bytes=90_000,
            freq_mhz=8.0,
            dsp=0.08,
            simd=0.06,
            cache_kb=1.0,
            bus_width=2.0,
            kernel_int8=0.05,
            kernel_int4=0.07,
            kernel_int2=0.07,
            ccm_bytes=0.0,
            fpu=0.02,
            dma=0.08,
            art_accelerator=0.02,
            ram_bank_count=0.15,
            unaligned_access_efficiency=0.05,
        ),
        gamma_mean={"std3x3": {2: 2.6e-6, 4: 2.1e-6, 8: 1.8e-6}, "dw_sep": {2: 2.2e-6, 4: 1.85e-6, 8: 1.65e-6}, "mbconv": {2: 2.9e-6, 4: 2.4e-6, 8: 2.1e-6}},
        gamma_std=1.2e-7,
        beta_mem=(2.9e-4, 1.4e-5),
        rho_launch=(0.20, 0.02),
        rho_copy=(0.15, 0.02),
    ),
    "balanced_mcu": DeviceFamilyConfig(
        name="balanced_mcu",
        static_mean=dict(
            sram_bytes=256 * 1024,
            flash_bytes=2_048 * 1024,
            freq_mhz=180,
            dsp=0.60,
            simd=0.52,
            cache_kb=12.0,
            bus_width=16.0,
            kernel_int8=0.92,
            kernel_int4=0.68,
            kernel_int2=0.34,
            runtime_type="cmsis_nn",
            ccm_bytes=16 * 1024,
            fpu=0.20,
            dma=0.74,
            art_accelerator=0.35,
            ram_bank_count=1.2,
            unaligned_access_efficiency=0.56,
        ),
        static_std=dict(
            sram_bytes=16_000,
            flash_bytes=150_000,
            freq_mhz=10.0,
            dsp=0.08,
            simd=0.08,
            cache_kb=1.4,
            bus_width=2.0,
            kernel_int8=0.04,
            kernel_int4=0.05,
            kernel_int2=0.05,
            ccm_bytes=2_000.0,
            fpu=0.04,
            dma=0.06,
            art_accelerator=0.06,
            ram_bank_count=0.18,
            unaligned_access_efficiency=0.05,
        ),
        gamma_mean={"std3x3": {2: 1.9e-6, 4: 1.55e-6, 8: 1.28e-6}, "dw_sep": {2: 1.7e-6, 4: 1.38e-6, 8: 1.18e-6}, "mbconv": {2: 2.1e-6, 4: 1.8e-6, 8: 1.56e-6}},
        gamma_std=1.0e-7,
        beta_mem=(1.9e-4, 1.0e-5),
        rho_launch=(0.15, 0.015),
        rho_copy=(0.10, 0.015),
    ),
    "high_performance_mcu": DeviceFamilyConfig(
        name="high_performance_mcu",
        static_mean=dict(
            sram_bytes=768 * 1024,
            flash_bytes=6_144 * 1024,
            freq_mhz=260,
            dsp=0.84,
            simd=0.88,
            cache_kb=48.0,
            bus_width=32.0,
            kernel_int8=0.98,
            kernel_int4=0.87,
            kernel_int2=0.60,
            runtime_type="vendor_runtime",
            ccm_bytes=64 * 1024,
            fpu=0.85,
            dma=0.92,
            art_accelerator=0.88,
            ram_bank_count=2.0,
            unaligned_access_efficiency=0.92,
        ),
        static_std=dict(
            sram_bytes=50_000,
            flash_bytes=280_000,
            freq_mhz=16.0,
            dsp=0.05,
            simd=0.05,
            cache_kb=5.0,
            bus_width=3.0,
            kernel_int8=0.02,
            kernel_int4=0.03,
            kernel_int2=0.04,
            ccm_bytes=8_000.0,
            fpu=0.04,
            dma=0.03,
            art_accelerator=0.04,
            ram_bank_count=0.15,
            unaligned_access_efficiency=0.03,
        ),
        gamma_mean={"std3x3": {2: 1.2e-6, 4: 9.2e-7, 8: 7.2e-7}, "dw_sep": {2: 1.1e-6, 4: 8.6e-7, 8: 7.4e-7}, "mbconv": {2: 1.35e-6, 4: 1.02e-6, 8: 8.2e-7}},
        gamma_std=7.0e-8,
        beta_mem=(1.1e-4, 8.0e-6),
        rho_launch=(0.09, 0.01),
        rho_copy=(0.06, 0.01),
    ),
    "low_bit_friendly_mcu": DeviceFamilyConfig(
        name="low_bit_friendly_mcu",
        static_mean=dict(
            sram_bytes=512 * 1024,
            flash_bytes=4_096 * 1024,
            freq_mhz=220,
            dsp=0.78,
            simd=0.74,
            cache_kb=28.0,
            bus_width=32.0,
            kernel_int8=0.96,
            kernel_int4=0.95,
            kernel_int2=0.84,
            runtime_type="custom_runtime",
            ccm_bytes=32 * 1024,
            fpu=0.45,
            dma=0.86,
            art_accelerator=0.64,
            ram_bank_count=2.0,
            unaligned_access_efficiency=0.86,
        ),
        static_std=dict(
            sram_bytes=30_000,
            flash_bytes=220_000,
            freq_mhz=12.0,
            dsp=0.05,
            simd=0.05,
            cache_kb=4.0,
            bus_width=3.0,
            kernel_int8=0.03,
            kernel_int4=0.03,
            kernel_int2=0.03,
            ccm_bytes=4_000.0,
            fpu=0.05,
            dma=0.04,
            art_accelerator=0.05,
            ram_bank_count=0.2,
            unaligned_access_efficiency=0.04,
        ),
        gamma_mean={"std3x3": {2: 1.05e-6, 4: 8.6e-7, 8: 8.1e-7}, "dw_sep": {2: 9.2e-7, 4: 7.9e-7, 8: 7.6e-7}, "mbconv": {2: 1.15e-6, 4: 9.1e-7, 8: 8.3e-7}},
        gamma_std=6.0e-8,
        beta_mem=(1.4e-4, 8.0e-6),
        rho_launch=(0.11, 0.01),
        rho_copy=(0.08, 0.01),
    ),
    "depthwise_unfriendly_mcu": DeviceFamilyConfig(
        name="depthwise_unfriendly_mcu",
        static_mean=dict(
            sram_bytes=384 * 1024,
            flash_bytes=3_072 * 1024,
            freq_mhz=200,
            dsp=0.68,
            simd=0.62,
            cache_kb=20.0,
            bus_width=16.0,
            kernel_int8=0.90,
            kernel_int4=0.62,
            kernel_int2=0.28,
            runtime_type="cmsis_nn",
            ccm_bytes=12 * 1024,
            fpu=0.30,
            dma=0.70,
            art_accelerator=0.52,
            ram_bank_count=1.0,
            unaligned_access_efficiency=0.46,
        ),
        static_std=dict(
            sram_bytes=18_000,
            flash_bytes=180_000,
            freq_mhz=9.0,
            dsp=0.06,
            simd=0.06,
            cache_kb=2.0,
            bus_width=2.0,
            kernel_int8=0.04,
            kernel_int4=0.05,
            kernel_int2=0.05,
            ccm_bytes=2_000.0,
            fpu=0.05,
            dma=0.05,
            art_accelerator=0.06,
            ram_bank_count=0.15,
            unaligned_access_efficiency=0.05,
        ),
        gamma_mean={"std3x3": {2: 1.7e-6, 4: 1.45e-6, 8: 1.2e-6}, "dw_sep": {2: 2.4e-6, 4: 2.1e-6, 8: 1.8e-6}, "mbconv": {2: 2.0e-6, 4: 1.72e-6, 8: 1.48e-6}},
        gamma_std=9.0e-8,
        beta_mem=(1.7e-4, 1.0e-5),
        rho_launch=(0.15, 0.015),
        rho_copy=(0.10, 0.015),
    ),
    "memory_bottleneck_mcu": DeviceFamilyConfig(
        name="memory_bottleneck_mcu",
        static_mean=dict(
            sram_bytes=288 * 1024,
            flash_bytes=2_048 * 1024,
            freq_mhz=176,
            dsp=0.56,
            simd=0.48,
            cache_kb=6.0,
            bus_width=8.0,
            kernel_int8=0.82,
            kernel_int4=0.46,
            kernel_int2=0.18,
            runtime_type="tflm",
            ccm_bytes=0,
            fpu=0.12,
            dma=0.22,
            art_accelerator=0.06,
            ram_bank_count=1.0,
            unaligned_access_efficiency=0.24,
        ),
        static_std=dict(
            sram_bytes=20_000,
            flash_bytes=120_000,
            freq_mhz=10.0,
            dsp=0.07,
            simd=0.07,
            cache_kb=1.0,
            bus_width=2.0,
            kernel_int8=0.04,
            kernel_int4=0.05,
            kernel_int2=0.05,
            ccm_bytes=0.0,
            fpu=0.04,
            dma=0.05,
            art_accelerator=0.04,
            ram_bank_count=0.1,
            unaligned_access_efficiency=0.05,
        ),
        gamma_mean={"std3x3": {2: 2.2e-6, 4: 1.88e-6, 8: 1.58e-6}, "dw_sep": {2: 2.08e-6, 4: 1.78e-6, 8: 1.52e-6}, "mbconv": {2: 2.42e-6, 4: 2.08e-6, 8: 1.82e-6}},
        gamma_std=1.0e-7,
        beta_mem=(3.6e-4, 1.5e-5),
        rho_launch=(0.19, 0.018),
        rho_copy=(0.20, 0.02),
    ),
}

MICRO_PROBE_LIBRARY = (
    {"probe_id": "std3x3_small_8b", "op": "std3x3", "quant": 8, "shape": (1, 16, 16, 16), "macs": 16 * 16 * 16 * 16 * 9, "bytes": 16 * 16 * 16 * 2, "kind": "conv"},
    {"probe_id": "std3x3_medium_4b", "op": "std3x3", "quant": 4, "shape": (1, 24, 8, 8), "macs": 8 * 8 * 24 * 24 * 9, "bytes": 24 * 8 * 8 * 2, "kind": "conv"},
    {"probe_id": "std3x3_large_8b", "op": "std3x3", "quant": 8, "shape": (1, 32, 8, 8), "macs": 8 * 8 * 32 * 32 * 9, "bytes": 32 * 8 * 8 * 2, "kind": "conv"},
    {"probe_id": "dw_small_8b", "op": "dw_sep", "quant": 8, "shape": (1, 16, 16, 16), "macs": 16 * 16 * (16 * 9 + 16 * 16), "bytes": 16 * 16 * 16 * 2, "kind": "depthwise"},
    {"probe_id": "dw_medium_4b", "op": "dw_sep", "quant": 4, "shape": (1, 24, 8, 8), "macs": 8 * 8 * (24 * 9 + 24 * 24), "bytes": 24 * 8 * 8 * 2, "kind": "depthwise"},
    {"probe_id": "pw_small_8b", "op": "dw_sep", "quant": 8, "shape": (1, 32, 16, 16), "macs": 16 * 16 * 32 * 24, "bytes": 32 * 16 * 16 * 2, "kind": "pointwise"},
    {"probe_id": "pw_medium_4b", "op": "dw_sep", "quant": 4, "shape": (1, 40, 8, 8), "macs": 8 * 8 * 40 * 32, "bytes": 40 * 8 * 8 * 2, "kind": "pointwise"},
    {"probe_id": "mb_expand_8b", "op": "mbconv", "quant": 8, "shape": (1, 16, 16, 16), "macs": 16 * 16 * 16 * 64, "bytes": 16 * 16 * 16 * 3, "kind": "expand"},
    {"probe_id": "mb_project_4b", "op": "mbconv", "quant": 4, "shape": (1, 24, 8, 8), "macs": 8 * 8 * 96 * 24, "bytes": 24 * 8 * 8 * 3, "kind": "project"},
    {"probe_id": "move_aligned_4b", "op": "dw_sep", "quant": 4, "shape": (1, 32, 8, 8), "macs": 0, "bytes": 32 * 8 * 8, "kind": "aligned_move"},
    {"probe_id": "move_misaligned_4b", "op": "dw_sep", "quant": 4, "shape": (1, 30, 7, 7), "macs": 0, "bytes": 30 * 7 * 7, "kind": "misaligned_move"},
    {"probe_id": "pool_act_2b", "op": "mbconv", "quant": 2, "shape": (1, 32, 4, 4), "macs": 32 * 16, "bytes": 32 * 4 * 4, "kind": "pool"},
    {"probe_id": "bitpack_2b", "op": "mbconv", "quant": 2, "shape": (1, 48, 4, 4), "macs": 48 * 8, "bytes": 48 * 4 * 4, "kind": "bitpack"},
    {"probe_id": "bitpack_4b", "op": "mbconv", "quant": 4, "shape": (1, 48, 4, 4), "macs": 48 * 8, "bytes": 48 * 4 * 4, "kind": "bitpack"},
    {"probe_id": "fc_8b", "op": "std3x3", "quant": 8, "shape": (1, 1, 1, 128), "macs": 128 * 64, "bytes": 128 * 8, "kind": "fc"},
)

REFERENCE_ARCHITECTURES = {
    "宽浅参考网络": ArchitectureSpec(
        name="宽浅参考网络",
        stages=(
            StageSpec("std3x3", 1.25, 1, 8),
            StageSpec("std3x3", 1.25, 1, 8),
            StageSpec("mbconv", 1.0, 1, 8),
            StageSpec("std3x3", 1.0, 1, 4),
            StageSpec("mbconv", 1.0, 1, 4),
        ),
    ),
    "深窄参考网络": ArchitectureSpec(
        name="深窄参考网络",
        stages=(
            StageSpec("dw_sep", 0.5, 2, 8),
            StageSpec("dw_sep", 0.75, 2, 8),
            StageSpec("dw_sep", 0.75, 2, 4),
            StageSpec("mbconv", 0.75, 2, 4),
            StageSpec("dw_sep", 0.75, 2, 4),
        ),
    ),
    "混合精度深度可分离参考网络": ArchitectureSpec(
        name="混合精度深度可分离参考网络",
        stages=(
            StageSpec("dw_sep", 0.75, 2, 8),
            StageSpec("dw_sep", 1.0, 2, 4),
            StageSpec("mbconv", 0.75, 2, 2),
            StageSpec("dw_sep", 1.0, 1, 2),
            StageSpec("mbconv", 1.0, 1, 4),
        ),
    ),
    "低比特参考网络": ArchitectureSpec(
        name="低比特参考网络",
        stages=(
            StageSpec("dw_sep", 0.5, 1, 2),
            StageSpec("mbconv", 0.75, 1, 2),
            StageSpec("dw_sep", 0.75, 2, 4),
            StageSpec("mbconv", 0.75, 1, 2),
            StageSpec("dw_sep", 1.0, 1, 4),
        ),
    ),
    "访存突发参考网络": ArchitectureSpec(
        name="访存突发参考网络",
        stages=(
            StageSpec("std3x3", 1.25, 2, 8),
            StageSpec("std3x3", 1.5, 1, 8),
            StageSpec("mbconv", 1.25, 1, 8),
            StageSpec("std3x3", 1.25, 2, 4),
            StageSpec("mbconv", 1.25, 1, 4),
        ),
    ),
}

DATASET_ACCURACY_PRIORS: dict[str, dict[str, float]] = {
    "synthetic_cifar10": {"chance": 0.10, "base": 0.62, "amplitude": 0.24, "target_log_macs": 14.15, "target_log_params": 9.20, "mac_sigma": 0.75, "param_sigma": 0.88},
    "cifar10": {"chance": 0.10, "base": 0.60, "amplitude": 0.25, "target_log_macs": 14.20, "target_log_params": 9.28, "mac_sigma": 0.78, "param_sigma": 0.90},
    "cifar100": {"chance": 0.01, "base": 0.20, "amplitude": 0.22, "target_log_macs": 14.25, "target_log_params": 9.32, "mac_sigma": 0.80, "param_sigma": 0.92},
}

# 将解析延迟项标定到 MCU 论文常见的几十到百毫秒区间，
# 避免 synthetic 结果因统一缩放过大而落到数万毫秒。
LATENCY_SCALE_MS = 5.0


def _sample_with_std(mean: float, std: float, rng: np.random.Generator, lower: float = 0.0) -> float:
    return max(lower, float(rng.normal(mean, std)))


def _stable_arch_seed(arch: ArchitectureSpec, dataset_name: str) -> int:
    digest = hashlib.sha256(f"{dataset_name}:{arch.compact_repr()}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def synthetic_architecture_accuracy(
    arch: ArchitectureSpec,
    dataset_name: str = "cifar10",
    noise_scale: float = 0.004,
    seed: int | None = None,
) -> float:
    priors = DATASET_ACCURACY_PRIORS.get(dataset_name, DATASET_ACCURACY_PRIORS["cifar10"])
    metrics = compute_stage_metrics(arch)
    total_macs = float(sum(metric.macs for metric in metrics))
    total_params = float(sum(metric.params for metric in metrics))
    log_macs = float(np.log(total_macs + 1.0))
    log_params = float(np.log(total_params + 1.0))

    widths = np.asarray([stage.width for stage in arch.stages], dtype=np.float32)
    depths = np.asarray([stage.depth for stage in arch.stages], dtype=np.float32)
    quants = np.asarray([stage.quant for stage in arch.stages], dtype=np.float32)
    ops = [stage.op for stage in arch.stages]

    width_mean = float(widths.mean())
    depth_total = float(depths.sum())
    quant_term = float(np.clip((quants.mean() - 2.0) / 6.0, 0.0, 1.0))
    mb_frac = float(sum(op == "mbconv" for op in ops) / len(ops))
    std_frac = float(sum(op == "std3x3" for op in ops) / len(ops))
    dw_frac = float(sum(op == "dw_sep" for op in ops) / len(ops))
    late_capacity = float(np.mean([metric.out_channels for metric in metrics[-2:]]))

    capacity_term = float(np.exp(-((log_macs - priors["target_log_macs"]) ** 2) / (2.0 * priors["mac_sigma"] ** 2)))
    param_term = float(np.exp(-((log_params - priors["target_log_params"]) ** 2) / (2.0 * priors["param_sigma"] ** 2)))
    width_term = float(np.exp(-((width_mean - 0.95) ** 2) / (2.0 * 0.18**2)))
    depth_term = float(np.clip((depth_total - 5.0) / 5.0, 0.0, 1.0))
    stage_balance = float(np.clip(1.0 - np.std(widths) / 0.32, 0.0, 1.0))
    op_term = float(np.clip(0.58 * mb_frac + 0.32 * std_frac + 0.18 * dw_frac, 0.0, 1.0))
    late_stage_term = float(np.clip((late_capacity - 44.0) / 28.0, 0.0, 1.0))

    score = (
        0.26 * capacity_term
        + 0.15 * param_term
        + 0.14 * depth_term
        + 0.12 * width_term
        + 0.10 * quant_term
        + 0.09 * stage_balance
        + 0.07 * late_stage_term
        + 0.05 * op_term
        + 0.08 * float(mb_frac > 0.0)
    )
    score -= 0.05 * max(0.0, dw_frac - 0.60)
    score -= 0.04 * float(any(stage.quant == 2 and stage.op == "std3x3" for stage in arch.stages))
    score -= 0.03 * max(0.0, log_macs - priors["target_log_macs"] - 0.45)

    rng = np.random.default_rng(seed if seed is not None else _stable_arch_seed(arch, dataset_name))
    score = float(np.clip(score + float(rng.normal(0.0, noise_scale)), 0.0, 1.0))
    accuracy = float(priors["base"] + priors["amplitude"] * score)
    return float(np.clip(accuracy, priors["chance"] + 0.01, 0.95))


def sample_hardware_static(family: str, index: int, seed: int | None = None) -> HardwareStaticSpec:
    cfg = DEVICE_FAMILIES[family]
    rng = np.random.default_rng(seed)
    payload: dict[str, Any] = {}
    for key, mean in cfg.static_mean.items():
        if isinstance(mean, str):
            payload[key] = mean
            continue
        std = cfg.static_std.get(key, 0.0)
        sampled = _sample_with_std(float(mean), std, rng, lower=0.0)
        if key in {"sram_bytes", "flash_bytes", "ccm_bytes"}:
            sampled = int(sampled)
        payload[key] = sampled
    return HardwareStaticSpec(name=f"{family}_{index:03d}", family=family, **payload)


def sample_response_coefficients(family: str, seed: int | None = None) -> HardwareResponseCoefficients:
    cfg = DEVICE_FAMILIES[family]
    rng = np.random.default_rng(seed)
    gamma = {op: {quant: _sample_with_std(mean, cfg.gamma_std, rng, lower=1e-7) for quant, mean in inner.items()} for op, inner in cfg.gamma_mean.items()}
    return HardwareResponseCoefficients(
        gamma=gamma,
        beta_mem=_sample_with_std(cfg.beta_mem[0], cfg.beta_mem[1], rng, lower=1e-6),
        rho_launch=_sample_with_std(cfg.rho_launch[0], cfg.rho_launch[1], rng, lower=0.01),
        rho_copy=_sample_with_std(cfg.rho_copy[0], cfg.rho_copy[1], rng, lower=0.01),
    )


def _freq_scale(static: HardwareStaticSpec) -> float:
    return 200.0 / max(40.0, float(static.freq_mhz))


def _kernel_support(static: HardwareStaticSpec, quant: int) -> float:
    support = {8: static.kernel_int8, 4: static.kernel_int4, 2: static.kernel_int2}[quant]
    return float(np.clip(support, 0.05, 1.0))


def _quant_latency_multiplier(static: HardwareStaticSpec, quant: int) -> float:
    return float(np.clip(1.18 - 0.32 * _kernel_support(static, quant), 0.72, 1.26))


def _op_latency_multiplier(static: HardwareStaticSpec, op: str) -> float:
    if op == "dw_sep":
        penalty = 1.0 + 0.18 * (1.0 - float(static.simd)) + 0.12 * float(static.bus_width <= 16.0)
        penalty += 0.16 * (1.0 - float(static.unaligned_access_efficiency))
        penalty -= 0.05 * float(static.ccm_bytes >= 32 * 1024)
        return float(np.clip(penalty, 0.82, 1.45))
    if op == "mbconv":
        penalty = 1.0 + 0.10 * (1.0 - float(static.dsp)) + 0.08 * (1.0 - float(static.fpu))
        penalty += 0.10 * float(static.ccm_bytes < 24 * 1024)
        penalty -= 0.04 * float(static.ram_bank_count >= 2.0)
        return float(np.clip(penalty, 0.84, 1.34))
    penalty = 1.0 + 0.08 * (1.0 - float(static.art_accelerator)) + 0.06 * float(static.bus_width <= 16.0)
    return float(np.clip(penalty, 0.84, 1.28))


def _memory_latency_multiplier(static: HardwareStaticSpec, *, aligned: bool = True, bitpack: bool = False) -> float:
    penalty = 1.0 + 0.20 * (1.0 - float(static.dma)) + 0.12 * float(static.bus_width <= 16.0)
    penalty += 0.10 * float(static.cache_kb < 12.0)
    penalty += 0.16 * (1.0 - float(static.unaligned_access_efficiency)) * (0.0 if aligned else 1.0)
    if bitpack:
        penalty += 0.18 * (1.0 - _kernel_support(static, 2))
    return float(np.clip(penalty, 0.86, 1.58))


def predict_cost_from_response(
    arch: ArchitectureSpec,
    static: HardwareStaticSpec,
    response: HardwareResponseCoefficients,
    noise_scale: float = 0.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    metrics = compute_stage_metrics(arch)
    latency = 0.0
    peak_sram = 0.0
    flash = 0.0
    for stage, metric in zip(arch.stages, metrics):
        gamma = response.gamma[stage.op][stage.quant]
        op_multiplier = _op_latency_multiplier(static, stage.op)
        quant_multiplier = _quant_latency_multiplier(static, stage.quant)
        mem_term = response.beta_mem * metric.bytes_moved / max(static.bus_width, 1.0) * _memory_latency_multiplier(static)
        launch_term = response.rho_launch * stage.depth
        copy_term = response.rho_copy * (metric.act_out * stage.quant / 8.0) / 4096.0
        latency += (gamma * metric.macs * _freq_scale(static) * LATENCY_SCALE_MS * op_multiplier * quant_multiplier) + mem_term + launch_term + copy_term
        sram_candidate = metric.act_in * stage.quant / 8.0 + metric.act_out * stage.quant / 8.0 + metric.workspace
        if static.cache_kb < 12:
            sram_candidate *= 1.08
        if static.ccm_bytes >= 24 * 1024:
            sram_candidate *= 0.95
        if static.ram_bank_count >= 2.0:
            sram_candidate *= 0.97
        peak_sram = max(peak_sram, sram_candidate)
        runtime_surcharge = 1.0
        if static.runtime_type == "tflm":
            runtime_surcharge = 1.12
        elif static.runtime_type == "vendor_runtime":
            runtime_surcharge = 0.94
        elif static.runtime_type == "custom_runtime":
            runtime_surcharge = 1.03
        flash += metric.params * stage.quant / 8.0 + metric.code_footprint * runtime_surcharge
    if noise_scale > 0:
        latency *= max(0.8, float(rng.normal(1.0, noise_scale)))
        peak_sram *= max(0.9, float(rng.normal(1.0, noise_scale * 0.5)))
        flash *= max(0.9, float(rng.normal(1.0, noise_scale * 0.35)))
    return float(latency), float(peak_sram), float(flash)


def measure_probe_suite(static: HardwareStaticSpec, response: HardwareResponseCoefficients, noise_scale: float = 0.03, seed: int | None = None) -> tuple[ProbeMeasurement, ...]:
    rng = np.random.default_rng(seed)
    rows = []
    for probe in MICRO_PROBE_LIBRARY:
        gamma = response.gamma[probe["op"]][probe["quant"]]
        op_multiplier = _op_latency_multiplier(static, probe["op"])
        quant_multiplier = _quant_latency_multiplier(static, probe["quant"])
        aligned = probe.get("kind") != "misaligned_move"
        bitpack = probe.get("kind") == "bitpack"
        mac_term = gamma * probe["macs"] * _freq_scale(static) * LATENCY_SCALE_MS * op_multiplier * quant_multiplier
        mem_term = response.beta_mem * probe["bytes"] / max(static.bus_width, 1.0) * _memory_latency_multiplier(static, aligned=aligned, bitpack=bitpack)
        runtime_term = response.rho_copy if "move" in str(probe.get("kind", "")) else response.rho_launch * 0.35
        if probe.get("kind") == "fc":
            runtime_term *= 0.75
        latency = (mac_term + mem_term + runtime_term) * max(0.9, float(rng.normal(1.0, noise_scale)))
        rows.append(
            ProbeMeasurement(
                probe_id=probe["probe_id"],
                op=probe["op"],
                quant=probe["quant"],
                input_shape=tuple(probe["shape"]),
                latency_ms=float(latency),
                latency_per_mac=float(latency / max(probe["macs"], 1.0)),
                latency_per_byte=float(latency / max(probe["bytes"], 1.0)),
            )
        )
    return tuple(rows)


def measure_reference_networks(static: HardwareStaticSpec, response: HardwareResponseCoefficients, noise_scale: float = 0.02, seed: int | None = None) -> tuple[ReferenceMeasurement, ...]:
    refs: list[ReferenceMeasurement] = []
    for idx, (name, arch) in enumerate(REFERENCE_ARCHITECTURES.items()):
        latency, sram, flash = predict_cost_from_response(arch, static, response, noise_scale=noise_scale, seed=None if seed is None else seed + idx)
        refs.append(ReferenceMeasurement(name=name, architecture=arch, latency_ms=latency, peak_sram_bytes=sram, flash_bytes=flash))
    return tuple(refs)


def create_device_record(family: str, index: int, seed: int | None = None) -> DeviceRecord:
    static = sample_hardware_static(family, index=index, seed=seed)
    response = sample_response_coefficients(family, seed=None if seed is None else seed + 17)
    probes = measure_probe_suite(static, response, seed=None if seed is None else seed + 29)
    references = measure_reference_networks(static, response, seed=None if seed is None else seed + 41)
    return DeviceRecord(static=static, response=response, probes=probes, references=references)


def sample_budget(static: HardwareStaticSpec, seed: int | None = None) -> BudgetSpec:
    rng = np.random.default_rng(seed)
    return BudgetSpec(
        t_max_ms=float(rng.uniform(9.0, 36.0) * _freq_scale(static) * 4.5),
        m_max_bytes=float(rng.uniform(0.38, 0.78) * static.sram_bytes),
        f_max_bytes=float(rng.uniform(0.18, 0.48) * static.flash_bytes),
    )


def build_arch_measurement_table(
    architectures: list[ArchitectureSpec],
    devices: list[DeviceRecord],
    accuracy_lookup: dict[str, float] | None = None,
    noise_scale: float = 0.02,
    dataset_name: str = "cifar10",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    accuracy_lookup = accuracy_lookup or {}
    for device in devices:
        for arch in architectures:
            latency, sram, flash = predict_cost_from_response(arch, device.static, device.response, noise_scale=noise_scale)
            accuracy = accuracy_lookup.get(arch.compact_repr())
            if accuracy is None or pd.isna(accuracy):
                accuracy = synthetic_architecture_accuracy(arch, dataset_name=dataset_name, noise_scale=0.0)
            rows.append(
                {
                    "device_name": device.static.name,
                    "family": device.static.family,
                    "arch_name": arch.name,
                    "arch_repr": arch.compact_repr(),
                    "architecture_json": json.dumps(arch.to_dict(), ensure_ascii=False),
                    "latency_ms": latency,
                    "peak_sram_bytes": sram,
                    "flash_bytes": flash,
                    "accuracy": float(accuracy),
                }
            )
    return pd.DataFrame(rows)


def export_device_directory(device: DeviceRecord, root: str | Path) -> None:
    root = ensure_dir(root)
    write_json(root / "hardware_static.json", device.static.to_dict())
    write_json(root / "hardware_response.json", device.response.to_dict())
    write_jsonl(root / "probe_results.jsonl", [probe.to_dict() for probe in device.probes])
    write_jsonl(root / "reference_results.jsonl", [ref.to_dict() for ref in device.references])


def export_device_corpus(devices: list[DeviceRecord], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    manifest = []
    for device in devices:
        path = output_dir / device.static.name
        export_device_directory(device, path)
        manifest.append({"device": device.static.name, "path": str(path)})
    write_json(output_dir / "manifest.json", {"devices": manifest})


def make_standard_baseline(name: str = "标准TinyNet基线") -> ArchitectureSpec:
    return default_architecture(name=name)
