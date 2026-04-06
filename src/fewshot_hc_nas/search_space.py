from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Iterable

import numpy as np

from .types import ArchitectureSpec, DEPTHS, NUM_SEARCHABLE_BLOCKS, OPS, QUANTS, BlockMetrics, BlockSpec, WIDTHS

STEM_CHANNELS = 16
BLOCK_BASE_CHANNELS = (16, 24, 32, 48, 64)
BLOCK_STRIDES = (1, 2, 1, 2, 1)
MBEXPAND_RATIO = 4
ARCH_TOKEN_DIM = len(OPS) + len(WIDTHS) + len(DEPTHS) + len(QUANTS) + 4
ARCH_FEATURE_DIM = NUM_SEARCHABLE_BLOCKS * ARCH_TOKEN_DIM

# Backward-compatible aliases for pre-refactor modules.
STAGE_BASE_CHANNELS = BLOCK_BASE_CHANNELS
STAGE_STRIDES = BLOCK_STRIDES


def make_divisible(value: float, divisor: int = 4) -> int:
    return max(divisor, int(round(value / divisor) * divisor))


def block_out_channels(block_index: int, width: float) -> int:
    return make_divisible(BLOCK_BASE_CHANNELS[block_index] * width)


def stage_out_channels(stage_index: int, width: float) -> int:
    return block_out_channels(stage_index, width)


def default_architecture(name: str = "tiny_net_base") -> ArchitectureSpec:
    return ArchitectureSpec(
        name=name,
        blocks=tuple(BlockSpec("std3x3", 1.0, 1, 8) for _ in range(NUM_SEARCHABLE_BLOCKS)),
    )


def architecture_from_rows(rows: Iterable[dict]) -> ArchitectureSpec:
    return ArchitectureSpec(blocks=tuple(BlockSpec(**row) for row in rows))


def sample_architecture(seed: int | None = None, name: str = "candidate") -> ArchitectureSpec:
    rng = random.Random(seed)
    return ArchitectureSpec(
        blocks=tuple(
            BlockSpec(
                op=rng.choice(OPS),
                width=rng.choice(WIDTHS),
                depth=rng.choice(DEPTHS),
                quant=rng.choice(QUANTS),
            )
            for _ in range(NUM_SEARCHABLE_BLOCKS)
        ),
        name=name,
    )


def mutate_architecture(
    arch: ArchitectureSpec,
    seed: int | None = None,
    mutation_rate: float = 0.25,
    name: str | None = None,
) -> ArchitectureSpec:
    rng = random.Random(seed)
    blocks: list[BlockSpec] = []
    for block in arch.blocks:
        op = rng.choice(OPS) if rng.random() < mutation_rate else block.op
        width = rng.choice(WIDTHS) if rng.random() < mutation_rate else block.width
        depth = rng.choice(DEPTHS) if rng.random() < mutation_rate else block.depth
        quant = rng.choice(QUANTS) if rng.random() < mutation_rate else block.quant
        blocks.append(BlockSpec(op=op, width=width, depth=depth, quant=quant))
    return ArchitectureSpec(blocks=tuple(blocks), name=name or arch.name)


def crossover_architectures(
    left: ArchitectureSpec,
    right: ArchitectureSpec,
    seed: int | None = None,
    name: str = "child",
) -> ArchitectureSpec:
    rng = random.Random(seed)
    blocks = [rng.choice((lb, rb)) for lb, rb in zip(left.blocks, right.blocks)]
    return ArchitectureSpec(blocks=tuple(blocks), name=name)


def _op_block_stats(
    op: str,
    in_ch: int,
    out_ch: int,
    in_hw: int,
    stride: int,
    quant: int,
) -> tuple[int, float, float, float, float, float, float]:
    out_hw = max(1, math.ceil(in_hw / stride))
    act_in = float(in_ch * in_hw * in_hw)
    act_out = float(out_ch * out_hw * out_hw)
    bytes_moved = (act_in + act_out) * quant / 8.0
    if op == "std3x3":
        macs = float(out_hw * out_hw * in_ch * out_ch * 9)
        params = float(in_ch * out_ch * 9)
        workspace = act_out * 0.06
        code_footprint = 2048.0
    elif op == "dw_sep":
        macs = float(out_hw * out_hw * (in_ch * 9 + in_ch * out_ch))
        params = float(in_ch * 9 + in_ch * out_ch)
        workspace = act_out * 0.05
        code_footprint = 1792.0
    elif op == "mbconv":
        hidden = max(out_ch, in_ch) * MBEXPAND_RATIO
        macs = float(out_hw * out_hw * (in_ch * hidden + hidden * 9 + hidden * out_ch))
        params = float(in_ch * hidden + hidden * 9 + hidden * out_ch)
        workspace = act_out * 0.11
        code_footprint = 3072.0
    else:
        raise ValueError(f"Unsupported op: {op}")
    total_bytes = bytes_moved + params * quant / 8.0
    return out_hw, macs, params, act_in, act_out, total_bytes, workspace + code_footprint


def max_width() -> float:
    return max(WIDTHS)


def compute_block_metrics(arch: ArchitectureSpec, input_hw: int = 32) -> list[BlockMetrics]:
    metrics: list[BlockMetrics] = []
    in_ch = STEM_CHANNELS
    in_hw = input_hw
    for idx, block in enumerate(arch.blocks):
        out_ch = block_out_channels(idx, block.width)
        block_macs = 0.0
        block_params = 0.0
        block_bytes = 0.0
        block_workspace = 0.0
        block_code = 0.0
        first_act_in = float(in_ch * in_hw * in_hw)
        current_in_ch = in_ch
        current_hw = in_hw
        last_act_out = 0.0
        for depth_idx in range(block.depth):
            stride = BLOCK_STRIDES[idx] if depth_idx == 0 else 1
            out_hw, macs, params, act_in, act_out, bytes_moved, workspace_plus_code = _op_block_stats(
                block.op,
                current_in_ch,
                out_ch,
                current_hw,
                stride,
                block.quant,
            )
            block_macs += macs
            block_params += params
            block_bytes += bytes_moved
            block_workspace += workspace_plus_code * 0.65
            block_code += workspace_plus_code * 0.35
            current_in_ch = out_ch
            current_hw = out_hw
            last_act_out = act_out
        metrics.append(
            BlockMetrics(
                block_index=idx,
                in_channels=in_ch,
                out_channels=out_ch,
                in_hw=in_hw,
                out_hw=current_hw,
                macs=block_macs,
                params=block_params,
                act_in=first_act_in,
                act_out=last_act_out,
                bytes_moved=block_bytes,
                workspace=block_workspace,
                code_footprint=block_code,
            )
        )
        in_ch = out_ch
        in_hw = current_hw
    return metrics


def compute_stage_metrics(arch: ArchitectureSpec, input_hw: int = 32) -> list[BlockMetrics]:
    return compute_block_metrics(arch, input_hw=input_hw)


def encode_architecture(arch: ArchitectureSpec) -> np.ndarray:
    metrics = compute_block_metrics(arch)
    tokens: list[float] = []
    for block, metric in zip(arch.blocks, metrics):
        tokens.extend([1.0 if block.op == op else 0.0 for op in OPS])
        tokens.extend([1.0 if block.width == width else 0.0 for width in WIDTHS])
        tokens.extend([1.0 if block.depth == depth else 0.0 for depth in DEPTHS])
        tokens.extend([1.0 if block.quant == quant else 0.0 for quant in QUANTS])
        tokens.extend(
            [
                math.log(metric.macs + 1.0),
                math.log(metric.params + 1.0),
                math.log(metric.act_in + 1.0),
                math.log(metric.act_out + 1.0),
            ]
        )
    return np.asarray(tokens, dtype=np.float32)


def structured_architecture_tensor(arch: ArchitectureSpec) -> np.ndarray:
    metrics = compute_block_metrics(arch)
    rows: list[list[float]] = []
    for block, metric in zip(arch.blocks, metrics):
        rows.append(
            [
                float(OPS.index(block.op)),
                float(block.width),
                float(block.depth),
                float(block.quant),
                metric.macs,
                metric.params,
                metric.act_in,
                metric.act_out,
                metric.bytes_moved,
                metric.workspace,
                metric.code_footprint,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def architecture_summary(arch: ArchitectureSpec) -> dict:
    block_metrics = [asdict(metric) for metric in compute_block_metrics(arch)]
    return {
        "architecture": arch.to_dict(),
        "feature_dim": int(encode_architecture(arch).shape[0]),
        "block_metrics": block_metrics,
        "stage_metrics": block_metrics,
    }


def instantiate_model(arch: ArchitectureSpec, num_classes: int = 10):
    from .models import TinyNet

    return TinyNet(arch, num_classes=num_classes)
