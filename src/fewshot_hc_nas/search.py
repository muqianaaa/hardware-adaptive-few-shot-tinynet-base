from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .models import decode_meta_architecture
from .search_space import ARCH_FEATURE_DIM, architecture_from_rows, crossover_architectures, encode_architecture, mutate_architecture, sample_architecture, structured_architecture_tensor
from .types import ArchitectureSpec, BudgetSpec, HardwareStaticSpec, OPS, QUANTS, WIDTHS, DEPTHS, BlockSpec


def build_heuristic_prior(static: HardwareStaticSpec, response_flat: dict[str, Any], budget: BudgetSpec, num_blocks: int = 5) -> dict[str, Any]:
    gamma = response_flat["gamma"]
    if isinstance(gamma, dict):
        op_cost = {op: float(np.mean(list(gamma[op].values()))) for op in OPS}
    else:
        op_cost = {op: float(gamma[idx].mean()) for idx, op in enumerate(OPS)}
    inv = np.asarray([1.0 / max(op_cost[op], 1e-9) for op in OPS], dtype=np.float64)
    op_probs = (inv / inv.sum()).tolist()

    memory_ratio = budget.m_max_bytes / max(float(static.sram_bytes), 1.0)
    flash_ratio = budget.f_max_bytes / max(float(static.flash_bytes), 1.0)
    ccm_ratio = float(static.ccm_bytes) / max(float(static.sram_bytes), 1.0)
    bandwidth_pressure = 1.0 if float(static.bus_width) <= 16.0 else 0.0
    width_scores = np.asarray(
        [
            1.0 / (w ** 1.35) if memory_ratio < 0.55 else 1.0 / (0.25 + abs((1.0 + 0.20 * ccm_ratio) - w))
            for w in WIDTHS
        ],
        dtype=np.float64,
    )
    if bandwidth_pressure > 0.0 or float(static.unaligned_access_efficiency) < 0.45:
        width_scores *= np.asarray([1.35, 1.20, 1.08, 0.95, 0.80, 0.68], dtype=np.float64)
    elif ccm_ratio > 0.20 and float(static.ram_bank_count) >= 2.0:
        width_scores *= np.asarray([0.80, 0.92, 1.00, 1.08, 1.18, 1.24], dtype=np.float64)
    width_probs = (width_scores / width_scores.sum()).tolist()

    depth_scores = np.asarray([1.5, 1.0], dtype=np.float64) if budget.t_max_ms < 20 else np.asarray([1.0, 1.25], dtype=np.float64)
    if bandwidth_pressure > 0.0 or memory_ratio < 0.60:
        depth_scores *= np.asarray([1.24, 0.82], dtype=np.float64)
    elif float(static.cache_kb) >= 24.0 and float(static.ram_bank_count) >= 2.0:
        depth_scores *= np.asarray([0.90, 1.14], dtype=np.float64)
    depth_probs = (depth_scores / depth_scores.sum()).tolist()

    low_bit_friendly = float(static.kernel_int4 + static.kernel_int2)
    quant_scores = np.asarray([0.7 + low_bit_friendly, 0.9 + static.kernel_int4, 1.2 + static.kernel_int8], dtype=np.float64)
    if memory_ratio < 0.45:
        quant_scores *= np.asarray([1.5, 1.1, 0.7], dtype=np.float64)
    if flash_ratio < 0.50:
        quant_scores *= np.asarray([1.35 + static.kernel_int2, 1.10 + 0.5 * static.kernel_int4, 0.62], dtype=np.float64)
    if float(static.kernel_int2) < 0.20:
        quant_scores[0] *= 0.55
    if float(static.kernel_int4) < 0.35:
        quant_scores[1] *= 0.82
    quant_probs = (quant_scores / quant_scores.sum()).tolist()
    return {
        "num_blocks": num_blocks,
        "op_probs": [op_probs for _ in range(num_blocks)],
        "width_probs": [width_probs for _ in range(num_blocks)],
        "depth_probs": [depth_probs for _ in range(num_blocks)],
        "quant_probs": [quant_probs for _ in range(num_blocks)],
    }


def build_generator_prior(generator_outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "num_blocks": int(generator_outputs["op_prob"].shape[1]),
        "op_probs": generator_outputs["op_prob"][0].detach().cpu().tolist(),
        "width_probs": generator_outputs["width_prob"][0].detach().cpu().tolist(),
        "depth_probs": generator_outputs["depth_prob"][0].detach().cpu().tolist(),
        "quant_probs": generator_outputs["quant_prob"][0].detach().cpu().tolist(),
    }


def generate_architecture_direct(generator, z: torch.Tensor, budget: BudgetSpec, device: str = "cpu", name: str = "meta_generated") -> ArchitectureSpec:
    budget_x = torch.as_tensor([[budget.t_max_ms, budget.m_max_bytes, budget.f_max_bytes]], dtype=torch.float32, device=device)
    outputs = generator(z if z.ndim == 2 else z.unsqueeze(0), budget_x)
    return decode_meta_architecture(outputs, name=name)


def sample_from_prior(prior: dict[str, Any], seed: int | None = None, name: str = "candidate") -> ArchitectureSpec:
    rng = random.Random(seed)
    rows = []
    for block_idx in range(prior["num_blocks"]):
        rows.append(
            {
                "op": rng.choices(OPS, weights=prior["op_probs"][block_idx], k=1)[0],
                "width": rng.choices(WIDTHS, weights=prior["width_probs"][block_idx], k=1)[0],
                "depth": rng.choices((1, 2), weights=prior["depth_probs"][block_idx], k=1)[0],
                "quant": rng.choices(QUANTS, weights=prior["quant_probs"][block_idx], k=1)[0],
            }
        )
    arch = architecture_from_rows(rows)
    return ArchitectureSpec(blocks=arch.blocks, name=name)


def _with_replaced_block(arch: ArchitectureSpec, block_index: int, **updates: Any) -> ArchitectureSpec:
    blocks = list(arch.blocks)
    block = blocks[block_index]
    blocks[block_index] = BlockSpec(
        op=str(updates.get("op", block.op)),
        width=float(updates.get("width", block.width)),
        depth=int(updates.get("depth", block.depth)),
        quant=int(updates.get("quant", block.quant)),
    )
    return ArchitectureSpec(blocks=tuple(blocks), name=arch.name)


def iter_local_neighbors(seed_arch: ArchitectureSpec, radius: int = 1) -> list[ArchitectureSpec]:
    if radius <= 0:
        return [seed_arch]

    def _expand_one_step(base_arch: ArchitectureSpec) -> dict[str, ArchitectureSpec]:
        items: dict[str, ArchitectureSpec] = {}
        for block_index, block in enumerate(base_arch.blocks):
            for op in OPS:
                if op != block.op:
                    arch = _with_replaced_block(base_arch, block_index, op=op)
                    items[arch.compact_repr()] = arch
            for width in WIDTHS:
                if width != block.width:
                    arch = _with_replaced_block(base_arch, block_index, width=width)
                    items[arch.compact_repr()] = arch
            for depth in DEPTHS:
                if depth != block.depth:
                    arch = _with_replaced_block(base_arch, block_index, depth=depth)
                    items[arch.compact_repr()] = arch
            for quant in QUANTS:
                if quant != block.quant:
                    arch = _with_replaced_block(base_arch, block_index, quant=quant)
                    items[arch.compact_repr()] = arch
        return items

    visited: dict[str, ArchitectureSpec] = {seed_arch.compact_repr(): seed_arch}
    frontier: dict[str, ArchitectureSpec] = {seed_arch.compact_repr(): seed_arch}
    for _ in range(radius):
        next_frontier: dict[str, ArchitectureSpec] = {}
        for arch in frontier.values():
            for arch_repr, neighbor in _expand_one_step(arch).items():
                if arch_repr not in visited:
                    visited[arch_repr] = neighbor
                    next_frontier[arch_repr] = neighbor
        if not next_frontier:
            break
        frontier = next_frontier
    return list(visited.values())


def local_refine_search(
    seed_arch: ArchitectureSpec,
    accuracy_predictor,
    cost_predictor,
    feasibility_head,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec,
    device: str = "cpu",
    radius: int = 1,
    static_context: torch.Tensor | None = None,
    dominant_dims: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for idx, arch in enumerate(iter_local_neighbors(seed_arch, radius=radius)):
        arch = ArchitectureSpec(blocks=arch.blocks, name=f"refine_{idx}")
        pred = predict_candidate(
            arch,
            accuracy_predictor=accuracy_predictor,
            cost_predictor=cost_predictor,
            feasibility_head=feasibility_head,
            z=z,
            response=response,
            calibration=calibration,
            budget=budget,
            device=device,
            static_context=static_context,
        )
        rows.append({"architecture": arch, "prediction": pred, "score": score_prediction(pred, budget, dominant_dims=dominant_dims)})
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def predict_candidate(
    arch: ArchitectureSpec,
    accuracy_predictor,
    cost_predictor,
    feasibility_head,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec | None,
    device: str,
    static_context: torch.Tensor | None = None,
) -> dict[str, float]:
    arch_x = torch.as_tensor(encode_architecture(arch), dtype=torch.float32, device=device).reshape(1, ARCH_FEATURE_DIM)
    structured_x = torch.as_tensor(structured_architecture_tensor(arch), dtype=torch.float32, device=device).unsqueeze(0)
    z_batch = z if z.ndim == 2 else z.unsqueeze(0)
    calibration_batch = calibration if calibration.ndim == 2 else calibration.unsqueeze(0)
    with torch.no_grad():
        acc = accuracy_predictor(arch_x)
        cost_outputs = cost_predictor(arch_x, structured_x, z_batch, response, calibration_batch, static_context=static_context)
        if budget is None:
            feasible = torch.full((1, 1), 0.5, dtype=arch_x.dtype, device=device)
        else:
            budget_x = torch.as_tensor([[budget.t_max_ms, budget.m_max_bytes, budget.f_max_bytes]], dtype=torch.float32, device=device)
            feasible = feasibility_head(arch_x, z_batch, cost_outputs, budget_x, calibration_batch)
    return {
        "accuracy": float(acc.item()),
        "latency_ms": float(cost_outputs["latency_ms"].item()),
        "peak_sram_bytes": float(cost_outputs["peak_sram_bytes"].item()),
        "flash_bytes": float(cost_outputs["flash_bytes"].item()),
        "feasible_prob": float(feasible.item()),
    }


def _score_weights(dominant_dims: tuple[str, ...] | None) -> tuple[float, float, float]:
    if not dominant_dims:
        return 0.18, 0.12, 0.10
    weights = {"T": 0.15, "M": 0.15, "F": 0.15}
    dominant_order = [str(dim).upper() for dim in dominant_dims]
    if dominant_order:
        weights[dominant_order[0]] += 0.13
    if len(dominant_order) > 1:
        weights[dominant_order[1]] += 0.07
    return weights["T"], weights["M"], weights["F"]


def score_prediction(pred: dict[str, float], budget: BudgetSpec, dominant_dims: tuple[str, ...] | None = None) -> float:
    latency_ratio = pred["latency_ms"] / max(budget.t_max_ms, 1.0)
    sram_ratio = pred["peak_sram_bytes"] / max(budget.m_max_bytes, 1.0)
    flash_ratio = pred["flash_bytes"] / max(budget.f_max_bytes, 1.0)
    overflow_penalty = max(0.0, latency_ratio - 1.0) + max(0.0, sram_ratio - 1.0) + max(0.0, flash_ratio - 1.0)
    feasible_bonus = 0.12 if latency_ratio <= 1.0 and sram_ratio <= 1.0 and flash_ratio <= 1.0 else 0.0
    latency_w, sram_w, flash_w = _score_weights(dominant_dims)
    return pred["accuracy"] - latency_w * latency_ratio - sram_w * sram_ratio - flash_w * flash_ratio - 0.65 * overflow_penalty + 0.15 * pred["feasible_prob"] + feasible_bonus


def evolutionary_search(
    accuracy_predictor,
    cost_predictor,
    feasibility_head,
    generator,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    static: HardwareStaticSpec,
    budget: BudgetSpec,
    device: str = "cpu",
    population_size: int = 32,
    rounds: int = 20,
    mutation_rate: float = 0.25,
    seed: int = 0,
    static_context: torch.Tensor | None = None,
    dominant_dims: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    budget_x = torch.as_tensor([[budget.t_max_ms, budget.m_max_bytes, budget.f_max_bytes]], dtype=torch.float32, device=device)
    with torch.no_grad():
        prior = build_generator_prior(generator(z if z.ndim == 2 else z.unsqueeze(0), budget_x))
    if not prior["op_probs"]:
        response_np = {
            "gamma": response["gamma"].detach().cpu().numpy()[0],
            "beta_mem": float(response["beta_mem"].detach().cpu().numpy()[0][0]),
            "rho_launch": float(response["rho_launch"].detach().cpu().numpy()[0][0]),
            "rho_copy": float(response["rho_copy"].detach().cpu().numpy()[0][0]),
        }
        prior = build_heuristic_prior(static, response_np, budget)

    rng = random.Random(seed)
    population = [sample_from_prior(prior, seed=rng.randint(0, 1_000_000), name=f"seed_{i}") for i in range(population_size)]
    scored: list[dict[str, Any]] = []
    for round_idx in range(rounds):
        scored = []
        for arch in population:
            pred = predict_candidate(
                arch,
                accuracy_predictor=accuracy_predictor,
                cost_predictor=cost_predictor,
                feasibility_head=feasibility_head,
                z=z,
                response=response,
                calibration=calibration,
                budget=budget,
                device=device,
                static_context=static_context,
            )
            scored.append({"architecture": arch, "prediction": pred, "score": score_prediction(pred, budget, dominant_dims=dominant_dims), "round": round_idx})
        scored.sort(key=lambda item: item["score"], reverse=True)
        elites = scored[: max(4, population_size // 4)]
        next_population = [ArchitectureSpec(blocks=item["architecture"].blocks, name=item["architecture"].name) for item in elites]
        while len(next_population) < population_size:
            parent_a = rng.choice(elites)["architecture"]
            parent_b = rng.choice(elites)["architecture"]
            child = crossover_architectures(parent_a, parent_b, seed=rng.randint(0, 1_000_000))
            child = mutate_architecture(child, seed=rng.randint(0, 1_000_000), mutation_rate=mutation_rate)
            next_population.append(child)
        population = next_population
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:20]


def random_search(
    accuracy_predictor,
    cost_predictor,
    feasibility_head,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec,
    device: str = "cpu",
    trials: int = 128,
    seed: int = 0,
    dominant_dims: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(trials):
        arch = sample_architecture(seed=rng.randint(0, 1_000_000), name=f"random_{idx}")
        pred = predict_candidate(
            arch,
            accuracy_predictor=accuracy_predictor,
            cost_predictor=cost_predictor,
            feasibility_head=feasibility_head,
            z=z,
            response=response,
            calibration=calibration,
            budget=budget,
            device=device,
        )
        rows.append({"architecture": arch, "prediction": pred, "score": score_prediction(pred, budget, dominant_dims=dominant_dims)})
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:20]
