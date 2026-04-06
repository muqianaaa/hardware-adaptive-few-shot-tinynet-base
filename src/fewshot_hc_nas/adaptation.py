from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .models import CALIBRATION_DIM
from .search_space import encode_architecture, structured_architecture_tensor
from .types import ArchitectureSpec


def _row_architecture(row: dict[str, Any]) -> ArchitectureSpec:
    if "architecture_json" in row:
        payload = row["architecture_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ArchitectureSpec.from_dict(payload)
    if "arch_repr" in row:
        return ArchitectureSpec.from_compact_repr(str(row["arch_repr"]), name=str(row.get("arch_name", "candidate")))
    raise KeyError("Row is missing both architecture_json and arch_repr.")


def _rows_to_tensors(support_rows: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arch_features = []
    arch_struct = []
    targets = []
    feasible = []
    budgets = []
    for row in support_rows:
        arch = _row_architecture(row)
        arch_features.append(encode_architecture(arch))
        arch_struct.append(structured_architecture_tensor(arch))
        targets.append([row["latency_ms"], row["peak_sram_bytes"], row["flash_bytes"]])
        if {"budget_t", "budget_m", "budget_f"} <= row.keys():
            budget = [row["budget_t"], row["budget_m"], row["budget_f"]]
        else:
            budget = [row["latency_ms"] * 1.08, row["peak_sram_bytes"] * 1.08, row["flash_bytes"] * 1.08]
        budgets.append(budget)
        feasible.append(
            [
                float(
                    (row["latency_ms"] <= budget[0])
                    and (row["peak_sram_bytes"] <= budget[1])
                    and (row["flash_bytes"] <= budget[2])
                )
            ]
        )
    arch_x = torch.as_tensor(np.stack(arch_features), dtype=torch.float32, device=device)
    structured_x = torch.as_tensor(np.stack(arch_struct), dtype=torch.float32, device=device)
    target_x = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=device)
    budget_x = torch.as_tensor(np.asarray(budgets), dtype=torch.float32, device=device)
    feasible_x = torch.as_tensor(np.asarray(feasible), dtype=torch.float32, device=device)
    return arch_x, structured_x, target_x, budget_x, feasible_x


def adapt_device_state(
    response_decoder,
    cost_predictor,
    feasibility_head,
    initial_z: torch.Tensor,
    support_rows: list[dict[str, Any]],
    initial_calibration: torch.Tensor | None = None,
    steps: int = 25,
    lr: float = 1e-2,
    device: str = "cpu",
    static_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    if initial_z.ndim == 1:
        initial_z = initial_z.unsqueeze(0)
    if initial_calibration is None:
        initial_calibration = torch.zeros(initial_z.shape[0], CALIBRATION_DIM, dtype=initial_z.dtype)
    if initial_calibration.ndim == 1:
        initial_calibration = initial_calibration.unsqueeze(0)
    z = torch.nn.Parameter(initial_z.detach().clone().to(device))
    calibration = torch.nn.Parameter(initial_calibration.detach().clone().to(device))
    optimizer = torch.optim.Adam([z, calibration], lr=lr)
    history: list[dict[str, float]] = []
    arch_x, structured_x, target_x, budget_x, feasible_x = _rows_to_tensors(support_rows, device)
    for step in range(steps):
        z_batch = z.expand(arch_x.shape[0], -1)
        calib_batch = calibration.expand(arch_x.shape[0], -1)
        static_batch = None
        if static_context is not None:
            static_batch = static_context if static_context.ndim == 2 else static_context.unsqueeze(0)
            if static_batch.shape[0] == 1 and arch_x.shape[0] > 1:
                static_batch = static_batch.expand(arch_x.shape[0], -1)
        response = response_decoder(z_batch)
        cost_outputs = cost_predictor(arch_x, structured_x, z_batch, response, calib_batch, static_context=static_batch)
        feasible_pred = feasibility_head(arch_x, z_batch, cost_outputs, budget_x, calib_batch)
        loss = (
            F.l1_loss(cost_outputs["latency_ms"], target_x[:, 0:1])
            + F.l1_loss(cost_outputs["peak_sram_bytes"], target_x[:, 1:2])
            + F.l1_loss(cost_outputs["flash_bytes"], target_x[:, 2:3])
            + 0.25 * F.binary_cross_entropy(feasible_pred, feasible_x)
            + 1e-3 * calibration.pow(2).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"step": float(step), "loss": float(loss.item())})
    return z.detach(), calibration.detach(), history


def adapt_device_embedding(
    response_decoder,
    cost_predictor,
    initial_z: torch.Tensor,
    support_rows: list[dict[str, Any]],
    steps: int = 25,
    lr: float = 1e-2,
    device: str = "cpu",
    static_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    class _FallbackFeasibility(torch.nn.Module):
        def forward(self, arch_x, z, cost_outputs, budget, calibration=None):
            ratios = (
                cost_outputs["latency_ms"] / budget[:, 0:1]
                + cost_outputs["peak_sram_bytes"] / budget[:, 1:2]
                + cost_outputs["flash_bytes"] / budget[:, 2:3]
            )
            return torch.sigmoid(-(ratios - 3.0))

    z, _, history = adapt_device_state(
        response_decoder=response_decoder,
        cost_predictor=cost_predictor,
        feasibility_head=_FallbackFeasibility(),
        initial_z=initial_z,
        support_rows=support_rows,
        steps=steps,
        lr=lr,
        device=device,
        static_context=static_context,
    )
    return z, history
