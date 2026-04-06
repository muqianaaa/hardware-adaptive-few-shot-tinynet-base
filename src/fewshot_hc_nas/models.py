from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_space import ARCH_FEATURE_DIM, MBEXPAND_RATIO, STEM_CHANNELS, BLOCK_BASE_CHANNELS, make_divisible, block_out_channels, max_width
from .types import ArchitectureSpec, BlockSpec, NUM_SEARCHABLE_BLOCKS, OPS, QUANTS, WIDTHS, DEPTHS

CALIBRATION_DIM = 4
GENERATOR_GROUP_DIMS = (len(OPS), len(WIDTHS), len(DEPTHS), len(QUANTS))


def _max_stage_channels() -> list[int]:
    return [make_divisible(base * max_width()) for base in BLOCK_BASE_CHANNELS]


MAX_STAGE_CHANNELS = _max_stage_channels()


class DynamicConv2d(nn.Module):
    def __init__(self, max_in: int, max_out: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(max_out, max_in, kernel_size, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(max_out))

    def forward(self, x: torch.Tensor, out_channels: int, stride: int | None = None) -> torch.Tensor:
        in_channels = x.shape[1]
        return F.conv2d(
            x,
            self.weight[:out_channels, :in_channels],
            self.bias[:out_channels],
            stride=self.stride if stride is None else stride,
            padding=self.padding,
        )


class DynamicPointwiseConv(nn.Module):
    def __init__(self, max_in: int, max_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_out, max_in, 1, 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(max_out))

    def forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        in_channels = x.shape[1]
        return F.conv2d(x, self.weight[:out_channels, :in_channels], self.bias[:out_channels])


class DynamicDepthwiseConv(nn.Module):
    def __init__(self, max_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(max_channels, 1, kernel_size, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(max_channels))

    def forward(self, x: torch.Tensor, stride: int | None = None) -> torch.Tensor:
        channels = x.shape[1]
        return F.conv2d(
            x,
            self.weight[:channels],
            self.bias[:channels],
            stride=self.stride if stride is None else stride,
            padding=self.padding,
            groups=channels,
        )


class StandardBlock(nn.Module):
    def __init__(self, max_in: int, max_out: int):
        super().__init__()
        self.conv = DynamicConv2d(max_in=max_in, max_out=max_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, out_channels: int, stride: int) -> torch.Tensor:
        return F.relu(self.conv(x, out_channels=out_channels, stride=stride), inplace=False)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, max_in: int, max_out: int):
        super().__init__()
        max_channels = max(max_in, max_out)
        self.dw = DynamicDepthwiseConv(max_channels=max_channels, kernel_size=3, stride=1, padding=1)
        self.pw = DynamicPointwiseConv(max_in=max_channels, max_out=max_out)

    def forward(self, x: torch.Tensor, out_channels: int, stride: int) -> torch.Tensor:
        x = F.relu(self.dw(x, stride=stride), inplace=False)
        x = F.relu(self.pw(x, out_channels=out_channels), inplace=False)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, max_in: int, max_out: int):
        super().__init__()
        hidden = max(max_in, max_out) * MBEXPAND_RATIO
        self.expand = DynamicPointwiseConv(max_in=max_in, max_out=hidden)
        self.dw = DynamicDepthwiseConv(max_channels=hidden, kernel_size=3, stride=1, padding=1)
        self.project = DynamicPointwiseConv(max_in=hidden, max_out=max_out)

    def forward(self, x: torch.Tensor, out_channels: int, stride: int) -> torch.Tensor:
        hidden = max(x.shape[1], out_channels) * MBEXPAND_RATIO
        x = F.relu(self.expand(x, out_channels=hidden), inplace=False)
        x = F.relu(self.dw(x, stride=stride), inplace=False)
        x = F.relu(self.project(x, out_channels=out_channels), inplace=False)
        return x


class TinyNet(nn.Module):
    def __init__(self, architecture: ArchitectureSpec, num_classes: int = 10):
        super().__init__()
        self.architecture = architecture
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        in_channels = STEM_CHANNELS
        for idx, block in enumerate(architecture.blocks):
            out_channels = block_out_channels(idx, block.width)
            block_layers = []
            for depth_idx in range(block.depth):
                stride = 1 if depth_idx > 0 else (1 if idx in {0, 2, 4} else 2)
                if block.op == "std3x3":
                    seq = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.ReLU(inplace=True),
                    )
                elif block.op == "dw_sep":
                    seq = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        nn.ReLU(inplace=True),
                    )
                elif block.op == "mbconv":
                    hidden = max(in_channels, out_channels) * MBEXPAND_RATIO
                    seq = nn.Sequential(
                        nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                        nn.ReLU(inplace=True),
                    )
                else:
                    raise ValueError(block.op)
                block_layers.append(seq)
                in_channels = out_channels
            self.blocks.append(nn.Sequential(*block_layers))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.head(torch.flatten(x, 1))


class Supernet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.stage_blocks = nn.ModuleList()
        prev_max = STEM_CHANNELS
        for max_out in MAX_STAGE_CHANNELS:
            first_bank = nn.ModuleDict(
                {
                    "std3x3": StandardBlock(prev_max, max_out),
                    "dw_sep": DepthwiseSeparableBlock(prev_max, max_out),
                    "mbconv": MBConvBlock(prev_max, max_out),
                }
            )
            repeat_bank = nn.ModuleDict(
                {
                    "std3x3": StandardBlock(max_out, max_out),
                    "dw_sep": DepthwiseSeparableBlock(max_out, max_out),
                    "mbconv": MBConvBlock(max_out, max_out),
                }
            )
            self.stage_blocks.append(nn.ModuleList([first_bank, repeat_bank]))
            prev_max = max_out
        self.head_weight = nn.Parameter(torch.randn(num_classes, MAX_STAGE_CHANNELS[-1]) * 0.02)
        self.head_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor, architecture: ArchitectureSpec) -> torch.Tensor:
        x = self.stem(x)
        for block_idx, block in enumerate(architecture.blocks):
            out_channels = block_out_channels(block_idx, block.width)
            first_block = self.stage_blocks[block_idx][0][block.op]
            repeat_block = self.stage_blocks[block_idx][1][block.op]
            stride = 1 if block_idx in {0, 2, 4} else 2
            x = first_block(x, out_channels=out_channels, stride=stride)
            for _ in range(block.depth - 1):
                x = repeat_block(x, out_channels=out_channels, stride=1)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)
        return F.linear(x, self.head_weight[:, : x.shape[1]], self.head_bias)


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0, final_activation: bool = False):
        super().__init__()
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if not is_last or final_activation:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AccuracyPredictor(nn.Module):
    def __init__(self, input_dim: int = ARCH_FEATURE_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP([input_dim, hidden_dim, hidden_dim, 1], dropout=0.08)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.phi = MLP([input_dim, hidden_dim, hidden_dim], dropout=0.0)
        self.rho = MLP([hidden_dim, hidden_dim, output_dim], dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.phi(x)
        return self.rho(features.mean(dim=1))


class HardwareEncoder(nn.Module):
    def __init__(self, static_dim: int = 20, probe_dim: int = 7, reference_dim: int = 6, hidden_dim: int = 160, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.static_mlp = MLP([static_dim, hidden_dim, hidden_dim], dropout=0.04)
        self.probe_encoder = DeepSetEncoder(probe_dim, hidden_dim, hidden_dim)
        self.reference_encoder = DeepSetEncoder(reference_dim, hidden_dim, hidden_dim)
        self.fusion = MLP([hidden_dim * 3, hidden_dim, embed_dim], dropout=0.04)

    def forward(self, static_x: torch.Tensor, probe_x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        static_feat = self.static_mlp(static_x)
        probe_feat = self.probe_encoder(probe_x)
        ref_feat = self.reference_encoder(ref_x)
        return self.fusion(torch.cat([static_feat, probe_feat, ref_feat], dim=-1))


class ResponseDecoder(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 160):
        super().__init__()
        output_dim = len(OPS) * len(QUANTS) + 3
        self.body = MLP([embed_dim, hidden_dim, hidden_dim, output_dim], dropout=0.04)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.body(z)
        gamma_raw = raw[:, : len(OPS) * len(QUANTS)]
        extras = raw[:, len(OPS) * len(QUANTS) :]
        gamma = F.softplus(gamma_raw).view(-1, len(OPS), len(QUANTS)) + 1e-6
        beta_mem = F.softplus(extras[:, 0:1]) + 1e-6
        rho_launch = F.softplus(extras[:, 1:2]) + 1e-3
        rho_copy = F.softplus(extras[:, 2:3]) + 1e-3
        return {"gamma": gamma, "beta_mem": beta_mem, "rho_launch": rho_launch, "rho_copy": rho_copy}


class StructuredCostPredictor(nn.Module):
    def __init__(self, arch_dim: int = ARCH_FEATURE_DIM, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.residual = MLP([arch_dim + embed_dim, hidden_dim, hidden_dim, 3], dropout=0.04)

    @staticmethod
    def _decode_static_context(static_context: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if static_context is None:
            freq_scale = torch.ones((batch_size, 1), device=device, dtype=dtype)
            bus_width = torch.full((batch_size, 1), 24.0, device=device, dtype=dtype)
            cache_penalty = torch.ones((batch_size, 1), device=device, dtype=dtype)
            return freq_scale, bus_width, cache_penalty

        if static_context.ndim == 1:
            static_context = static_context.unsqueeze(0)
        if static_context.shape[0] == 1 and batch_size > 1:
            static_context = static_context.expand(batch_size, -1)

        freq_mhz = torch.clamp(torch.exp(static_context[:, 2:3]) - 1.0, min=40.0)
        cache_kb = torch.clamp(torch.exp(static_context[:, 5:6]) - 1.0, min=0.0)
        bus_width = torch.clamp(static_context[:, 6:7] * 64.0, min=8.0)
        freq_scale = 200.0 / freq_mhz
        cache_penalty = torch.where(cache_kb < 12.0, torch.full_like(cache_kb, 1.08), torch.ones_like(cache_kb))
        return freq_scale, bus_width, cache_penalty

    def forward(
        self,
        arch_x: torch.Tensor,
        structured_x: torch.Tensor,
        z: torch.Tensor,
        response: dict[str, torch.Tensor],
        calibration: torch.Tensor | None = None,
        static_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        op_ids = structured_x[:, :, 0].long()
        quant_bits = structured_x[:, :, 3].long()
        quant_indices = torch.zeros_like(quant_bits)
        quant_indices = torch.where(quant_bits == 4, torch.ones_like(quant_indices), quant_indices)
        quant_indices = torch.where(quant_bits == 8, torch.full_like(quant_indices, 2), quant_indices)
        batch_index = torch.arange(structured_x.shape[0], device=structured_x.device)
        gamma_rows = [
            response["gamma"][batch_index, op_ids[:, block_idx], quant_indices[:, block_idx]]
            for block_idx in range(structured_x.shape[1])
        ]
        gamma = torch.stack(gamma_rows, dim=1)

        macs = structured_x[:, :, 4]
        params = structured_x[:, :, 5]
        act_in = structured_x[:, :, 6]
        act_out = structured_x[:, :, 7]
        bytes_moved = structured_x[:, :, 8]
        workspace = structured_x[:, :, 9]
        code = structured_x[:, :, 10]
        depth = structured_x[:, :, 2]
        freq_scale, bus_width, cache_penalty = self._decode_static_context(static_context, arch_x.shape[0], arch_x.device, arch_x.dtype)

        latency_formula = (
            gamma * macs * freq_scale * 5.0
            + response["beta_mem"] * bytes_moved / bus_width
            + response["rho_launch"] * depth
            + response["rho_copy"] * (act_out * quant_bits / 8.0) / 4096.0
        ).sum(dim=1, keepdim=True)
        sram_formula = ((act_in + act_out) * quant_bits / 8.0 + workspace).amax(dim=1, keepdim=True) * cache_penalty
        flash_formula = (params * quant_bits / 8.0 + code).sum(dim=1, keepdim=True)

        residual = self.residual(torch.cat([arch_x, z], dim=-1))
        if calibration is None:
            calibration = torch.zeros(arch_x.shape[0], CALIBRATION_DIM, device=arch_x.device, dtype=arch_x.dtype)
        latency = F.softplus(latency_formula + residual[:, 0:1] + calibration[:, 0:1])
        sram = F.softplus(sram_formula + residual[:, 1:2] + calibration[:, 1:2])
        flash = F.softplus(flash_formula + residual[:, 2:3] + calibration[:, 2:3])
        return {"latency_ms": latency, "peak_sram_bytes": sram, "flash_bytes": flash}


class FeasibilityHead(nn.Module):
    def __init__(self, arch_dim: int = ARCH_FEATURE_DIM, embed_dim: int = 64, hidden_dim: int = 192):
        super().__init__()
        self.net = MLP([arch_dim + embed_dim + 3, hidden_dim, hidden_dim, 1], dropout=0.04)

    def forward(
        self,
        arch_x: torch.Tensor,
        z: torch.Tensor,
        cost_outputs: dict[str, torch.Tensor],
        budget: torch.Tensor,
        calibration: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ratios = torch.cat(
            [
                cost_outputs["latency_ms"] / budget[:, 0:1],
                cost_outputs["peak_sram_bytes"] / budget[:, 1:2],
                cost_outputs["flash_bytes"] / budget[:, 2:3],
            ],
            dim=1,
        )
        raw = self.net(torch.cat([arch_x, z, ratios], dim=1))
        if calibration is not None:
            raw = raw + calibration[:, 3:4]
        return torch.sigmoid(raw)


class BlackBoxCostPredictor(nn.Module):
    def __init__(self, arch_dim: int = ARCH_FEATURE_DIM, static_dim: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP([arch_dim + static_dim, hidden_dim, hidden_dim, 4], dropout=0.08)

    def forward(self, arch_x: torch.Tensor, static_x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(torch.cat([arch_x, static_x], dim=-1))
        return {
            "latency_ms": F.softplus(raw[:, 0:1]),
            "peak_sram_bytes": F.softplus(raw[:, 1:2]),
            "flash_bytes": F.softplus(raw[:, 2:3]),
            "feasible_prob": torch.sigmoid(raw[:, 3:4]),
        }


class MetaArchitectureGenerator(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 160):
        super().__init__()
        total_dim = NUM_SEARCHABLE_BLOCKS * sum(GENERATOR_GROUP_DIMS)
        self.body = MLP([embed_dim + 3, hidden_dim, hidden_dim, total_dim], dropout=0.04)

    def forward(self, z: torch.Tensor, budget: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.body(torch.cat([z, budget], dim=1))
        cursor = 0
        groups: dict[str, list[torch.Tensor]] = {"op_logits": [], "width_logits": [], "depth_logits": [], "quant_logits": []}
        for _ in range(NUM_SEARCHABLE_BLOCKS):
            for name, dim in zip(groups.keys(), GENERATOR_GROUP_DIMS):
                groups[name].append(raw[:, cursor : cursor + dim])
                cursor += dim
        outputs = {name: torch.stack(values, dim=1) for name, values in groups.items()}
        outputs.update({name.replace("_logits", "_prob"): torch.softmax(tensor, dim=-1) for name, tensor in outputs.items()})
        return outputs

    def decode_architecture(self, z: torch.Tensor, budget: torch.Tensor, name: str = "meta_generated") -> ArchitectureSpec:
        outputs = self.forward(z, budget)
        return decode_meta_architecture(outputs, name=name)


def decode_meta_architecture(outputs: dict[str, torch.Tensor], name: str = "meta_generated") -> ArchitectureSpec:
    op_idx = outputs["op_prob"][0].argmax(dim=-1).tolist()
    width_idx = outputs["width_prob"][0].argmax(dim=-1).tolist()
    depth_idx = outputs["depth_prob"][0].argmax(dim=-1).tolist()
    quant_idx = outputs["quant_prob"][0].argmax(dim=-1).tolist()
    blocks = []
    for block_index in range(NUM_SEARCHABLE_BLOCKS):
        blocks.append(
            BlockSpec(
                op=OPS[int(op_idx[block_index])],
                width=WIDTHS[int(width_idx[block_index])],
                depth=DEPTHS[int(depth_idx[block_index])],
                quant=QUANTS[int(quant_idx[block_index])],
            )
        )
    return ArchitectureSpec(blocks=tuple(blocks), name=name)


# Backward-compatible alias for existing checkpoints and imports.
BudgetConditionedGenerator = MetaArchitectureGenerator


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def bundle_state_dict(bundle: dict[str, nn.Module]) -> dict[str, Any]:
    return {name: module.state_dict() for name, module in bundle.items()}
