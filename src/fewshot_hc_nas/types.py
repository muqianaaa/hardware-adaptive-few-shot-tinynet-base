from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

OPS = ("std3x3", "dw_sep", "mbconv")
WIDTHS = (0.375, 0.5, 0.75, 1.0, 1.25, 1.5)
DEPTHS = (1, 2)
QUANTS = (2, 4, 8)
RUNTIME_TYPES = ("cmsis_nn", "tflm", "custom_runtime", "vendor_runtime")
NUM_SEARCHABLE_BLOCKS = 5


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class BlockSpec:
    op: str
    width: float
    depth: int
    quant: int

    def __post_init__(self) -> None:
        _require(self.op in OPS, f"Unsupported op: {self.op}")
        _require(self.width in WIDTHS, f"Unsupported width: {self.width}")
        _require(self.depth in DEPTHS, f"Unsupported depth: {self.depth}")
        _require(self.quant in QUANTS, f"Unsupported quant: {self.quant}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, init=False)
class ArchitectureSpec:
    blocks: tuple[BlockSpec, ...]
    name: str = "candidate"

    def __init__(self, blocks: tuple[BlockSpec, ...] | None = None, name: str = "candidate", stages: tuple[BlockSpec, ...] | None = None):
        resolved_blocks = blocks if blocks is not None else stages
        _require(resolved_blocks is not None, "ArchitectureSpec requires blocks or stages.")
        resolved_blocks = tuple(resolved_blocks)
        _require(len(resolved_blocks) == NUM_SEARCHABLE_BLOCKS, f"ArchitectureSpec expects exactly {NUM_SEARCHABLE_BLOCKS} blocks.")
        object.__setattr__(self, "blocks", resolved_blocks)
        object.__setattr__(self, "name", name)

    @property
    def stages(self) -> tuple["BlockSpec", ...]:
        return self.blocks

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "blocks": [block.to_dict() for block in self.blocks]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchitectureSpec":
        block_payloads = list(payload.get("blocks", payload.get("stages", [])))
        if len(block_payloads) == NUM_SEARCHABLE_BLOCKS - 1:
            block_payloads.append({"op": "std3x3", "width": 1.0, "depth": 1, "quant": 8})
        blocks = tuple(BlockSpec(**block) for block in block_payloads)
        return cls(blocks=blocks, name=payload.get("name", "candidate"))

    @classmethod
    def from_compact_repr(cls, value: str, name: str = "candidate") -> "ArchitectureSpec":
        parts = [part.strip() for part in str(value).split("|") if part.strip()]
        _require(len(parts) == NUM_SEARCHABLE_BLOCKS, f"ArchitectureSpec compact repr expects {NUM_SEARCHABLE_BLOCKS} blocks.")
        blocks = []
        for part in parts:
            op, width, depth, quant = part.split(":")
            blocks.append(BlockSpec(op=op, width=float(width), depth=int(depth), quant=int(quant)))
        return cls(blocks=tuple(blocks), name=name)

    def compact_repr(self) -> str:
        return "|".join(f"{block.op}:{block.width}:{block.depth}:{block.quant}" for block in self.blocks)


@dataclass(frozen=True)
class BlockMetrics:
    block_index: int
    in_channels: int
    out_channels: int
    in_hw: int
    out_hw: int
    macs: float
    params: float
    act_in: float
    act_out: float
    bytes_moved: float
    workspace: float
    code_footprint: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stage_index(self) -> int:
        return self.block_index


@dataclass(frozen=True)
class HardwareStaticSpec:
    name: str
    family: str
    sram_bytes: int
    flash_bytes: int
    freq_mhz: float
    dsp: float
    simd: float
    cache_kb: float
    bus_width: float
    kernel_int8: float
    kernel_int4: float
    kernel_int2: float
    runtime_type: str
    ccm_bytes: int = 0
    fpu: float = 0.0
    dma: float = 0.0
    art_accelerator: float = 0.0
    ram_bank_count: float = 1.0
    unaligned_access_efficiency: float = 0.5

    def __post_init__(self) -> None:
        _require(self.runtime_type in RUNTIME_TYPES, f"Unsupported runtime_type: {self.runtime_type}")

    def to_feature_list(self) -> list[float]:
        runtime_onehot = [1.0 if self.runtime_type == rt else 0.0 for rt in RUNTIME_TYPES]
        return [
            float(self.sram_bytes),
            float(self.flash_bytes),
            float(self.freq_mhz),
            float(self.dsp),
            float(self.simd),
            float(self.cache_kb),
            float(self.bus_width),
            float(self.kernel_int8),
            float(self.kernel_int4),
            float(self.kernel_int2),
            float(self.ccm_bytes),
            float(self.fpu),
            float(self.dma),
            float(self.art_accelerator),
            float(self.ram_bank_count),
            float(self.unaligned_access_efficiency),
            *runtime_onehot,
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeMeasurement:
    probe_id: str
    op: str
    quant: int
    input_shape: tuple[int, ...]
    latency_ms: float
    latency_per_mac: float
    latency_per_byte: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_shape"] = list(self.input_shape)
        return data


@dataclass(frozen=True)
class ReferenceMeasurement:
    name: str
    architecture: ArchitectureSpec
    latency_ms: float
    peak_sram_bytes: float
    flash_bytes: float
    accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture.to_dict(),
            "latency_ms": self.latency_ms,
            "peak_sram_bytes": self.peak_sram_bytes,
            "flash_bytes": self.flash_bytes,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class HardwareResponseCoefficients:
    gamma: dict[str, dict[int, float]]
    beta_mem: float
    rho_launch: float
    rho_copy: float

    def flatten(self) -> list[float]:
        values: list[float] = []
        for op in OPS:
            for quant in QUANTS:
                values.append(float(self.gamma[op][quant]))
        values.extend([float(self.beta_mem), float(self.rho_launch), float(self.rho_copy)])
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma": {op: {str(k): v for k, v in inner.items()} for op, inner in self.gamma.items()},
            "beta_mem": self.beta_mem,
            "rho_launch": self.rho_launch,
            "rho_copy": self.rho_copy,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HardwareResponseCoefficients":
        gamma = {op: {int(k): float(v) for k, v in inner.items()} for op, inner in payload["gamma"].items()}
        return cls(
            gamma=gamma,
            beta_mem=float(payload["beta_mem"]),
            rho_launch=float(payload["rho_launch"]),
            rho_copy=float(payload["rho_copy"]),
        )


@dataclass(frozen=True)
class BudgetSpec:
    t_max_ms: float
    m_max_bytes: float
    f_max_bytes: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostPrediction:
    latency_ms: float
    peak_sram_bytes: float
    flash_bytes: float
    feasible_prob: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceRecord:
    static: HardwareStaticSpec
    response: HardwareResponseCoefficients
    probes: tuple[ProbeMeasurement, ...]
    references: tuple[ReferenceMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "static": self.static.to_dict(),
            "response": self.response.to_dict(),
            "probes": [probe.to_dict() for probe in self.probes],
            "references": [ref.to_dict() for ref in self.references],
        }


@dataclass(frozen=True)
class DeviceState:
    embedding: tuple[float, ...]
    calibration: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"embedding": list(self.embedding), "calibration": list(self.calibration)}


@dataclass
class ExperimentArtifact:
    name: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backward-compatible aliases for pre-refactor modules.
StageSpec = BlockSpec
StageMetrics = BlockMetrics
