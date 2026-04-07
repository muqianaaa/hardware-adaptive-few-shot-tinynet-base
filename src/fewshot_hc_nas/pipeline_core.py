from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adaptation import adapt_device_state
from .backends import CSVReplayBackend, CommandBackend, HardwareBackend, HybridBackend, SyntheticBackend
from .datasets import ArchitectureAccuracyDataset, FewShotTaskDataset, build_image_datasets, dataset_num_classes
from .hardware import (
    DEVICE_FAMILIES,
    REFERENCE_ARCHITECTURES,
    build_arch_measurement_table,
    create_device_record,
    export_device_directory,
    make_standard_baseline,
    sample_budget,
    synthetic_architecture_accuracy,
)
from .io import append_markdown_log, ensure_dir, read_dataframe, read_json, read_jsonl, read_yaml, write_dataframe, write_json, write_jsonl
from .models import (
    CALIBRATION_DIM,
    ARCH_FEATURE_DIM,
    AccuracyPredictor,
    BlackBoxCostPredictor,
    BudgetConditionedGenerator,
    FeasibilityHead,
    HardwareEncoder,
    MetaArchitectureGenerator,
    ResponseDecoder,
    StructuredCostPredictor,
    Supernet,
    bundle_state_dict,
)
from .paper_viz import build_method_overview_diagram, format_table_for_paper, plot_ablation_panels, plot_board_ablation_panels, plot_board_improvement, plot_board_method_panels, plot_kshot_curve, plot_main_result_panels
from .search import build_heuristic_prior, evolutionary_search, generate_architecture_direct, local_refine_search, predict_candidate, random_search, sample_from_prior, score_prediction
from .search_space import ARCH_TOKEN_DIM, crossover_architectures, default_architecture, encode_architecture, mutate_architecture, sample_architecture, structured_architecture_tensor
from .types import ArchitectureSpec, BudgetSpec, DeviceRecord, HardwareResponseCoefficients, HardwareStaticSpec, ProbeMeasurement, ReferenceMeasurement, BlockSpec, OPS, WIDTHS, DEPTHS, QUANTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str | None = None) -> str:
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _artifact_roots(root: str | Path, dataset_name: str) -> tuple[Path, Path]:
    root = Path(root)
    return ensure_dir(root / "data" / "generated" / dataset_name), ensure_dir(root / "data" / "checkpoints" / dataset_name)


def _log(stage: str, lines: list[str], root: str | Path = ".") -> None:
    root = Path(root)
    append_markdown_log(root / "WORK_LOG.md", stage, lines)
    append_markdown_log(root / "research-log.md", stage, lines)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    def _fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in frame.columns) + " |")
    return "\n".join(lines)


def _mobilenet_like_architecture() -> ArchitectureSpec:
    return ArchitectureSpec(
        name="轻量深度可分离基线",
        blocks=(
            BlockSpec("dw_sep", 0.75, 2, 8),
            BlockSpec("dw_sep", 1.0, 2, 8),
            BlockSpec("mbconv", 1.0, 2, 8),
            BlockSpec("dw_sep", 1.0, 1, 4),
            BlockSpec("mbconv", 1.0, 1, 4),
        ),
    )


def _board_baseline_architectures() -> dict[str, ArchitectureSpec]:
    return {
        "standard_cnn": make_standard_baseline(name="标准 TinyNet 基线"),
        "mobilenet_like": _mobilenet_like_architecture(),
        "shallow_wide": REFERENCE_ARCHITECTURES["宽浅参考网络"],
        "deep_narrow": REFERENCE_ARCHITECTURES["深窄参考网络"],
        "depthwise_heavy_mixed_precision": REFERENCE_ARCHITECTURES["混合精度深度可分离参考网络"],
    }


def _default_response_coefficients() -> HardwareResponseCoefficients:
    return HardwareResponseCoefficients(
        gamma={
            "std3x3": {2: 2.0e-6, 4: 1.7e-6, 8: 1.4e-6},
            "dw_sep": {2: 1.92e-6, 4: 1.62e-6, 8: 1.36e-6},
            "mbconv": {2: 2.22e-6, 4: 1.90e-6, 8: 1.62e-6},
        },
        beta_mem=2.8e-4,
        rho_launch=0.17,
        rho_copy=0.14,
    )


def _resolve_command_config(config: dict[str, Any], root: Path) -> dict[str, Any] | str:
    raw = config.get("command_backend", config.get("command_backend_config", {}))
    if isinstance(raw, (str, Path)):
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        return str(path)
    return dict(raw)


def _make_hardware_backend(config: dict[str, Any], root: str | Path = ".") -> HardwareBackend:
    root = Path(root)
    backend_mode = str(config.get("backend", "synthetic")).lower()
    if backend_mode == "csv_replay":
        return CSVReplayBackend()
    if backend_mode == "command":
        return CommandBackend(_resolve_command_config(config, root))
    if backend_mode == "hybrid_command_csv_replay":
        return HybridBackend(
            command_backend=CommandBackend(_resolve_command_config(config, root)),
            fallback_backend=CSVReplayBackend(),
            command_devices=config.get("command_devices", []),
            command_families=config.get("command_families", []),
        )
    return SyntheticBackend(noise_scale=float(config.get("measurement_noise_scale", 0.0)))


def _allowed_quants(config: dict[str, Any]) -> set[int]:
    raw = config.get("allowed_quants", [])
    return {int(item) for item in raw}


def _arch_supported_by_quants(arch: ArchitectureSpec, allowed_quants: set[int]) -> bool:
    if not allowed_quants:
        return True
    return all(block.quant in allowed_quants for block in arch.blocks)


def _filter_candidate_pool(candidates: list[dict[str, Any]], allowed_quants: set[int]) -> list[dict[str, Any]]:
    if not allowed_quants:
        return candidates
    filtered = [item for item in candidates if _arch_supported_by_quants(item["architecture"], allowed_quants)]
    return filtered or candidates


def _static_features(static: HardwareStaticSpec) -> np.ndarray:
    runtime_onehot = [1.0 if static.runtime_type == rt else 0.0 for rt in ("cmsis_nn", "tflm", "custom_runtime", "vendor_runtime")]
    return np.asarray(
        [
            np.log(static.sram_bytes + 1.0),
            np.log(static.flash_bytes + 1.0),
            np.log(static.freq_mhz + 1.0),
            static.dsp,
            static.simd,
            np.log(static.cache_kb + 1.0),
            static.bus_width / 64.0,
            static.kernel_int8,
            static.kernel_int4,
            static.kernel_int2,
            np.log(static.ccm_bytes + 1.0),
            static.fpu,
            static.dma,
            static.art_accelerator,
            static.ram_bank_count / 4.0,
            static.unaligned_access_efficiency,
            *runtime_onehot,
        ],
        dtype=np.float32,
    )


def _probe_tensor(probes: list[ProbeMeasurement]) -> np.ndarray:
    op_vocab = ("std3x3", "dw_sep", "mbconv", "fc", "move", "pool")
    rows = []
    for idx, probe in enumerate(probes):
        op_index = op_vocab.index(probe.op) if probe.op in op_vocab else len(op_vocab) - 1
        rows.append(
            [
                float(op_index) / max(len(op_vocab) - 1, 1),
                float(probe.quant) / 8.0,
                np.log(np.prod(probe.input_shape) + 1.0),
                np.log(probe.latency_ms + 1e-6),
                np.log(abs(probe.latency_per_mac) + 1e-9),
                np.log(abs(probe.latency_per_byte) + 1e-9),
                float(idx) / max(len(probes) - 1, 1),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _reference_tensor(references: list[ReferenceMeasurement]) -> np.ndarray:
    rows = []
    for ref in references:
        rows.append(
            [
                np.log(ref.latency_ms + 1e-6),
                np.log(ref.peak_sram_bytes + 1.0),
                np.log(ref.flash_bytes + 1.0),
                float(np.mean([stage.width for stage in ref.architecture.stages])),
                float(np.mean([stage.depth for stage in ref.architecture.stages])),
                float(np.mean([stage.quant for stage in ref.architecture.stages])) / 8.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _hardware_ablation_flags(config: dict[str, Any]) -> tuple[bool, bool, bool]:
    static_only = bool(config.get("static_only", False))
    disable_probes = static_only or bool(config.get("disable_probes", False))
    disable_refs = static_only or bool(config.get("disable_refs", False))
    disable_response = bool(config.get("disable_response_decoder", False))
    return disable_probes, disable_refs, disable_response


def _device_feature_arrays(
    device_record: DeviceRecord,
    disable_probes: bool = False,
    disable_refs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static_arr = _static_features(device_record.static)
    probe_arr = _probe_tensor(list(device_record.probes))
    ref_arr = _reference_tensor(list(device_record.references))
    if disable_probes:
        probe_arr = np.zeros_like(probe_arr)
    if disable_refs:
        ref_arr = np.zeros_like(ref_arr)
    return static_arr, probe_arr, ref_arr


def _budget_tensor(budget: BudgetSpec, device: str, batch_size: int = 1) -> torch.Tensor:
    return torch.as_tensor([[budget.t_max_ms, budget.m_max_bytes, budget.f_max_bytes]] * batch_size, dtype=torch.float32, device=device)


def _normalize_static_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    runtime_aliases = {
        "onnxrt_edge": "custom_runtime",
        "onnx_runtime": "custom_runtime",
        "tflite_micro": "tflm",
        "vendor_sdk": "vendor_runtime",
    }
    payload["runtime_type"] = runtime_aliases.get(payload.get("runtime_type"), payload.get("runtime_type", "custom_runtime"))
    payload.setdefault("ccm_bytes", 0)
    payload.setdefault("fpu", 0.0)
    payload.setdefault("dma", 0.0)
    payload.setdefault("art_accelerator", 0.0)
    payload.setdefault("ram_bank_count", 1.0)
    payload.setdefault("unaligned_access_efficiency", 0.5)
    return payload


def _manifest_item_name(item: dict[str, Any]) -> str:
    return str(item.get("device", item.get("name")))


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]]).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:8], 16)


DEFAULT_BUDGET_PROFILES: dict[str, dict[str, Any]] = {
    "low_memory_mcu_000": {"latency_scale": 0.88, "sram_scale": 0.62, "flash_scale": 0.60, "dominant_dims": ("M",)},
    "memory_bottleneck_mcu_000": {"latency_scale": 0.92, "sram_scale": 0.56, "flash_scale": 0.72, "dominant_dims": ("M", "T")},
    "depthwise_unfriendly_mcu_000": {"latency_scale": 0.68, "sram_scale": 0.90, "flash_scale": 0.80, "dominant_dims": ("T",)},
    "low_bit_friendly_mcu_001": {"latency_scale": 0.84, "sram_scale": 0.82, "flash_scale": 0.40, "dominant_dims": ("F",)},
    "stm32f405rgt6_000": {"latency_scale": 0.30, "sram_scale": 0.82, "flash_scale": 0.48, "dominant_dims": ("T", "F")},
}

DEFAULT_FAMILY_BUDGET_PROFILES: dict[str, dict[str, Any]] = {
    "low_memory_mcu": {"latency_scale": 0.88, "sram_scale": 0.62, "flash_scale": 0.60, "dominant_dims": ("M",)},
    "memory_bottleneck_mcu": {"latency_scale": 0.92, "sram_scale": 0.56, "flash_scale": 0.72, "dominant_dims": ("M", "T")},
    "depthwise_unfriendly_mcu": {"latency_scale": 0.68, "sram_scale": 0.90, "flash_scale": 0.80, "dominant_dims": ("T",)},
    "low_bit_friendly_mcu": {"latency_scale": 0.84, "sram_scale": 0.82, "flash_scale": 0.40, "dominant_dims": ("F",)},
    "stm32f405rgt6_real": {"latency_scale": 0.30, "sram_scale": 0.82, "flash_scale": 0.48, "dominant_dims": ("T", "F")},
}


def _budget_profile_for_device(device_name: str, family: str, config: dict[str, Any]) -> dict[str, Any]:
    by_name = {str(key): value for key, value in config.get("budget_profiles", {}).items()}
    by_family = {str(key): value for key, value in config.get("family_budget_profiles", {}).items()}
    profile = dict(DEFAULT_FAMILY_BUDGET_PROFILES.get(str(family), {}))
    profile.update(by_family.get(str(family), {}))
    profile.update(DEFAULT_BUDGET_PROFILES.get(str(device_name), {}))
    profile.update(by_name.get(str(device_name), {}))
    if not profile:
        scale = float(config.get("standard_budget_scale", 1.0))
        profile = {"latency_scale": scale, "sram_scale": scale, "flash_scale": scale, "dominant_dims": ("T", "M", "F")}
    dominant_dims = tuple(str(dim).upper() for dim in profile.get("dominant_dims", ("T", "M", "F")))
    return {
        "latency_scale": float(profile.get("latency_scale", profile.get("scale", 1.0))),
        "sram_scale": float(profile.get("sram_scale", profile.get("scale", 1.0))),
        "flash_scale": float(profile.get("flash_scale", profile.get("scale", 1.0))),
        "dominant_dims": dominant_dims or ("T", "M", "F"),
    }


def _budget_from_standard_measurement(
    standard_row: dict[str, Any],
    device_name: str,
    family: str,
    config: dict[str, Any],
) -> tuple[BudgetSpec, dict[str, Any]]:
    profile = _budget_profile_for_device(device_name, family, config)
    budget = BudgetSpec(
        t_max_ms=float(standard_row["latency_ms"]) * profile["latency_scale"],
        m_max_bytes=float(standard_row["peak_sram_bytes"]) * profile["sram_scale"],
        f_max_bytes=float(standard_row["flash_bytes"]) * profile["flash_scale"],
    )
    return budget, profile


def _over_budget_dims(row: dict[str, Any], budget: BudgetSpec) -> str:
    dims: list[str] = []
    if float(row["latency_ms"]) > budget.t_max_ms:
        dims.append("T")
    if float(row["peak_sram_bytes"]) > budget.m_max_bytes:
        dims.append("M")
    if float(row["flash_bytes"]) > budget.f_max_bytes:
        dims.append("F")
    return "".join(dims) or "none"


def _dominant_margin(row: dict[str, Any], budget: BudgetSpec, dominant_dims: tuple[str, ...]) -> tuple[float, float, float]:
    ratios = {
        "T": float(row["latency_ms"]) / max(budget.t_max_ms, 1.0),
        "M": float(row["peak_sram_bytes"]) / max(budget.m_max_bytes, 1.0),
        "F": float(row["flash_bytes"]) / max(budget.f_max_bytes, 1.0),
    }
    dominant = [ratios.get(dim, 0.0) for dim in dominant_dims]
    return tuple(dominant + [float(row["accuracy"]), float(row.get("measured_score", row.get("pred_score", 0.0)))])


def _load_device_record(device_dir: str | Path) -> DeviceRecord:
    device_dir = Path(device_dir)
    static = HardwareStaticSpec(**_normalize_static_payload(read_json(device_dir / "hardware_static.json")))
    response_path = device_dir / "hardware_response.json"
    response = HardwareResponseCoefficients.from_dict(read_json(response_path)) if response_path.exists() else _default_response_coefficients()
    probes = [
        ProbeMeasurement(
            probe_id=row["probe_id"],
            op=row["op"],
            quant=int(row["quant"]),
            input_shape=tuple(row["input_shape"]),
            latency_ms=float(row["latency_ms"]),
            latency_per_mac=float(
                row.get(
                    "latency_per_mac",
                    float(row["latency_ms"]) / max(float(row.get("macs", 1.0)), 1.0),
                )
            ),
            latency_per_byte=float(
                row.get(
                    "latency_per_byte",
                    float(row["latency_ms"]) / max(float(row.get("bytes", 1.0)), 1.0),
                )
            ),
        )
        for row in read_jsonl(device_dir / "probe_results.jsonl")
    ]
    references = [ReferenceMeasurement(name=row["name"], architecture=ArchitectureSpec.from_dict(row["architecture"]), latency_ms=float(row["latency_ms"]), peak_sram_bytes=float(row["peak_sram_bytes"]), flash_bytes=float(row["flash_bytes"]), accuracy=None if row.get("accuracy") is None else float(row["accuracy"])) for row in read_jsonl(device_dir / "reference_results.jsonl")]
    return DeviceRecord(static=static, response=response, probes=tuple(probes), references=tuple(references))


def _load_base_rows_for_device(
    device_dir: str | Path,
    device_name: str,
    measurements_by_device: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    device_dir = Path(device_dir)
    replay_path = device_dir / "arch_measurements.jsonl"
    if replay_path.exists():
        return read_jsonl(replay_path)
    return measurements_by_device.get(device_name, [])


def _row_architecture(row: dict[str, Any]) -> ArchitectureSpec:
    payload = row["architecture_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ArchitectureSpec.from_dict(payload)


def _rows_to_arch_batch(rows: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch_x, struct_x, target_x = [], [], []
    for row in rows:
        arch = _row_architecture(row)
        arch_x.append(encode_architecture(arch))
        struct_x.append(structured_architecture_tensor(arch))
        target_x.append([float(row["latency_ms"]), float(row["peak_sram_bytes"]), float(row["flash_bytes"])])
    return (
        torch.as_tensor(np.stack(arch_x), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(struct_x), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(target_x), dtype=torch.float32, device=device),
    )


def _stamp_budget(rows: list[dict[str, Any]], budget: BudgetSpec) -> list[dict[str, Any]]:
    return [{**row, "budget_t": budget.t_max_ms, "budget_m": budget.m_max_bytes, "budget_f": budget.f_max_bytes} for row in rows]


def _sample_support_query_rows(
    rows: list[dict[str, Any]],
    support_size: int,
    query_size: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = [rows[index] for index in rng.permutation(len(rows))]
    support_rows = shuffled[:support_size]
    query_rows = shuffled[support_size : support_size + query_size]
    return support_rows, query_rows


def _budget_from_rows(rows: list[dict[str, Any]], rng: np.random.Generator) -> BudgetSpec:
    lat = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
    sram = np.asarray([float(row["peak_sram_bytes"]) for row in rows], dtype=np.float64)
    flash = np.asarray([float(row["flash_bytes"]) for row in rows], dtype=np.float64)
    return BudgetSpec(
        t_max_ms=float(np.quantile(lat, rng.uniform(0.36, 0.56)) * rng.uniform(1.02, 1.10)),
        m_max_bytes=float(np.quantile(sram, rng.uniform(0.44, 0.70)) * rng.uniform(1.02, 1.10)),
        f_max_bytes=float(np.quantile(flash, rng.uniform(0.44, 0.70)) * rng.uniform(1.02, 1.10)),
    )


def _normalize_ablation_mode(ablation_mode: str | None) -> str:
    mode = str(ablation_mode or "full")
    aliases = {
        "no_refinement": "no_local_refine",
        "no_local_refinement": "no_local_refine",
    }
    return aliases.get(mode, mode)


def _select_support_seed_architectures(
    measurement_frame: pd.DataFrame,
    limit: int,
    allowed_quants: set[int],
    seed: int,
) -> list[ArchitectureSpec]:
    baselines = _board_baseline_architectures()
    selected: list[ArchitectureSpec] = []
    seen: set[str] = set()
    for arch in baselines.values():
        if not _arch_supported_by_quants(arch, allowed_quants):
            continue
        selected.append(arch)
        seen.add(arch.compact_repr())
    rows = measurement_frame.drop_duplicates("arch_repr").to_dict(orient="records")
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows.sort(key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)
    for row in rows:
        arch = _row_architecture(row)
        if arch.compact_repr() in seen or not _arch_supported_by_quants(arch, allowed_quants):
            continue
        selected.append(arch)
        seen.add(arch.compact_repr())
        if len(selected) >= limit:
            break
    return selected[:limit]


def _register_device_in_manifest(manifest_path: Path, device_item: dict[str, Any]) -> None:
    manifest = read_json(manifest_path) if manifest_path.exists() else {"dataset_name": "synthetic_cifar10", "splits": {"test": []}}
    splits = manifest.setdefault("splits", {})
    items = splits.setdefault(str(device_item.get("split", "test")), [])
    device_name = str(device_item["device"])
    existing = {str(item["device"]): item for item in items}
    existing[device_name] = device_item
    splits[str(device_item.get("split", "test"))] = list(existing.values())
    write_json(manifest_path, manifest)


def _collect_real_device_support_rows(
    backend: HardwareBackend,
    config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    dataset_name = config.get("dataset_name", "synthetic_cifar10")
    generated_dir, _ = _artifact_roots(root, dataset_name)
    device_dir = Path(config["device_dir"])
    if not device_dir.is_absolute():
        device_dir = root / device_dir
    ensure_dir(device_dir)
    if bool(config.get("force_refresh_board_cache", False)):
        for filename in (
            "hardware_static.json",
            "probe_results.jsonl",
            "reference_results.jsonl",
            "arch_measurements.jsonl",
            "task_budgets.jsonl",
        ):
            path = device_dir / filename
            if path.exists():
                path.unlink()
    static = backend.load_static(device_dir)
    probes = backend.run_micro_probes(device_dir)
    references = backend.run_reference_nets(device_dir)
    allowed = _allowed_quants(config)
    support_seed_count = int(config.get("support_seed_count", max(int(config.get("support_size", 8)), 16)))
    measurement_frame = read_dataframe(generated_dir / "arch_measurements.parquet")
    accuracy_frame = read_dataframe(generated_dir / "accuracy_dataset.parquet")
    accuracy_lookup = dict(zip(accuracy_frame["arch_repr"], accuracy_frame["accuracy"]))
    support_architectures = _select_support_seed_architectures(measurement_frame, support_seed_count, allowed, int(config.get("seed", 0)))
    measured = backend.measure_candidates(device_dir, support_architectures)
    rows: list[dict[str, Any]] = []
    for arch, row in zip(support_architectures, measured):
        if row.get("unsupported"):
            continue
        rows.append(
            {
                "device_name": static.name,
                "family": static.family,
                "arch_name": arch.name,
                "arch_repr": arch.compact_repr(),
                "architecture_json": json.dumps(arch.to_dict(), ensure_ascii=False),
                "latency_ms": float(row["latency_ms"]),
                "peak_sram_bytes": float(row["peak_sram_bytes"]),
                "flash_bytes": float(row["flash_bytes"]),
                "accuracy": float(accuracy_lookup.get(arch.compact_repr(), synthetic_architecture_accuracy(arch, dataset_name=dataset_name, noise_scale=0.0))),
            }
        )
    write_jsonl(device_dir / "arch_measurements.jsonl", rows)
    if rows:
        rng = np.random.default_rng(int(config.get("seed", 0)))
        budgets_per_device = int(config.get("budgets_per_device", 4))
        write_jsonl(device_dir / "task_budgets.jsonl", [{"budget_id": idx, **_budget_from_rows(rows, rng).to_dict()} for idx in range(budgets_per_device)])
    if bool(config.get("register_in_manifest", True)):
        _register_device_in_manifest(
            generated_dir / "devices" / "manifest.json",
            {
                "device": static.name,
                "path": str(device_dir),
                "family": static.family,
                "split": str(config.get("manifest_split", "test")),
            },
        )
    return {
        "device_dir": str(device_dir),
        "hardware_static": str(device_dir / "hardware_static.json"),
        "probe_results": str(device_dir / "probe_results.jsonl"),
        "reference_results": str(device_dir / "reference_results.jsonl"),
        "arch_measurements": str(device_dir / "arch_measurements.jsonl"),
        "num_support_rows": len(rows),
        "device_name": static.name,
        "family": static.family,
        "num_probes": len(probes),
        "num_references": len(references),
    }


def _device_tensor_batch(
    devices: list[DeviceRecord],
    device: str,
    disable_probes: bool = False,
    disable_refs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.stack([_device_feature_arrays(item, disable_probes, disable_refs)[0] for item in devices]), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack([_device_feature_arrays(item, disable_probes, disable_refs)[1] for item in devices]), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack([_device_feature_arrays(item, disable_probes, disable_refs)[2] for item in devices]), dtype=torch.float32, device=device),
    )


def _device_bundle(
    device_record: DeviceRecord,
    device: str,
    disable_probes: bool = False,
    disable_refs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    static_arr, probe_arr, ref_arr = _device_feature_arrays(device_record, disable_probes, disable_refs)
    return (
        torch.as_tensor(static_arr, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(probe_arr, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(ref_arr, dtype=torch.float32, device=device).unsqueeze(0),
    )


def _load_accuracy_predictor(root: Path, device: str, dataset_name: str) -> AccuracyPredictor:
    _, checkpoint_dir = _artifact_roots(root, dataset_name)
    ckpt = checkpoint_dir / "accuracy_predictor.pt"
    payload = torch.load(ckpt, map_location=device) if ckpt.exists() else {"input_dim": ARCH_FEATURE_DIM}
    model = AccuracyPredictor(input_dim=int(payload.get("input_dim", ARCH_FEATURE_DIM))).to(device)
    if ckpt.exists():
        model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _load_model_bundle(bundle_path: Path, device: str) -> tuple[HardwareEncoder, ResponseDecoder, StructuredCostPredictor, FeasibilityHead, MetaArchitectureGenerator, BlackBoxCostPredictor, torch.Tensor, torch.Tensor]:
    payload = torch.load(bundle_path, map_location=device)
    encoder = HardwareEncoder().to(device)
    decoder = ResponseDecoder().to(device)
    cost_predictor = StructuredCostPredictor().to(device)
    feasibility_head = FeasibilityHead().to(device)
    generator = MetaArchitectureGenerator().to(device)
    blackbox = BlackBoxCostPredictor().to(device)
    encoder.load_state_dict(payload["encoder"])
    decoder.load_state_dict(payload["decoder"])
    cost_predictor.load_state_dict(payload["cost_predictor"])
    feasibility_head.load_state_dict(payload["feasibility_head"])
    generator.load_state_dict(payload["generator"])
    blackbox.load_state_dict(payload["blackbox"])
    mean_z = payload.get("mean_z", torch.zeros(1, 64)).to(device)
    mean_calibration = payload.get("mean_calibration", torch.zeros(1, CALIBRATION_DIM)).to(device)
    for module in (encoder, decoder, cost_predictor, feasibility_head, generator, blackbox):
        module.eval()
    return encoder, decoder, cost_predictor, feasibility_head, generator, blackbox, mean_z, mean_calibration


def _evaluate_architecture_accuracy(model: Supernet, loader: DataLoader, arch: ArchitectureSpec, device: str, limit_batches: int) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= limit_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images, arch).argmax(dim=1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
    return correct / max(total, 1)


def _resolve_accuracy_targets(frame: pd.DataFrame, num_classes: int, config: dict[str, Any]) -> tuple[pd.Series, str]:
    mode = str(config.get("accuracy_label_mode", "auto")).lower()
    oracle_weight = float(config.get("accuracy_oracle_weight", 0.9))
    chance_level = 1.0 / max(num_classes, 1)
    if mode == "supernet":
        return frame["supernet_accuracy"], "supernet"
    if mode == "oracle":
        return frame["oracle_accuracy"], "oracle"
    if mode == "hybrid":
        return oracle_weight * frame["oracle_accuracy"] + (1.0 - oracle_weight) * frame["supernet_accuracy"], "hybrid"
    if float(frame["supernet_accuracy"].std(ddof=0)) < float(config.get("accuracy_auto_min_std", 5e-3)) or float(frame["supernet_accuracy"].mean()) <= chance_level + float(config.get("accuracy_auto_chance_margin", 0.05)):
        return oracle_weight * frame["oracle_accuracy"] + (1.0 - oracle_weight) * frame["supernet_accuracy"], "auto_oracle_fallback"
    return frame["supernet_accuracy"], "auto_supernet"


def _response_targets(devices: list[DeviceRecord], device: str) -> torch.Tensor:
    return torch.as_tensor(np.stack([np.asarray(item.response.flatten(), dtype=np.float32) for item in devices]), dtype=torch.float32, device=device)


def _zero_response_like(response: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in response.items()}


def _heuristic_evolutionary_search(accuracy_predictor: AccuracyPredictor, cost_predictor: StructuredCostPredictor, feasibility_head: FeasibilityHead, z: torch.Tensor, response: dict[str, torch.Tensor], calibration: torch.Tensor, static: HardwareStaticSpec, budget: BudgetSpec, device: str, population_size: int, rounds: int, seed: int, static_context: torch.Tensor | None = None, dominant_dims: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    response_np = {"gamma": response["gamma"].detach().cpu().numpy()[0], "beta_mem": float(response["beta_mem"].detach().cpu().numpy()[0][0]), "rho_launch": float(response["rho_launch"].detach().cpu().numpy()[0][0]), "rho_copy": float(response["rho_copy"].detach().cpu().numpy()[0][0])}
    prior = build_heuristic_prior(static, response_np, budget)
    rng = random.Random(seed)
    population = [sample_from_prior(prior, seed=rng.randint(0, 1_000_000), name=f"heuristic_{idx}") for idx in range(population_size)]
    scored: list[dict[str, Any]] = []
    for round_idx in range(rounds):
        scored = []
        for arch in population:
            pred = predict_candidate(arch, accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, budget, device, static_context=static_context)
            scored.append({"architecture": arch, "prediction": pred, "score": score_prediction(pred, budget, dominant_dims=dominant_dims), "round": round_idx, "candidate_source": "heuristic_search"})
        scored.sort(key=lambda item: item["score"], reverse=True)
        elites = scored[: max(4, population_size // 4)]
        next_population = [ArchitectureSpec(blocks=item["architecture"].blocks, name=item["architecture"].name) for item in elites]
        while len(next_population) < population_size:
            child = crossover_architectures(rng.choice(elites)["architecture"], rng.choice(elites)["architecture"], seed=rng.randint(0, 1_000_000), name="heuristic_child")
            next_population.append(mutate_architecture(child, seed=rng.randint(0, 1_000_000), mutation_rate=0.25))
        population = next_population
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: min(len(scored), max(72, (population_size * 3) // 4))]


def _coarse_architecture_prediction(arch: ArchitectureSpec, accuracy_predictor: AccuracyPredictor, budget: BudgetSpec, device: str) -> dict[str, float]:
    arch_x = torch.as_tensor(encode_architecture(arch), dtype=torch.float32, device=device).reshape(1, -1)
    structured_x = structured_architecture_tensor(arch)
    baseline_struct = structured_architecture_tensor(default_architecture())
    mac_ratio = float(structured_x[:, 4].sum() / max(float(baseline_struct[:, 4].sum()), 1.0))
    byte_ratio = float(structured_x[:, 8].sum() / max(float(baseline_struct[:, 8].sum()), 1.0))
    latency_proxy = float(budget.t_max_ms * (0.72 * mac_ratio + 0.28 * byte_ratio))
    peak_sram_bytes = float(np.max((structured_x[:, 6] + structured_x[:, 7]) * structured_x[:, 3] / 8.0 + structured_x[:, 9]))
    flash_bytes = float(np.sum(structured_x[:, 5] * structured_x[:, 3] / 8.0 + structured_x[:, 10]))
    with torch.no_grad():
        accuracy = float(accuracy_predictor(arch_x).item())
    feasible_prob = float(latency_proxy <= budget.t_max_ms and peak_sram_bytes <= budget.m_max_bytes and flash_bytes <= budget.f_max_bytes)
    return {
        "accuracy": accuracy,
        "latency_ms": latency_proxy,
        "peak_sram_bytes": peak_sram_bytes,
        "flash_bytes": flash_bytes,
        "feasible_prob": feasible_prob,
    }


def _random_baseline_prediction(arch: ArchitectureSpec, accuracy_predictor: AccuracyPredictor, budget: BudgetSpec, device: str) -> dict[str, float]:
    arch_x = torch.as_tensor(encode_architecture(arch), dtype=torch.float32, device=device).reshape(1, -1)
    structured_x = structured_architecture_tensor(arch)
    peak_sram_bytes = float(np.max((structured_x[:, 6] + structured_x[:, 7]) * structured_x[:, 3] / 8.0 + structured_x[:, 9]))
    flash_bytes = float(np.sum(structured_x[:, 5] * structured_x[:, 3] / 8.0 + structured_x[:, 10]))
    with torch.no_grad():
        accuracy = float(accuracy_predictor(arch_x).item())
    return {
        "accuracy": accuracy,
        "latency_ms": float("nan"),
        "peak_sram_bytes": peak_sram_bytes,
        "flash_bytes": flash_bytes,
        "feasible_prob": float("nan"),
    }


def _hardware_agnostic_search(
    accuracy_predictor: AccuracyPredictor,
    budget: BudgetSpec,
    device: str,
    trials: int,
    seed: int,
    dominant_dims: tuple[str, ...] | None = None,
    keep_top: int = 12,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    generic_dims = ("T", "M", "F")
    while len(rows) < trials:
        arch = sample_architecture(seed=rng.randint(0, 1_000_000), name=f"agnostic_{len(rows)}")
        arch_repr = arch.compact_repr()
        if arch_repr in seen:
            continue
        seen.add(arch_repr)
        prediction = _coarse_architecture_prediction(arch, accuracy_predictor, budget, device)
        rows.append(
            {
                "architecture": arch,
                "prediction": prediction,
                "score": score_prediction(prediction, budget, dominant_dims=generic_dims),
                "candidate_source": "task_only_search",
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: min(len(rows), keep_top)]


def _fit_blackbox_device_calibration(
    blackbox: BlackBoxCostPredictor,
    support_rows: list[dict[str, Any]],
    static_x: torch.Tensor,
    device: str,
) -> dict[str, float]:
    if not support_rows:
        return {"latency_scale": 1.0, "sram_scale": 1.0, "flash_scale": 1.0}
    latency_scales, sram_scales, flash_scales = [], [], []
    static_batch = static_x.view(1, -1)
    with torch.no_grad():
        for row in support_rows:
            arch = _row_architecture(row)
            arch_x = torch.as_tensor(encode_architecture(arch), dtype=torch.float32, device=device).view(1, -1)
            pred = blackbox(arch_x, static_batch)
            latency_scales.append(float(row["latency_ms"]) / max(float(pred["latency_ms"].item()), 1e-6))
            sram_scales.append(float(row["peak_sram_bytes"]) / max(float(pred["peak_sram_bytes"].item()), 1e-6))
            flash_scales.append(float(row["flash_bytes"]) / max(float(pred["flash_bytes"].item()), 1e-6))
    return {
        "latency_scale": float(np.clip(np.median(latency_scales), 0.25, 4.0)),
        "sram_scale": float(np.clip(np.median(sram_scales), 0.25, 4.0)),
        "flash_scale": float(np.clip(np.median(flash_scales), 0.25, 4.0)),
    }


def _predict_blackbox_candidate(
    arch: ArchitectureSpec,
    blackbox: BlackBoxCostPredictor,
    accuracy_predictor: AccuracyPredictor,
    static_x: torch.Tensor,
    budget: BudgetSpec,
    device: str,
    calibration: dict[str, float] | None = None,
) -> dict[str, float]:
    arch_x = torch.as_tensor(encode_architecture(arch), dtype=torch.float32, device=device).view(1, -1)
    static_batch = static_x.view(1, -1)
    with torch.no_grad():
        accuracy = float(accuracy_predictor(arch_x).item())
        pred = blackbox(arch_x, static_batch)
    calibration = calibration or {"latency_scale": 1.0, "sram_scale": 1.0, "flash_scale": 1.0}
    latency_ms = float(pred["latency_ms"].item()) * calibration["latency_scale"]
    peak_sram_bytes = float(pred["peak_sram_bytes"].item()) * calibration["sram_scale"]
    flash_bytes = float(pred["flash_bytes"].item()) * calibration["flash_scale"]
    feasible_prob = float(latency_ms <= budget.t_max_ms and peak_sram_bytes <= budget.m_max_bytes and flash_bytes <= budget.f_max_bytes)
    return {
        "accuracy": accuracy,
        "latency_ms": latency_ms,
        "peak_sram_bytes": peak_sram_bytes,
        "flash_bytes": flash_bytes,
        "feasible_prob": feasible_prob,
    }


def _pure_random_candidates(
    accuracy_predictor: AccuracyPredictor,
    cost_predictor: StructuredCostPredictor,
    feasibility_head: FeasibilityHead,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec,
    device: str,
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    while len(rows) < min(trials, 48):
        arch = sample_architecture(seed=rng.randint(0, 1_000_000), name=f"random_{len(rows)}")
        arch_repr = arch.compact_repr()
        if arch_repr in seen:
            continue
        seen.add(arch_repr)
        prediction = _random_baseline_prediction(arch, accuracy_predictor, budget, device)
        rows.append(
            {
                "architecture": arch,
                "prediction": prediction,
                "score": rng.random(),
                "candidate_source": "random_sample",
            }
        )
    rng.shuffle(rows)
    return rows


def _predicted_random_candidates(
    accuracy_predictor: AccuracyPredictor,
    cost_predictor: StructuredCostPredictor,
    feasibility_head: FeasibilityHead,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec,
    device: str,
    trials: int,
    seed: int,
    static_context: torch.Tensor | None = None,
    dominant_dims: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    while len(rows) < min(trials, 128):
        arch = sample_architecture(seed=rng.randint(0, 1_000_000), name=f"guided_random_{len(rows)}")
        arch_repr = arch.compact_repr()
        if arch_repr in seen:
            continue
        seen.add(arch_repr)
        prediction = predict_candidate(
            arch,
            accuracy_predictor,
            cost_predictor,
            feasibility_head,
            z,
            response,
            calibration,
            budget,
            device,
            static_context=static_context,
        )
        rows.append(
            {
                "architecture": arch,
                "prediction": prediction,
                "score": score_prediction(prediction, budget, dominant_dims=dominant_dims),
                "candidate_source": "guided_random_search",
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: min(len(rows), 96)]


def _blackbox_random_search(blackbox: BlackBoxCostPredictor, accuracy_predictor: AccuracyPredictor, static_x: torch.Tensor, budget: BudgetSpec, device: str, trials: int, seed: int, dominant_dims: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    calibration = {"latency_scale": 1.0, "sram_scale": 1.0, "flash_scale": 1.0}
    for idx in range(trials):
        arch = sample_architecture(seed=rng.randint(0, 1_000_000), name=f"blackbox_{idx}")
        arch_repr = arch.compact_repr()
        if arch_repr in seen:
            continue
        seen.add(arch_repr)
        prediction = _predict_blackbox_candidate(arch, blackbox, accuracy_predictor, static_x, budget, device, calibration)
        rows.append({"architecture": arch, "prediction": prediction, "score": score_prediction(prediction, budget, dominant_dims=dominant_dims), "candidate_source": "blackbox_search"})
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: min(len(rows), 16)]


def _apply_hardware_observation_modes(
    static_x: torch.Tensor,
    probe_x: torch.Tensor,
    ref_x: torch.Tensor,
    probe_mode: str,
    ref_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probe_out = torch.zeros_like(probe_x) if probe_mode == "zero" else probe_x
    ref_out = torch.zeros_like(ref_x) if ref_mode == "zero" else ref_x
    return static_x, probe_out, ref_out


def _apply_response_mode(
    response: dict[str, torch.Tensor],
    disable_response: bool,
) -> dict[str, torch.Tensor]:
    return _zero_response_like(response) if disable_response else response


def _merge_candidate_pools(*pools: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for pool in pools:
        for item in pool:
            key = item["architecture"].compact_repr()
            incumbent = merged.get(key)
            if incumbent is None or float(item["score"]) > float(incumbent["score"]):
                merged[key] = item
    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def _meta_generated_candidates(
    accuracy_predictor: AccuracyPredictor,
    cost_predictor: StructuredCostPredictor,
    feasibility_head: FeasibilityHead,
    generator: MetaArchitectureGenerator,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec,
    device: str,
    refine_radius: int = 1,
    seed: int = 0,
    prior_samples: int = 12,
    static_context: torch.Tensor | None = None,
    dominant_dims: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    generated_arch = generate_architecture_direct(generator, z, budget, device=device, name="meta_generated")
    direct_prediction = predict_candidate(
        generated_arch,
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
    direct_row = {"architecture": generated_arch, "prediction": direct_prediction, "score": score_prediction(direct_prediction, budget, dominant_dims=dominant_dims), "candidate_source": "generated_direct"}
    budget_x = torch.as_tensor([[budget.t_max_ms, budget.m_max_bytes, budget.f_max_bytes]], dtype=torch.float32, device=device)
    with torch.no_grad():
        generator_outputs = generator(z if z.ndim == 2 else z.unsqueeze(0), budget_x)
    prior = {
        "num_blocks": int(generator_outputs["op_prob"].shape[1]),
        "op_probs": generator_outputs["op_prob"][0].detach().cpu().tolist(),
        "width_probs": generator_outputs["width_prob"][0].detach().cpu().tolist(),
        "depth_probs": generator_outputs["depth_prob"][0].detach().cpu().tolist(),
        "quant_probs": generator_outputs["quant_prob"][0].detach().cpu().tolist(),
    }
    prior_rows = []
    rng = random.Random(seed)
    seen_prior = {generated_arch.compact_repr()}
    for idx in range(max(0, prior_samples)):
        arch = sample_from_prior(prior, seed=rng.randint(0, 1_000_000), name=f"meta_prior_{idx}")
        arch_repr = arch.compact_repr()
        if arch_repr in seen_prior:
            continue
        seen_prior.add(arch_repr)
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
        prior_rows.append({"architecture": arch, "prediction": pred, "score": score_prediction(pred, budget, dominant_dims=dominant_dims), "candidate_source": "generated_prior"})
    prior_rows.sort(key=lambda item: item["score"], reverse=True)
    refined_rows = local_refine_search(
        generated_arch,
        accuracy_predictor=accuracy_predictor,
        cost_predictor=cost_predictor,
        feasibility_head=feasibility_head,
        z=z,
        response=response,
        calibration=calibration,
        budget=budget,
        device=device,
        radius=refine_radius,
        static_context=static_context,
        dominant_dims=dominant_dims,
    )
    for row in refined_rows:
        row.setdefault("candidate_source", "local_refine")
    if refine_radius <= 0:
        merged = _merge_candidate_pools([direct_row], prior_rows, limit=max(64, len(prior_rows) + 1))
    else:
        merged = _merge_candidate_pools([direct_row], refined_rows, prior_rows, limit=max(96, len(refined_rows) + len(prior_rows) + 1))
    return merged


def _search_candidates(method: str, accuracy_predictor: AccuracyPredictor, cost_predictor: StructuredCostPredictor, feasibility_head: FeasibilityHead, generator: MetaArchitectureGenerator, blackbox: BlackBoxCostPredictor, z: torch.Tensor, response: dict[str, torch.Tensor], calibration: torch.Tensor, static: HardwareStaticSpec, static_x: torch.Tensor, budget: BudgetSpec, device: str, population_size: int, rounds: int, random_trials: int, seed: int, use_generator: bool, refine_radius: int = 1, support_rows: list[dict[str, Any]] | None = None, dominant_dims: tuple[str, ...] | None = None, include_task_only_exploration: bool = True, strict_article_flow: bool = False) -> list[dict[str, Any]]:
    if method == "random_search":
        return _pure_random_candidates(accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, budget, device=device, trials=random_trials, seed=seed)
    if method == "blackbox_cost_mlp":
        calibration_stats = _fit_blackbox_device_calibration(blackbox, support_rows or [], static_x, device)
        rng = random.Random(seed)
        rows = []
        seen: set[str] = set()
        while len(rows) < min(random_trials, 128):
            arch = sample_architecture(seed=rng.randint(0, 1_000_000), name=f"blackbox_{len(rows)}")
            arch_repr = arch.compact_repr()
            if arch_repr in seen:
                continue
            seen.add(arch_repr)
            prediction = _predict_blackbox_candidate(arch, blackbox, accuracy_predictor, static_x, budget, device, calibration_stats)
            rows.append({"architecture": arch, "prediction": prediction, "score": score_prediction(prediction, budget, dominant_dims=dominant_dims), "candidate_source": "blackbox_search"})
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[: min(len(rows), 32)]
    if method == "hardware_agnostic":
        return _hardware_agnostic_search(accuracy_predictor, budget, device, trials=48, seed=seed, dominant_dims=dominant_dims)
    if method == "zero_shot":
        return _heuristic_evolutionary_search(accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, static, budget, device, population_size, rounds, seed, static_context=static_x.view(1, -1), dominant_dims=dominant_dims)
    if use_generator and method in {"few_shot"} and strict_article_flow:
        return _meta_generated_candidates(
            accuracy_predictor=accuracy_predictor,
            cost_predictor=cost_predictor,
            feasibility_head=feasibility_head,
            generator=generator,
            z=z,
            response=response,
            calibration=calibration,
            budget=budget,
            device=device,
            refine_radius=refine_radius,
            seed=seed,
            prior_samples=0,
            static_context=static_x.view(1, -1),
            dominant_dims=dominant_dims,
        )
    if use_generator and method in {"few_shot"}:
        direct_candidates = _meta_generated_candidates(
            accuracy_predictor=accuracy_predictor,
            cost_predictor=cost_predictor,
            feasibility_head=feasibility_head,
            generator=generator,
            z=z,
            response=response,
            calibration=calibration,
            budget=budget,
            device=device,
            refine_radius=refine_radius,
            seed=seed,
            prior_samples=max(96, population_size),
            static_context=static_x.view(1, -1),
            dominant_dims=dominant_dims,
        )
        search_candidates = evolutionary_search(
            accuracy_predictor,
            cost_predictor,
            feasibility_head,
            generator,
            z,
            response,
            calibration,
            static,
            budget,
            device=device,
            population_size=max(population_size, 96),
            rounds=max(rounds, 48),
            seed=seed,
            static_context=static_x.view(1, -1),
            dominant_dims=dominant_dims,
        )
        heuristic_candidates = _heuristic_evolutionary_search(
            accuracy_predictor,
            cost_predictor,
            feasibility_head,
            z,
            response,
            calibration,
            static,
            budget,
            device,
            max(48, (population_size * 3) // 4),
            max(48, rounds // 2),
            seed + 211,
            static_context=static_x.view(1, -1),
            dominant_dims=dominant_dims,
        )
        random_candidates = _predicted_random_candidates(
            accuracy_predictor,
            cost_predictor,
            feasibility_head,
            z,
            response,
            calibration,
            budget,
            device,
            trials=max(256, random_trials),
            seed=seed + 431,
            static_context=static_x.view(1, -1),
            dominant_dims=dominant_dims,
        )
        exploration_candidates: list[dict[str, Any]] = []
        if include_task_only_exploration:
            exploration_candidates = _hardware_agnostic_search(
                accuracy_predictor,
                budget,
                device,
                trials=max(128, random_trials // 2),
                seed=seed + 719,
                dominant_dims=dominant_dims,
                keep_top=32,
            )
            for row in exploration_candidates:
                row.setdefault("candidate_source", "task_exploration")
        for row in search_candidates:
            row.setdefault("candidate_source", "generator_search")
        pools: list[list[dict[str, Any]]] = [direct_candidates, search_candidates, heuristic_candidates, random_candidates]
        if exploration_candidates:
            pools.append(exploration_candidates)
        return _merge_candidate_pools(*pools, limit=max(160, len(search_candidates) + len(direct_candidates) + len(heuristic_candidates) // 2))
    if not use_generator:
        return _heuristic_evolutionary_search(accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, static, budget, device, population_size, rounds, seed, static_context=static_x.view(1, -1), dominant_dims=dominant_dims)
    generator_candidates = evolutionary_search(accuracy_predictor, cost_predictor, feasibility_head, generator, z, response, calibration, static, budget, device=device, population_size=population_size, rounds=rounds, seed=seed, static_context=static_x.view(1, -1), dominant_dims=dominant_dims)
    heuristic_candidates = _heuristic_evolutionary_search(accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, static, budget, device, population_size=max(12, population_size // 2), rounds=max(8, rounds // 2), seed=seed + 97, static_context=static_x.view(1, -1), dominant_dims=dominant_dims)
    for row in generator_candidates:
        row.setdefault("candidate_source", "generator_search")
    merged = _merge_candidate_pools(generator_candidates, heuristic_candidates, limit=max(16, len(generator_candidates)))
    return merged


def _augment_candidates_with_seed_architectures(
    candidates: list[dict[str, Any]],
    seed_architectures: list[ArchitectureSpec],
    accuracy_predictor: AccuracyPredictor,
    cost_predictor: StructuredCostPredictor,
    feasibility_head: FeasibilityHead,
    z: torch.Tensor,
    response: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    budget: BudgetSpec,
    device: str,
    limit: int = 10,
    static_context: torch.Tensor | None = None,
    dominant_dims: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    seen = {item["architecture"].compact_repr() for item in candidates}
    augmented = list(candidates)
    for arch in seed_architectures:
        if arch.compact_repr() in seen:
            continue
        pred = predict_candidate(arch, accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, budget, device, static_context=static_context)
        augmented.append({"architecture": arch, "prediction": pred, "score": score_prediction(pred, budget, dominant_dims=dominant_dims), "round": -1, "candidate_source": "support_seed"})
        seen.add(arch.compact_repr())
    augmented.sort(key=lambda item: item["score"], reverse=True)
    return augmented[:limit]


def prepare_data(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    data_root = root / config.get("data_root", "data")
    dataset_name = config.get("dataset_name", "cifar10")
    generated_dir, _ = _artifact_roots(root, dataset_name)
    datasets = build_image_datasets(dataset_name, data_root, val_size=int(config.get("val_size", 5_000)), seed=int(config.get("seed", 0)), allow_synthetic_fallback=bool(config.get("allow_synthetic_fallback", True)), force_synthetic=bool(config.get("force_synthetic", False)), cifar100_local_path=config.get("cifar100_local_path"), cifar100_label_mode=config.get("cifar100_label_mode", "fine"))
    manifest = {"dataset_name": dataset_name, "train_size": len(datasets["train"]), "val_size": len(datasets["val"]), "test_size": len(datasets["test"]), "data_root": str(data_root)}
    manifest_path = generated_dir / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    _log("准备数据", [f"数据集：{dataset_name}", f"清单文件：{manifest_path}"], root=root)
    return {**manifest, "manifest_path": str(manifest_path)}


def train_supernet(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    set_seed(int(config.get("seed", 0)))
    device = get_device(config.get("device"))
    dataset_name = config.get("dataset_name", "cifar10")
    generated_dir, checkpoint_dir = _artifact_roots(root, dataset_name)
    datasets = build_image_datasets(dataset_name, root / config.get("data_root", "data"), val_size=int(config.get("val_size", 5_000)), seed=int(config.get("seed", 0)), allow_synthetic_fallback=bool(config.get("allow_synthetic_fallback", True)), force_synthetic=bool(config.get("force_synthetic", False)), cifar100_local_path=config.get("cifar100_local_path"), cifar100_label_mode=config.get("cifar100_label_mode", "fine"))
    train_loader = DataLoader(datasets["train"], batch_size=int(config.get("batch_size", 64)), shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=int(config.get("batch_size", 64)), shuffle=False)
    model = Supernet(num_classes=int(config.get("num_classes", dataset_num_classes(dataset_name, config.get("cifar100_label_mode", "fine"))))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    baseline_arch = default_architecture(name="标准 TinyNet 基线")
    history = []
    global_step = 0
    for epoch in range(int(config.get("epochs", 2))):
        model.train()
        train_losses = []
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= int(config.get("limit_train_batches", 80)):
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, sample_architecture(seed=global_step, name=f"supernet_train_{global_step}"))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            global_step += 1
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx >= int(config.get("limit_val_batches", 20)):
                    break
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images, baseline_arch).argmax(dim=1)
                total += int(labels.numel())
                correct += int((pred == labels).sum().item())
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_acc": float(correct / max(total, 1))})
    checkpoint_path = checkpoint_dir / "supernet.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
    history_path = write_dataframe(generated_dir / "supernet_history.csv", pd.DataFrame(history))
    return {"checkpoint": str(checkpoint_path), "history": str(history_path)}


def build_accuracy_dataset(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    device = get_device(config.get("device"))
    set_seed(int(config.get("seed", 0)))
    dataset_name = config.get("dataset_name", "cifar10")
    generated_dir, checkpoint_dir = _artifact_roots(root, dataset_name)
    datasets = build_image_datasets(dataset_name, root / config.get("data_root", "data"), val_size=int(config.get("val_size", 5_000)), seed=int(config.get("seed", 0)), allow_synthetic_fallback=bool(config.get("allow_synthetic_fallback", True)), force_synthetic=bool(config.get("force_synthetic", False)), cifar100_local_path=config.get("cifar100_local_path"), cifar100_label_mode=config.get("cifar100_label_mode", "fine"))
    loader = DataLoader(datasets["val"], batch_size=int(config.get("batch_size", 128)), shuffle=False)
    model = Supernet(num_classes=int(config.get("num_classes", dataset_num_classes(dataset_name, config.get("cifar100_label_mode", "fine"))))).to(device)
    ckpt = checkpoint_dir / "supernet.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    rows = []
    seen = set()
    seed_cursor = int(config.get("seed", 0))
    while len(rows) < int(config.get("num_architectures", 256)):
        arch = sample_architecture(seed=seed_cursor, name=f"候选网络{len(rows):04d}")
        seed_cursor += 1
        if arch.compact_repr() in seen:
            continue
        seen.add(arch.compact_repr())
        rows.append({"arch_name": arch.name, "arch_repr": arch.compact_repr(), "architecture_json": json.dumps(arch.to_dict(), ensure_ascii=False), "supernet_accuracy": _evaluate_architecture_accuracy(model, loader, arch, device, int(config.get("limit_val_batches", 20))), "oracle_accuracy": synthetic_architecture_accuracy(arch, dataset_name=dataset_name, noise_scale=float(config.get("oracle_accuracy_noise", 0.004)))})
    frame = pd.DataFrame(rows)
    frame["accuracy"], label_source = _resolve_accuracy_targets(frame, int(config.get("num_classes", dataset_num_classes(dataset_name, config.get("cifar100_label_mode", "fine")))), config)
    dataset_path = write_dataframe(generated_dir / "accuracy_dataset.parquet", frame)
    features = np.stack([encode_architecture(ArchitectureSpec.from_dict(json.loads(row))) for row in frame["architecture_json"]])
    targets = frame["accuracy"].to_numpy(dtype=np.float32)
    split = max(1, int(0.8 * len(frame)))
    predictor = AccuracyPredictor(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=float(config.get("predictor_lr", 1e-3)))
    train_loader = DataLoader(ArchitectureAccuracyDataset(features[:split], targets[:split]), batch_size=int(config.get("predictor_batch_size", 64)), shuffle=True)
    val_loader = DataLoader(ArchitectureAccuracyDataset(features[split:] if split < len(frame) else features[:1], targets[split:] if split < len(frame) else targets[:1]), batch_size=int(config.get("predictor_batch_size", 64)), shuffle=False)
    hist = []
    for epoch in range(int(config.get("predictor_epochs", 40))):
        predictor.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = F.mse_loss(predictor(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        predictor.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_losses.append(float(F.mse_loss(predictor(x), y).item()))
        hist.append({"epoch": float(epoch), "train_mse": float(np.mean(train_losses)), "val_mse": float(np.mean(val_losses) if val_losses else 0.0)})
    predictor_ckpt = checkpoint_dir / "accuracy_predictor.pt"
    torch.save({"model_state": predictor.state_dict(), "input_dim": features.shape[1]}, predictor_ckpt)
    history_path = write_dataframe(generated_dir / "accuracy_predictor_history.csv", pd.DataFrame(hist))
    _log("构建精度数据集", [f"数据集：{dataset_name}", f"精度标签来源：{label_source}", f"精度数据文件：{dataset_path}"], root=root)
    return {"dataset_path": str(dataset_path), "predictor_checkpoint": str(predictor_ckpt), "history": str(history_path)}


def build_synth_devices(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    set_seed(int(config.get("seed", 0)))
    dataset_name = config.get("dataset_name", "cifar10")
    generated_dir, _ = _artifact_roots(root, dataset_name)
    device_root = ensure_dir(generated_dir / "devices")
    accuracy_path = generated_dir / "accuracy_dataset.parquet"
    accuracy_lookup = dict(zip(read_dataframe(accuracy_path)["arch_repr"], read_dataframe(accuracy_path)["accuracy"])) if accuracy_path.exists() or accuracy_path.with_suffix(".csv").exists() else {}

    split_specs = {"train": int(config.get("train_per_family", 4)), "val": int(config.get("val_per_family", 2)), "test": int(config.get("test_per_family", 2))}
    architectures: list[ArchitectureSpec] = [make_standard_baseline(name="鏍囧噯 TinyNet 鍩虹嚎"), _mobilenet_like_architecture(), *REFERENCE_ARCHITECTURES.values()]
    seen = {arch.compact_repr() for arch in architectures}
    cursor = 0
    while len(architectures) < int(config.get("num_architectures", 192)):
        arch = sample_architecture(seed=int(config.get("seed", 0)) + cursor, name=f"候选网络{len(architectures):04d}")
        cursor += 1
        if arch.compact_repr() not in seen:
            architectures.append(arch)
            seen.add(arch.compact_repr())

    manifest = {"dataset_name": dataset_name, "splits": {}}
    all_devices: list[DeviceRecord] = []
    for split_name, count in split_specs.items():
        split_dir = ensure_dir(device_root / split_name)
        items = []
        for family_idx, family in enumerate(DEVICE_FAMILIES):
            for local_idx in range(count):
                seed = int(config.get("seed", 0)) + family_idx * 10_000 + local_idx * 37 + {"train": 0, "val": 1000, "test": 2000}[split_name]
                record = create_device_record(family, index=local_idx, seed=seed)
                device_dir = split_dir / record.static.name
                export_device_directory(record, device_dir)
                items.append({"device": record.static.name, "path": str(device_dir), "family": family, "split": split_name})
                all_devices.append(record)
        manifest["splits"][split_name] = items

    measurement_frame = build_arch_measurement_table(architectures=architectures, devices=all_devices, accuracy_lookup=accuracy_lookup, noise_scale=float(config.get("measurement_noise_scale", 0.02)), dataset_name=dataset_name)
    measurement_path = write_dataframe(generated_dir / "arch_measurements.parquet", measurement_frame)
    grouped = measurement_frame.groupby("device_name")
    for split_items in manifest["splits"].values():
        for item in split_items:
            rows = grouped.get_group(item["device"]).to_dict(orient="records")
            device_dir = Path(item["path"])
            write_jsonl(device_dir / "arch_measurements.jsonl", rows)
            rng = np.random.default_rng(int(config.get("seed", 0)) + len(rows))
            write_jsonl(device_dir / "task_budgets.jsonl", [{"budget_id": idx, **_budget_from_rows(rows, rng).to_dict()} for idx in range(int(config.get("budgets_per_device", 4)))])
    manifest_path = device_root / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "measurement_table": str(measurement_path)}


def _oracle_architecture(rows: list[dict[str, Any]], budget: BudgetSpec) -> ArchitectureSpec:
    feasible = [row for row in rows if float(row["latency_ms"]) <= budget.t_max_ms and float(row["peak_sram_bytes"]) <= budget.m_max_bytes and float(row["flash_bytes"]) <= budget.f_max_bytes]
    chosen = max(feasible or rows, key=lambda row: float(row.get("accuracy", 0.0)) - max(0.0, float(row["latency_ms"]) - budget.t_max_ms) / max(budget.t_max_ms, 1.0))
    return ArchitectureSpec.from_dict(json.loads(chosen["architecture_json"]))


def _oracle_token_targets(arch: ArchitectureSpec, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    op_x = torch.as_tensor([OPS.index(stage.op) for stage in arch.stages], dtype=torch.long, device=device)
    width_x = torch.as_tensor([WIDTHS.index(stage.width) for stage in arch.stages], dtype=torch.long, device=device)
    depth_x = torch.as_tensor([DEPTHS.index(stage.depth) for stage in arch.stages], dtype=torch.long, device=device)
    quant_x = torch.as_tensor([QUANTS.index(stage.quant) for stage in arch.stages], dtype=torch.long, device=device)
    return op_x, width_x, depth_x, quant_x


def train_hardware_models(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    set_seed(int(config.get("seed", 0)))
    dataset_name = config.get("dataset_name", "cifar10")
    device = get_device(config.get("device"))
    generated_dir, checkpoint_dir = _artifact_roots(root, dataset_name)
    bundle_filename = str(config.get("bundle_filename", "hardware_model_bundle.pt"))
    artifact_tag = str(config.get("artifact_tag", "")).strip()
    row_feasible_weight = float(config.get("row_feasible_weight", 0.25))
    meta_feasible_weight = float(config.get("meta_feasible_weight", 0.25))
    generator_loss_weight = float(config.get("generator_loss_weight", 0.30))
    response_aux_weight = float(config.get("response_aux_weight", 0.10))
    budget_scale_min = float(config.get("budget_scale_min", 0.82))
    budget_scale_max = float(config.get("budget_scale_max", 1.22))
    probe_mode = str(config.get("probe_mode", "full")).lower()
    ref_mode = str(config.get("ref_mode", "full")).lower()
    disable_response = bool(config.get("disable_response", False))
    disable_calibration = bool(config.get("disable_calibration", False))
    manifest = read_json(generated_dir / "devices" / "manifest.json")
    measurement_frame = read_dataframe(generated_dir / "arch_measurements.parquet")
    device_lookup = {Path(item["path"]).name: _load_device_record(item["path"]) for split in manifest["splits"].values() for item in split}
    train_devices = [device_lookup[_manifest_item_name(item)] for item in manifest["splits"]["train"]]
    train_frame = measurement_frame[measurement_frame["device_name"].isin([item.static.name for item in train_devices])].copy()
    measurements_by_device = {name: group.to_dict(orient="records") for name, group in train_frame.groupby("device_name")}

    encoder, decoder = HardwareEncoder().to(device), ResponseDecoder().to(device)
    cost_predictor, feasibility_head = StructuredCostPredictor().to(device), FeasibilityHead().to(device)
    generator, blackbox = BudgetConditionedGenerator().to(device), BlackBoxCostPredictor().to(device)
    response_optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(config.get("response_lr", config.get("lr", 1e-3))))
    main_optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(cost_predictor.parameters()) + list(feasibility_head.parameters()) + list(generator.parameters()), lr=float(config.get("meta_lr", config.get("lr", 1e-3))))
    blackbox_optim = torch.optim.Adam(blackbox.parameters(), lr=float(config.get("blackbox_lr", 1e-3)))

    response_hist = []
    static_x, probe_x, ref_x = _device_tensor_batch(train_devices, device)
    static_x, probe_x, ref_x = _apply_hardware_observation_modes(static_x, probe_x, ref_x, probe_mode, ref_mode)
    response_y = _response_targets(train_devices, device)
    if not disable_response:
        for epoch in range(int(config.get("response_epochs", 24))):
            z = encoder(static_x, probe_x, ref_x)
            response = decoder(z)
            pred = torch.cat([response["gamma"].reshape(z.shape[0], -1), response["beta_mem"], response["rho_launch"], response["rho_copy"]], dim=1)
            loss = F.mse_loss(pred, response_y)
            response_optim.zero_grad()
            loss.backward()
            response_optim.step()
            response_hist.append({"epoch": float(epoch), "response_mse": float(loss.item())})

    rows = train_frame.to_dict(orient="records")
    row_hist = []
    rng = np.random.default_rng(int(config.get("seed", 0)))
    for epoch in range(int(config.get("row_epochs", 16))):
        rng.shuffle(rows)
        row_losses = []
        blackbox_losses = []
        for start in range(0, len(rows), int(config.get("batch_size", 32))):
            batch_rows = rows[start : start + int(config.get("batch_size", 32))]
            device_names = [row["device_name"] for row in batch_rows]
            arch_x, struct_x, target_x = _rows_to_arch_batch(batch_rows, device)
            static_x = torch.as_tensor(np.stack([_static_features(device_lookup[name].static) for name in device_names]), dtype=torch.float32, device=device)
            probe_x = torch.as_tensor(np.stack([_probe_tensor(list(device_lookup[name].probes)) for name in device_names]), dtype=torch.float32, device=device)
            ref_x = torch.as_tensor(np.stack([_reference_tensor(list(device_lookup[name].references)) for name in device_names]), dtype=torch.float32, device=device)
            static_x, probe_x, ref_x = _apply_hardware_observation_modes(static_x, probe_x, ref_x, probe_mode, ref_mode)
            z = encoder(static_x, probe_x, ref_x)
            response = _apply_response_mode(decoder(z), disable_response)
            budget_scale = torch.rand((arch_x.shape[0], 3), device=device) * max(budget_scale_max - budget_scale_min, 1e-6) + budget_scale_min
            budget_x = torch.cat([target_x[:, 0:1] * budget_scale[:, 0:1], target_x[:, 1:2] * budget_scale[:, 1:2], target_x[:, 2:3] * budget_scale[:, 2:3]], dim=1)
            feasible_true = ((target_x[:, 0:1] <= budget_x[:, 0:1]) & (target_x[:, 1:2] <= budget_x[:, 1:2]) & (target_x[:, 2:3] <= budget_x[:, 2:3])).float()
            calibration = torch.zeros(arch_x.shape[0], CALIBRATION_DIM, device=device)
            cost = cost_predictor(arch_x, struct_x, z, response, calibration, static_context=static_x)
            feasible = feasibility_head(arch_x, z, cost, budget_x, calibration)
            loss = F.l1_loss(cost["latency_ms"], target_x[:, 0:1]) + F.l1_loss(cost["peak_sram_bytes"], target_x[:, 1:2]) + F.l1_loss(cost["flash_bytes"], target_x[:, 2:3]) + row_feasible_weight * F.binary_cross_entropy(feasible, feasible_true)
            main_optim.zero_grad()
            loss.backward()
            main_optim.step()
            row_losses.append(float(loss.item()))

            blackbox_pred = blackbox(arch_x.detach(), static_x.detach())
            blackbox_loss = F.l1_loss(blackbox_pred["latency_ms"], target_x[:, 0:1]) + F.l1_loss(blackbox_pred["peak_sram_bytes"], target_x[:, 1:2]) + F.l1_loss(blackbox_pred["flash_bytes"], target_x[:, 2:3]) + row_feasible_weight * F.binary_cross_entropy(blackbox_pred["feasible_prob"], feasible_true)
            blackbox_optim.zero_grad()
            blackbox_loss.backward()
            blackbox_optim.step()
            blackbox_losses.append(float(blackbox_loss.item()))
        row_hist.append({"epoch": float(epoch), "row_loss": float(np.mean(row_losses)), "blackbox_loss": float(np.mean(blackbox_losses))})

    task_dataset = FewShotTaskDataset(devices=train_devices, measurements_by_device=measurements_by_device, support_size=int(config.get("support_size", 8)), query_size=int(config.get("query_size", 24)), seed=int(config.get("seed", 0)))
    meta_hist = []
    for episode in range(int(config.get("meta_episodes", 32))):
        task = task_dataset.sample_task()
        budget = task["budget"]
        support_rows = _stamp_budget(task["support_rows"], budget)
        query_rows = _stamp_budget(task["query_rows"], budget)
        static_x, probe_x, ref_x = _device_bundle(task["device"], device)
        static_x, probe_x, ref_x = _apply_hardware_observation_modes(static_x, probe_x, ref_x, probe_mode, ref_mode)
        initial_z = encoder(static_x, probe_x, ref_x)
        adapted_z, adapted_calibration, _ = adapt_device_state(decoder, cost_predictor, feasibility_head, initial_z, support_rows, initial_calibration=torch.zeros(1, CALIBRATION_DIM, device=device), steps=int(config.get("adapt_steps", 8)), lr=float(config.get("adapt_lr", 5e-3)), device=device, static_context=static_x)
        if disable_calibration:
            adapted_calibration = torch.zeros_like(adapted_calibration)
        arch_x, struct_x, target_x = _rows_to_arch_batch(query_rows, device)
        z_batch = adapted_z.expand(arch_x.shape[0], -1)
        calib_batch = adapted_calibration.expand(arch_x.shape[0], -1)
        response = _apply_response_mode(decoder(z_batch), disable_response)
        budget_x = _budget_tensor(budget, device=device, batch_size=arch_x.shape[0])
        feasible_true = ((target_x[:, 0:1] <= budget_x[:, 0:1]) & (target_x[:, 1:2] <= budget_x[:, 1:2]) & (target_x[:, 2:3] <= budget_x[:, 2:3])).float()
        response_batch = {name: value.expand(arch_x.shape[0], *value.shape[1:]) for name, value in response.items()}
        cost = cost_predictor(arch_x, struct_x, z_batch, response_batch, calib_batch, static_context=static_x.expand(arch_x.shape[0], -1))
        feasible = feasibility_head(arch_x, z_batch, cost, budget_x, calib_batch)
        query_loss = F.l1_loss(cost["latency_ms"], target_x[:, 0:1]) + F.l1_loss(cost["peak_sram_bytes"], target_x[:, 1:2]) + F.l1_loss(cost["flash_bytes"], target_x[:, 2:3]) + meta_feasible_weight * F.binary_cross_entropy(feasible, feasible_true)
        op_x, width_x, depth_x, quant_x = _oracle_token_targets(_oracle_architecture(query_rows + support_rows, budget), device)
        generator_out = generator(adapted_z, _budget_tensor(budget, device))
        generator_loss = 0.0
        for idx in range(op_x.shape[0]):
            generator_loss = generator_loss + F.cross_entropy(generator_out["op_logits"][:, idx, :], op_x[idx : idx + 1]) + F.cross_entropy(generator_out["width_logits"][:, idx, :], width_x[idx : idx + 1]) + F.cross_entropy(generator_out["depth_logits"][:, idx, :], depth_x[idx : idx + 1]) + F.cross_entropy(generator_out["quant_logits"][:, idx, :], quant_x[idx : idx + 1])
        response_now = _apply_response_mode(decoder(initial_z), disable_response)
        if disable_response:
            response_loss = torch.zeros((), device=device)
        else:
            response_loss = F.mse_loss(torch.cat([response_now["gamma"].reshape(1, -1), response_now["beta_mem"], response_now["rho_launch"], response_now["rho_copy"]], dim=1), _response_targets([task["device"]], device))
        total_loss = query_loss + generator_loss_weight * generator_loss + response_aux_weight * response_loss
        main_optim.zero_grad()
        total_loss.backward()
        main_optim.step()
        meta_hist.append({"episode": float(episode), "query_loss": float(query_loss.item()), "generator_loss": float(generator_loss.item()), "response_loss": float(response_loss.item()), "total_loss": float(total_loss.item())})

    adapted_states = []
    for device_record in train_devices:
        rows_for_device = measurements_by_device[device_record.static.name]
        budget = _budget_from_rows(rows_for_device, np.random.default_rng(int(config.get("seed", 0))))
        support_rows = _stamp_budget(rows_for_device[: int(config.get("support_size", 8))], budget)
        static_x, probe_x, ref_x = _device_bundle(device_record, device)
        static_x, probe_x, ref_x = _apply_hardware_observation_modes(static_x, probe_x, ref_x, probe_mode, ref_mode)
        z0 = encoder(static_x, probe_x, ref_x)
        z1, c1, _ = adapt_device_state(decoder, cost_predictor, feasibility_head, z0, support_rows, initial_calibration=torch.zeros(1, CALIBRATION_DIM, device=device), steps=int(config.get("adapt_steps", 8)), lr=float(config.get("adapt_lr", 5e-3)), device=device, static_context=static_x)
        if disable_calibration:
            c1 = torch.zeros_like(c1)
        adapted_states.append((z1.detach().cpu(), c1.detach().cpu()))
    mean_z = torch.cat([item[0] for item in adapted_states], dim=0).mean(dim=0, keepdim=True)
    mean_calibration = torch.cat([item[1] for item in adapted_states], dim=0).mean(dim=0, keepdim=True)

    bundle_path = checkpoint_dir / bundle_filename
    torch.save({**bundle_state_dict({"encoder": encoder, "decoder": decoder, "cost_predictor": cost_predictor, "feasibility_head": feasibility_head, "generator": generator, "blackbox": blackbox}), "mean_z": mean_z.cpu(), "mean_calibration": mean_calibration.cpu(), "dataset_name": dataset_name, "arch_feature_dim": ARCH_FEATURE_DIM, "arch_token_dim": ARCH_TOKEN_DIM, "row_feasible_weight": row_feasible_weight, "meta_feasible_weight": meta_feasible_weight, "generator_loss_weight": generator_loss_weight, "response_aux_weight": response_aux_weight}, bundle_path)
    suffix = f"_{artifact_tag}" if artifact_tag else ""
    response_history_path = write_dataframe(generated_dir / f"hardware_response_history{suffix}.csv", pd.DataFrame(response_hist))
    row_history_path = write_dataframe(generated_dir / f"hardware_row_history{suffix}.csv", pd.DataFrame(row_hist))
    meta_history_path = write_dataframe(generated_dir / f"hardware_meta_history{suffix}.csv", pd.DataFrame(meta_hist))
    build_method_overview_diagram(root / "to_human" / f"{dataset_name}_鏂规硶鎬诲浘.png")
    build_method_overview_diagram(root / "paper" / "figures" / f"{dataset_name}_鏂规硶鎬诲浘.png")
    return {"bundle_path": str(bundle_path), "response_history": str(response_history_path), "row_history": str(row_history_path), "meta_history": str(meta_history_path)}


def _evaluate_query_prediction(rows: list[dict[str, Any]], budget: BudgetSpec, accuracy_predictor: AccuracyPredictor, cost_predictor: StructuredCostPredictor, feasibility_head: FeasibilityHead, z: torch.Tensor, response: dict[str, torch.Tensor], calibration: torch.Tensor, device: str, static_context: torch.Tensor | None = None) -> dict[str, float]:
    if not rows:
        return {"latency_mae": 0.0, "sram_mae": 0.0, "flash_mae": 0.0, "feasible_acc": 0.0, "accuracy_mae": 0.0}
    arch_x, struct_x, target_x = _rows_to_arch_batch(rows, device)
    z_batch = z.expand(arch_x.shape[0], -1)
    calib_batch = calibration.expand(arch_x.shape[0], -1)
    budget_x = _budget_tensor(budget, device=device, batch_size=arch_x.shape[0])
    if response["gamma"].shape[0] == 1 and arch_x.shape[0] > 1:
        response = {
            "gamma": response["gamma"].expand(arch_x.shape[0], -1, -1),
            "beta_mem": response["beta_mem"].expand(arch_x.shape[0], -1),
            "rho_launch": response["rho_launch"].expand(arch_x.shape[0], -1),
            "rho_copy": response["rho_copy"].expand(arch_x.shape[0], -1),
        }
    true_acc = torch.as_tensor([[float(row.get("accuracy", 0.0))] for row in rows], dtype=torch.float32, device=device)
    feasible_true = torch.as_tensor([[float(float(row["latency_ms"]) <= budget.t_max_ms and float(row["peak_sram_bytes"]) <= budget.m_max_bytes and float(row["flash_bytes"]) <= budget.f_max_bytes)] for row in rows], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_acc = accuracy_predictor(arch_x)
        static_batch = None
        if static_context is not None:
            static_batch = static_context if static_context.ndim == 2 else static_context.unsqueeze(0)
            if static_batch.shape[0] == 1 and arch_x.shape[0] > 1:
                static_batch = static_batch.expand(arch_x.shape[0], -1)
        cost = cost_predictor(arch_x, struct_x, z_batch, response, calib_batch, static_context=static_batch)
        feasible = feasibility_head(arch_x, z_batch, cost, budget_x, calib_batch)
    return {
        "latency_mae": float(torch.mean(torch.abs(cost["latency_ms"] - target_x[:, 0:1])).item()),
        "sram_mae": float(torch.mean(torch.abs(cost["peak_sram_bytes"] - target_x[:, 1:2])).item()),
        "flash_mae": float(torch.mean(torch.abs(cost["flash_bytes"] - target_x[:, 2:3])).item()),
        "feasible_acc": float(torch.mean(((feasible >= 0.5) == (feasible_true >= 0.5)).float()).item()),
        "accuracy_mae": float(torch.mean(torch.abs(pred_acc - true_acc)).item()),
    }


def _evaluate_blackbox_query_prediction(rows: list[dict[str, Any]], budget: BudgetSpec, accuracy_predictor: AccuracyPredictor, blackbox: BlackBoxCostPredictor, static_x: torch.Tensor, device: str, support_rows: list[dict[str, Any]] | None = None) -> dict[str, float]:
    if not rows:
        return {"latency_mae": 0.0, "sram_mae": 0.0, "flash_mae": 0.0, "feasible_acc": 0.0, "accuracy_mae": 0.0}
    calibration_stats = _fit_blackbox_device_calibration(blackbox, support_rows or [], static_x, device)
    true_acc = np.asarray([float(row.get("accuracy", 0.0)) for row in rows], dtype=np.float32)
    pred_acc = []
    latencies, srams, flashes, feasible_hits = [], [], [], []
    for row in rows:
        arch = _row_architecture(row)
        prediction = _predict_blackbox_candidate(arch, blackbox, accuracy_predictor, static_x, budget, device, calibration_stats)
        pred_acc.append(prediction["accuracy"])
        latencies.append(abs(prediction["latency_ms"] - float(row["latency_ms"])))
        srams.append(abs(prediction["peak_sram_bytes"] - float(row["peak_sram_bytes"])))
        flashes.append(abs(prediction["flash_bytes"] - float(row["flash_bytes"])))
        feasible_hits.append(
            float(prediction["feasible_prob"] >= 0.5)
            == float(float(row["latency_ms"]) <= budget.t_max_ms and float(row["peak_sram_bytes"]) <= budget.m_max_bytes and float(row["flash_bytes"]) <= budget.f_max_bytes)
        )
    return {
        "latency_mae": float(np.mean(latencies)),
        "sram_mae": float(np.mean(srams)),
        "flash_mae": float(np.mean(flashes)),
        "feasible_acc": float(np.mean(feasible_hits)),
        "accuracy_mae": float(np.mean(np.abs(np.asarray(pred_acc, dtype=np.float32) - true_acc))),
    }


def _evaluate_hardware_agnostic_query_prediction(
    rows: list[dict[str, Any]],
    budget: BudgetSpec,
    accuracy_predictor: AccuracyPredictor,
    device: str,
) -> dict[str, float]:
    if not rows:
        return {"latency_mae": 0.0, "sram_mae": 0.0, "flash_mae": 0.0, "feasible_acc": 0.0, "accuracy_mae": 0.0}
    true_acc = np.asarray([float(row.get("accuracy", 0.0)) for row in rows], dtype=np.float32)
    pred_acc, latencies, srams, flashes, feasible_hits = [], [], [], [], []
    for row in rows:
        arch = _row_architecture(row)
        prediction = _coarse_architecture_prediction(arch, accuracy_predictor, budget, device)
        pred_acc.append(prediction["accuracy"])
        latencies.append(abs(prediction["latency_ms"] - float(row["latency_ms"])))
        srams.append(abs(prediction["peak_sram_bytes"] - float(row["peak_sram_bytes"])))
        flashes.append(abs(prediction["flash_bytes"] - float(row["flash_bytes"])))
        feasible_hits.append(
            float(prediction["feasible_prob"] >= 0.5)
            == float(
                float(row["latency_ms"]) <= budget.t_max_ms
                and float(row["peak_sram_bytes"]) <= budget.m_max_bytes
                and float(row["flash_bytes"]) <= budget.f_max_bytes
            )
        )
    return {
        "latency_mae": float(np.mean(latencies)),
        "sram_mae": float(np.mean(srams)),
        "flash_mae": float(np.mean(flashes)),
        "feasible_acc": float(np.mean(feasible_hits)),
        "accuracy_mae": float(np.mean(np.abs(np.asarray(pred_acc, dtype=np.float32) - true_acc))),
    }


def _measure_candidates_with_accuracy(backend: HardwareBackend, device_dir: Path, candidates: list[dict[str, Any]], accuracy_lookup: dict[str, float], dataset_name: str, topk_measure: int) -> list[dict[str, Any]]:
    rows = []
    for candidate, measured in zip(candidates[:topk_measure], backend.measure_candidates(device_dir, [item["architecture"] for item in candidates[:topk_measure]])):
        arch = candidate["architecture"]
        latency_ms = float(measured.get("latency_ms", 1.0e12))
        peak_sram_bytes = float(measured.get("peak_sram_bytes", 1.0e12))
        flash_bytes = float(measured.get("flash_bytes", 1.0e12))
        rows.append({"arch_name": arch.name, "arch_repr": arch.compact_repr(), "accuracy": float(accuracy_lookup.get(arch.compact_repr(), synthetic_architecture_accuracy(arch, dataset_name=dataset_name, noise_scale=0.0))), "latency_ms": latency_ms, "peak_sram_bytes": peak_sram_bytes, "flash_bytes": flash_bytes, "pred_score": float(candidate["score"]), "pred_accuracy": float(candidate["prediction"]["accuracy"]), "pred_latency_ms": float(candidate["prediction"]["latency_ms"]), "pred_peak_sram_bytes": float(candidate["prediction"]["peak_sram_bytes"]), "pred_flash_bytes": float(candidate["prediction"]["flash_bytes"]), "pred_feasible_prob": float(candidate["prediction"]["feasible_prob"]), "candidate_source": str(candidate.get("candidate_source", "unknown")), "candidate_rank": int(candidates.index(candidate))})
    return rows


def _measurement_pool_size(requested_topk: int) -> int:
    return max(8, requested_topk)


def _support_rows_as_measured_candidates(rows: list[dict[str, Any]], budget: BudgetSpec, accuracy_lookup: dict[str, float], dataset_name: str, dominant_dims: tuple[str, ...] = ("T", "M", "F")) -> list[dict[str, Any]]:
    measured_rows = []
    seen = set()
    for row in rows:
        arch_repr = row["arch_repr"]
        if arch_repr in seen:
            continue
        raw_accuracy = row.get("accuracy")
        resolved_accuracy = raw_accuracy if raw_accuracy is not None else accuracy_lookup.get(
            arch_repr,
            synthetic_architecture_accuracy(_row_architecture(row), dataset_name=dataset_name, noise_scale=0.0),
        )
        measured = {
            "arch_name": row["arch_name"],
            "arch_repr": arch_repr,
            "accuracy": float(resolved_accuracy),
            "latency_ms": float(row["latency_ms"]),
            "peak_sram_bytes": float(row["peak_sram_bytes"]),
            "flash_bytes": float(row["flash_bytes"]),
        }
        pred = {
            "accuracy": measured["accuracy"],
            "latency_ms": measured["latency_ms"],
            "peak_sram_bytes": measured["peak_sram_bytes"],
            "flash_bytes": measured["flash_bytes"],
            "feasible_prob": float(measured["latency_ms"] <= budget.t_max_ms and measured["peak_sram_bytes"] <= budget.m_max_bytes and measured["flash_bytes"] <= budget.f_max_bytes),
        }
        measured_rows.append(
            {
                **measured,
                "pred_score": float(score_prediction(pred, budget, dominant_dims=dominant_dims)),
                "pred_accuracy": measured["accuracy"],
                "pred_latency_ms": measured["latency_ms"],
                "pred_peak_sram_bytes": measured["peak_sram_bytes"],
                "pred_flash_bytes": measured["flash_bytes"],
                "pred_feasible_prob": pred["feasible_prob"],
                "candidate_source": "support_measurement",
                "candidate_rank": -1,
            }
        )
        seen.add(arch_repr)
    return measured_rows


def _resolve_architecture_by_repr(candidates: list[dict[str, Any]], arch_repr: str, support_rows: list[dict[str, Any]] | None = None) -> ArchitectureSpec:
    for item in candidates:
        if item["architecture"].compact_repr() == arch_repr:
            return item["architecture"]
    for row in support_rows or []:
        if row.get("arch_repr") == arch_repr:
            return _row_architecture(row)
    raise KeyError(f"Unable to resolve architecture for {arch_repr}")


def _select_best_measured_candidate(
    rows: list[dict[str, Any]],
    budget: BudgetSpec,
    method: str = "generic",
    accuracy_band: float = 0.005,
    dominant_dims: tuple[str, ...] = ("T", "M", "F"),
) -> dict[str, Any]:
    for row in rows:
        row["over_budget_dims"] = _over_budget_dims(row, budget)
        row["measured_score"] = float(
            score_prediction(
                {
                    "accuracy": float(row["accuracy"]),
                    "latency_ms": float(row["latency_ms"]),
                    "peak_sram_bytes": float(row["peak_sram_bytes"]),
                    "flash_bytes": float(row["flash_bytes"]),
                    "feasible_prob": float(row["latency_ms"] <= budget.t_max_ms and row["peak_sram_bytes"] <= budget.m_max_bytes and row["flash_bytes"] <= budget.f_max_bytes),
                },
                budget,
                dominant_dims=dominant_dims,
            )
        )
    feasible = [row for row in rows if row["over_budget_dims"] == "none"]
    if feasible:
        max_accuracy = max(float(row["accuracy"]) for row in feasible)
        band_rows = [row for row in feasible if float(row["accuracy"]) >= max_accuracy - accuracy_band]

        def _feasible_priority(row: dict[str, Any]) -> tuple[float, ...]:
            ratios = {
                "T": float(row["latency_ms"]) / max(budget.t_max_ms, 1.0),
                "M": float(row["peak_sram_bytes"]) / max(budget.m_max_bytes, 1.0),
                "F": float(row["flash_bytes"]) / max(budget.f_max_bytes, 1.0),
            }
            dominant = tuple(ratios.get(dim, 0.0) for dim in dominant_dims)
            return dominant + (float(row["accuracy"]), float(row["measured_score"]))

        band_rows.sort(key=_feasible_priority, reverse=True)
        return band_rows[0]

    def _overflow_components(row: dict[str, Any]) -> tuple[float, float, float, float]:
        latency_excess = max(0.0, float(row["latency_ms"]) / max(budget.t_max_ms, 1.0) - 1.0)
        sram_excess = max(0.0, float(row["peak_sram_bytes"]) / max(budget.m_max_bytes, 1.0) - 1.0)
        flash_excess = max(0.0, float(row["flash_bytes"]) / max(budget.f_max_bytes, 1.0) - 1.0)
        overflow = latency_excess + sram_excess + flash_excess
        return overflow, latency_excess, sram_excess, flash_excess

    def _infeasible_priority(row: dict[str, Any]) -> tuple[float, float, float]:
        overflow, latency_excess, sram_excess, flash_excess = _overflow_components(row)
        dominant_shortfall = -sum(
            {
                "T": latency_excess,
                "M": sram_excess,
                "F": flash_excess,
            }.get(dim, 0.0)
            for dim in dominant_dims
        )
        return (-overflow, dominant_shortfall, float(row["accuracy"]))
    ranked = sorted(rows, key=_infeasible_priority, reverse=True)
    min_overflow = _overflow_components(ranked[0])[0]
    near_feasible = [row for row in ranked if _overflow_components(row)[0] <= min_overflow + 0.15]
    if near_feasible:
        near_feasible.sort(
            key=lambda row: (
                float(row["accuracy"]),
                -_overflow_components(row)[0],
                float(row["measured_score"]),
            ),
            reverse=True,
        )
        return near_feasible[0]
    return ranked[0]


def _select_device_items(split_items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_names = [str(name) for name in config.get("device_names", [])]
    if explicit_names:
        by_name = {_manifest_item_name(item): item for item in split_items}
        selected = [by_name[name] for name in explicit_names if name in by_name]
        if selected:
            return selected
    representative_families = list(config.get("representative_families", []))
    max_devices = int(config.get("max_devices", len(split_items)))
    if representative_families:
        grouped_items: dict[str, list[dict[str, Any]]] = {}
        for item in split_items:
            grouped_items.setdefault(str(item["family"]), []).append(item)
        selected = []
        for family in representative_families:
            family_items = grouped_items.get(str(family), [])
            if family_items:
                selected.append(family_items[0])
        return selected[:max_devices]

    selection_strategy = str(config.get("device_selection_strategy", "round_robin_by_family")).lower()
    if selection_strategy == "round_robin_by_family":
        grouped_items: dict[str, list[dict[str, Any]]] = {}
        for item in split_items:
            grouped_items.setdefault(str(item["family"]), []).append(item)
        ordered_families = list(grouped_items.keys())
        items = []
        round_idx = 0
        while len(items) < max_devices:
            added = False
            for family in ordered_families:
                family_items = grouped_items[family]
                if round_idx < len(family_items):
                    items.append(family_items[round_idx])
                    added = True
                    if len(items) >= max_devices:
                        break
            if not added:
                break
            round_idx += 1
        return items
    return split_items[:max_devices]


def _write_board_debug_artifacts(
    output_dir: Path,
    device_name: str,
    method: str,
    candidates: list[dict[str, Any]],
    measured_rows: list[dict[str, Any]],
    selected_row: dict[str, Any],
) -> None:
    device_dir = ensure_dir(output_dir / device_name)
    candidate_frame = pd.DataFrame(
        [
            {
                "method": method,
                "candidate_rank": idx,
                "arch_name": item["architecture"].name,
                "arch_repr": item["architecture"].compact_repr(),
                "candidate_source": item.get("candidate_source", "unknown"),
                "pred_score": item.get("score"),
                "pred_accuracy": item["prediction"]["accuracy"],
                "pred_latency_ms": item["prediction"]["latency_ms"],
                "pred_peak_sram_bytes": item["prediction"]["peak_sram_bytes"],
                "pred_flash_bytes": item["prediction"]["flash_bytes"],
                "pred_feasible_prob": item["prediction"]["feasible_prob"],
            }
            for idx, item in enumerate(candidates)
        ]
    )
    measured_frame = pd.DataFrame(measured_rows)
    if not measured_frame.empty:
        measured_frame["method"] = method
        measured_frame["is_selected"] = measured_frame["arch_repr"].eq(selected_row["arch_repr"]).astype(int)
    selected_frame = pd.DataFrame([{**selected_row, "method": method}])
    write_dataframe(device_dir / f"{method}_鍊欓€夋睜.csv", candidate_frame)
    write_dataframe(device_dir / f"{method}_澶嶆祴缁撴灉.csv", measured_frame)
    write_dataframe(device_dir / f"{method}_鏈€缁堥€夋嫨.csv", selected_frame)


def meta_eval(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    set_seed(int(config.get("seed", 0)))
    dataset_name = config.get("dataset_name", "cifar10")
    device = get_device(config.get("device"))
    generated_dir, checkpoint_dir = _artifact_roots(root, dataset_name)
    manifest = read_json(generated_dir / "devices" / "manifest.json")
    split_items = manifest["splits"][config.get("device_split", "val")]
    items = _select_device_items(split_items, config)
    measurement_frame = read_dataframe(generated_dir / "arch_measurements.parquet")
    measurements_by_device = {name: group.to_dict(orient="records") for name, group in measurement_frame.groupby("device_name")}
    accuracy_lookup = dict(zip(read_dataframe(generated_dir / "accuracy_dataset.parquet")["arch_repr"], read_dataframe(generated_dir / "accuracy_dataset.parquet")["accuracy"]))
    accuracy_predictor = _load_accuracy_predictor(root, device, dataset_name)
    encoder, decoder, cost_predictor, feasibility_head, generator, blackbox, mean_z, mean_calibration = _load_model_bundle(checkpoint_dir / config.get("bundle_filename", "hardware_model_bundle.pt"), device)
    methods = config.get("methods", ["few_shot", "zero_shot", "hardware_agnostic", "random_search", "blackbox_cost_mlp"])
    ablation_mode = _normalize_ablation_mode(str(config.get("ablation_mode", "full")))
    include_task_only_exploration = bool(config.get("include_task_only_exploration", True))
    backend = SyntheticBackend(noise_scale=float(config.get("measurement_noise_scale", 0.0)))
    baselines = _board_baseline_architectures()
    standard_name = config.get("standard_baseline_name", "standard_cnn")
    rows = []
    kshot_rows = []

    base_seed = int(config.get("seed", 0))
    base_seed = int(config.get("seed", 0))
    for device_idx, item in enumerate(items):
        device_dir = Path(item["path"])
        device_record = _load_device_record(device_dir)
        base_rows = measurements_by_device[device_record.static.name]
        device_seed = _stable_seed(base_seed, dataset_name, device_record.static.name)
        rng = np.random.default_rng(device_seed)
        standard_measured = backend.measure_candidates(device_dir, [baselines[standard_name]])[0]
        budget, budget_profile = _budget_from_standard_measurement(
            standard_measured,
            device_record.static.name,
            device_record.static.family,
            config,
        )
        raw_support_rows, raw_query_rows = _sample_support_query_rows(
            base_rows,
            int(config.get("support_size", 8)),
            int(config.get("query_size", 24)),
            rng,
        )
        support_rows = _stamp_budget(raw_support_rows, budget)
        query_rows = _stamp_budget(raw_query_rows, budget)
        static_arr = _static_features(device_record.static)
        probe_arr = np.zeros_like(_probe_tensor(list(device_record.probes))) if ablation_mode in {"no_probes", "static_only"} else _probe_tensor(list(device_record.probes))
        ref_arr = np.zeros_like(_reference_tensor(list(device_record.references))) if ablation_mode in {"no_refs", "static_only"} else _reference_tensor(list(device_record.references))
        static_x = torch.as_tensor(static_arr, dtype=torch.float32, device=device).unsqueeze(0)
        probe_x = torch.as_tensor(probe_arr, dtype=torch.float32, device=device).unsqueeze(0)
        ref_x = torch.as_tensor(ref_arr, dtype=torch.float32, device=device).unsqueeze(0)
        initial_z = encoder(static_x, probe_x, ref_x)
        zero_cal = torch.zeros(1, CALIBRATION_DIM, dtype=torch.float32, device=device)
        zero_response = decoder(initial_z)

        zero_metrics = _evaluate_query_prediction(query_rows, budget, accuracy_predictor, cost_predictor, feasibility_head, initial_z, zero_response, zero_cal, device, static_context=static_x)
        kshot_rows.append(
            {
                "dataset_name": dataset_name,
                "device_name": device_record.static.name,
                "method": "zero_shot",
                "support_size": 0,
                "latency_mae": zero_metrics["latency_mae"],
                "sram_mae": zero_metrics["sram_mae"],
                "flash_mae": zero_metrics["flash_mae"],
            }
        )

        for current_support_size in config.get("support_sizes", []):
            current_support_rows, _ = _sample_support_query_rows(
                base_rows,
                int(current_support_size),
                int(config.get("query_size", 24)),
                np.random.default_rng(_stable_seed(device_seed, f"kshot_{current_support_size}")),
            )
            adapted_z, adapted_c, _ = adapt_device_state(
                decoder,
                cost_predictor,
                feasibility_head,
                initial_z,
                _stamp_budget(current_support_rows, budget),
                initial_calibration=zero_cal,
                steps=int(config.get("adapt_steps", 12)),
                lr=float(config.get("adapt_lr", 5e-3)),
                device=device,
                static_context=static_x,
            )
            metrics = _evaluate_query_prediction(query_rows, budget, accuracy_predictor, cost_predictor, feasibility_head, adapted_z, decoder(adapted_z), adapted_c, device, static_context=static_x)
            kshot_rows.append({"dataset_name": dataset_name, "device_name": device_record.static.name, "method": "few_shot", "support_size": int(current_support_size), "latency_mae": metrics["latency_mae"], "sram_mae": metrics["sram_mae"], "flash_mae": metrics["flash_mae"]})

        for method in methods:
            if method == "few_shot":
                if ablation_mode == "no_adaptation":
                    z, calibration = initial_z, zero_cal
                else:
                    z, calibration, _ = adapt_device_state(decoder, cost_predictor, feasibility_head, initial_z, support_rows, initial_calibration=zero_cal, steps=int(config.get("adapt_steps", 12)), lr=float(config.get("adapt_lr", 5e-3)), device=device, static_context=static_x)
                if ablation_mode == "no_calibration":
                    calibration = torch.zeros_like(calibration)
                response = decoder(z)
                if ablation_mode == "no_response":
                    response = _zero_response_like(response)
            elif method == "hardware_agnostic":
                z, calibration, response = mean_z.to(device), mean_calibration.to(device), decoder(mean_z.to(device))
            else:
                z, calibration, response = initial_z, zero_cal, zero_response
            if ablation_mode == "no_response":
                response = _zero_response_like(response)

            if method == "blackbox_cost_mlp":
                prediction_metrics = _evaluate_blackbox_query_prediction(query_rows, budget, accuracy_predictor, blackbox, static_x[0], device, support_rows=support_rows)
            elif method == "hardware_agnostic":
                prediction_metrics = _evaluate_hardware_agnostic_query_prediction(
                    query_rows,
                    budget,
                    accuracy_predictor,
                    device,
                )
            elif method == "random_search":
                prediction_metrics = {
                    "latency_mae": float("nan"),
                    "sram_mae": float("nan"),
                    "flash_mae": float("nan"),
                    "feasible_acc": float("nan"),
                    "accuracy_mae": float("nan"),
                }
            else:
                prediction_metrics = _evaluate_query_prediction(query_rows, budget, accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, device, static_context=static_x)
            candidates = _search_candidates(method, accuracy_predictor, cost_predictor, feasibility_head, generator, blackbox, z, response, calibration, device_record.static, static_x[0], budget, device, int(config.get("population_size", 48)), int(config.get("rounds", 24)), int(config.get("random_trials", 192)), _stable_seed(device_seed, method), ablation_mode != "no_generator", 0 if ablation_mode == "no_local_refine" else int(config.get("refine_radius", 2)), support_rows=support_rows, dominant_dims=budget_profile["dominant_dims"], include_task_only_exploration=include_task_only_exploration, strict_article_flow=bool(config.get("strict_article_flow", False)))
            if method == "few_shot" and bool(config.get("allow_support_candidates", False)):
                support_architectures = [_row_architecture(row) for row in support_rows]
                candidates = _augment_candidates_with_seed_architectures(candidates, support_architectures, accuracy_predictor, cost_predictor, feasibility_head, z, response, calibration, budget, device, limit=max(int(config.get("topk_measure", 8)), 10), static_context=static_x, dominant_dims=budget_profile["dominant_dims"])
            measured_rows = _measure_candidates_with_accuracy(backend, device_dir, candidates, accuracy_lookup, dataset_name, _measurement_pool_size(int(config.get("topk_measure", 8))))
            if method == "few_shot" and bool(config.get("allow_support_candidates", False)):
                measured_rows.extend(_support_rows_as_measured_candidates(support_rows, budget, accuracy_lookup, dataset_name, dominant_dims=budget_profile["dominant_dims"]))
            selected = _select_best_measured_candidate(measured_rows, budget, method=method, dominant_dims=budget_profile["dominant_dims"])
            rows.append({"dataset_name": dataset_name, "device_name": device_record.static.name, "family": device_record.static.family, "method": method, "ablation_mode": ablation_mode, "support_size": int(config.get("support_size", 8)), "budget_t": budget.t_max_ms, "budget_m": budget.m_max_bytes, "budget_f": budget.f_max_bytes, "accuracy": float(selected["accuracy"]), "latency_ms": float(selected["latency_ms"]), "peak_sram_bytes": float(selected["peak_sram_bytes"]), "flash_bytes": float(selected["flash_bytes"]), "budget_feasible": float(selected["over_budget_dims"] == "none"), "over_budget_dims": selected["over_budget_dims"], "latency_mae": prediction_metrics["latency_mae"], "sram_mae": prediction_metrics["sram_mae"], "flash_mae": prediction_metrics["flash_mae"], "feasible_acc": prediction_metrics["feasible_acc"], "accuracy_mae": prediction_metrics["accuracy_mae"], "arch_name": selected["arch_name"], "arch_repr": selected["arch_repr"]})

    result_frame = pd.DataFrame(rows)
    out_dir = ensure_dir(generated_dir / "meta_eval")
    results_path = write_dataframe(out_dir / f"{config.get('device_split', 'val')}_{ablation_mode}_results.csv", result_frame)
    paper_path = write_dataframe(root / "paper" / "tables" / f"{dataset_name}_{config.get('device_split', 'val')}_{ablation_mode}_results.csv", format_table_for_paper(result_frame))
    summary = result_frame.groupby(["dataset_name", "ablation_mode", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), latency_ms=("latency_ms", "mean"), peak_sram_bytes=("peak_sram_bytes", "mean"), flash_bytes=("flash_bytes", "mean"), budget_feasible=("budget_feasible", "mean"), latency_mae=("latency_mae", "mean"), sram_mae=("sram_mae", "mean"), flash_mae=("flash_mae", "mean")) if not result_frame.empty else pd.DataFrame()
    summary_path = write_dataframe(out_dir / f"{config.get('device_split', 'val')}_{ablation_mode}_summary.csv", summary)
    if not summary.empty:
        if ablation_mode == "full":
            plot_main_result_panels(summary, root / "to_human" / f"{dataset_name}_{config.get('device_split', 'val')}_主结果多面板.png", metric_columns=["accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"], ylabel_map={"accuracy": "准确率", "latency_ms": "延迟 / ms", "peak_sram_bytes": "峰值 SRAM / byte", "flash_bytes": "Flash / byte"})
            plot_main_result_panels(summary, root / "paper" / "figures" / f"{dataset_name}_{config.get('device_split', 'val')}_主结果多面板.png", metric_columns=["accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"], ylabel_map={"accuracy": "准确率", "latency_ms": "延迟 / ms", "peak_sram_bytes": "峰值 SRAM / byte", "flash_bytes": "Flash / byte"})
        else:
            plot_ablation_panels(summary, root / "to_human" / f"{dataset_name}_{config.get('device_split', 'val')}_{ablation_mode}_消融图.png", metric_columns=["latency_mae", "sram_mae", "flash_mae"], ylabel_map={"latency_mae": "延迟预测误差", "sram_mae": "SRAM 预测误差", "flash_mae": "Flash 预测误差"})
            plot_ablation_panels(summary, root / "paper" / "figures" / f"{dataset_name}_{config.get('device_split', 'val')}_{ablation_mode}_消融图.png", metric_columns=["latency_mae", "sram_mae", "flash_mae"], ylabel_map={"latency_mae": "延迟预测误差", "sram_mae": "SRAM 预测误差", "flash_mae": "Flash 预测误差"})
    payload = {"results": str(results_path), "summary": str(summary_path), "paper_table": str(paper_path)}
    if kshot_rows:
        kshot_summary = pd.DataFrame(kshot_rows).groupby(["dataset_name", "method", "support_size"], as_index=False).agg(latency_mae=("latency_mae", "mean"), metric_std=("latency_mae", "std")).fillna({"metric_std": 0.0})
        kshot_path = write_dataframe(out_dir / f"{config.get('device_split', 'val')}_{ablation_mode}_kshot_curve.csv", kshot_summary)
        plot_kshot_curve(kshot_summary, root / "to_human" / f"{dataset_name}_{config.get('device_split', 'val')}_Kshot鏇茬嚎.png")
        plot_kshot_curve(kshot_summary, root / "paper" / "figures" / f"{dataset_name}_{config.get('device_split', 'val')}_Kshot鏇茬嚎.png")
        payload["kshot_table"] = str(kshot_path)
    return payload


def deploy_new_device(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    set_seed(int(config.get("seed", 0)))
    dataset_name = config.get("dataset_name", "cifar10")
    device = get_device(config.get("device"))
    generated_dir, checkpoint_dir = _artifact_roots(root, dataset_name)
    backend = _make_hardware_backend(config, root=root)
    device_dir = Path(config["device_dir"])
    if not device_dir.is_absolute():
        device_dir = root / device_dir
    if not (device_dir / "arch_measurements.jsonl").exists() and str(config.get("backend", "")).lower() in {"command", "hybrid_command_csv_replay"}:
        _collect_real_device_support_rows(backend, config, root)
    static = backend.load_static(device_dir)
    probes = backend.run_micro_probes(device_dir)
    references = backend.run_reference_nets(device_dir)
    encoder, decoder, cost_predictor, feasibility_head, generator, _, _, _ = _load_model_bundle(checkpoint_dir / "hardware_model_bundle.pt", device)
    accuracy_predictor = _load_accuracy_predictor(root, device, dataset_name)
    accuracy_lookup = dict(zip(read_dataframe(generated_dir / "accuracy_dataset.parquet")["arch_repr"], read_dataframe(generated_dir / "accuracy_dataset.parquet")["accuracy"]))
    static_x = torch.as_tensor(_static_features(static), dtype=torch.float32, device=device).unsqueeze(0)
    probe_x = torch.as_tensor(_probe_tensor(probes), dtype=torch.float32, device=device).unsqueeze(0)
    ref_x = torch.as_tensor(_reference_tensor(references), dtype=torch.float32, device=device).unsqueeze(0)
    initial_z = encoder(static_x, probe_x, ref_x)
    support_rows = read_jsonl(device_dir / "arch_measurements.jsonl")[: int(config.get("support_size", 8))] if (device_dir / "arch_measurements.jsonl").exists() else []
    budget = BudgetSpec(**config["budget"]) if config.get("budget") is not None else (_budget_from_rows(support_rows, np.random.default_rng(int(config.get("seed", 0)))) if support_rows else sample_budget(static, seed=int(config.get("seed", 0))))
    budget_profile = _budget_profile_for_device(static.name, static.family, config)
    if support_rows:
        adapted_z, adapted_c, history = adapt_device_state(decoder, cost_predictor, feasibility_head, initial_z, _stamp_budget(support_rows, budget), initial_calibration=torch.zeros(1, CALIBRATION_DIM, device=device), steps=int(config.get("adapt_steps", 16)), lr=float(config.get("adapt_lr", 5e-3)), device=device, static_context=static_x)
    else:
        adapted_z, adapted_c, history = initial_z, torch.zeros(1, CALIBRATION_DIM, device=device), []
    response = decoder(adapted_z)
    generated_arch = generate_architecture_direct(generator, adapted_z, budget, device=device, name="deploy_generated")
    candidates = local_refine_search(
        generated_arch,
        accuracy_predictor=accuracy_predictor,
        cost_predictor=cost_predictor,
        feasibility_head=feasibility_head,
        z=adapted_z,
        response=response,
        calibration=adapted_c,
        budget=budget,
        device=device,
        radius=int(config.get("refine_radius", 1)),
        static_context=static_x,
        dominant_dims=budget_profile["dominant_dims"],
    )
    candidates = _filter_candidate_pool(candidates, _allowed_quants(config))
    if support_rows and bool(config.get("allow_support_candidates", False)):
        support_architectures = [_row_architecture(row) for row in support_rows]
        candidates = _augment_candidates_with_seed_architectures(candidates, support_architectures, accuracy_predictor, cost_predictor, feasibility_head, adapted_z, response, adapted_c, budget, device, limit=max(int(config.get("topk_measure", 10)), 10), static_context=static_x, dominant_dims=budget_profile["dominant_dims"])
    measured_rows = _measure_candidates_with_accuracy(backend, device_dir, candidates, accuracy_lookup, dataset_name, _measurement_pool_size(int(config.get("topk_measure", 10))))
    if support_rows and bool(config.get("allow_support_candidates", False)):
        measured_rows.extend(_support_rows_as_measured_candidates(support_rows, budget, accuracy_lookup, dataset_name, dominant_dims=budget_profile["dominant_dims"]))
    selected = _select_best_measured_candidate(measured_rows, budget, dominant_dims=budget_profile["dominant_dims"])
    deploy_dir = ensure_dir(generated_dir / "deployments" / static.name)
    best_arch = _resolve_architecture_by_repr(candidates, selected["arch_repr"], support_rows)
    write_json(deploy_dir / "best_arch.json", {"device_name": static.name, "budget": budget.to_dict(), "generated_architecture": generated_arch.to_dict(), "architecture": best_arch.to_dict(), "metrics": {"accuracy": selected["accuracy"], "latency_ms": selected["latency_ms"], "peak_sram_bytes": selected["peak_sram_bytes"], "flash_bytes": selected["flash_bytes"]}})
    write_json(deploy_dir / "topk_candidates.json", {"device_name": static.name, "budget": budget.to_dict(), "generated_architecture": generated_arch.to_dict(), "refined_candidates": measured_rows})
    predictions_path = write_dataframe(deploy_dir / "predictions.csv", pd.DataFrame(measured_rows))
    adaptation_path = write_dataframe(deploy_dir / "adaptation_history.csv", pd.DataFrame(history))
    return {
        "best_architecture": str(deploy_dir / "best_arch.json"),
        "refined_candidates": str(deploy_dir / "topk_candidates.json"),
        "predictions": str(predictions_path),
        "adaptation_history": str(adaptation_path),
    }


def collect_real_board_support(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    backend = _make_hardware_backend(config, root=root)
    return _collect_real_device_support_rows(backend, config, root)


def benchmark_new_boards(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    set_seed(int(config.get("seed", 0)))
    dataset_name = config.get("dataset_name", "cifar100")
    device = get_device(config.get("device"))
    generated_dir, checkpoint_dir = _artifact_roots(root, dataset_name)
    manifest = read_json(generated_dir / "devices" / "manifest.json")
    split_items = manifest["splits"][config.get("device_split", "test")]
    items = _select_device_items(split_items, config)
    board_count = len(items)
    measurements_by_device = {name: group.to_dict(orient="records") for name, group in read_dataframe(generated_dir / "arch_measurements.parquet").groupby("device_name")}
    accuracy_lookup = dict(zip(read_dataframe(generated_dir / "accuracy_dataset.parquet")["arch_repr"], read_dataframe(generated_dir / "accuracy_dataset.parquet")["accuracy"]))
    accuracy_predictor = _load_accuracy_predictor(root, device, dataset_name)
    encoder, decoder, cost_predictor, feasibility_head, generator, blackbox, mean_z, mean_calibration = _load_model_bundle(checkpoint_dir / config.get("bundle_filename", "hardware_model_bundle.pt"), device)
    backend = _make_hardware_backend(config, root=root)
    baselines = _board_baseline_architectures()
    standard_name = config.get("standard_baseline_name", "standard_cnn")
    benchmark_methods = list(config.get("methods", ["few_shot", "zero_shot", "hardware_agnostic", "random_search", "blackbox_cost_mlp"]))
    ablation_mode = _normalize_ablation_mode(str(config.get("ablation_mode", "full")))
    include_task_only_exploration = bool(config.get("include_task_only_exploration", True))
    selected_rows = []
    delta_rows = []
    output_subdir = str(config.get("output_subdir", "board_benchmark"))
    debug_subdir = str(config.get("debug_subdir", f"{output_subdir}_debug"))
    figure_tag = str(config.get("figure_tag", "")).strip()
    debug_root = ensure_dir(generated_dir / debug_subdir)
    base_seed = int(config.get("seed", 0))
    for device_idx, item in enumerate(items):
        device_dir = Path(item["path"])
        device_record = _load_device_record(device_dir)
        static_arr = _static_features(device_record.static)
        probe_arr = np.zeros_like(_probe_tensor(list(device_record.probes))) if ablation_mode in {"no_probes", "static_only"} else _probe_tensor(list(device_record.probes))
        ref_arr = np.zeros_like(_reference_tensor(list(device_record.references))) if ablation_mode in {"no_refs", "static_only"} else _reference_tensor(list(device_record.references))
        static_x = torch.as_tensor(static_arr, dtype=torch.float32, device=device).unsqueeze(0)
        probe_x = torch.as_tensor(probe_arr, dtype=torch.float32, device=device).unsqueeze(0)
        ref_x = torch.as_tensor(ref_arr, dtype=torch.float32, device=device).unsqueeze(0)
        initial_z = encoder(static_x, probe_x, ref_x)
        zero_c = torch.zeros(1, CALIBRATION_DIM, device=device)
        base_rows = _load_base_rows_for_device(device_dir, device_record.static.name, measurements_by_device)
        if not base_rows and str(config.get("backend", "")).lower() in {"command", "hybrid_command_csv_replay"}:
            collect_cfg = {
                **config,
                "device_dir": str(device_dir),
                "support_seed_count": max(int(config.get("support_size", 8)), int(config.get("support_seed_count", 16))),
            }
            _collect_real_device_support_rows(backend, collect_cfg, root)
            base_rows = _load_base_rows_for_device(device_dir, device_record.static.name, measurements_by_device)
        device_seed = _stable_seed(base_seed, dataset_name, device_record.static.name)
        rng = np.random.default_rng(device_seed)
        standard_measured = backend.measure_candidates(device_dir, [baselines[standard_name]])[0]
        budget, budget_profile = _budget_from_standard_measurement(
            standard_measured,
            device_record.static.name,
            device_record.static.family,
            config,
        )
        raw_support_rows, _ = _sample_support_query_rows(
            base_rows,
            int(config.get("support_size", 8)),
            0,
            rng,
        )
        support_rows = _stamp_budget(raw_support_rows, budget)
        adapted_z, adapted_c, _ = adapt_device_state(decoder, cost_predictor, feasibility_head, initial_z, support_rows, initial_calibration=zero_c, steps=int(config.get("adapt_steps", 12)), lr=float(config.get("adapt_lr", 5e-3)), device=device, static_context=static_x)
        standard_over_budget_dims = _over_budget_dims(standard_measured, budget)
        standard_row = {"dataset_name": dataset_name, "device_name": device_record.static.name, "family": device_record.static.family, "method": standard_name, "arch_repr": baselines[standard_name].compact_repr(), "accuracy": float(accuracy_lookup.get(baselines[standard_name].compact_repr(), synthetic_architecture_accuracy(baselines[standard_name], dataset_name=dataset_name, noise_scale=0.0))), "latency_ms": float(standard_measured["latency_ms"]), "peak_sram_bytes": float(standard_measured["peak_sram_bytes"]), "flash_bytes": float(standard_measured["flash_bytes"]), "budget_t": budget.t_max_ms, "budget_m": budget.m_max_bytes, "budget_f": budget.f_max_bytes, "budget_feasible": float(standard_over_budget_dims == "none"), "over_budget_dims": standard_over_budget_dims}
        selected_rows.append(standard_row)
        for method in benchmark_methods:
            if method == "few_shot":
                if ablation_mode == "no_adaptation":
                    z, c = initial_z, zero_c
                else:
                    z, c = adapted_z, adapted_c
                if ablation_mode == "no_calibration":
                    c = torch.zeros_like(c)
                response = decoder(z)
                if ablation_mode == "no_response":
                    response = _zero_response_like(response)
            elif method == "hardware_agnostic":
                z, c, response = mean_z.to(device), mean_calibration.to(device), decoder(mean_z.to(device))
            else:
                z, c, response = initial_z, zero_c, decoder(initial_z)
            if ablation_mode == "no_response":
                response = _zero_response_like(response)
            candidates = _search_candidates(
                method,
                accuracy_predictor,
                cost_predictor,
                feasibility_head,
                generator,
                blackbox,
                z,
                response,
                c,
                device_record.static,
                static_x[0],
                budget,
                device,
                int(config.get("population_size", 48)),
                int(config.get("rounds", 24)),
                int(config.get("random_trials", 192)),
                _stable_seed(device_seed, method),
                ablation_mode != "no_generator",
                0 if ablation_mode == "no_local_refine" else int(config.get("refine_radius", 2)),
                support_rows=support_rows,
                dominant_dims=budget_profile["dominant_dims"],
                include_task_only_exploration=include_task_only_exploration,
                strict_article_flow=bool(config.get("strict_article_flow", False)),
            )
            candidates = _filter_candidate_pool(candidates, _allowed_quants(config))
            if method == "few_shot" and bool(config.get("allow_support_candidates", False)):
                support_architectures = [_row_architecture(row) for row in support_rows]
                candidates = _augment_candidates_with_seed_architectures(candidates, support_architectures, accuracy_predictor, cost_predictor, feasibility_head, z, response, c, budget, device, limit=max(int(config.get("topk_measure", 8)), 10), static_context=static_x, dominant_dims=budget_profile["dominant_dims"])
            measured_rows = _measure_candidates_with_accuracy(backend, device_dir, candidates, accuracy_lookup, dataset_name, _measurement_pool_size(int(config.get("topk_measure", 8))))
            if method == "few_shot" and bool(config.get("allow_support_candidates", False)):
                measured_rows.extend(_support_rows_as_measured_candidates(support_rows, budget, accuracy_lookup, dataset_name, dominant_dims=budget_profile["dominant_dims"]))
            selected = _select_best_measured_candidate(measured_rows, budget, method=method, dominant_dims=budget_profile["dominant_dims"])
            _write_board_debug_artifacts(debug_root, device_record.static.name, method, candidates, measured_rows, selected)
            selected_rows.append({"dataset_name": dataset_name, "device_name": device_record.static.name, "family": device_record.static.family, "method": method, "arch_repr": selected["arch_repr"], "accuracy": float(selected["accuracy"]), "latency_ms": float(selected["latency_ms"]), "peak_sram_bytes": float(selected["peak_sram_bytes"]), "flash_bytes": float(selected["flash_bytes"]), "budget_t": budget.t_max_ms, "budget_m": budget.m_max_bytes, "budget_f": budget.f_max_bytes, "budget_feasible": float(selected["over_budget_dims"] == "none"), "over_budget_dims": selected["over_budget_dims"]})
            if method == "few_shot":
                delta_rows.append({"dataset_name": dataset_name, "device_name": device_record.static.name, "family": device_record.static.family, "accuracy_delta": float(selected["accuracy"] - standard_row["accuracy"]), "latency_reduction_ratio": float((standard_row["latency_ms"] - selected["latency_ms"]) / max(standard_row["latency_ms"], 1e-6)), "sram_reduction_ratio": float((standard_row["peak_sram_bytes"] - selected["peak_sram_bytes"]) / max(standard_row["peak_sram_bytes"], 1e-6)), "flash_reduction_ratio": float((standard_row["flash_bytes"] - selected["flash_bytes"]) / max(standard_row["flash_bytes"], 1e-6)), "few_shot_dominates_standard": float(selected["accuracy"] >= standard_row["accuracy"] and selected["latency_ms"] <= standard_row["latency_ms"] and selected["peak_sram_bytes"] <= standard_row["peak_sram_bytes"] and selected["flash_bytes"] <= standard_row["flash_bytes"])})
    selected_frame = pd.DataFrame(selected_rows)
    delta_frame = pd.DataFrame(delta_rows)
    summary_frame = selected_frame.groupby(["dataset_name", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), latency_ms=("latency_ms", "mean"), peak_sram_bytes=("peak_sram_bytes", "mean"), flash_bytes=("flash_bytes", "mean"), budget_feasible=("budget_feasible", "mean"))
    output_dir = ensure_dir(generated_dir / output_subdir)
    selected_path = write_dataframe(output_dir / "selected_rows.csv", selected_frame)
    summary_path = write_dataframe(output_dir / "summary_by_method.csv", summary_frame)
    delta_path = write_dataframe(output_dir / "device_delta_vs_standard.csv", delta_frame)
    tag_suffix = f"_{figure_tag}" if figure_tag else ""
    comparison_frame = selected_frame[
        selected_frame["method"].isin(
            [standard_name, "few_shot", "zero_shot", "hardware_agnostic", "random_search", "blackbox_cost_mlp"]
        )
    ].copy()
    board_ylabel_map = {
        "accuracy": "准确率",
        "latency_ms": "延迟 / ms",
        "peak_sram_bytes": "峰值 SRAM / byte",
        "flash_bytes": "Flash / byte",
    }
    plot_board_method_panels(
        comparison_frame,
        root / "to_human" / f"{dataset_name}_{board_count}_board_method_panels{tag_suffix}.png",
        metric_columns=["accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"],
        xlabel_map=board_ylabel_map,
    )
    plot_board_method_panels(
        comparison_frame,
        root / "paper" / "figures" / f"{dataset_name}_{board_count}_board_method_panels{tag_suffix}.png",
        metric_columns=["accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"],
        xlabel_map=board_ylabel_map,
    )
    plot_board_improvement(delta_frame, root / "to_human" / f"{dataset_name}_{board_count}_board_improvement{tag_suffix}.png")
    plot_board_improvement(
        delta_frame,
        root / "paper" / "figures" / f"{dataset_name}_{board_count}_board_improvement{tag_suffix}.png",
    )
    if bool(config.get("export_legacy_paper_tables", False)):
        for target, frame in (
            (root / "paper" / "tables" / f"{dataset_name}_board_selected_rows.csv", format_table_for_paper(selected_frame)),
            (root / "paper" / "tables" / f"{dataset_name}_board_summary.csv", format_table_for_paper(summary_frame)),
            (root / "paper" / "tables" / f"{dataset_name}_board_delta_vs_standard.csv", format_table_for_paper(delta_frame)),
        ):
            try:
                write_dataframe(target, frame)
            except PermissionError:
                pass
    return {"selected_rows": str(selected_path), "summary_table": str(summary_path), "device_delta_table": str(delta_path)}


def export_result_tables(results: str | Path | pd.DataFrame, root: str | Path = ".", dataset_name: str | None = None) -> dict[str, Any]:
    root = Path(root)
    frame = results if isinstance(results, pd.DataFrame) else read_dataframe(results)
    if dataset_name is not None:
        frame = frame[frame["dataset_name"] == dataset_name].copy()
    out_dir = ensure_dir(root / "paper" / "tables")
    paper_frame = format_table_for_paper(frame, rename_columns=False)
    full_label = "完整方案"
    ablation_keep = [
        "完整方案",
        "去除微算子探针",
        "去除参考网络探针",
        "去除全部探针，仅保留静态硬件特征",
        "去除响应解码层",
        "去除设备校准",
        "去除元学习适配",
        "去除局部精修",
        "去除可部署性判别损失",
        "去除网络参数生成损失",
        "去除硬件响应辅助监督项",
    ]
    if "ablation_mode" in paper_frame.columns:
        main_source = paper_frame[paper_frame["ablation_mode"] == full_label].copy()
        if main_source.empty:
            main_source = paper_frame.copy()
    else:
        main_source = paper_frame.copy()
    main_table = main_source.groupby(["dataset_name", "method"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        latency_ms=("latency_ms", "mean"),
        peak_sram_bytes=("peak_sram_bytes", "mean"),
        flash_bytes=("flash_bytes", "mean"),
    )

    ablation_mask = paper_frame["method"] == "本文方法" if "method" in paper_frame.columns else pd.Series(False, index=paper_frame.index)
    if "ablation_mode" in paper_frame.columns:
        ablation_mask &= paper_frame["ablation_mode"].isin(ablation_keep)
    ablation_source = paper_frame[ablation_mask].copy()
    ablation_table = ablation_source.groupby(["dataset_name", "ablation_mode", "method"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        latency_ms=("latency_ms", "mean"),
        peak_sram_bytes=("peak_sram_bytes", "mean"),
        flash_bytes=("flash_bytes", "mean"),
    ) if not ablation_source.empty else pd.DataFrame(columns=["dataset_name", "ablation_mode", "method", "accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"])

    if "latency_mae" in paper_frame.columns and "sram_mae" in paper_frame.columns and "flash_mae" in paper_frame.columns:
        cost_mae_source = main_source.copy()
        cost_mae_table = cost_mae_source.groupby(["dataset_name", "method"], as_index=False).agg(
            latency_mae=("latency_mae", "mean"),
            sram_mae=("sram_mae", "mean"),
            flash_mae=("flash_mae", "mean"),
        )
    else:
        cost_mae_table = pd.DataFrame(columns=["dataset_name", "method", "latency_mae", "sram_mae", "flash_mae"])
    main_table = format_table_for_paper(main_table)
    ablation_table = format_table_for_paper(ablation_table)
    cost_mae_table = format_table_for_paper(cost_mae_table)
    suffix = dataset_name or "overall"
    main_path = write_dataframe(out_dir / f"{suffix}_main_results.csv", main_table)
    ablation_path = write_dataframe(out_dir / f"{suffix}_ablation_results.csv", ablation_table)
    cost_mae_path = write_dataframe(out_dir / f"{suffix}_cost_mae_results.csv", cost_mae_table)
    if dataset_name is not None:
        write_dataframe(out_dir / "模块与损失函数消融结果表.csv", ablation_table)
    ablation_plot_frame = ablation_source.groupby(["dataset_name", "ablation_mode"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        latency_ms=("latency_ms", "mean"),
        peak_sram_bytes=("peak_sram_bytes", "mean"),
        flash_bytes=("flash_bytes", "mean"),
    ) if not ablation_source.empty else pd.DataFrame()
    if not ablation_plot_frame.empty:
        plot_ablation_panels(
            ablation_plot_frame,
            root / "to_human" / f"{suffix}_combined_ablation_panels.png",
            metric_columns=["accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"],
            ylabel_map={"accuracy": "准确率", "latency_ms": "延迟 / ms", "peak_sram_bytes": "峰值 SRAM / byte", "flash_bytes": "Flash / byte"},
        )
        plot_ablation_panels(
            ablation_plot_frame,
            root / "paper" / "figures" / f"{suffix}_combined_ablation_panels.png",
            metric_columns=["accuracy", "latency_ms", "peak_sram_bytes", "flash_bytes"],
            ylabel_map={"accuracy": "准确率", "latency_ms": "延迟 / ms", "peak_sram_bytes": "峰值 SRAM / byte", "flash_bytes": "Flash / byte"},
        )
    markdown_exports = [
        (out_dir / f"{suffix}_main_results.md", f"主要结果（{suffix}）", main_table),
        (out_dir / f"{suffix}_ablation_results.md", f"消融结果（{suffix}）", ablation_table),
        (out_dir / f"{suffix}_cost_mae_results.md", f"代价预测误差（{suffix}）", cost_mae_table),
    ]
    for path, title, table in markdown_exports:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"# {title}\n\n{_dataframe_to_markdown(table)}")
    return {"main_table": str(main_path), "ablation_table": str(ablation_path), "cost_mae_table": str(cost_mae_path)}


def run_experiment_suite(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    dataset_name = config.get("dataset_name", "cifar10")
    main_cfg = dict(config.get("main_eval", {}))
    main_cfg.setdefault("dataset_name", dataset_name)
    main_cfg.setdefault("ablation_mode", "full")
    main_cfg.setdefault("support_sizes", config.get("support_sizes", [2, 4, 8, 16]))
    result_paths = [meta_eval(main_cfg, root=root)["results"]]
    module_ablation_train_base = dict(
        config.get(
            "module_ablation_train_base",
            config.get("loss_ablation_train_base", config.get("train_hardware_models", {})),
        )
    )
    retrainable_module_ablations: dict[str, dict[str, Any]] = {
        "no_probes": {"probe_mode": "zero"},
        "no_refs": {"ref_mode": "zero"},
        "static_only": {"probe_mode": "zero", "ref_mode": "zero"},
        "no_response": {"disable_response": True, "response_aux_weight": 0.0, "response_epochs": 0},
    }
    for ablation_mode in config.get("ablations", ["no_probes", "no_refs", "static_only", "no_response", "no_calibration", "no_adaptation", "no_local_refine"]):
        ablation_cfg = dict(main_cfg)
        ablation_cfg["ablation_mode"] = ablation_mode
        ablation_cfg["methods"] = config.get("ablation_methods", ["few_shot"])
        if ablation_mode in retrainable_module_ablations:
            train_cfg = dict(module_ablation_train_base)
            train_cfg.setdefault("dataset_name", dataset_name)
            train_cfg.update(retrainable_module_ablations[ablation_mode])
            train_cfg.setdefault("bundle_filename", f"hardware_model_bundle_{ablation_mode}.pt")
            train_cfg.setdefault("artifact_tag", ablation_mode)
            train_hardware_models(train_cfg, root=root)
            ablation_cfg["bundle_filename"] = train_cfg["bundle_filename"]
        result_paths.append(meta_eval(ablation_cfg, root=root)["results"])
    for item in config.get("loss_ablations", []):
        name = str(item["name"])
        train_cfg = dict(config.get("loss_ablation_train_base", {}))
        train_cfg.setdefault("dataset_name", dataset_name)
        train_cfg.update(item.get("train_overrides", {}))
        train_cfg.setdefault("bundle_filename", f"hardware_model_bundle_{name}.pt")
        train_cfg.setdefault("artifact_tag", name)
        train_hardware_models(train_cfg, root=root)
        loss_eval_cfg = dict(main_cfg)
        loss_eval_cfg["ablation_mode"] = name
        loss_eval_cfg["methods"] = ["few_shot"]
        loss_eval_cfg["bundle_filename"] = train_cfg["bundle_filename"]
        loss_eval_cfg.update(item.get("eval_overrides", {}))
        result_paths.append(meta_eval(loss_eval_cfg, root=root)["results"])
    merged = pd.concat([read_dataframe(path) for path in result_paths], ignore_index=True)
    generated_dir, _ = _artifact_roots(root, dataset_name)
    merged_path = write_dataframe(generated_dir / "experiment_suite.csv", merged)
    return {"merged_results": str(merged_path), **export_result_tables(merged, root=root, dataset_name=dataset_name)}
