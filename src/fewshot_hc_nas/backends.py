from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .board_serial import JsonlSerialBoardClient, SerialBoardConfig
from .hardware import predict_cost_from_response
from .io import read_json, read_jsonl, read_yaml, write_json, write_jsonl
from .types import ArchitectureSpec, HardwareResponseCoefficients, HardwareStaticSpec, ProbeMeasurement, ReferenceMeasurement


def _static_from_dict(payload: dict) -> HardwareStaticSpec:
    return HardwareStaticSpec(**payload)


def _probe_from_dict(payload: dict) -> ProbeMeasurement:
    latency_ms = float(payload["latency_ms"])
    macs = float(payload.get("macs", 0.0))
    bytes_moved = float(payload.get("bytes", 0.0))
    latency_per_mac = payload.get("latency_per_mac")
    latency_per_byte = payload.get("latency_per_byte")
    return ProbeMeasurement(
        probe_id=payload["probe_id"],
        op=payload["op"],
        quant=int(payload["quant"]),
        input_shape=tuple(payload["input_shape"]),
        latency_ms=latency_ms,
        latency_per_mac=float(latency_per_mac) if latency_per_mac is not None else (latency_ms / max(macs, 1.0)),
        latency_per_byte=float(latency_per_byte) if latency_per_byte is not None else (latency_ms / max(bytes_moved, 1.0)),
    )


def _reference_from_dict(payload: dict) -> ReferenceMeasurement:
    return ReferenceMeasurement(
        name=payload["name"],
        architecture=ArchitectureSpec.from_dict(payload["architecture"]),
        latency_ms=float(payload["latency_ms"]),
        peak_sram_bytes=float(payload["peak_sram_bytes"]),
        flash_bytes=float(payload["flash_bytes"]),
        accuracy=None if payload.get("accuracy") is None else float(payload["accuracy"]),
    )


class HardwareBackend(ABC):
    @abstractmethod
    def load_static(self, device_dir: str | Path) -> HardwareStaticSpec:
        raise NotImplementedError

    @abstractmethod
    def run_micro_probes(self, device_dir: str | Path) -> list[ProbeMeasurement]:
        raise NotImplementedError

    @abstractmethod
    def run_reference_nets(self, device_dir: str | Path) -> list[ReferenceMeasurement]:
        raise NotImplementedError

    @abstractmethod
    def measure_candidates(self, device_dir: str | Path, architectures: Iterable[ArchitectureSpec]) -> list[dict]:
        raise NotImplementedError


def _merge_rows_by_arch(path: Path, rows: list[dict]) -> None:
    existing = read_jsonl(path) if path.exists() else []
    merged = {str(row.get("arch_repr", "")): row for row in existing}
    for row in rows:
        key = str(row.get("arch_repr", ""))
        if key:
            merged[key] = row
    write_jsonl(path, list(merged.values()))


class SyntheticBackend(HardwareBackend):
    def __init__(self, noise_scale: float = 0.01):
        self.noise_scale = noise_scale

    def load_static(self, device_dir: str | Path) -> HardwareStaticSpec:
        return _static_from_dict(read_json(Path(device_dir) / "hardware_static.json"))

    def _load_response(self, device_dir: str | Path) -> HardwareResponseCoefficients:
        return HardwareResponseCoefficients.from_dict(read_json(Path(device_dir) / "hardware_response.json"))

    def run_micro_probes(self, device_dir: str | Path) -> list[ProbeMeasurement]:
        return [_probe_from_dict(row) for row in read_jsonl(Path(device_dir) / "probe_results.jsonl")]

    def run_reference_nets(self, device_dir: str | Path) -> list[ReferenceMeasurement]:
        return [_reference_from_dict(row) for row in read_jsonl(Path(device_dir) / "reference_results.jsonl")]

    def measure_candidates(self, device_dir: str | Path, architectures: Iterable[ArchitectureSpec]) -> list[dict]:
        static = self.load_static(device_dir)
        response = self._load_response(device_dir)
        rows = []
        for arch in architectures:
            latency, sram, flash = predict_cost_from_response(arch, static, response, noise_scale=self.noise_scale)
            rows.append(
                {
                    "arch_name": arch.name,
                    "arch_repr": arch.compact_repr(),
                    "latency_ms": latency,
                    "peak_sram_bytes": sram,
                    "flash_bytes": flash,
                }
            )
        return rows


class CSVReplayBackend(HardwareBackend):
    def load_static(self, device_dir: str | Path) -> HardwareStaticSpec:
        return _static_from_dict(read_json(Path(device_dir) / "hardware_static.json"))

    def run_micro_probes(self, device_dir: str | Path) -> list[ProbeMeasurement]:
        return [_probe_from_dict(row) for row in read_jsonl(Path(device_dir) / "probe_results.jsonl")]

    def run_reference_nets(self, device_dir: str | Path) -> list[ReferenceMeasurement]:
        return [_reference_from_dict(row) for row in read_jsonl(Path(device_dir) / "reference_results.jsonl")]

    def measure_candidates(self, device_dir: str | Path, architectures: Iterable[ArchitectureSpec]) -> list[dict]:
        table_path = Path(device_dir) / "arch_measurements.jsonl"
        if not table_path.exists():
            return SyntheticBackend(noise_scale=0.0).measure_candidates(device_dir, architectures)
        rows = read_jsonl(table_path)
        lookup = {row["arch_repr"]: row for row in rows}
        architectures = list(architectures)
        fallback_rows = SyntheticBackend(noise_scale=0.0).measure_candidates(device_dir, architectures)
        fallback_lookup = {row["arch_repr"]: row for row in fallback_rows}
        measured = []
        for arch in architectures:
            arch_repr = arch.compact_repr()
            measured.append(
                lookup.get(
                    arch_repr,
                    fallback_lookup.get(
                        arch_repr,
                        {
                            "arch_name": arch.name,
                            "arch_repr": arch_repr,
                            "latency_ms": 1.0e12,
                            "peak_sram_bytes": 1.0e12,
                            "flash_bytes": 1.0e12,
                            "unsupported": True,
                        },
                    ),
                )
            )
        return measured


class CommandBackend(HardwareBackend):
    def __init__(self, command_config: str | Path | dict[str, object], client: JsonlSerialBoardClient | None = None):
        self.command_config = command_config
        self._raw_config = read_yaml(command_config) if isinstance(command_config, (str, Path)) else dict(command_config)
        self._client = client
        self.supported_quants = tuple(int(v) for v in self._raw_config.get("supported_quants", [4, 8]))
        self.recover_command = self._raw_config.get("recover_command")

    def _board_client(self) -> JsonlSerialBoardClient:
        if self._client is None:
            self._client = JsonlSerialBoardClient(SerialBoardConfig.from_any(self._raw_config))
        return self._client

    def _recover_board(self) -> None:
        self._board_client().close()
        if not self.recover_command:
            return
        if isinstance(self.recover_command, str):
            subprocess.run(self.recover_command, check=True, timeout=600, shell=True)
            return
        subprocess.run([str(item) for item in self.recover_command], check=True, timeout=600)

    def _unsupported_measurement(self, arch: ArchitectureSpec) -> dict:
        return {
            "arch_name": arch.name,
            "arch_repr": arch.compact_repr(),
            "latency_ms": 1.0e12,
            "peak_sram_bytes": 1.0e12,
            "flash_bytes": 1.0e12,
            "unsupported": True,
        }

    def load_static(self, device_dir: str | Path) -> HardwareStaticSpec:
        device_dir = Path(device_dir)
        static_path = device_dir / "hardware_static.json"
        if static_path.exists():
            return _static_from_dict(read_json(static_path))
        response = self._board_client().command({"cmd": "get_static"})
        write_json(static_path, response["static"])
        return _static_from_dict(response["static"])

    def run_micro_probes(self, device_dir: str | Path) -> list[ProbeMeasurement]:
        device_dir = Path(device_dir)
        path = device_dir / "probe_results.jsonl"
        if path.exists():
            return [_probe_from_dict(row) for row in read_jsonl(path)]
        response = self._board_client().command({"cmd": "run_probe_suite"})
        rows = list(response["rows"])
        write_jsonl(path, rows)
        return [_probe_from_dict(row) for row in rows]

    def run_reference_nets(self, device_dir: str | Path) -> list[ReferenceMeasurement]:
        device_dir = Path(device_dir)
        path = device_dir / "reference_results.jsonl"
        if path.exists():
            return [_reference_from_dict(row) for row in read_jsonl(path)]
        response = self._board_client().command({"cmd": "run_reference_suite"})
        rows = list(response["rows"])
        write_jsonl(path, rows)
        return [_reference_from_dict(row) for row in rows]

    def measure_candidates(self, device_dir: str | Path, architectures: Iterable[ArchitectureSpec]) -> list[dict]:
        device_dir = Path(device_dir)
        table_path = device_dir / "arch_measurements.jsonl"
        cached_rows = read_jsonl(table_path) if table_path.exists() else []
        cached_lookup = {str(row.get("arch_repr", "")): row for row in cached_rows}
        measured_rows: list[dict] = []
        for arch in architectures:
            if any(block.quant not in self.supported_quants for block in arch.blocks):
                measured_rows.append(self._unsupported_measurement(arch))
                continue
            cached = cached_lookup.get(arch.compact_repr())
            if cached is not None:
                measured_rows.append(dict(cached))
                continue
            try:
                response = self._board_client().command(
                    {
                        "cmd": "measure_arch",
                        "arch_name": arch.name,
                        "arch_repr": arch.compact_repr(),
                        "arch": arch.to_dict(),
                    }
                )
                row = dict(response["row"])
                row.setdefault("arch_name", arch.name)
                row.setdefault("arch_repr", arch.compact_repr())
                row.setdefault("architecture_json", json.dumps(arch.to_dict(), ensure_ascii=False))
                row.setdefault("accuracy", None)
                measured_rows.append(row)
            except Exception:
                self._recover_board()
                measured_rows.append(self._unsupported_measurement(arch))
        persisted = [row for row in measured_rows if row.get("arch_repr")]
        if persisted:
            _merge_rows_by_arch(table_path, persisted)
        return measured_rows


class HybridBackend(HardwareBackend):
    def __init__(
        self,
        command_backend: CommandBackend,
        fallback_backend: HardwareBackend,
        command_devices: Iterable[str] | None = None,
        command_families: Iterable[str] | None = None,
    ):
        self.command_backend = command_backend
        self.fallback_backend = fallback_backend
        self.command_devices = {str(item) for item in (command_devices or [])}
        self.command_families = {str(item) for item in (command_families or [])}

    def _use_command(self, device_dir: str | Path) -> bool:
        device_dir = Path(device_dir)
        if device_dir.name in self.command_devices:
            return True
        static_path = device_dir / "hardware_static.json"
        if static_path.exists():
            family = str(read_json(static_path).get("family", ""))
            if family in self.command_families:
                return True
        return "real" in {part.lower() for part in device_dir.parts}

    def _backend(self, device_dir: str | Path) -> HardwareBackend:
        return self.command_backend if self._use_command(device_dir) else self.fallback_backend

    def load_static(self, device_dir: str | Path) -> HardwareStaticSpec:
        return self._backend(device_dir).load_static(device_dir)

    def run_micro_probes(self, device_dir: str | Path) -> list[ProbeMeasurement]:
        return self._backend(device_dir).run_micro_probes(device_dir)

    def run_reference_nets(self, device_dir: str | Path) -> list[ReferenceMeasurement]:
        return self._backend(device_dir).run_reference_nets(device_dir)

    def measure_candidates(self, device_dir: str | Path, architectures: Iterable[ArchitectureSpec]) -> list[dict]:
        return self._backend(device_dir).measure_candidates(device_dir, architectures)
