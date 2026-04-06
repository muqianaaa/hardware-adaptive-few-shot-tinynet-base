from __future__ import annotations

import json
import tempfile
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_hc_nas.adaptation import adapt_device_embedding, adapt_device_state
from fewshot_hc_nas.backends import CSVReplayBackend, CommandBackend, HybridBackend, SyntheticBackend
from fewshot_hc_nas.hardware import build_arch_measurement_table, create_device_record, export_device_directory
from fewshot_hc_nas.models import FeasibilityHead, ResponseDecoder, StructuredCostPredictor
from fewshot_hc_nas.search_space import sample_architecture
from fewshot_hc_nas.types import ArchitectureSpec


class _FakeBoardClient:
    def __init__(self) -> None:
        self.commands: list[dict] = []

    def command(self, payload: dict) -> dict:
        self.commands.append(dict(payload))
        cmd = payload["cmd"]
        if cmd == "get_static":
            return {
                "ok": True,
                "cmd": cmd,
                "static": {
                    "name": "stm32f405rgt6_000",
                    "family": "stm32f405rgt6_real",
                    "sram_bytes": 196608,
                    "flash_bytes": 1048576,
                    "freq_mhz": 168.0,
                    "dsp": 1.0,
                    "simd": 1.0,
                    "cache_kb": 0.0,
                    "bus_width": 32.0,
                    "kernel_int8": 1.0,
                    "kernel_int4": 1.0,
                    "kernel_int2": 0.0,
                    "runtime_type": "cmsis_nn",
                },
            }
        if cmd == "run_probe_suite":
            return {
                "ok": True,
                "cmd": cmd,
                "rows": [
                    {
                        "probe_id": f"probe_{idx}",
                        "op": "std3x3",
                        "quant": 8 if idx < 8 else 2,
                        "input_shape": [1, 16, 16, 16],
                        "latency_ms": 1.0 + idx,
                        "latency_per_mac": 1.0e-6 + idx * 1.0e-7,
                        "latency_per_byte": 1.0e-4 + idx * 1.0e-5,
                    }
                    for idx in range(9)
                ],
            }
        if cmd == "run_reference_suite":
            return {
                "ok": True,
                "cmd": cmd,
                "rows": [
                    {
                        "name": f"ref_{idx}",
                        "architecture": {
                            "name": f"ref_{idx}",
                            "blocks": [
                                {"op": "std3x3", "width": 1.0, "depth": 1, "quant": 8},
                                {"op": "dw_sep", "width": 0.75, "depth": 1, "quant": 8},
                                {"op": "mbconv", "width": 1.0, "depth": 1, "quant": 4},
                                {"op": "dw_sep", "width": 1.0, "depth": 1, "quant": 4},
                                {"op": "mbconv", "width": 1.0, "depth": 1, "quant": 4},
                            ],
                        },
                        "latency_ms": 10.0 + idx,
                        "peak_sram_bytes": 16384.0 + idx,
                        "flash_bytes": 65536.0 + idx,
                        "accuracy": None,
                    }
                    for idx in range(3)
                ],
            }
        if cmd == "measure_arch":
            arch_name = payload["arch_name"]
            arch_repr = payload["arch_repr"]
            return {
                "ok": True,
                "cmd": cmd,
                "row": {
                    "arch_name": arch_name,
                    "arch_repr": arch_repr,
                    "latency_ms": 12.5,
                    "peak_sram_bytes": 54321.0,
                    "flash_bytes": 76543.0,
                },
            }
        raise AssertionError(f"Unexpected command: {payload}")


class BackendAdaptationTest(unittest.TestCase):
    def test_synthetic_and_csv_backend(self) -> None:
        device = create_device_record("balanced_mcu", index=0, seed=21)
        archs = [sample_architecture(seed=idx) for idx in range(4)]
        table = build_arch_measurement_table(archs, [device])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_device_directory(device, root)
            root.joinpath("arch_measurements.jsonl").write_text(
                "\n".join(table.to_json(orient="records", lines=True).splitlines()),
                encoding="utf-8",
            )
            syn = SyntheticBackend()
            csvb = CSVReplayBackend()
            self.assertEqual(len(syn.run_micro_probes(root)), len(device.probes))
            self.assertEqual(len(csvb.measure_candidates(root, archs)), 4)

    def test_command_backend_and_hybrid_backend(self) -> None:
        archs = [
            ArchitectureSpec.from_dict(
                {
                    "name": "arch_0",
                    "blocks": [
                        {"op": "std3x3", "width": 0.75, "depth": 1, "quant": 8},
                        {"op": "dw_sep", "width": 0.75, "depth": 1, "quant": 4},
                        {"op": "mbconv", "width": 1.0, "depth": 1, "quant": 8},
                        {"op": "std3x3", "width": 1.0, "depth": 1, "quant": 4},
                        {"op": "mbconv", "width": 1.0, "depth": 1, "quant": 8},
                    ],
                }
            ),
            ArchitectureSpec.from_dict(
                {
                    "name": "arch_1",
                    "blocks": [
                        {"op": "dw_sep", "width": 0.5, "depth": 2, "quant": 8},
                        {"op": "dw_sep", "width": 0.75, "depth": 1, "quant": 4},
                        {"op": "mbconv", "width": 0.75, "depth": 1, "quant": 8},
                        {"op": "dw_sep", "width": 1.0, "depth": 1, "quant": 4},
                        {"op": "mbconv", "width": 1.0, "depth": 1, "quant": 8},
                    ],
                }
            ),
        ]
        unsupported = sample_architecture(seed=2, name="arch_2").to_dict()
        unsupported["blocks"][0]["quant"] = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            device_dir = root / "devices" / "real" / "stm32f405rgt6_000"
            fallback_dir = root / "devices" / "balanced_mcu_000"
            device_dir.mkdir(parents=True, exist_ok=True)
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fake_client = _FakeBoardClient()
            backend = CommandBackend(
                {"serial": {"port": "COM5"}, "supported_quants": [4, 8]},
                client=fake_client,
            )
            static = backend.load_static(device_dir)
            probes = backend.run_micro_probes(device_dir)
            refs = backend.run_reference_nets(device_dir)
            measured = backend.measure_candidates(
                device_dir,
                [*archs, type(archs[0]).from_dict(unsupported)],
            )
            self.assertEqual(static.name, "stm32f405rgt6_000")
            self.assertEqual(len(probes), 9)
            self.assertEqual(len(refs), 3)
            self.assertEqual(len(measured), 3)
            self.assertIn("architecture_json", measured[0])
            self.assertTrue(measured[2]["unsupported"])
            rows = (device_dir / "arch_measurements.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            persisted = json.loads(rows[0])
            self.assertIn("architecture_json", persisted)

            device = create_device_record("balanced_mcu", index=0, seed=21)
            export_device_directory(device, fallback_dir)
            table = build_arch_measurement_table(archs, [device])
            fallback_dir.joinpath("arch_measurements.jsonl").write_text(
                "\n".join(table.to_json(orient="records", lines=True).splitlines()),
                encoding="utf-8",
            )
            hybrid = HybridBackend(
                command_backend=backend,
                fallback_backend=CSVReplayBackend(),
                command_devices={"stm32f405rgt6_000"},
            )
            self.assertEqual(len(hybrid.measure_candidates(device_dir, archs)), 2)
            self.assertEqual(len(hybrid.measure_candidates(fallback_dir, archs)), 2)

    def test_adaptation_updates_embedding_and_calibration(self) -> None:
        device = create_device_record("balanced_mcu", index=0, seed=31)
        archs = [sample_architecture(seed=idx) for idx in range(3)]
        rows = build_arch_measurement_table(archs, [device]).to_dict(orient="records")
        decoder = ResponseDecoder()
        predictor = StructuredCostPredictor()
        feasibility = FeasibilityHead()
        z0 = torch.randn(1, 64)
        z1, calibration, history = adapt_device_state(decoder, predictor, feasibility, z0, rows, steps=2, lr=1e-3)
        self.assertEqual(tuple(z1.shape), (1, 64))
        self.assertEqual(tuple(calibration.shape), (1, 4))
        self.assertEqual(len(history), 2)

    def test_compat_wrapper_still_returns_embedding(self) -> None:
        device = create_device_record("balanced_mcu", index=0, seed=41)
        archs = [sample_architecture(seed=idx + 10) for idx in range(3)]
        rows = build_arch_measurement_table(archs, [device]).to_dict(orient="records")
        decoder = ResponseDecoder()
        predictor = StructuredCostPredictor()
        z0 = torch.randn(1, 64)
        z1, history = adapt_device_embedding(decoder, predictor, z0, rows, steps=2, lr=1e-3)
        self.assertEqual(tuple(z1.shape), (1, 64))
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
