from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_hc_nas.hardware import MICRO_PROBE_LIBRARY, REFERENCE_ARCHITECTURES, create_device_record, predict_cost_from_response, synthetic_architecture_accuracy
from fewshot_hc_nas.search_space import sample_architecture


class HardwareGenerationTest(unittest.TestCase):
    def test_device_record_shapes(self) -> None:
        device = create_device_record("balanced_mcu", index=0, seed=7)
        self.assertEqual(len(device.probes), len(MICRO_PROBE_LIBRARY))
        self.assertEqual(len(device.references), len(REFERENCE_ARCHITECTURES))

    def test_cost_prediction_positive(self) -> None:
        device = create_device_record("high_performance_mcu", index=0, seed=11)
        arch = sample_architecture(seed=13)
        latency, sram, flash = predict_cost_from_response(arch, device.static, device.response)
        self.assertGreater(latency, 0.0)
        self.assertGreater(sram, 0.0)
        self.assertGreater(flash, 0.0)

    def test_synthetic_accuracy_varies(self) -> None:
        left = sample_architecture(seed=17)
        right = sample_architecture(seed=23)
        left_acc = synthetic_architecture_accuracy(left, dataset_name="cifar100", noise_scale=0.0)
        right_acc = synthetic_architecture_accuracy(right, dataset_name="cifar100", noise_scale=0.0)
        self.assertGreaterEqual(left_acc, 0.0)
        self.assertLessEqual(left_acc, 1.0)
        self.assertGreaterEqual(right_acc, 0.0)
        self.assertLessEqual(right_acc, 1.0)
        self.assertNotEqual(left.compact_repr(), right.compact_repr())
        self.assertNotEqual(left_acc, right_acc)


if __name__ == "__main__":
    unittest.main()
