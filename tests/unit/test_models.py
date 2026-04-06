from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_hc_nas.models import (
    ARCH_FEATURE_DIM,
    AccuracyPredictor,
    BudgetConditionedGenerator,
    FeasibilityHead,
    HardwareEncoder,
    ResponseDecoder,
    StructuredCostPredictor,
)


class ModelShapeTest(unittest.TestCase):
    def test_hardware_stack_shapes(self) -> None:
        encoder = HardwareEncoder()
        decoder = ResponseDecoder()
        predictor = StructuredCostPredictor()
        feasibility_head = FeasibilityHead()
        generator = BudgetConditionedGenerator()
        static_x = torch.randn(2, 20)
        probe_x = torch.randn(2, 15, 7)
        ref_x = torch.randn(2, 5, 6)
        z = encoder(static_x, probe_x, ref_x)
        response = decoder(z)
        arch_x = torch.randn(2, ARCH_FEATURE_DIM)
        structured_x = torch.randn(2, 5, 11)
        structured_x[:, :, 0] = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2]], dtype=torch.float32)
        structured_x[:, :, 3] = torch.tensor([[2, 4, 8, 4, 8], [8, 4, 2, 8, 4]], dtype=torch.float32)
        budget = torch.tensor([[10.0, 20000.0, 50000.0], [12.0, 25000.0, 60000.0]])
        calibration = torch.zeros(2, 4)
        pred = predictor(arch_x, structured_x, z, response, calibration)
        self.assertEqual(pred["latency_ms"].shape, (2, 1))
        feasible = feasibility_head(arch_x, z, pred, budget, calibration)
        self.assertEqual(feasible.shape, (2, 1))
        generated = generator(z, budget)
        self.assertEqual(tuple(generated["op_logits"].shape), (2, 5, 3))
        self.assertEqual(tuple(generated["width_logits"].shape), (2, 5, 6))
        self.assertEqual(tuple(generated["depth_logits"].shape), (2, 5, 2))
        self.assertEqual(tuple(generated["quant_logits"].shape), (2, 5, 3))

    def test_accuracy_predictor_shape(self) -> None:
        predictor = AccuracyPredictor()
        output = predictor(torch.randn(4, ARCH_FEATURE_DIM))
        self.assertEqual(output.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
