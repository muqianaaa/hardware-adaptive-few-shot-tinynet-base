from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_hc_nas.search_space import compute_stage_metrics, encode_architecture, mutate_architecture, sample_architecture
from fewshot_hc_nas.types import ArchitectureSpec


class SearchSpaceTest(unittest.TestCase):
    def test_sample_encode_and_mutate(self) -> None:
        arch = sample_architecture(seed=1)
        self.assertEqual(len(arch.stages), 5)
        encoded = encode_architecture(arch)
        self.assertEqual(encoded.ndim, 1)
        self.assertGreater(encoded.shape[0], 0)
        metrics = compute_stage_metrics(arch)
        self.assertEqual(len(metrics), 5)
        mutated = mutate_architecture(arch, seed=2, mutation_rate=1.0)
        self.assertNotEqual(mutated.compact_repr(), arch.compact_repr())

    def test_architecture_round_trip(self) -> None:
        arch = sample_architecture(seed=3)
        loaded = ArchitectureSpec.from_dict(json.loads(json.dumps(arch.to_dict())))
        self.assertEqual(arch.compact_repr(), loaded.compact_repr())


if __name__ == "__main__":
    unittest.main()
