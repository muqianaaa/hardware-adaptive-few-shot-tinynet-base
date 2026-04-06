from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    modules = [
        "tests.unit.test_search_space",
        "tests.unit.test_hardware",
        "tests.unit.test_models",
        "tests.integration.test_backends_and_adaptation",
    ]
    python_executable = sys.executable or "python"
    cmd = [python_executable, "-m", "unittest", *modules, "-v"]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
