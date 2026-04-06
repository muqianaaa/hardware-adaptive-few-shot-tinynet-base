from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_hc_nas.io import read_yaml  # noqa: E402


def build_parser(default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    return parser


def load_config(path: str) -> dict:
    return read_yaml(ROOT / path)


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))

