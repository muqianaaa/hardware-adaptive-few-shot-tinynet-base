from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_hc_nas.board_serial import JsonlSerialBoardClient, SerialBoardConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "hardware" / "stm32f405rgt6_uart.yaml"),
    )
    parser.add_argument(
        "--cmd",
        default="ping",
        choices=["ping", "get_static", "run_probe_suite", "run_reference_suite"],
    )
    args = parser.parse_args()

    client = JsonlSerialBoardClient(SerialBoardConfig.from_any(args.config))
    response = client.command({"cmd": args.cmd})
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
