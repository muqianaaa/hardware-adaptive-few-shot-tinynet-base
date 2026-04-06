from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import build_synth_devices


def main() -> None:
    args = build_parser("configs/data/synthetic_devices.yaml").parse_args()
    emit(build_synth_devices(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

