from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import train_hardware_models


def main() -> None:
    args = build_parser("configs/train/hardware_models.yaml").parse_args()
    emit(train_hardware_models(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

