from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import train_supernet


def main() -> None:
    args = build_parser("configs/train/supernet.yaml").parse_args()
    emit(train_supernet(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

