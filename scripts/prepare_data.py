from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import prepare_data


def main() -> None:
    args = build_parser("configs/data/cifar10.yaml").parse_args()
    emit(prepare_data(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

