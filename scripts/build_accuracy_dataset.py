from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import build_accuracy_dataset


def main() -> None:
    args = build_parser("configs/train/accuracy_dataset.yaml").parse_args()
    emit(build_accuracy_dataset(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

