from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import benchmark_new_boards


def main() -> None:
    args = build_parser("configs/eval/board_benchmark_synthetic_cifar10.yaml").parse_args()
    emit(benchmark_new_boards(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()
