from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import collect_real_board_support


def main() -> None:
    args = build_parser("configs/eval/collect_stm32f405rgt6_support_command.yaml").parse_args()
    emit(collect_real_board_support(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()
