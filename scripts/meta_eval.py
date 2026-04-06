from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import meta_eval


def main() -> None:
    args = build_parser("configs/eval/meta_eval.yaml").parse_args()
    emit(meta_eval(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

