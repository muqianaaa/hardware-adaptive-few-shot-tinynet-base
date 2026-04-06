from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import deploy_new_device


def main() -> None:
    args = build_parser("configs/eval/deploy_template.yaml").parse_args()
    emit(deploy_new_device(load_config(args.config), root=ROOT))


if __name__ == "__main__":
    main()

