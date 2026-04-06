from __future__ import annotations

from _common import ROOT, build_parser, emit, load_config
from fewshot_hc_nas.pipeline import run_experiment_suite


def main() -> None:
    args = build_parser("configs/eval/experiment_suite.yaml").parse_args()
    cfg = load_config(args.config)
    emit(run_experiment_suite(cfg.get("experiment_suite", cfg), root=ROOT))


if __name__ == "__main__":
    main()
