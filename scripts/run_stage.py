from __future__ import annotations

import argparse

from _common import ROOT, emit, load_config
from fewshot_hc_nas.pipeline import (
    benchmark_new_boards,
    build_accuracy_dataset,
    build_synth_devices,
    collect_real_board_support,
    deploy_new_device,
    meta_eval,
    prepare_data,
    run_experiment_suite,
    train_hardware_models,
    train_supernet,
)


STAGE_REGISTRY = {
    "prepare_data": prepare_data,
    "train_supernet": train_supernet,
    "build_accuracy_dataset": build_accuracy_dataset,
    "build_synth_devices": build_synth_devices,
    "train_hardware_models": train_hardware_models,
    "meta_eval": meta_eval,
    "run_experiment_suite": run_experiment_suite,
    "benchmark_new_boards": benchmark_new_boards,
    "collect_real_board_support": collect_real_board_support,
    "deploy_new_device": deploy_new_device,
}

CONFIG_KEY_ALIASES = {
    "run_experiment_suite": "experiment_suite",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_REGISTRY))
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_key = CONFIG_KEY_ALIASES.get(args.stage, args.stage)
    stage_cfg = cfg.get(config_key, cfg)
    result = STAGE_REGISTRY[args.stage](stage_cfg, root=ROOT)
    emit(result)


if __name__ == "__main__":
    main()
