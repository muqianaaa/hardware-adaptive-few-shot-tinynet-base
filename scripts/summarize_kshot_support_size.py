from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_selected_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "device_name",
        "method",
        "accuracy",
        "latency_ms",
        "peak_sram_bytes",
        "flash_bytes",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame


def _resolve_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def _mean_row(frame: pd.DataFrame, method: str, k: int, series: str) -> dict:
    rows = frame.loc[frame["method"].eq(method)].copy()
    if rows.empty:
        raise ValueError(f"No rows for method={method} in input frame")
    return {
        "k": k,
        "series": series,
        "accuracy": float(rows["accuracy"].mean()),
        "latency_ms": float(rows["latency_ms"].mean()),
        "peak_sram_bytes": float(rows["peak_sram_bytes"].mean()),
        "flash_bytes": float(rows["flash_bytes"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k8",
        default="data/generated/synthetic_cifar10/board_benchmark_kshot_k8/selected_rows.csv",
        help="selected_rows.csv for the K=8 run; this file should include both zero_shot and few_shot.",
    )
    parser.add_argument(
        "--k4",
        default="data/generated/synthetic_cifar10/board_benchmark_kshot_k4/selected_rows.csv",
        help="selected_rows.csv for the K=4 run.",
    )
    parser.add_argument(
        "--k16",
        default="data/generated/synthetic_cifar10/board_benchmark_kshot_k16/selected_rows.csv",
        help="selected_rows.csv for the K=16 run.",
    )
    parser.add_argument(
        "--output",
        default="data/generated/synthetic_cifar10/kshot_support_size_summary.csv",
        help="Output CSV path relative to the repository root.",
    )
    args = parser.parse_args()

    k8_path = _resolve_existing(
        ROOT / args.k8,
        ROOT / "data/generated/synthetic_cifar10/board_benchmark_paper_final/selected_rows.csv",
        ROOT / "data/generated/synthetic_cifar10/board_benchmark_main/selected_rows.csv",
    )
    k4_path = ROOT / args.k4
    k16_path = ROOT / args.k16
    output_path = ROOT / args.output

    frame_k8 = _load_selected_rows(k8_path)
    frame_k4 = _load_selected_rows(k4_path)
    frame_k16 = _load_selected_rows(k16_path)

    rows = [
        _mean_row(frame_k8, "zero_shot", 0, "Zero-Shot Transfer"),
        _mean_row(frame_k4, "few_shot", 4, "Proposed Method"),
        _mean_row(frame_k8, "few_shot", 8, "Proposed Method"),
        _mean_row(frame_k16, "few_shot", 16, "Proposed Method"),
    ]
    summary = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
