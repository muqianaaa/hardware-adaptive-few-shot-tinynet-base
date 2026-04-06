from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_markdown_log(path: str | Path, heading: str, lines: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {heading}\n\n")
        for line in lines:
            handle.write(f"- {line}\n")


def copy_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if target.suffix == ".parquet":
        try:
            frame.to_parquet(target, index=False)
            return target
        except Exception:
            fallback = target.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            return fallback
    frame.to_csv(target, index=False)
    return target


def read_dataframe(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists() and target.suffix == ".parquet":
        csv_fallback = target.with_suffix(".csv")
        if csv_fallback.exists():
            target = csv_fallback
    if target.suffix == ".parquet":
        return pd.read_parquet(target)
    return pd.read_csv(target)
