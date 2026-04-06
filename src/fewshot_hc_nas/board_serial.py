from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_yaml


@dataclass(frozen=True)
class SerialBoardConfig:
    port: str
    baudrate: int = 115200
    timeout_s: float = 5.0
    write_timeout_s: float = 5.0
    open_delay_s: float = 2.0
    command_delay_s: float = 0.02
    retries: int = 2
    reset_input_buffer_on_command: bool = True

    @classmethod
    def from_any(cls, config: str | Path | dict[str, Any]) -> "SerialBoardConfig":
        if isinstance(config, (str, Path)):
            payload = read_yaml(config)
        else:
            payload = dict(config)
        serial_cfg = payload.get("serial", payload)
        return cls(
            port=str(serial_cfg["port"]),
            baudrate=int(serial_cfg.get("baudrate", 115200)),
            timeout_s=float(serial_cfg.get("timeout_s", 5.0)),
            write_timeout_s=float(serial_cfg.get("write_timeout_s", 5.0)),
            open_delay_s=float(serial_cfg.get("open_delay_s", 2.0)),
            command_delay_s=float(serial_cfg.get("command_delay_s", 0.02)),
            retries=int(serial_cfg.get("retries", 2)),
            reset_input_buffer_on_command=bool(serial_cfg.get("reset_input_buffer_on_command", True)),
        )


class JsonlSerialBoardClient:
    def __init__(self, config: SerialBoardConfig, serial_module: Any | None = None):
        self.config = config
        self._serial_module = serial_module
        self._serial = None

    def _module(self) -> Any:
        if self._serial_module is not None:
            return self._serial_module
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in integration environment
            raise RuntimeError("pyserial is required for the STM32 command backend.") from exc
        self._serial_module = serial
        return serial

    def open(self) -> None:
        if self._serial is not None:
            return
        serial = self._module()
        self._serial = serial.Serial(
            self.config.port,
            self.config.baudrate,
            timeout=self.config.timeout_s,
            write_timeout=self.config.write_timeout_s,
        )
        time.sleep(self.config.open_delay_s)
        if hasattr(self._serial, "reset_input_buffer"):
            self._serial.reset_input_buffer()
        if hasattr(self._serial, "reset_output_buffer"):
            self._serial.reset_output_buffer()

    def close(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        finally:
            self._serial = None

    def __enter__(self) -> "JsonlSerialBoardClient":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.open()
        assert self._serial is not None
        last_error: Exception | None = None
        expected_cmd = payload.get("cmd")
        max_attempts = 1 if expected_cmd == "measure_arch" else (self.config.retries + 1)
        for _ in range(max_attempts):
            try:
                if self.config.reset_input_buffer_on_command and hasattr(self._serial, "reset_input_buffer"):
                    self._serial.reset_input_buffer()
                message = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
                self._serial.write(message)
                if hasattr(self._serial, "flush"):
                    self._serial.flush()
                time.sleep(self.config.command_delay_s)
                deadline = time.monotonic() + self.config.timeout_s
                while time.monotonic() < deadline:
                    raw = self._serial.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    response = json.loads(line)
                    response_cmd = response.get("cmd")
                    if response_cmd == "boot":
                        continue
                    if expected_cmd is not None and response_cmd not in {expected_cmd, None}:
                        continue
                    if expected_cmd == "measure_arch":
                        expected_arch = payload.get("arch_repr")
                        response_arch = response.get("row", {}).get("arch_repr") if isinstance(response.get("row"), dict) else None
                        if expected_arch is not None and response_arch not in {expected_arch, None}:
                            continue
                    if not response.get("ok", False):
                        raise RuntimeError(str(response.get("error", "Unknown board error.")))
                    return response
                raise TimeoutError(f"Timed out waiting for board response to {payload.get('cmd')}.")
            except Exception as exc:  # pragma: no cover - depends on hardware transport
                last_error = exc
                self.close()
                self.open()
                time.sleep(0.1)
        assert last_error is not None
        raise RuntimeError(f"Board command failed: {payload}") from last_error
