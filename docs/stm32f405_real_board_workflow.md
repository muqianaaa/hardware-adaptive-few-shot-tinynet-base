# STM32F405RGT6 Real-Board Workflow

## 1. Firmware project

- Project root: `firmware/stm32f405_runner`
- Keil project: `firmware/stm32f405_runner/MDK-ARM/AI_Proc.uvprojx`
- Main runner source: `firmware/stm32f405_runner/Core/Src/main.c`

The firmware keeps the CubeMX-generated clock, GPIO, USART1, and TIM2 setup and replaces the application loop with a UART JSONL benchmark runner.

## 2. UART transport

- UART: `USART1`
- MCU pins: `PA9` (TX), `PA10` (RX)
- Baudrate: `115200`
- Host config: `configs/hardware/stm32f405rgt6_uart.yaml`

Update the `port` field in the YAML before host-side execution.

## 3. Supported commands

The runner accepts one JSON command per line:

- `{"cmd":"ping"}`
- `{"cmd":"get_static"}`
- `{"cmd":"run_probe_suite"}`
- `{"cmd":"run_reference_suite"}`
- `{"cmd":"measure_arch","arch_name":"...","arch_repr":"std3x3:0.75:1:8|..."}`

The board returns one JSON object per line.

## 4. Host-side sanity checks

```bash
python scripts/ping_serial_board.py --config configs/hardware/stm32f405rgt6_uart.yaml --cmd ping
python scripts/ping_serial_board.py --config configs/hardware/stm32f405rgt6_uart.yaml --cmd get_static
python scripts/ping_serial_board.py --config configs/hardware/stm32f405rgt6_uart.yaml --cmd run_probe_suite
python scripts/ping_serial_board.py --config configs/hardware/stm32f405rgt6_uart.yaml --cmd run_reference_suite
```

## 5. Collect the real-board support set

```bash
python scripts/run_stage.py --stage collect_real_board_support --config configs/eval/collect_stm32f405rgt6_support_command.yaml
```

This step creates the real device directory under:

```text
data/generated/synthetic_cifar10/devices/real/stm32f405rgt6_000
```

Expected outputs include:

- `hardware_static.json`
- `probe_results.jsonl`
- `reference_results.jsonl`
- `arch_measurements.jsonl`
- `task_budgets.jsonl`

## 6. Deploy on the real board

```bash
python scripts/run_stage.py --stage deploy_new_device --config configs/eval/deploy_stm32f405rgt6_command.yaml
```

This stage performs:

- few-shot adaptation on the host;
- parameter generation on the host;
- local refinement on the host;
- top-k candidate measurement on the real board.

## 7. Five-board benchmark

```bash
python scripts/run_stage.py --stage benchmark_new_boards --config configs/eval/board_benchmark_synthetic_cifar10_with_real_stm32f405rgt6.yaml
```

This benchmark combines:

- four replayed synthetic MCU devices;
- one live STM32F405RGT6 board.
