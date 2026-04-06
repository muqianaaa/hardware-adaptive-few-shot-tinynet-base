# Contributing

Thanks for your interest in improving this project.

## Development setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the test suite:

```bash
python scripts/run_tests.py
```

## Contribution scope

Contributions are especially welcome in these areas:

- code cleanup and documentation;
- reproducibility improvements;
- real-board tooling and protocol robustness;
- test coverage for hardware-conditioned search and adaptation;
- usability improvements for configs and scripts.

## Pull request guidelines

- Keep changes focused and easy to review.
- Update configs or docs when behavior changes.
- Preserve relative paths in public-facing configs and scripts.
- Avoid committing generated artifacts under `data/generated/` and `data/checkpoints/`.
- If you touch the STM32 firmware, note the exact files changed and how they were validated.

## Style

- Prefer clear, minimal changes over large unrelated refactors.
- Keep code comments concise and only where they improve readability.
- For Python changes, make sure the included tests still pass.
