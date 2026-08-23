"""
Pytest collection configuration.

The `debug/` directory and several `tests/` files are MANUAL test harnesses
(standalone scripts run via `python <file>.py`), not pytest test modules.
They use custom result tracking (e.g. `TestResults`, `check()`, `PASS`/`FAIL`)
and their own `asyncio.run(main())` entry points, so they must NOT be
collected by pytest. They remain runnable via their documented commands.
"""

collect_ignore = [
    # Manual debug harnesses (run via `python debug/<file>.py`)
    "debug",
    # Manual harnesses in tests/ (run via `python tests/<file>.py`)
    "tests/test_runtime.py",
    "tests/test_response_guarantee.py",
    "tests/test_production_regression.py",
    "tests/test_e2e.py",
]
