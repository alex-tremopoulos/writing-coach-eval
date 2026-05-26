# Writing Coach Interfaces

This package isolates Writing Coach version wiring from the shared batch runner in `src/scripts/store_output.py`.

## Files

- `base.py`: abstract `WritingCoachInterface` with common output normalization.
- `v2.py`: Writing Coach V2 implementation (graph + config + shims).
- `v3.py`: Writing Coach V3 implementation (graph + config + shims).
- `factory.py`: `create_writing_coach(version)` selector.
- `utils.py`: shared runtime helpers (`WC_APP_SRC` path install and shims).

## Add a new version

1. Create a new adapter file (for example, `v4.py`) that subclasses `WritingCoachInterface`.
2. Implement initialization and `run_query(...)`.
3. Register it in `factory.py` and export it in `__init__.py`.
4. Use it via:

```bash
python src/scripts/store_output.py <input.csv> --version v4
```

