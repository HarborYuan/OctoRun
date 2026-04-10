# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (dev + optional benchmark extras)
uv sync --extra dev
uv sync --extra dev --extra benchmark  # includes torch for GPU benchmarking

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_runner.py

# Run a single test case
uv run pytest tests/test_runner.py::TestProcessManager::test_init

# CLI usage
uv run octorun save_config --script your_script.py  # generate config.json
uv run octorun run --config config.json             # run workload
uv run octorun run --config config.json --kwargs '{"batch_size": 64}'  # override kwargs
uv run octorun list_gpus --detailed                  # list GPUs with metrics
uv run octorun benchmark                            # GPU benchmark

# Build and publish (CI handles this on release)
uv build
```

## Architecture

OctoRun distributes a Python script across multiple GPUs by dividing work into numbered chunks and assigning each chunk to a GPU subprocess. The key design constraint is that user scripts must accept `--gpu_id`, `--chunk_id`, and `--total_chunks` arguments.

### Core components

**`cli.py`** — Entry point (`octorun.cli:cli_main`). Handles GPU detection via `nvidia-smi`, parses subcommands (`run`, `save_config`, `list_gpus`, `benchmark`), merges JSON config with CLI kwargs overrides, and delegates to `ProcessManager`.

**`runner.py`** (`ProcessManager`) — The main execution loop. Spawns one Python subprocess per GPU per chunk, polls subprocess status, handles retries on failure, and writes per-chunk log files. The `run()` method continuously assigns available chunks to idle GPUs until all chunks complete or max retries are exhausted.

**`lock_manager.py`** (`ChunkLockManager`) — File-based distributed coordination layer. Uses atomic exclusive file creation to prevent duplicate chunk processing across machines sharing a filesystem. Locks live in `chunk_lock_dir`; completed chunks are tracked in separate `.done` files containing machine ID and timestamp. `get_next_available_chunk()` randomly selects among unlocked, uncompleted chunks to reduce contention.

**`gpu_benchmark.py`** + **`benchmarks/`** — Optional benchmarking (requires `benchmark` extra). Tests compute (TFLOPs via matrix multiply) and memory bandwidth via subprocess workers, displays results in a live table.

### Distributed multi-machine execution

Multiple machines can share work by pointing at the same `chunk_lock_dir` on a shared filesystem. The `ChunkLockManager` handles coordination — each machine picks up unchunked work atomically. OctoRun does **not** manage `CUDA_VISIBLE_DEVICES`; user scripts select the device using the provided `gpu_id`.

### Script compatibility contract

User scripts must implement these three arguments:
- `--gpu_id`: GPU device index; script calls `torch.cuda.set_device(gpu_id)` or uses as `local_rank` for CPU tasks
- `--chunk_id`: which slice of data this process handles
- `--total_chunks`: total number of chunks (for computing data slices)

Chunks must be independent and deterministic — the same `chunk_id` always processes the same data slice.

### Config schema

`config.json` fields (see `default_config.json`):
- `script_path`: path to user script
- `gpus`: list of GPU IDs or `"auto"` (filters GPUs with >100MB free memory)
- `total_chunks`: number of work chunks to divide the job into
- `kwargs`: dict of extra `--key value` args passed to each subprocess; CLI `--kwargs` JSON overrides these
- `log_dir`: directory for session log and per-chunk `chunk_N.log` files
- `chunk_lock_dir`: directory for `.lock` and `completed/chunk_N.completed` files (shared across machines for distributed runs)
- `monitor_interval`: seconds between polling subprocess status
- `restart_failed` + `max_retries`: retry policy for non-zero exit codes
- `memory_threshold`: unused in current code (reserved)
