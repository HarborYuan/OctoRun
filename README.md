<div align="center">

# 🐙 OctoRun

**Distributed Parallel Execution Made Simple**

*Run Python scripts across multiple GPUs with intelligent chunk management, failure recovery, and live job monitoring*

[![PyPI version](https://img.shields.io/pypi/v/octorun.svg)](https://pypi.org/project/octorun/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

</div>

## Overview

OctoRun dispatches one Python worker process per GPU, divides your dataset into chunks, and coordinates work across multiple machines through shared lock files. Each worker independently claims a chunk, processes it, and writes its output atomically — making it safe to run OctoRun on many nodes simultaneously without any central coordinator.

## Key Features

- **Automatic GPU Detection** — detects and allocates all available GPUs
- **Chunk-Based Work Distribution** — divides keys by stride, not by file, so any `total_chunks` value works regardless of input sharding
- **Failure Recovery** — stale locks (heartbeat timeout 5 min) are automatically reclaimed; configurable retries
- **Multi-Machine Safe** — multiple OctoRun instances on different nodes share the same lock directory on a shared filesystem
- **Live Job Monitoring** — `octorun status` shows active sessions, completed chunks, and stale locks at a glance

## Installation

```bash
# Via uv (recommended)
uv tool install octorun

# In a project venv
uv add octorun

# Via pip
pip install octorun
```

Optional extras:

```bash
# GPU benchmark tooling (requires PyTorch)
pip install "octorun[benchmark]"
```

## Quick Start

```bash
# 1. Generate a default config
octorun save_config --script ./your_script.py

# 2. Run across all available GPUs
octorun run --config config.json

# 3. Monitor progress from any machine
octorun status ./logs
```

## Commands

### `run` (`r`)

Launch workers across all available GPUs.

```bash
octorun run --config config.json
octorun run --config config.json --kwargs '{"batch_size": 64}'
```

CLI `--kwargs` override any `kwargs` values from the config file.

---

### `status` (`st`)

Show a live summary of a running or completed job.

```bash
octorun status <log_dir>
octorun status <log_dir> --alive-threshold 120
```

`log_dir` is the same directory you set as `log_dir` in your config. It contains the `*_session_*.log` files and the `locks/` subdirectory.

Example output:

```
OctoRun Job Status — ./logs/stage3
──────────────────────────────────────────────────────────────
  Locks total  : 164
  Completed    : 29
  Active       : 48  (6 sessions)
  Stale locks  : 87  (dead workers, will be auto-reclaimed)
──────────────────────────────────────────────────────────────

Active sessions (6):
  node1               8 chunks  [25, 26, 28, 32, ...]  (heartbeat 43s ago)
  node2               8 chunks  [16, 18, 20, 21, ...]  (heartbeat 47s ago)
  ...

Dead sessions — stale locks (4 node(s)):
  node3               8 chunks  [161, 162, ...]  (last seen 1h31m ago)
  ...
```

| Field | Meaning |
|-------|---------|
| **Locks total** | Lock files currently on disk (active + stale) |
| **Completed** | Chunks that finished successfully (persistent `.completed` markers) |
| **Active** | Chunks held by workers with a live heartbeat |
| **Stale locks** | Locks from crashed/killed workers — reclaimed automatically when the next chunk finishes on any active node |

`--alive-threshold` (default 300 s) sets how long a session can be silent before it is considered dead.

---

### `save_config` (`s`)

Write a default `config.json` to the current directory.

```bash
octorun save_config --script ./your_script.py
```

---

### `list_gpus` (`l`)

List available GPUs.

```bash
octorun list_gpus
octorun list_gpus --detailed
```

---

### `benchmark` (`b`)

Continuously measure GPU TFLOPs and communication bandwidth.

```bash
octorun benchmark
octorun benchmark --gpus 0,1,2 --test-duration 10 --interval 30
```

Requires `octorun[benchmark]`.

## Configuration

Generate a starter config with `octorun save_config`, then edit as needed:

```json
{
    "script_path": "your_script.py",
    "gpus": "auto",
    "total_chunks": 1000,
    "log_dir": "./logs",
    "chunk_lock_dir": "./logs/locks",
    "monitor_interval": 60,
    "restart_failed": true,
    "max_retries": 2,
    "kwargs": {
        "input_dir": "/data/input",
        "output_dir": "/data/output",
        "batch_size": 32
    }
}
```

| Option | Description | Default |
|--------|-------------|---------|
| `script_path` | Path to your Python worker script | — |
| `gpus` | `"auto"` or a list of GPU IDs | `"auto"` |
| `total_chunks` | Total number of chunks to divide work into | `128` |
| `log_dir` | Directory for session and chunk logs | `"./logs"` |
| `chunk_lock_dir` | Directory for lock files | `"./logs/locks"` |
| `monitor_interval` | Seconds between process-status checks | `60` |
| `restart_failed` | Retry failed chunks | `false` |
| `max_retries` | Maximum retries per chunk | `3` |
| `kwargs` | Extra arguments forwarded to your script | `{}` |

## Writing a Worker Script

Your script must accept three OctoRun arguments plus any custom ones:

```python
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    # Required by OctoRun
    p.add_argument("--gpu_id",       type=int, required=True)
    p.add_argument("--chunk_id",     type=int, required=True)
    p.add_argument("--total_chunks", type=int, required=True)
    # Your own arguments (forwarded via kwargs)
    p.add_argument("--input_dir",    type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--batch_size",   type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()

    # Shard your dataset by stride
    all_keys = sorted(load_all_keys(args.input_dir))
    my_keys  = all_keys[args.chunk_id::args.total_chunks]

    # Skip if already done (idempotent)
    output_path = get_output_path(args.output_dir, args.chunk_id, args.total_chunks)
    if output_path.exists():
        return

    # Process and write atomically
    results = process(my_keys, gpu=args.gpu_id, batch_size=args.batch_size)
    write_atomic(results, output_path)

if __name__ == "__main__":
    main()
```

## Multi-Machine Usage

Run OctoRun on each machine, pointing at the **same shared `log_dir` and `chunk_lock_dir`**:

```bash
# machine A
octorun run --config config.json   # claims chunks 0, 1, 2, ...

# machine B (simultaneously)
octorun run --config config.json   # claims the next available chunks
```

Lock files on the shared filesystem prevent any chunk from being processed twice.
Monitor all machines from anywhere:

```bash
octorun status ./logs
```

## Contributing

Fork the repository, create a feature branch, and open a pull request.

## License

MIT License.
