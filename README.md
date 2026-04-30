<div align="center">

# 🐙 OctoRun

**Distributed Parallel Execution Made Simple**

*Run Python scripts across multiple GPUs and multiple machines with chunk-based work distribution, automatic failure recovery, and live job monitoring.*

[![PyPI version](https://img.shields.io/pypi/v/octorun.svg)](https://pypi.org/project/octorun/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## Overview

OctoRun launches one Python worker per GPU, splits the work into `total_chunks` shards by stride (not by file), and coordinates concurrent runs across machines via lock files on a shared filesystem. There is no central scheduler — each worker picks the next free chunk, refreshes a heartbeat while running, and writes a completion marker on success. Locks left behind by crashed workers are automatically reclaimed.

## Features

- **Auto GPU detection** — one worker per available CUDA device, or supply an explicit slot list (CPU works too).
- **Stride-based chunking** — `total_chunks` is independent of input sharding; any value works.
- **Crash-safe locks** — heartbeat-refreshed locks; stale ones (including 0-byte locks left by mid-write deaths) auto-reclaim after 5 min.
- **Multi-node by default** — point N machines at the same shared `log_dir` and they cooperate without coordination.
- **Live status** — `octorun status <log_dir>` prints active sessions, completed chunks, and stale locks.

## What's new in 1.2.0

- **`octorun status` now reaps stale locks.** Each invocation sweeps the lock dir and removes anything that no live worker can refresh, then reports the post-cleanup state. Pass `--no-clean` for read-only behavior.
- **Aggressive stale detection.** Malformed locks and pre-1.0.0 format locks (no `HEARTBEAT` marker) are now treated as stale unconditionally — they cannot be refreshed, so a live owner is impossible. Empty-file and timestamp-based rules from 1.1.0 are unchanged.
- **Stale-count bug fix.** The reported `Stale locks` figure used to subtract `Completed` from the total lock count, which underflowed (clamped to 0) once a job had more completions than residual locks — masking real dead-worker locks. Now computed correctly as `lock_count - active_chunk_count`.

## Install

```bash
uv tool install octorun                  # recommended — global CLI
uv add octorun                           # in a project venv
pip install octorun                      # via pip
pip install "octorun[benchmark]"         # GPU benchmark extras
```

## Quick start

```bash
octorun save_config --script ./worker.py    # write a default config.json
octorun run --config config.json            # launch on all detected GPUs
octorun status ./logs                       # check progress (any machine)
```

## CLI

| Command | Purpose |
|---------|---------|
| `octorun run --config <cfg> [--kwargs '<json>']` | Launch workers; CLI `--kwargs` overrides config kwargs. |
| `octorun status <log_dir> [--alive-threshold N] [--no-clean]` | Live job summary; reaps stale locks before reporting unless `--no-clean`. |
| `octorun save_config [--script <path>]` | Write a default `config.json` here. |
| `octorun list_gpus [--detailed]` | Show detected GPUs (with usage and processes when `-d`). |
| `octorun benchmark [--gpus auto] [--test-duration s] [--interval s]` | Continuous TFLOPS / bandwidth probe. Requires `[benchmark]` extra. |
| `octorun install-skill [--dest <dir>] [--force]` | Install the bundled Claude Code skill into `~/.claude/skills/octorun`. |

### Status output

```
OctoRun Job Status — ./logs/stage3
──────────────────────────────────────────────────────────────
  Locks total  : 164
  Completed    : 29
  Active       : 48  (6 sessions)
  Stale locks  : 87  (cleaned 87 this run)
──────────────────────────────────────────────────────────────

Active sessions (6):
  node1               8 chunks  [25, 26, 28, 32, ...]  (heartbeat 43s ago)
  ...
Dead sessions — stale locks (4 node(s)):
  node3               8 chunks  [161, 162, ...]  (last seen 1h31m ago)
```

`--alive-threshold` (default 300 s) controls how silent a session can be before its locks are flagged stale.

## Configuration

```json
{
    "script_path": "./worker.py",
    "gpus": "auto",
    "total_chunks": 1000,
    "log_dir": "./logs",
    "chunk_lock_dir": "./logs/locks",
    "monitor_interval": 60,
    "restart_failed": true,
    "max_retries": 2,
    "success_codes": [0],
    "kwargs": {
        "input_dir": "/data/input",
        "output_dir": "/data/output",
        "batch_size": 32
    }
}
```

| Option | Description | Default |
|--------|-------------|---------|
| `script_path` | Worker script (absolute or relative to CWD) | — |
| `gpus` | `"auto"` or explicit list (`[0,1,2,3]`). On CPU-only nodes use a slot list, e.g. `[0,1]`. | `"auto"` |
| `total_chunks` | Number of stride shards. Independent of input file count. | `128` |
| `log_dir` | Session and chunk logs (must be shared FS for multi-node) | `"./logs"` |
| `chunk_lock_dir` | Lock files; **must be on a POSIX FS** (see [Locks](#how-locks-work)) | `"./logs/locks"` |
| `monitor_interval` | Seconds between worker health checks | `60` |
| `restart_failed` | Retry chunks whose worker exits with a non-success code | `false` |
| `max_retries` | Per-chunk retry budget | `3` |
| `success_codes` | Worker exit codes treated as success | `[0]` |
| `kwargs` | Forwarded to the worker script as CLI flags | `{}` |

#### `success_codes`

Default `[0]`. Add codes here for known-benign non-zero exits — e.g. CPython interpreter shutdown errors (often `120`) on slow networked filesystems where the chunk's data is already written:

```json
"success_codes": [0, 120]
```

## Writing a worker

Workers receive `--gpu_id`, `--chunk_id`, `--total_chunks`, plus everything in `kwargs` as flags.

### `--gpu_id` is a local rank, not a device index

OctoRun does **not** set `CUDA_VISIBLE_DEVICES`. `--gpu_id` is the worker's index on the current machine (a `local_rank`). Two patterns:

```python
# A — direct device index (fine when no other processes share the GPUs)
torch.cuda.set_device(args.gpu_id)

# B — pin the device by setting CUDA_VISIBLE_DEVICES BEFORE importing torch
import os, argparse
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu_id", type=int, default=0)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.gpu_id)
import torch
model = model.to("cuda:0")
```

Pattern **B** is recommended for transformers / large-model loaders, which often initialize CUDA at import time.

### Minimal template

```python
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_id",       type=int, required=True)
    p.add_argument("--chunk_id",     type=int, required=True)
    p.add_argument("--total_chunks", type=int, required=True)
    p.add_argument("--input_dir",    type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--batch_size",   type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    all_keys = sorted(load_all_keys(args.input_dir))
    my_keys  = all_keys[args.chunk_id::args.total_chunks]   # stride shard

    output_path = get_output_path(args.output_dir, args.chunk_id, args.total_chunks)
    if output_path.exists():
        return                                              # idempotent skip

    results = process(my_keys, gpu=args.gpu_id, batch_size=args.batch_size)
    write_atomic(results, output_path)                      # tmp + rename

if __name__ == "__main__":
    main()
```

## Multi-machine usage

Point each machine at the same shared `log_dir` and `chunk_lock_dir`:

```bash
# machine A
octorun run --config config.json
# machine B (in parallel)
octorun run --config config.json
# any machine
octorun status ./logs
```

Lock files prevent any chunk from being processed twice. Workers write a `completed/chunk_<id>.completed` marker on success, which is honored across all machines.

## How locks work

Each in-progress chunk owns a 3-line lock file under `chunk_lock_dir`:

```
<pid>
<iso-8601 timestamp>     # refreshed every monitor_interval
HEARTBEAT
```

A lock is considered **stale** (and reclaimable) when any of:

- its `HEARTBEAT` timestamp is older than `_STALE_TIMEOUT_SECONDS` (5 min);
- it is 0 bytes and its mtime is older than the same threshold (these form when a process dies between `O_CREAT|O_EXCL` and the subsequent write, or when a network FS swallows the write);
- it is malformed or pre-1.0.0 format (no `HEARTBEAT` marker) — since 1.2.0 these are reaped on sight (no live worker can refresh them).

Stale locks are reclaimed lazily on the next `acquire_lock`, and eagerly by `octorun status` (which sweeps the lock dir before reporting; pass `--no-clean` to disable).

### Filesystem requirements

Locking relies on `O_CREAT | O_EXCL` having atomic, cross-client semantics. Use a real POSIX filesystem (local disk, NFS, CephFS).

**Do not place `chunk_lock_dir` on object-storage-backed filesystems** like HDFS-fuse or s3fs — they fall back to non-atomic stat-then-create sequences and cache metadata client-side, so two workers can claim the same lock. Output data may live on HDFS / S3, but locks must not. If you must run that way, treat the lock as best-effort and write outputs idempotently (`tmp + rename`).

## Claude Code skill

Install the bundled skill so an LLM agent in Claude Code can answer OctoRun questions with up-to-date semantics:

```bash
octorun install-skill          # installs to ~/.claude/skills/octorun
octorun install-skill --force  # overwrite an existing copy
```

The skill records the version of octorun it shipped with — re-run `install-skill` after upgrading.

## Contributing

Fork, branch, PR.

## License

MIT.
