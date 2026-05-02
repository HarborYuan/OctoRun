---
name: octorun
description: Use this skill when the user asks for help running, configuring, or debugging OctoRun — a CLI for distributed parallel Python execution across GPUs that uses chunk-based work distribution and lock files. Trigger on requests mentioning `octorun`, "chunk lock", "stale lock", "octorun status", multi-GPU dispatch, or distributing work across nodes via shared filesystem.
---

# octorun — distributed parallel execution

_Bundled with octorun {{OCTORUN_VERSION}}._ When the installed `octorun --version` differs significantly from this, the user may need to upgrade or you may need to verify behavior against the actual installed version.

OctoRun dispatches one Python worker per GPU, splits work into chunks (stride-sharded keys, not files), and coordinates across nodes through file-based locks on a shared filesystem. Multiple machines can run the same config concurrently — they negotiate via the lock directory.

## Install / upgrade

```bash
uv tool install octorun           # first install
uv tool install --upgrade octorun # later upgrades
```

## Config skeleton

```json
{
  "script_path": "/abs/path/to/worker.py",
  "log_dir": "/shared/path/logs",
  "lock_dir": "/shared/path/logs/locks",
  "total_chunks": 10000,
  "gpus": "auto",
  "kwargs": { "input_dir": "...", "output_dir": "..." }
}
```

- `total_chunks` is independent of input sharding — workers slice keys by `chunk_id % total_chunks`.
- `gpus`: `"auto"` to detect, or an explicit list (`[0,1,2,3]`). Use an explicit list on CPU-only machines (slot count, not GPU IDs).
- `log_dir` and `lock_dir` MUST be on a shared filesystem if you want multi-node coordination.

Worker scripts receive `--gpu_id`, `--chunk_id`, `--total_chunks` plus everything in `kwargs`.

## Commands

```bash
octorun save_config --script ./worker.py    # write default config.json
octorun run --config config.json            # launch workers
octorun run --config config.json --kwargs '{"batch_size": 64}'
octorun status <log_dir>                    # active sessions + completed + stale locks
octorun list_gpus [-d]                      # show detected GPUs
octorun benchmark                           # continuous TFLOPS test (extras: torch)
octorun install-skill                       # install this skill into ~/.claude/skills/
```

## Lock model

Each chunk has a lock file `<lock_dir>/chunk_<id>.lock` with three lines: pid, ISO timestamp, `HEARTBEAT`. Completed chunks get a marker at `<lock_dir>/completed/chunk_<id>.completed`.

- Locks are refreshed periodically by the running worker.
- If a worker's lock timestamp is older than 5 min (`_STALE_TIMEOUT_SECONDS`), another worker may reclaim it.
- 0-byte locks older than 5 min are also stale (added in 1.1.0) — they form when a process dies between `O_CREAT|O_EXCL` and the subsequent write, or when a network FS swallows the write.
- Pre-1.0.0 locks (no `HEARTBEAT` marker) are never auto-cleaned for backward compat.

## Diagnosing a stuck job

1. `octorun status <log_dir>` — look for chunks neither active nor completed.
2. Inspect the lock file: 3-line + recent timestamp = healthy worker; 0-byte or old timestamp = stale.
3. If running pre-1.1.0 with 0-byte stale locks, upgrade or `rm` them manually.
4. Re-launch workers — they will reclaim stale locks on next `acquire_lock`.

In 1.2.1+ `octorun status` derives alive/stale counts from lock-file heartbeat timestamps directly, so it stays accurate even when session-log mtime lags on HDFS-fuse / NFS (append-mode writes get buffered until close). Pre-1.2.1, status would report "0 active sessions" while workers were healthily heartbeating their locks.

## Common pitfalls

- **`gpus: "auto"` on CPU-only nodes** raises `ValueError`. Set an explicit slot list.
- **`log_dir` on local disk** breaks multi-node coordination — every node sees its own state.
- **Worker hardcodes `python`** in some pipelines — prefix with `PATH="<venv>/bin:$PATH"` if your project uses per-stage venvs.
- **`kwargs` on CLI overrides config kwargs** — useful for one-off param tweaks without editing the JSON.
