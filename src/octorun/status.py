"""Job status reporting for OctoRun.

Reads session logs and lock files written by ProcessManager to produce a
human-readable summary of an in-progress or completed job:

  - How many chunks are actively running (alive heartbeat)
  - How many are done (completed markers)
  - How many locks are stale (dead workers, pending reclaim)
"""

import datetime
import re
from pathlib import Path
from typing import Optional

# Must match runner._HEARTBEAT_INTERVAL and lock_manager._STALE_TIMEOUT_SECONDS
_ALIVE_THRESHOLD_SECONDS = 300


def _parse_log_timestamp(line: str) -> Optional[datetime.datetime]:
    """Extract datetime from a session log line: '[2026-04-10 18:38:40] ...'"""
    m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if m:
        return datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    return None


def _parse_running_chunks(line: str) -> list[int]:
    """Extract chunk IDs from 'Running chunks: [0, 1, 2, ...]'"""
    m = re.search(r'Running chunks: \[([^\]]*)\]', line)
    if not m:
        return []
    content = m.group(1).strip()
    if not content:
        return []
    try:
        return [int(x.strip()) for x in content.split(',') if x.strip()]
    except ValueError:
        return []


def _read_session_tail(log_path: Path) -> tuple[Optional[datetime.datetime], list[int]]:
    """Return (last_timestamp, running_chunks) from the tail of a session log."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except OSError:
        return None, []

    last_ts: Optional[datetime.datetime] = None
    running_chunks: list[int] = []
    for line in reversed(lines[-20:]):
        line = line.strip()
        if last_ts is None:
            ts = _parse_log_timestamp(line)
            if ts:
                last_ts = ts
        if not running_chunks:
            chunks = _parse_running_chunks(line)
            if chunks:
                running_chunks = chunks
        if last_ts is not None and running_chunks:
            break
    return last_ts, running_chunks


def _read_lock_heartbeats(lock_dir: Path) -> dict[int, datetime.datetime]:
    """Return {chunk_id: heartbeat_ts} parsed from each chunk_*.lock file.

    The lock format (lock_manager.ChunkLockManager) is three lines: pid,
    ISO-8601 timestamp, HEARTBEAT marker. Workers rewrite the file in place
    every _HEARTBEAT_INTERVAL seconds, and each rewrite is a fresh small file
    — its mtime updates immediately even on networked filesystems (HDFS-fuse,
    NFS) that buffer append-mode writes to long-lived session logs. The
    timestamp on line 2 is therefore the ground-truth heartbeat for whether
    a worker is alive, regardless of session-log lag.
    """
    heartbeats: dict[int, datetime.datetime] = {}
    if not lock_dir.exists():
        return heartbeats
    for lock_file in lock_dir.glob('chunk_*.lock'):
        try:
            with open(lock_file, 'r') as f:
                lines = f.read().strip().split('\n')
            if len(lines) < 3 or lines[2] != 'HEARTBEAT':
                continue
            ts = datetime.datetime.fromisoformat(lines[1])
            chunk_id = int(lock_file.stem[len('chunk_'):])
        except (OSError, ValueError):
            continue
        heartbeats[chunk_id] = ts
    return heartbeats


def get_status(
    log_dir: str,
    alive_threshold: int = _ALIVE_THRESHOLD_SECONDS,
    cleanup: bool = True,
) -> dict:
    """Collect job status from log_dir.

    When ``cleanup`` is True (default), stale locks are removed from
    ``log_dir/locks/`` before the lock count is taken so the reported numbers
    reflect post-cleanup state.

    Returns a dict with keys:
      lock_count, completed_count, alive_sessions, dead_sessions,
      active_chunk_count, stale_lock_count, cleaned_lock_count
    where each session is (node_name, age_seconds, chunk_ids).
    """
    log_path = Path(log_dir)
    lock_dir = log_path / 'locks'
    completed_dir = lock_dir / 'completed'

    cleaned_lock_count = 0
    if cleanup and lock_dir.exists():
        from .lock_manager import ChunkLockManager
        cleaned_lock_count = ChunkLockManager(str(lock_dir)).cleanup_stale_locks()

    lock_count = len(list(lock_dir.glob('*.lock'))) if lock_dir.exists() else 0
    completed_count = len(list(completed_dir.glob('*.completed'))) if completed_dir.exists() else 0

    # Lock-file heartbeats are the source of truth for alive vs stale.
    # Session-log mtime can lag hours behind reality on HDFS-fuse / NFS
    # because session logs are append-mode and the FS buffers writes until
    # close or hflush; lock files are tiny rewrite-in-place and update
    # immediately.
    now = datetime.datetime.now()
    lock_heartbeats = _read_lock_heartbeats(lock_dir)
    active_lock_chunks = {
        chunk_id for chunk_id, ts in lock_heartbeats.items()
        if (now - ts).total_seconds() <= alive_threshold
    }

    # Keep only the most recent session log per node — used for the per-node
    # display (hostname + last-known chunk list), not for alive/stale counts.
    latest: dict[str, tuple[datetime.datetime, list[int]]] = {}
    for session_log in sorted(log_path.glob('*_session_*.log')):
        node = session_log.name.split('_session_')[0]
        ts, chunks = _read_session_tail(session_log)
        if ts is None:
            continue
        if node not in latest or ts > latest[node][0]:
            latest[node] = (ts, chunks)

    alive_sessions: list[tuple[str, int, list[int]]] = []
    dead_sessions: list[tuple[str, int, list[int]]] = []
    for node, (ts, chunks) in sorted(latest.items()):
        age = int((now - ts).total_seconds())
        # Promote a session to "alive" when its session log looks stale but
        # any chunk it claimed to be running still has a fresh lock heartbeat
        # — this rescues sessions whose log mtime is lagging on a buffered FS.
        log_fresh = age <= alive_threshold
        any_lock_fresh = any(c in active_lock_chunks for c in chunks)
        if log_fresh or any_lock_fresh:
            alive_sessions.append((node, age, chunks))
        else:
            dead_sessions.append((node, age, chunks))

    active_chunk_count = len(active_lock_chunks)
    # Residual locks (no fresh heartbeat) are stale: cleanup will reap them
    # next cycle, or a live worker will reclaim them on next acquire_lock.
    stale_lock_count = max(0, lock_count - active_chunk_count)

    return {
        'lock_count': lock_count,
        'completed_count': completed_count,
        'alive_sessions': alive_sessions,
        'dead_sessions': dead_sessions,
        'active_chunk_count': active_chunk_count,
        'stale_lock_count': stale_lock_count,
        'cleaned_lock_count': cleaned_lock_count,
    }


def print_status(
    log_dir: str,
    alive_threshold: int = _ALIVE_THRESHOLD_SECONDS,
    cleanup: bool = True,
) -> None:
    """Print a status summary for an OctoRun job to stdout.

    Args:
        log_dir: Directory containing session logs and a locks/ subdirectory.
        alive_threshold: Seconds since last heartbeat before a session is dead.
        cleanup: If True (default), reap stale locks before reporting.
    """
    s = get_status(log_dir, alive_threshold, cleanup=cleanup)
    width = 62

    print(f"\nOctoRun Job Status — {log_dir}")
    print('─' * width)
    print(f"  Locks total  : {s['lock_count']}")
    print(f"  Completed    : {s['completed_count']}")
    n_alive = len(s['alive_sessions'])
    print(f"  Active       : {s['active_chunk_count']}  ({n_alive} session{'s' if n_alive != 1 else ''})")
    if cleanup:
        print(f"  Stale locks  : {s['stale_lock_count']}  (cleaned {s['cleaned_lock_count']} this run)")
    else:
        print(f"  Stale locks  : {s['stale_lock_count']}  (dead workers, will be auto-reclaimed)")
    print('─' * width)

    if s['alive_sessions']:
        print(f"\nActive sessions ({n_alive}):")
        for node, age, chunks in sorted(s['alive_sessions'], key=lambda x: x[0]):
            chunk_str = f"[{', '.join(map(str, chunks))}]"
            print(f"  {node:<24}  {len(chunks):>2} chunks  {chunk_str}  (heartbeat {age}s ago)")

    if s['dead_sessions']:
        print(f"\nDead sessions — stale locks ({len(s['dead_sessions'])} node(s)):")
        for node, age, chunks in sorted(s['dead_sessions'], key=lambda x: -x[1]):
            mins = age // 60
            time_str = f"{mins}m ago" if mins < 60 else f"{mins // 60}h{mins % 60:02d}m ago"
            chunk_str = f"[{', '.join(map(str, chunks))}]"
            print(f"  {node:<24}  {len(chunks):>2} chunks  {chunk_str}  (last seen {time_str})")

    print()
