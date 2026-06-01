import datetime
import threading
import time
from .lock_manager import ChunkLockManager
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

_HEARTBEAT_INTERVAL = 30  # seconds


class ProcessManager:
    """
    Manages the lifecycle of processes, including starting, stopping, and monitoring them.
    """

    def __init__(self, config):
        self.config = config

        self.processes = {}
        self.chunk_lock_manager = ChunkLockManager(config['chunk_lock_dir'])

        self.start_time = datetime.datetime.now()
        
        self.completed_chunks = self.chunk_lock_manager.get_completed_chunks()
        self.failed_chunks = set()        # ever failed at least once this session (reporting)
        self.permanently_failed = set()   # exhausted max_retries -> terminal, excluded from assignment
        self.retry_count = {}
        self.retry_after = {}             # chunk_id -> monotonic ts before which we won't re-pick it (backoff)

        # Exponential backoff for retrying a failed chunk: base * 2**(attempt-1), capped.
        self.retry_base_delay = self.config.get('retry_base_delay', 5)      # seconds
        self.retry_max_delay = self.config.get('retry_max_delay', 1800)     # 30 min cap

        # Setup logging
        self.setup_logging()
        self.log_message(f"ProcessManager initialized with configuration: {json.dumps(self.config, indent=2)}")

        # Log already completed chunks
        if self.completed_chunks:
            self.log_message(f"Found {len(self.completed_chunks)} previously completed chunks: {sorted(self.completed_chunks)}")
    
    def setup_logging(self):
        """Setup logging directory and open the session log file once.

        Holding a long-lived line-buffered fd avoids re-opening the file on
        every log message — re-opens trigger a stat() round-trip, which on
        networked filesystems (HDFS-fuse) can return transient EOVERFLOW under
        load and crash the runner.
        """
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)

        # Create session log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        machine_name = os.uname().nodename
        self.session_log = os.path.join(log_dir, f"{machine_name}_session_{timestamp}.log")

        self._session_fp = open(self.session_log, 'w', buffering=1)
        self._session_fp.write(f"Session Started: {self.start_time}\n")
        self._session_fp.write(f"Configuration: {json.dumps(self.config, indent=2)}\n")
        self._session_fp.write("-" * 80 + "\n")
        self._session_fp.flush()

    def log_message(self, message: str):
        """Log message to session log and print.

        A failed log write must never kill the runner — the message has
        already been emitted to stdout via print(), and abandoning every
        in-flight chunk over a transient FS hiccup is a far worse outcome.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)

        try:
            self._session_fp.write(log_entry + "\n")
        except OSError as e:
            sys.stderr.write(
                f"[octorun] suppressed session-log write error: {e}\n"
            )

    def read_and_print_process_errors(self, chunk_id: int, log_file: str, num_lines: int = 20):
        """Read and print the last few lines of a failed process log file"""
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                if lines:
                    # Get the last num_lines lines
                    error_lines = lines[-num_lines:] if len(lines) >= num_lines else lines
                    
                    # Log error details with improved formatting
                    self.log_message("")  # Empty line for spacing
                    self.log_message(f"ERROR DETAILS - Chunk {chunk_id} (showing last {min(len(lines), num_lines)} lines):")
                    self.log_message("=" * 80)
                    
                    for i, line in enumerate(error_lines, 1):
                        # Add line numbers and clean formatting
                        line_content = line.rstrip()
                        if line_content:  # Skip empty lines
                            self.log_message(f"{i:2d}: {line_content}")
                    
                    self.log_message("=" * 80)
                    self.log_message("")  # Empty line for spacing
                else:
                    self.log_message(f"Log file for chunk {chunk_id} is empty")
            else:
                self.log_message(f"Log file for chunk {chunk_id} not found: {log_file}")
        except Exception as e:
            self.log_message(f"Error reading log file for chunk {chunk_id}: {e}")

    def start_process(
            self, 
            gpu_id: int, 
            chunk_id: int, 
            total_chunks: int,
            kwargs: Optional[Dict] = None,
        ) -> Optional[subprocess.Popen]:
        """Start a process on a specific GPU"""
        try:
            cmd = [
                'python', 
                self.config['script_path'],
                '--gpu_id', str(gpu_id),
                '--chunk_id', str(chunk_id),
                '--total_chunks', str(total_chunks),
            ]
            if kwargs:
                for key, value in kwargs.items():
                    cmd.append(f'--{key}')
                    cmd.append(str(value))
            # Setup environment
            env = os.environ.copy()

            # Setup logging — append, not truncate. Preserves prior attempts'
            # output across retries, and avoids a 30s lease-conflict stall on
            # networked filesystems (HDFS-fuse) where O_TRUNC blocks while
            # another writer holds the file.
            log_file = os.path.join(self.config['log_dir'], f"chunk_{chunk_id}.log")

            log_f = open(log_file, 'a')
            log_f.write(f"=== Starting process on GPU {gpu_id}, chunk {chunk_id} "
                        f"({datetime.datetime.now().isoformat()}) ===\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write("-" * 50 + "\n")
            log_f.flush()

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env
            )
            log_f.close()

            self.processes[chunk_id] = {
                'process': process,
                'gpu_id': gpu_id,
                'start_time': datetime.datetime.now(),
                'log_file': log_file,
                'status': 'running'
            }
            
            self.log_message(f"Started process for GPU {gpu_id}, chunk {chunk_id} (PID: {process.pid})")
            return process
            
        except Exception as e:
            self.log_message(f"Error starting process for GPU {gpu_id}, chunk {chunk_id}: {e}")
            return None
        
    def check_processes(self) -> Dict:
        """Check status of all running processes"""
        status = {
            'running': [],
            'completed': [],
            'failed': []
        }
        
        for chunk_id, proc_info in list(self.processes.items()):
            process = proc_info['process']
            
            if process.poll() is None:
                # Still running
                status['running'].append(chunk_id)
                continue

            # Process finished -> reap it and free its GPU slot for the run loop.
            return_code = process.returncode
            success_codes = self.config.get('success_codes', [0])
            if return_code in success_codes:
                status['completed'].append(chunk_id)
                self.completed_chunks.add(chunk_id)
                proc_info['status'] = 'completed'

                # Mark chunk as completed for all machines + release lock
                self.chunk_lock_manager.mark_chunk_completed(chunk_id)
                self.chunk_lock_manager.release_chunk(chunk_id)

                # Clear any retry/backoff bookkeeping (e.g. a retry that finally passed)
                self.retry_count.pop(chunk_id, None)
                self.retry_after.pop(chunk_id, None)
                self.failed_chunks.discard(chunk_id)

                duration = datetime.datetime.now() - proc_info['start_time']
                self.log_message(f"GPU {proc_info['gpu_id']}, chunk {chunk_id} completed successfully (Duration: {duration})")
            else:
                status['failed'].append(chunk_id)
                self.failed_chunks.add(chunk_id)  # ever-failed, for the final summary
                proc_info['status'] = 'failed'

                # Release the lock so this node (after backoff) or any other node can retry
                self.chunk_lock_manager.release_chunk(chunk_id)

                self.log_message(f"GPU {proc_info['gpu_id']}, chunk {chunk_id} failed with return code {return_code}")
                self.read_and_print_process_errors(chunk_id, proc_info['log_file'])

                # Real retry with exponential backoff (capped). Past max_retries -> terminal.
                attempts = self.retry_count.get(chunk_id, 0)
                if self.config.get('restart_failed', True) and attempts < self.config.get('max_retries', 0):
                    self.retry_count[chunk_id] = attempts + 1
                    delay = min(self.retry_base_delay * (2 ** attempts), self.retry_max_delay)
                    self.retry_after[chunk_id] = time.monotonic() + delay
                    self.log_message(f"Scheduling retry {attempts + 1}/{self.config.get('max_retries', 0)} "
                                     f"for chunk {chunk_id} after {delay:.0f}s backoff")
                else:
                    self.permanently_failed.add(chunk_id)
                    self.retry_after.pop(chunk_id, None)
                    self.log_message(f"Chunk {chunk_id} permanently failed after {attempts} retries")

            del self.processes[chunk_id]

        return status

    def get_completion_status(self, total_chunks: int) -> Dict:
        """Get overall completion status across all machines"""
        all_completed = self.chunk_lock_manager.get_completed_chunks()
        currently_running = {p_id for p_id, p_info in self.processes.items() 
                           if p_info['status'] == 'running'}
        
        return {
            'completed_count': len(all_completed),
            'running_count': len(currently_running),
            'failed_count': len(self.permanently_failed),
            'completion_percentage': (len(all_completed) / total_chunks) * 100,
            'completed_chunks': sorted(all_completed),
            'running_chunks': sorted(currently_running)
        }

    def run(self, gpu_ids: List[int], total_chunks: int, kwargs: Optional[Dict] = None):
        """Main execution loop with lock-based chunk assignment"""
        self.log_message(f"Starting processing with {len(gpu_ids)} GPUs and {total_chunks} chunks")
        self.log_message(f"GPU IDs: {gpu_ids}")
        self.log_message(f"Using lock-based chunk assignment from: {self.config['chunk_lock_dir']}")

        self._start_heartbeat()
        try:
            # Persistent GPU slots. Every tick: reap finished processes, then refill
            # EVERY idle GPU. A GPU that can't get a chunk right now is NOT retired —
            # it's retried next tick, so stale-lock reclaim keeps working at the tail
            # and backed-off retries become eligible again once their delay elapses.
            while True:
                # 1. Reap finished processes (frees GPU slots, updates retry/backoff state)
                self.check_processes()

                # 2. Global completion check
                completion_status = self.get_completion_status(total_chunks)
                global_completed = completion_status['completed_count']
                if global_completed >= total_chunks:
                    self.log_message("All chunks completed!")
                    break
                if global_completed + len(self.permanently_failed) >= total_chunks:
                    self.log_message("All chunks either completed or permanently failed!")
                    break

                # 3. Refill every idle GPU. Chunks still in backoff are skipped here so
                #    this node doesn't re-grab a just-failed chunk before its delay (other
                #    nodes are free to pick it up — that flexibility is intentional).
                busy_gpus = {p['gpu_id'] for p in self.processes.values()}
                now = time.monotonic()
                backoff = {c for c, t in self.retry_after.items() if t > now}
                exclude = (self.completed_chunks | self.permanently_failed
                           | set(self.processes.keys()) | backoff)
                for gpu_id in gpu_ids:
                    if gpu_id in busy_gpus:
                        continue
                    next_chunk = self.chunk_lock_manager.get_next_available_chunk(total_chunks, exclude)
                    if next_chunk is not None:
                        self.start_process(gpu_id, next_chunk, total_chunks, kwargs=kwargs)
                        exclude.add(next_chunk)
                        busy_gpus.add(gpu_id)
                    # else: leave this GPU idle; it is retried on the next tick

                # 4. Status update
                self.log_message(f"Status: {len(self.processes)} running, {global_completed} completed globally, "
                                 f"{len(self.permanently_failed)} permanently failed")
                self.log_message(f"Progress: {completion_status['completion_percentage']:.1f}% ({global_completed}/{total_chunks})")
                self.log_message(f"Running chunks: {sorted(self.processes.keys())}")

                time.sleep(self.config['monitor_interval'])

        except KeyboardInterrupt:
            self.log_message("Interrupted by user. Terminating processes...")
            self.cleanup()
            self.print_summary()
            return

        self._stop_heartbeat()

        # Final summary
        self.print_summary()
    
    def _start_heartbeat(self):
        """Start a background thread that refreshes lock timestamps every 30 seconds."""
        self._heartbeat_stop = threading.Event()

        def _loop():
            while not self._heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
                self.chunk_lock_manager.refresh_all_locks()

        self._heartbeat_thread = threading.Thread(target=_loop, daemon=True, name="octorun-heartbeat")
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        if hasattr(self, '_heartbeat_stop'):
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=5)

    def cleanup(self):
        """Cleanup all running processes and release chunk locks"""
        self._stop_heartbeat()

        for chunk_id, proc_info in self.processes.items():
            process = proc_info['process']
            if process.poll() is None:
                self.log_message(f"Terminating process for chunk {chunk_id} (PID: {process.pid})")
                process.terminate()

                # Wait a bit, then kill if necessary
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

        # Release all acquired chunk locks
        self.chunk_lock_manager.release_all_locks()
        self.log_message("All chunk locks released")
    
    def print_summary(self):
        """Print final execution summary"""
        total_time = datetime.datetime.now() - self.start_time
        global_completed = self.chunk_lock_manager.get_completed_chunks()
        
        self.log_message("\n" + "=" * 80)
        self.log_message("FINAL SUMMARY")
        self.log_message("=" * 80)
        self.log_message(f"Total execution time: {total_time}")
        self.log_message(f"Completed chunks (globally): {len(global_completed)}")
        self.log_message(f"Completed chunks (this session): {len(self.completed_chunks)}")
        self.log_message(f"Chunks that failed at least once (this session): {len(self.failed_chunks)}")
        self.log_message(f"Permanently failed chunks (exhausted retries): {len(self.permanently_failed)}")

        if self.permanently_failed:
            self.log_message(f"Permanently failed chunk IDs: {sorted(self.permanently_failed)}")

        self.log_message(f"Session log: {self.session_log}")
        self.log_message(f"Individual logs: {self.config['log_dir']}/chunk_*.log")
