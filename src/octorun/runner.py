import datetime
import time
from .lock_manager import ChunkLockManager
import json
import os
import subprocess
from typing import Dict, List, Optional


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
        self.failed_chunks = set()
        self.retry_count = {}

        # Setup logging
        self.setup_logging()

        # Log already completed chunks
        if self.completed_chunks:
            self.log_message(f"Found {len(self.completed_chunks)} previously completed chunks: {sorted(self.completed_chunks)}")
    
    def setup_logging(self):
        """Setup logging directory"""
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)

        # Create session log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = os.path.join(log_dir, f"session_{timestamp}.log")

        with open(self.session_log, 'w') as f:
            f.write(f"Session Started: {self.start_time}\n")
            f.write(f"Configuration: {json.dumps(self.config, indent=2)}\n")
            f.write("-" * 80 + "\n")

    def log_message(self, message: str):
        """Log message to session log and print"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.session_log, 'a') as f:
            f.write(log_entry + "\n")

    def start_process(self, gpu_id: int, chunk_id: int, total_chunks: int) -> Optional[subprocess.Popen]:
        """Start a process on a specific GPU"""
        try:
            cmd = [
                'python', 
                self.config['script_path'],
                '--gpu_id', str(gpu_id),
                '--chunk_id', str(chunk_id),
                '--total_chunks', str(total_chunks),
            ]
            # Setup environment
            env = os.environ.copy()

            # Setup logging
            log_file = os.path.join(self.config['log_dir'], f"chunk_{chunk_id}.log")

            with open(log_file, 'w') as f:
                f.write(f"Starting process on GPU {gpu_id}, chunk {chunk_id}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("-" * 50 + "\n")

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                env=env
            )

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
            else:
                # Process finished
                return_code = process.returncode
                if return_code == 0:
                    status['completed'].append(chunk_id)
                    self.completed_chunks.add(chunk_id)
                    proc_info['status'] = 'completed'
                    
                    # Mark chunk as completed for all machines
                    self.chunk_lock_manager.mark_chunk_completed(chunk_id)
                    
                    # Release chunk lock
                    self.chunk_lock_manager.release_chunk(chunk_id)
                    
                    duration = datetime.datetime.now() - proc_info['start_time']
                    self.log_message(f"GPU {proc_info['gpu_id']}, chunk {chunk_id} completed successfully (Duration: {duration})")
                else:
                    status['failed'].append(chunk_id)
                    self.failed_chunks.add(chunk_id)
                    proc_info['status'] = 'failed'
                    
                    # Release chunk lock
                    self.chunk_lock_manager.release_chunk(chunk_id)
                    
                    self.log_message(f"GPU {proc_info['gpu_id']}, chunk {chunk_id} failed with return code {return_code}")
                    
                    # Handle retries
                    if self.config['restart_failed'] and self.retry_count.get(chunk_id, 0) < self.config['max_retries']:
                        self.retry_count[chunk_id] = self.retry_count.get(chunk_id, 0) + 1
                        self.log_message(f"Scheduling retry {self.retry_count[chunk_id]}/{self.config['max_retries']} for chunk {chunk_id}")
        
        return status

    def get_completion_status(self, total_chunks: int) -> Dict:
        """Get overall completion status across all machines"""
        all_completed = self.chunk_lock_manager.get_completed_chunks()
        currently_running = {p_id for p_id, p_info in self.processes.items() 
                           if p_info['status'] == 'running'}
        
        return {
            'completed_count': len(all_completed),
            'running_count': len(currently_running),
            'remaining_count': total_chunks - len(all_completed) - len(currently_running),
            'completion_percentage': (len(all_completed) / total_chunks) * 100,
            'completed_chunks': sorted(all_completed),
            'running_chunks': sorted(currently_running)
        }

    def run(self, gpu_ids: List[int], total_chunks: int):
        """Main execution loop with lock-based chunk assignment"""
        self.log_message(f"Starting processing with {len(gpu_ids)} GPUs and {total_chunks} chunks")
        self.log_message(f"GPU IDs: {gpu_ids}")
        self.log_message(f"Using lock-based chunk assignment from: {self.config['chunk_lock_dir']}")

        try:
            # Initial process assignment - try to get available chunks for each GPU
            for gpu_id in gpu_ids:
                chunk_id = self.chunk_lock_manager.get_next_available_chunk(
                    total_chunks, 
                    self.completed_chunks | self.failed_chunks
                )
                
                if chunk_id is not None:
                    self.start_process(gpu_id, chunk_id, total_chunks)
                    time.sleep(2)  # Stagger starts
                else:
                    self.log_message(f"No available chunks for GPU {gpu_id} at startup")
            
            # Monitor loop
            while True:
                # Check global completion status
                completion_status = self.get_completion_status(total_chunks)
                
                if completion_status['completed_count'] >= total_chunks:
                    self.log_message("All chunks completed!")
                    break
                
                time.sleep(self.config['monitor_interval'])
                
                status = self.check_processes()
                
                # For each GPU that finished a chunk, try to assign a new one
                finished_chunks = status['completed'] + status['failed']
                for chunk_id in finished_chunks:
                    if chunk_id in self.processes:
                        gpu_id = self.processes[chunk_id]['gpu_id']
                        
                        # Try to get next available chunk for this GPU
                        next_chunk = self.chunk_lock_manager.get_next_available_chunk(
                            total_chunks,
                            self.completed_chunks | self.failed_chunks | 
                            {p_id for p_id, p_info in self.processes.items() 
                             if p_info['status'] == 'running'}
                        )

                        if next_chunk is not None:
                            self.start_process(gpu_id, next_chunk, total_chunks)
                        else:
                            self.log_message(f"No more chunks available for GPU {gpu_id}")

                # Print status update
                running_count = len(status['running'])
                global_completed = completion_status['completed_count']
                failed_count = len([c for c in self.failed_chunks 
                                  if self.retry_count.get(c, 0) >= self.config['max_retries']])
                
                self.log_message(f"Status: {running_count} running, {global_completed} completed globally, {failed_count} permanently failed")
                self.log_message(f"Progress: {completion_status['completion_percentage']:.1f}% ({global_completed}/{total_chunks})")
                self.log_message(f"Running chunks: {completion_status['running_chunks']}")
        
        except KeyboardInterrupt:
            self.log_message("Interrupted by user. Terminating processes...")
            self.cleanup()
        
        # Final summary
        self.print_summary()
    
    def cleanup(self):
        """Cleanup all running processes and release chunk locks"""
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
        self.log_message(f"Failed chunks (this session): {len(self.failed_chunks)}")
        
        if self.failed_chunks:
            self.log_message(f"Failed chunk IDs: {sorted(self.failed_chunks)}")

        self.log_message(f"Session log: {self.session_log}")
        self.log_message(f"Individual logs: {self.config['log_dir']}/chunk_*.log")
