"""
GPU Benchmark Module for OctoRun
Continuously tests GPU speed (TFLOPs) and memory bandwidth.
"""

import time
import datetime
import threading
import signal
import sys
from typing import List, Dict, Optional, Any
import subprocess
import json
import os


class GPUBenchmark:
    """
    Continuous GPU performance testing for speed (TFLOPs) and memory bandwidth.
    """

    def __init__(self, gpu_ids: List[int], test_duration: float = 5.0, test_interval: float = 10.0):
        """
        Initialize GPU benchmark.
        
        Args:
            gpu_ids: List of GPU IDs to test
            test_duration: Duration of each test in seconds
            test_interval: Interval between tests in seconds
        """
        self.gpu_ids = gpu_ids
        self.test_duration = test_duration
        self.test_interval = test_interval
        self.running = False
        self.threads: List[threading.Thread] = []
        self.results: Dict[int, Dict[str, Any]] = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize results storage for each GPU
        for gpu_id in gpu_ids:
            self.results[gpu_id] = {
                'compute_history': [],
                'memory_history': [],
                'last_test_time': None,
                'status': 'initializing'
            }
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n🛑 Received signal {signum}. Stopping benchmark...")
        self.stop()
        sys.exit(0)
    
    def _test_gpu_compute_performance(self, gpu_id: int) -> Dict[str, Any]:
        """
        Test GPU compute performance (TFLOPs) using matrix multiplication.
        
        Args:
            gpu_id: GPU ID to test
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Get path to compute benchmark script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, 'benchmarks', 'compute_benchmark.py')
            
            # Run the test script
            result = subprocess.run(
                ['python', script_path, str(gpu_id), str(self.test_duration)],
                capture_output=True,
                text=True,
                timeout=self.test_duration + 10
            )
            
            if result.returncode == 0:
                performance_data = json.loads(result.stdout.strip())
                return performance_data
            else:
                return {
                    "error": f"Script failed: {result.stderr}",
                    "returncode": result.returncode
                }
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _test_gpu_memory_bandwidth(self, gpu_id: int) -> Dict[str, Any]:
        """
        Test GPU memory bandwidth.
        
        Args:
            gpu_id: GPU ID to test
            
        Returns:
            Dictionary containing memory bandwidth metrics
        """
        try:
            # Get path to memory benchmark script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, 'benchmarks', 'memory_benchmark.py')
            
            # Run the test script (use fixed 5 second duration for memory tests)
            result = subprocess.run(
                ['python', script_path, str(gpu_id), str(self.test_duration)],
                capture_output=True,
                text=True,
                timeout=self.test_duration + 10
            )
            
            if result.returncode == 0:
                bandwidth_data = json.loads(result.stdout.strip())
                return bandwidth_data
            else:
                return {"error": f"Script failed: {result.stderr}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    
    def _benchmark_worker(self, gpu_id: int):
        """
        Worker function for continuous GPU benchmarking.
        
        Args:
            gpu_id: GPU ID to benchmark
        """
        # print(f"🚀 Starting benchmark worker for GPU {gpu_id}")
        
        while self.running:
            try:
                self.results[gpu_id]['status'] = 'testing'
                test_start = time.time()
                
                compute_result = self._test_gpu_compute_performance(gpu_id)
                
                memory_result = self._test_gpu_memory_bandwidth(gpu_id)
                
                test_end = time.time()
                
                # Store results
                timestamp = datetime.datetime.now()
                self.results[gpu_id]['compute_history'].append({
                    'timestamp': timestamp,
                    'result': compute_result
                })
                self.results[gpu_id]['memory_history'].append({
                    'timestamp': timestamp,
                    'result': memory_result
                })
                self.results[gpu_id]['last_test_time'] = timestamp
                self.results[gpu_id]['status'] = 'idle'
                
                # Wait for next test cycle
                elapsed = test_end - test_start
                sleep_time = max(0, self.test_interval - elapsed)
                
                if sleep_time > 0 and self.running:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"❌ Error in benchmark worker for GPU {gpu_id}: {e}")
                self.results[gpu_id]['status'] = 'error'
                time.sleep(5)  # Wait before retry
    
    
    def _print_results_table(self):
        """Prints a formatted table of the latest benchmark results."""
        # ANSI escape code to clear the screen and move cursor to top-left
        print("\033[H\033[J", end="")
        
        # Header
        header = (
            f"{'GPU':<5} | {'Status':<12} | {'Compute (TFLOPs)':<20} | "
            f"{'Memory (GB/s)':<18} | {'Last Test':<20}"
        )
        print(header)
        print("-" * len(header))
        
        # Rows
        for gpu_id in sorted(self.results.keys()):
            result = self.results[gpu_id]
            status = result.get('status', 'N/A')
            
            # Compute
            last_compute = result['compute_history'][-1]['result'] if result['compute_history'] else {}
            if 'tflops' in last_compute:
                compute_str = f"{last_compute['tflops']:.2f}"
            elif 'estimated_utilization_percent' in last_compute:
                compute_str = f"{last_compute['estimated_utilization_percent']:.1f}% util"
            else:
                compute_str = "N/A"

            # Memory
            last_memory = result['memory_history'][-1]['result'] if result['memory_history'] else {}
            if 'bandwidth_gbps' in last_memory:
                memory_str = f"{last_memory['bandwidth_gbps']:.2f}"
            else:
                memory_str = "N/A"

            # Last test time
            last_test_time = result.get('last_test_time')
            if last_test_time:
                time_diff = datetime.datetime.now() - last_test_time
                last_test_str = f"{int(time_diff.total_seconds())}s ago"
            else:
                last_test_str = "Never"
            
            row = (
                f"{gpu_id:<5} | {status:<12} | {compute_str:<20} | "
                f"{memory_str:<18} | {last_test_str:<20}"
            )
            print(row)
        
        print("\nPress Ctrl+C to stop.")
    
    
    def start(self):
        """Start continuous GPU benchmarking."""
        if self.running:
            print("⚠️  Benchmark already running")
            return
        
        print(f"🚀 Starting continuous GPU benchmark for GPUs: {self.gpu_ids}")
        print(f"⏱️  Test duration: {self.test_duration}s, Interval: {self.test_interval}s")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        self.running = True
        
        # Start worker threads for each GPU
        for gpu_id in self.gpu_ids:
            thread = threading.Thread(
                target=self._benchmark_worker,
                args=(gpu_id,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        try:
            # Keep main thread alive and refresh results
            while self.running:
                self._print_results_table()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping benchmark...")
            self.stop()
    
    def stop(self):
        """Stop benchmarking."""
        if not self.running:
            return
        
        print("🛑 Stopping GPU benchmark...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        print("✅ GPU benchmark stopped")


def run_gpu_benchmark(gpu_ids: Optional[List[int]] = None, 
                     test_duration: float = 5.0, 
                     test_interval: float = 10.0):
    """
    Run continuous GPU benchmark.
    
    Args:
        gpu_ids: List of GPU IDs to test (None for auto-detect)
        test_duration: Duration of each test in seconds
        test_interval: Interval between tests in seconds
    """
    # Auto-detect GPUs if not specified
    if gpu_ids is None:
        from .cli import get_available_gpus
        gpu_ids = get_available_gpus()
        
        if not gpu_ids:
            print("❌ No GPUs found!")
            return
        
        print(f"🔍 Auto-detected GPUs: {gpu_ids}")
    
    # Validate GPU IDs
    try:
        from .cli import get_available_gpus
        available_gpus = get_available_gpus()
        
        invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in available_gpus]
        if invalid_gpus:
            print(f"❌ Invalid GPU IDs: {invalid_gpus}")
            print(f"📋 Available GPUs: {available_gpus}")
            return
    except Exception as e:
        print(f"⚠️  Could not validate GPU IDs: {e}")
    
    # Create and start benchmark
    benchmark = GPUBenchmark(gpu_ids, test_duration, test_interval)
    benchmark.start()
