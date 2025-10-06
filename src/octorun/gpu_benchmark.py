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

    def __init__(self, gpu_ids: List[int], test_duration: float = 10.0, test_interval: float = 30.0):
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
                ['python', script_path, str(gpu_id), '5.0'],
                capture_output=True,
                text=True,
                timeout=10
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
        print(f"🚀 Starting benchmark worker for GPU {gpu_id}")
        
        while self.running:
            try:
                self.results[gpu_id]['status'] = 'testing'
                test_start = time.time()
                
                print(f"🧪 Testing GPU {gpu_id} compute performance...")
                compute_result = self._test_gpu_compute_performance(gpu_id)
                
                print(f"💾 Testing GPU {gpu_id} memory bandwidth...")
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
                
                # Print results
                self._print_gpu_results(gpu_id, compute_result, memory_result)
                
                # Wait for next test cycle
                elapsed = test_end - test_start
                sleep_time = max(0, self.test_interval - elapsed)
                
                if sleep_time > 0 and self.running:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"❌ Error in benchmark worker for GPU {gpu_id}: {e}")
                self.results[gpu_id]['status'] = 'error'
                time.sleep(5)  # Wait before retry
    
    
    def _print_gpu_results(self, gpu_id: int, compute_result: Dict, memory_result: Dict):
        """Print GPU performance results."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n📊 GPU {gpu_id} Performance Results ({timestamp})")
        print("=" * 60)
        
        if 'error' not in compute_result:
            print(f"🔥 Compute Performance:")
            framework = compute_result.get('framework', 'unknown')
            if 'tflops' in compute_result:
                print(f"   TFLOPs: {compute_result.get('tflops', 0):.2f}")
                print(f"   Operations: {compute_result.get('operations', 0)}")
                print(f"   Matrix Size: {compute_result.get('matrix_size', 0)}")
            elif 'estimated_utilization_percent' in compute_result:
                print(f"   GPU Utilization: {compute_result.get('estimated_utilization_percent', 0):.1f}%")
                print(f"   GPU Name: {compute_result.get('gpu_name', 'Unknown')}")
            elif 'utilization_percent' in compute_result:
                print(f"   GPU Utilization: {compute_result.get('utilization_percent', 0)}%")
                print(f"   GPU Name: {compute_result.get('gpu_name', 'Unknown')}")
            
            if 'note' in compute_result:
                print(f"   Note: {compute_result['note']}")
            print(f"   Framework: {framework}")
        else:
            print(f"❌ Compute Test Error: {compute_result['error']}")
        
        if 'error' not in memory_result:
            print(f"💾 Memory Performance:")
            framework = memory_result.get('framework', 'unknown')
            if 'bandwidth_gbps' in memory_result:
                print(f"   Bandwidth: {memory_result.get('bandwidth_gbps', 0):.2f} GB/s")
                print(f"   Operations: {memory_result.get('operations', 0)}")
            elif 'memory_total_mb' in memory_result:
                total_mb = memory_result.get('memory_total_mb', 0)
                free_mb = memory_result.get('memory_free_mb', 0)
                used_mb = memory_result.get('memory_used_mb', 0)
                print(f"   Total Memory: {total_mb} MB")
                print(f"   Free Memory: {free_mb} MB")
                print(f"   Used Memory: {used_mb} MB")
                print(f"   Usage: {(used_mb / total_mb * 100):.1f}%" if total_mb > 0 else "   Usage: N/A")
                
            if 'note' in memory_result:
                print(f"   Note: {memory_result['note']}")
            print(f"   Framework: {framework}")
        else:
            print(f"❌ Memory Test Error: {memory_result['error']}")
        
        print("=" * 60)
    
    
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
            # Keep main thread alive
            while self.running:
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
                     test_duration: float = 10.0, 
                     test_interval: float = 30.0):
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
