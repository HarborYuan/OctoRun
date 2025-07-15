"""
GPU Benchmark Module for OctoRun
Continuously tests GPU speed (TFLOPs) and communication performance.
"""

import time
import datetime
import threading
import signal
import sys
from typing import List, Dict, Literal, Optional, Any
import subprocess
import json
import os


class GPUBenchmark:
    """
    Continuous GPU performance testing for speed (TFLOPs) and communication.
    """

    def __init__(self, gpu_ids: List[int], test_duration: float = 10.0, test_interval: float = 30.0, mode: Literal['single', 'p2p'] = 'single'):
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
        self.mode = mode
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
                'communication_history': [],
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
            # Create a simple Python script for GPU compute testing
            test_script = f'''
import sys
import json
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def test_compute_performance_torch(gpu_id, duration=10.0):
    """Test GPU compute performance using PyTorch."""
    if not TORCH_AVAILABLE:
        return {{"error": "PyTorch not available"}}
    
    if not torch.cuda.is_available():
        return {{"error": "CUDA not available"}}
    
    if gpu_id >= torch.cuda.device_count():
        return {{"error": f"GPU {{gpu_id}} not available"}}
    
    device = torch.device(f"cuda:{{gpu_id}}")
    torch.cuda.set_device(device)
    
    # Warm up
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    
    # Actual test
    matrix_size = 4096  # Larger matrix for more intensive computation
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    
    operations = 0
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        torch.cuda.synchronize()
        op_start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        op_end = time.time()
        
        operations += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate TFLOPs
    # Matrix multiplication: 2 * n^3 FLOPs for n x n matrices
    flops_per_operation = 2 * (matrix_size ** 3)
    total_flops = operations * flops_per_operation
    tflops = (total_flops / total_time) / 1e12
    
    # Get memory info
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9   # GB
    
    return {{
        "tflops": tflops,
        "operations": operations,
        "duration": total_time,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "matrix_size": matrix_size,
        "framework": "pytorch"
    }}

def test_compute_performance_nvidia_ml(gpu_id, duration=10.0):
    """Test GPU performance using nvidia-ml-py (fallback)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        
        # Get GPU info
        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Simulate compute test by monitoring GPU utilization
        start_time = time.time()
        utilization_samples = []
        
        while (time.time() - start_time) < duration:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_samples.append(util.gpu)
                time.sleep(0.1)
            except:
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        avg_utilization = sum(utilization_samples) / len(utilization_samples) if utilization_samples else 0
        
        return {{
            "estimated_utilization_percent": avg_utilization,
            "gpu_name": name,
            "duration": total_time,
            "memory_total_gb": memory_info.total / 1e9,
            "memory_free_gb": memory_info.free / 1e9,
            "framework": "nvidia-ml-py",
            "note": "No PyTorch available - showing GPU utilization instead of TFLOPs"
        }}
        
    except ImportError:
        return {{"error": "Neither PyTorch nor pynvml available for GPU testing"}}
    except Exception as e:
        return {{"error": f"nvidia-ml-py error: {{str(e)}}"}}

def fallback_compute_test(gpu_id, duration=10.0):
    """Fallback test using nvidia-smi."""
    import subprocess
    
    try:
        # Check if GPU exists
        result = subprocess.run([
            "nvidia-smi", "-i", str(gpu_id), 
            "--query-gpu=name,utilization.gpu,memory.total,memory.free", 
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        
        line = result.stdout.strip()
        parts = line.split(', ')
        
        return {{
            "gpu_name": parts[0],
            "utilization_percent": int(parts[1]),
            "memory_total_mb": int(parts[2]),
            "memory_free_mb": int(parts[3]),
            "framework": "nvidia-smi",
            "note": "Limited GPU info - install PyTorch for detailed performance testing"
        }}
        
    except Exception as e:
        return {{"error": f"nvidia-smi error: {{str(e)}}"}}

if __name__ == "__main__":
    gpu_id = {gpu_id}
    duration = {self.test_duration}
    
    # Try different methods in order of preference
    if TORCH_AVAILABLE:
        result = test_compute_performance_torch(gpu_id, duration)
    else:
        # Try nvidia-ml-py first, then fallback to nvidia-smi
        result = test_compute_performance_nvidia_ml(gpu_id, duration)
        if "error" in result:
            result = fallback_compute_test(gpu_id, duration)
    
    print(json.dumps(result))
'''
            
            # Write test script to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                script_path = f.name
            
            try:
                # Run the test script
                result = subprocess.run(
                    ['python', script_path],
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
            finally:
                # Clean up temporary file
                try:
                    os.unlink(script_path)
                except:
                    pass
                    
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
            test_script = f'''
import json
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def test_memory_bandwidth_torch(gpu_id, duration=5.0):
    """Test GPU memory bandwidth using PyTorch."""
    if not TORCH_AVAILABLE:
        return {{"error": "PyTorch not available"}}
    
    if not torch.cuda.is_available():
        return {{"error": "CUDA not available"}}
    
    if gpu_id >= torch.cuda.device_count():
        return {{"error": f"GPU {{gpu_id}} not available"}}
    
    device = torch.device(f"cuda:{{gpu_id}}")
    torch.cuda.set_device(device)
    
    # Test different memory operations
    size = 256 * 1024 * 1024  # 256M elements
    
    # Memory copy test
    data = torch.randn(size, device=device, dtype=torch.float32)
    bytes_per_element = 4  # float32
    
    operations = 0
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        torch.cuda.synchronize()
        # Copy operation
        data_copy = data.clone()
        torch.cuda.synchronize()
        operations += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate bandwidth
    bytes_transferred = operations * size * bytes_per_element
    bandwidth_gbps = (bytes_transferred / total_time) / 1e9  # GB/s
    
    return {{
        "bandwidth_gbps": bandwidth_gbps,
        "operations": operations,
        "duration": total_time,
        "data_size_mb": (size * bytes_per_element) / 1e6,
        "framework": "pytorch"
    }}

def fallback_memory_test(gpu_id, duration=5.0):
    """Fallback memory test using nvidia-smi."""
    import subprocess
    
    try:
        result = subprocess.run([
            "nvidia-smi", "-i", str(gpu_id),
            "--query-gpu=name,memory.total,memory.free,memory.used", 
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        
        line = result.stdout.strip()
        parts = line.split(', ')
        
        return {{
            "gpu_name": parts[0],
            "memory_total_mb": int(parts[1]),
            "memory_free_mb": int(parts[2]),
            "memory_used_mb": int(parts[3]),
            "framework": "nvidia-smi",
            "note": "Limited memory info - install PyTorch for bandwidth testing"
        }}
        
    except Exception as e:
        return {{"error": f"nvidia-smi error: {{str(e)}}"}}

if __name__ == "__main__":
    gpu_id = {gpu_id}
    duration = 5.0
    
    if TORCH_AVAILABLE:
        result = test_memory_bandwidth_torch(gpu_id, duration)
    else:
        result = fallback_memory_test(gpu_id, duration)
    
    print(json.dumps(result))
'''
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                script_path = f.name
            
            try:
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    bandwidth_data = json.loads(result.stdout.strip())
                    return bandwidth_data
                else:
                    return {"error": f"Script failed: {result.stderr}"}
            finally:
                try:
                    os.unlink(script_path)
                except:
                    pass
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _test_gpu_communication(self, gpu_ids: List[int]) -> Dict[str, Any]:
        """
        Test GPU-to-GPU communication performance (if multiple GPUs).
        
        Args:
            gpu_ids: List of GPU IDs to test communication between
            
        Returns:
            Dictionary containing communication metrics
        """
        if len(gpu_ids) < 2:
            return {"error": "Need at least 2 GPUs for communication test"}
        
        try:
            test_script = f'''
import json
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def test_gpu_communication_torch(gpu_ids, duration=5.0):
    """Test GPU-to-GPU communication using PyTorch."""
    if not TORCH_AVAILABLE:
        return {{"error": "PyTorch not available for communication testing"}}
    
    if not torch.cuda.is_available():
        return {{"error": "CUDA not available"}}
    
    if len(gpu_ids) < 2:
        return {{"error": "Need at least 2 GPUs"}}
    
    results = {{}}
    
    # Test P2P copy between each pair of GPUs
    for i, gpu1 in enumerate(gpu_ids):
        for j, gpu2 in enumerate(gpu_ids):
            if i >= j:  # Only test upper triangle to avoid duplicates
                continue
                
            try:
                device1 = torch.device(f"cuda:{{gpu1}}")
                device2 = torch.device(f"cuda:{{gpu2}}")
                
                # Test data transfer
                size = 64 * 1024 * 1024  # 64M elements
                data = torch.randn(size, device=device1, dtype=torch.float32)
                
                operations = 0
                start_time = time.time()
                
                while (time.time() - start_time) < duration:
                    torch.cuda.synchronize()
                    data_copy = data.to(device2)
                    torch.cuda.synchronize()
                    operations += 1
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Calculate bandwidth
                bytes_per_element = 4  # float32
                bytes_transferred = operations * size * bytes_per_element
                bandwidth_gbps = (bytes_transferred / total_time) / 1e9
                
                results[f"gpu{{gpu1}}_to_gpu{{gpu2}}"] = {{
                    "bandwidth_gbps": bandwidth_gbps,
                    "operations": operations,
                    "duration": total_time,
                    "framework": "pytorch"
                }}
                
            except Exception as e:
                results[f"gpu{{gpu1}}_to_gpu{{gpu2}}"] = {{"error": str(e)}}
    
    return results

def fallback_communication_test(gpu_ids):
    """Fallback communication test - just check GPU topology."""
    import subprocess
    
    try:
        # Get GPU topology info
        result = subprocess.run([
            "nvidia-smi", "topo", "-m"
        ], capture_output=True, text=True, check=True)
        
        return {{
            "note": "PyTorch not available - showing GPU topology instead",
            "topology_info": result.stdout.strip(),
            "framework": "nvidia-smi"
        }}
        
    except Exception as e:
        return {{"error": f"nvidia-smi topology error: {{str(e)}}"}}

if __name__ == "__main__":
    gpu_ids = {gpu_ids}
    duration = 3.0
    
    if TORCH_AVAILABLE:
        result = test_gpu_communication_torch(gpu_ids, duration)
    else:
        result = fallback_communication_test(gpu_ids)
    
    print(json.dumps(result))
'''
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                script_path = f.name
            
            try:
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    comm_data = json.loads(result.stdout.strip())
                    return comm_data
                else:
                    return {"error": f"Script failed: {result.stderr}"}
            finally:
                try:
                    os.unlink(script_path)
                except:
                    pass
                    
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
    
    def _communication_worker(self):
        """
        Worker function for continuous GPU communication testing.
        """
        if len(self.gpu_ids) < 2:
            print("⚠️  Skipping communication tests (need at least 2 GPUs)")
            return
        
        print(f"📡 Starting communication benchmark worker for GPUs {self.gpu_ids}")
        
        while self.running:
            try:
                print(f"📡 Testing GPU communication...")
                comm_result = self._test_gpu_communication(self.gpu_ids)
                
                # Store results
                timestamp = datetime.datetime.now()
                for gpu_id in self.gpu_ids:
                    self.results[gpu_id]['communication_history'].append({
                        'timestamp': timestamp,
                        'result': comm_result
                    })
                
                # Print results
                self._print_communication_results(comm_result)
                
                # Wait for next test cycle
                if self.running:
                    time.sleep(self.test_interval)
                    
            except Exception as e:
                print(f"❌ Error in communication benchmark: {e}")
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
    
    def _print_communication_results(self, comm_result: Dict):
        """Print GPU communication results."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n📡 GPU Communication Results ({timestamp})")
        print("=" * 60)
        
        if 'error' in comm_result:
            print(f"❌ Communication Test Error: {comm_result['error']}")
        elif 'note' in comm_result:
            # Fallback mode - showing topology instead of bandwidth
            print(f"ℹ️  {comm_result['note']}")
            if 'topology_info' in comm_result:
                print("🗺️  GPU Topology:")
                # Print first few lines of topology info
                lines = comm_result['topology_info'].split('\n')[:10]
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                if len(comm_result['topology_info'].split('\n')) > 10:
                    print("   ... (truncated)")
        else:
            # Normal communication test results
            bandwidth_results = []
            for pair, result in comm_result.items():
                if 'error' not in result:
                    bandwidth = result.get('bandwidth_gbps', 0)
                    bandwidth_results.append((pair, bandwidth))
                    print(f"📡 {pair}: {bandwidth:.2f} GB/s")
                else:
                    print(f"❌ {pair}: {result['error']}")
            
            # Show summary if we have results
            if bandwidth_results:
                bandwidths = [bw for _, bw in bandwidth_results]
                avg_bw = sum(bandwidths) / len(bandwidths)
                max_bw = max(bandwidths)
                min_bw = min(bandwidths)
                print(f"\n📊 Summary:")
                print(f"   Average: {avg_bw:.2f} GB/s")
                print(f"   Maximum: {max_bw:.2f} GB/s")
                print(f"   Minimum: {min_bw:.2f} GB/s")
        
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
        if self.mode == 'single':
            for gpu_id in self.gpu_ids:
                thread = threading.Thread(
                    target=self._benchmark_worker,
                    args=(gpu_id,),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
        
        # Start communication worker thread
        if self.mode == 'p2p':
            assert len(self.gpu_ids) > 1, "P2P mode requires at least 2 GPUs"
            comm_thread = threading.Thread(
                target=self._communication_worker,
                daemon=True
            )
            comm_thread.start()
            self.threads.append(comm_thread)
        
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        summary = {}
        
        for gpu_id in self.gpu_ids:
            gpu_data = self.results[gpu_id]
            
            # Compute statistics
            compute_tflops = [
                result['result'].get('tflops', 0) 
                for result in gpu_data['compute_history'] 
                if 'error' not in result['result']
            ]
            
            memory_bandwidth = [
                result['result'].get('bandwidth_gbps', 0) 
                for result in gpu_data['memory_history'] 
                if 'error' not in result['result']
            ]
            
            summary[f'gpu_{gpu_id}'] = {
                'status': gpu_data['status'],
                'test_count': len(gpu_data['compute_history']),
                'last_test': gpu_data['last_test_time'].isoformat() if gpu_data['last_test_time'] else None,
                'compute_performance': {
                    'avg_tflops': sum(compute_tflops) / len(compute_tflops) if compute_tflops else 0,
                    'max_tflops': max(compute_tflops) if compute_tflops else 0,
                    'min_tflops': min(compute_tflops) if compute_tflops else 0,
                },
                'memory_performance': {
                    'avg_bandwidth_gbps': sum(memory_bandwidth) / len(memory_bandwidth) if memory_bandwidth else 0,
                    'max_bandwidth_gbps': max(memory_bandwidth) if memory_bandwidth else 0,
                    'min_bandwidth_gbps': min(memory_bandwidth) if memory_bandwidth else 0,
                }
            }
        
        return summary


def run_gpu_benchmark(gpu_ids: Optional[List[int]] = None, 
                     test_duration: float = 10.0, 
                     test_interval: float = 30.0,
                     mode: Literal['single', 'p2p'] = 'single',
                     ):
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
    benchmark = GPUBenchmark(gpu_ids, test_duration, test_interval, mode=mode)
    benchmark.start()
