"""
GPU Memory Bandwidth Benchmark Script
Tests GPU memory bandwidth performance.
Requires PyTorch with CUDA support.
"""

import sys
import json
import time

try:
    import torch
except ImportError:
    print(json.dumps({"error": "PyTorch is not installed. Please install PyTorch with CUDA support."}))
    sys.exit(1)


def test_memory_bandwidth_torch(gpu_id, duration=5.0):
    """Test GPU memory bandwidth using PyTorch."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    if gpu_id >= torch.cuda.device_count():
        return {"error": f"GPU {gpu_id} not available"}
    
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Test different memory operations
    size = 128 * 1024 * 1024  # 128M elements
    dtype = torch.float32 # cpu does not support bf16
    
    # Memory copy test
    data = torch.randn(size, device='cpu', dtype=dtype)
    bytes_per_element = data.element_size()
    
    operations = 0
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        torch.cuda.synchronize()
        # Copy operation
        data_copy = data.to(device=device)
        torch.cuda.synchronize()
        operations += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate bandwidth
    bytes_transferred = operations * size * bytes_per_element
    bandwidth_gbps = (bytes_transferred / total_time) / 1e9  # GB/s
    
    memory_used_gb = torch.cuda.memory_allocated(device) / 1e9
    
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "operations": operations,
        "duration": total_time,
        "data_size_mb": (size * bytes_per_element) / 1e6,
        "size": size,
        "dtype": dtype.__str__().split('.')[-1],
        "memory_used_gb": memory_used_gb,
        "framework": "pytorch"
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python memory_benchmark.py <gpu_id> <duration>"}))
        sys.exit(1)
    
    gpu_id = int(sys.argv[1])
    duration = float(sys.argv[2])
    
    result = test_memory_bandwidth_torch(gpu_id, duration)
    print(json.dumps(result))
