"""
GPU Compute Performance Benchmark Script
Tests GPU compute performance (TFLOPs) using matrix multiplication.
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


def test_compute_performance_torch(gpu_id, duration=10.0):
    """Test GPU compute performance using PyTorch."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    if gpu_id >= torch.cuda.device_count():
        return {"error": f"GPU {gpu_id} not available"}
    
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Warm up
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    
    # Actual test
    matrix_size = 16_384  # Larger matrix for more intensive computation
    dtype = torch.bfloat16
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)

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
    
    return {
        "tflops": tflops,
        "operations": operations,
        "duration": total_time,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "matrix_size": matrix_size,
        "dtype": dtype.__str__().split('.')[-1],
        "framework": "pytorch"
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python compute_benchmark.py <gpu_id> <duration>"}))
        sys.exit(1)
    
    gpu_id = int(sys.argv[1])
    duration = float(sys.argv[2])
    
    result = test_compute_performance_torch(gpu_id, duration)
    print(json.dumps(result))
