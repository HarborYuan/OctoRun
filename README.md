<div align="center">

# 🐙 OctoRun

**Distributed Parallel Execution Made Simple**

*A powerful command-line tool for running Python scripts across multiple GPUs with intelligent task management and monitoring*

[![PyPI version](https://img.shields.io/pypi/v/octorun.svg)](https://pypi.org/project/octorun/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/HarborYuan/OctoRun/actions)

---

</div>

## 📋 Overview

**OctoRun** is designed to help you run computationally intensive Python scripts across multiple GPUs efficiently. It automatically manages GPU allocation, chunks your workload, handles failures with retry mechanisms, and provides comprehensive monitoring and logging.

## ✨ Key Features

- 🔍 **Automatic GPU Detection**: Automatically detects and utilizes available GPUs
- 🧩 **Intelligent Chunk Management**: Divides work into chunks and distributes across GPUs
- 🔄 **Failure Recovery**: Automatic retry mechanism for failed chunks
- 📊 **Comprehensive Logging**: Detailed logging for monitoring and debugging
- ⚙️ **Flexible Configuration**: JSON-based configuration with CLI overrides
- 🎯 **Kwargs Support**: Pass custom arguments to your scripts via config or CLI
- 💾 **Memory Monitoring**: Monitor GPU memory usage and thresholds
- 🔒 **Lock Management**: Prevent duplicate processing of chunks

## 🤖 For LLMs: Working with OctoRun

<details>
<summary>
Click to expand
</summary>

When helping users work with OctoRun, here's what you need to know:

### 🎯 Core Concept
OctoRun distributes computational work across multiple GPUs by:
1. **Chunking**: Divides work into numbered chunks (0 to N-1)
2. **GPU Assignment**: Assigns each chunk to an available GPU
3. **Parallel Execution**: Runs chunks simultaneously across GPUs
4. **Monitoring**: Tracks progress and handles failures
5. **Lock-based System**: Uses file locks to prevent duplicate processing - requires all machines to have access to the same shared file system

### 🔧 Required Script Arguments
User scripts **must** accept these three arguments:
- `--gpu_id`: GPU device ID (int) - For CPU-based programs, this serves as `local_rank` and doesn't require GPU environment
- `--chunk_id`: Current chunk number (int) 
- `--total_chunks`: Total number of chunks (int)

### 📝 Typical User Workflow
1. **Script Preparation**: User has a Python script that needs parallelization
2. **Configuration**: Generate config with `octorun save_config --script script.py`
3. **Execution**: Run with `octorun run --config config.json`
4. **Monitoring**: Check logs and GPU usage during execution
5. **Multi-machine**: Ensure shared file system access when running across multiple machines

### 🛠️ Common Tasks You Can Help With
- **Script Modification**: Help adapt existing scripts to work with OctoRun's required arguments
- **Configuration**: Assist with creating and customizing `config.json` files
- **Kwargs Usage**: Help pass custom parameters through config or CLI
- **Error Debugging**: Analyze logs and troubleshoot execution issues
- **Performance Optimization**: Suggest chunk count and GPU allocation strategies
- **Multi-machine Setup**: Help configure shared file system access for distributed execution

### 🚨 Important Notes
- For GPU-based programs: Scripts must handle GPU device setting (e.g., `torch.cuda.set_device(args.gpu_id)`)
- For CPU-based programs: `gpu_id` serves as `local_rank` for parallel processing identification
- Chunk processing should be independent and deterministic (same `chunk_id` should always produce the same result)
- Use `chunk_id` to determine which portion of data to process
- Always validate that required arguments are properly implemented
- All machines must have access to the same shared file system for lock coordination
- OctoRun does not modify `CUDA_VISIBLE_DEVICES` - your script should handle GPU device selection using the provided `gpu_id`

</details>


## 🚀 Installation

### Quick Run via uv (Without Installation) 
```bash
uvx octorun [run, save_config, list_gpus]
```

### Via uv (Installation, Globally) 
```bash
uv tool install octorun
```

### Via uv (Install in Your Own Project)
```bash
uv add octorun
```

### Via pip
```bash
pip install octorun
```

## ⚡ Quick Start

### 1️⃣ Create Configuration
```bash
octorun save_config --script ./your_script.py
# or use the short form:
octorun s --script ./your_script.py
```

### 2️⃣ Run Your Script
```bash
octorun run [--config config.json]
# or use the short form:
octorun r
```

### 3️⃣ Monitor GPU Usage
```bash
octorun list_gpus [--detailed]
# or use the short form:
octorun l -d
```

### 4️⃣ View Logs
```bash
# Monitor session progress
tail -f logs/session_*.log

# Monitor individual chunks
tail -f logs/chunk_*.log
```

## ⚙️ Configuration

### 📄 Basic Configuration

The configuration file (`config.json`) contains the following options:

```json
{
    "script_path": "./your_script.py",
    "gpus": "auto",
    "total_chunks": 128,
    "log_dir": "./logs",
    "chunk_lock_dir": "./logs/locks",
    "monitor_interval": 60,
    "restart_failed": false,
    "max_retries": 3,
    "memory_threshold": 90,
    "kwargs": {
        "batch_size": 32,
        "learning_rate": 0.001
    }
}
```

### 🔧 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `script_path` | Path to your Python script | - |
| `gpus` | GPU configuration ("auto" or list of GPU IDs) | "auto" |
| `total_chunks` | Number of chunks to divide work into | 128 |
| `log_dir` | Directory for log files | "./logs" |
| `chunk_lock_dir` | Directory for chunk lock files | "./logs/locks" |
| `monitor_interval` | Monitoring interval in seconds | 60 |
| `restart_failed` | Whether to restart failed processes | false |
| `max_retries` | Maximum retries for failed chunks | 3 |
| `memory_threshold` | Memory threshold percentage | 90 |
| `kwargs` | Custom arguments to pass to script | {} |

## 🎯 Using Kwargs

OctoRun supports passing additional keyword arguments to your scripts through both the configuration file and command line interface.

### 📋 Configuration File

Add kwargs to your `config.json`:

```json
{
    "script_path": "./train_model.py",
    "gpus": "auto",
    "total_chunks": 128,
    "kwargs": {
        "batch_size": 64,
        "learning_rate": 0.01,
        "model_type": "transformer",
        "epochs": 10,
        "output_dir": "./results"
    }
}
```

### 🖥️ Command Line Interface

Override or add kwargs via command line:

```bash
# Override config kwargs
octorun run --config config.json --kwargs '{"batch_size": 128, "learning_rate": 0.005}'

# Add new kwargs
octorun run --config config.json --kwargs '{"model_type": "bert", "max_length": 512}'
```

### 🎯 Priority

<div align="center">

**CLI kwargs** > **Config file kwargs**

*CLI kwargs override config file kwargs for the same keys while preserving other config kwargs*

</div>

## 🔧 Script Implementation

Your script must accept the required OctoRun arguments plus any custom kwargs:

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # 🔧 Required OctoRun arguments
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--chunk_id', type=int, required=True)
    parser.add_argument('--total_chunks', type=int, required=True)
    
    # 🎯 Your custom arguments (Optional)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./output')
    
    args = parser.parse_args()
    
    # 🎮 Device handling - Set the GPU device
    # This is an exmple when using PyTorch
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"🎮 Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # ✨ Use the arguments in your script
    print(f"🚀 Processing chunk {args.chunk_id}/{args.total_chunks} on GPU {args.gpu_id}")
    print(f"🎯 Training with batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # Your processing logic here
    ...

if __name__ == "__main__":
    main()
```

## 🎮 Commands

### 🚀 `run` (r)

Run your script with the specified configuration:

```bash
octorun run --config config.json [--kwargs '{"key": "value"}']
```

### 💾 `save_config` (s)

Generate a default configuration file:

```bash
octorun save_config [--script ./your_script.py]
```

### 🔍 `list_gpus` (l)

List available GPUs:

```bash
octorun list_gpus [--detailed]
```

## 📚 Examples

### 🤖 Example 1: Machine Learning Training

<details>
<summary>Click to expand</summary>

Config file (`ml_config.json`):
```json
{
    "script_path": "./train_model.py",
    "total_chunks": 64,
    "kwargs": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "model_type": "resnet50",
        "epochs": 100,
        "dataset_path": "/data/imagenet"
    }
}
```

Command:
```bash
octorun run --config ml_config.json --kwargs '{"batch_size": 64, "learning_rate": 0.01}'
```

</details>

### 📊 Example 2: Data Processing

<details>
<summary>Click to expand</summary>

```bash
octorun run --config config.json --kwargs '{"input_dir": "/data/raw", "output_dir": "/data/processed", "compression": "gzip"}'
```

</details>

## 📊 Monitoring and Logging

OctoRun provides comprehensive logging:

| Log Type | Location | Description |
|----------|----------|-------------|
| 📋 **Session logs** | `logs/session_TIMESTAMP.log` | Overall session information |
| 🧩 **Chunk logs** | `logs/chunk_N.log` | Individual chunk processing logs |
| 🔒 **Lock files** | `logs/locks/` | Chunk completion tracking |

### 📊 Real-time Monitoring

```bash
# Monitor session progress
tail -f logs/session_*.log

# Monitor specific chunk
tail -f logs/chunk_42.log

# Monitor GPU usage
watch -n 1 'octorun list_gpus --detailed'
```

## 🛠️ Error Handling

- 🔄 **Automatic retry** mechanism for failed chunks
- 📊 **Configurable** maximum retry attempts
- 💾 **Memory threshold** monitoring
- 📝 **Comprehensive** error logging

<div align="center">

*Robust error handling ensures your jobs complete successfully*

</div>

## 📋 Requirements

- 🐍 **Python** ≥ 3.10
- 🎮 **NVIDIA GPUs** with CUDA support
- 🔧 **nvidia-smi** tool available in PATH

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✨ Make your changes
4. 🧪 Add tests
5. 📤 Submit a pull request

<div align="center">

[![Contributors](https://img.shields.io/badge/contributors-welcome-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

</div>

## 📄 License

This project is licensed under the **MIT License**.

## 👨‍💻 Author

**Haobo Yuan** - [haoboyuan@ucmerced.edu](mailto:haoboyuan@ucmerced.edu)

## 🙏 Acknowledgements

The project is highly relied on AI tools for code generation and documentation, enhancing productivity and code quality.

---

<div align="center">

**Made with ❤️ and 🤖 AI assistance**

*Star ⭐ this repo if you find it useful!*

</div>
