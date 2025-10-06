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

## 🚀 Installation

You can install OctoRun using `pip` or `uv`.

### Via pip
```bash
pip install octorun
```

### Via uv
```bash
# Install globally
uv tool install octorun

# Install in your project
uv add octorun
```

## ⚡ Quick Start

1.  **Create Configuration**:
    ```bash
    octorun save_config --script ./your_script.py
    ```

2.  **Run Your Script**:
    ```bash
    octorun run
    ```

3.  **Monitor GPUs**:
    ```bash
    octorun list_gpus -d
    ```

## 🎮 Commands

### `run` (r)

Run your script with the specified configuration.

```bash
octorun run --config config.json [--kwargs '{"key": "value"}']
```

### `save_config` (s)

Generate a default configuration file.

```bash
octorun save_config --script ./your_script.py
```

### `list_gpus` (l)

List available GPUs and their current usage.

```bash
octorun list_gpus [--detailed]
```

The `detailed` flag provides a more comprehensive view of GPU stats, including memory usage, temperature, and running processes.

### `benchmark` (b)

Run a benchmark to determine the optimal number of parallel processes for your GPUs.

```bash
octorun benchmark
```

This command runs a series of tests to help you configure the `gpus` parameter in your `config.json` for the best performance.

## ⚙️ Configuration

OctoRun uses a `config.json` file for configuration. You can generate a default one with `octorun save_config`.

| Option             | Description                                  | Default        |
| ------------------ | -------------------------------------------- | -------------- |
| `script_path`      | Path to your Python script                   | -              |
| `gpus`             | "auto" or list of GPU IDs                    | "auto"         |
| `total_chunks`     | Number of chunks to divide work into         | 128            |
| `log_dir`          | Directory for log files                      | "./logs"       |
| `chunk_lock_dir`   | Directory for chunk lock files               | "./logs/locks" |
| `monitor_interval` | Monitoring interval in seconds               | 60             |
| `restart_failed`   | Whether to restart failed processes          | false          |
| `max_retries`      | Maximum retries for failed chunks            | 3              |
| `memory_threshold` | Memory threshold percentage                  | 90             |
| `kwargs`           | Custom arguments to pass to your script      | {}             |

## 🎯 Using Kwargs

You can pass custom arguments to your script via the `kwargs` object in your `config.json` or directly through the CLI.

**CLI kwargs will override config file kwargs.**

```bash
octorun run --kwargs '{"batch_size": 128, "learning_rate": 0.005}'
```

## 🔧 Script Implementation

Your script must accept the following arguments:

-   `--gpu_id`: GPU device ID (int)
-   `--chunk_id`: Current chunk number (int)
-   `--total_chunks`: Total number of chunks (int)

Here is an example of how to structure your script:

```python
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    
    # Required OctoRun arguments
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--chunk_id', type=int, required=True)
    parser.add_argument('--total_chunks', type=int, required=True)
    
    # Your custom arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./output')
    
    args = parser.parse_args()
    
    # Set the GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}")
    
    print(f"Processing chunk {args.chunk_id}/{args.total_chunks}")
    
    # Your logic here

if __name__ == "__main__":
    main()
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## 📄 License

This project is licensed under the **MIT License**.