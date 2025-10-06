"""Command-line interface for octorun."""

import argparse
import json
import os
import sys
import subprocess
import pwd
from typing import Dict, List, Optional, Dict

from . import __version__ as version

from .runner import ProcessManager


def get_available_gpus() -> List[int]:
    """
    Get a list of available GPU IDs using nvidia-smi.
    
    Returns:
        List[int]: List of available GPU IDs. Returns empty list if no GPUs found or nvidia-smi not available.
    """
    try:
        # Run nvidia-smi to get GPU information
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_ids = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                gpu_id = int(parts[0])
                gpu_name = parts[1]
                free_memory = int(parts[2])
                total_memory = int(parts[3])
                
                # Only include GPUs with some free memory (more than 100MB)
                if free_memory > 100:
                    gpu_ids.append(gpu_id)
        
        return gpu_ids
    
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        # nvidia-smi not found or error occurred
        print(f"Warning: Could not detect GPUs: {e}")
        return []


def get_gpu_info(gpu_id: int) -> Optional[Dict]:
    """
    Get detailed information about a specific GPU.
    
    Args:
        gpu_id (int): The GPU ID to query
        
    Returns:
        Optional[Dict]: GPU information dictionary or None if error
    """
    try:
        # Get basic GPU information
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index,name,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        line = result.stdout.strip()
        if not line:
            return None
            
        parts = line.split(', ')
        gpu_info = {
            'id': int(parts[0]),
            'name': parts[1],
            'memory_free': int(parts[2]),
            'memory_total': int(parts[3]),
            'utilization_gpu': int(parts[4]),
            'utilization_memory': int(parts[5]),
            'temperature': int(parts[6]),
            'power_draw': float(parts[7]),
            'power_limit': float(parts[8])
        }
        
        # Get running processes on this GPU
        try:
            process_result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            processes = []
            for process_line in process_result.stdout.strip().split('\n'):
                if process_line.strip():
                    process_parts = process_line.split(', ')
                    if len(process_parts) >= 3:
                        pid = int(process_parts[0])
                        username = get_process_username(pid)
                        processes.append({
                            'pid': pid,
                            'process_name': process_parts[1],
                            'used_memory': int(process_parts[2]),
                            'username': username
                        })
            
            gpu_info['processes'] = processes
            
        except (subprocess.CalledProcessError, ValueError):
            gpu_info['processes'] = []
        
        return gpu_info
    
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not get GPU {gpu_id} info: {e}")
        return None


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="octorun",
        description="A command-line tool for octorun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {}".format(version),
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )
    
    # Run command (example for running something)
    run_parser = subparsers.add_parser(
        "run",
        aliases=["r"],
        description="Run a task or script",
        help="Run a task or script",
    )
    run_parser.set_defaults(command="run")
    run_parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    run_parser.add_argument(
        "--kwargs",
        type=str,
        help="Additional keyword arguments as JSON string (e.g., '{\"batch_size\": 32, \"lr\": 0.001}')",
    )
    save_config_parser = subparsers.add_parser(
        "save_config",
        aliases=["s"],
        help="Save default configuration to ./config.json",
    )
    save_config_parser.set_defaults(command="save_config")
    save_config_parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Script to run",
    )
    
    # List GPUs command
    list_gpus_parser = subparsers.add_parser(
        "list_gpus",
        aliases=["l"],
        description="List available GPUs",
        help="List available GPUs",
    )
    list_gpus_parser.set_defaults(command="list_gpus")
    list_gpus_parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed GPU information",
    )
    
    # GPU Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        aliases=["b"],
        description="Continuously test GPU performance (TFLOPs and communication)",
        help="Continuously test GPU performance (TFLOPs and communication)",
    )
    benchmark_parser.set_defaults(command="benchmark")
    benchmark_parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        default="auto",
        help="GPU IDs to test (comma-separated, e.g., '0,1,2' or 'auto' for all available)",
    )
    benchmark_parser.add_argument(
        "--test-duration",
        "-t",
        type=float,
        default=10.,
        help="Duration of each test in seconds (default: 10.0)",
    )
    benchmark_parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=15.,
        help="Interval between tests in seconds (default: 30.0)",
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the run command."""
    # Here you would implement the actual run logic
    print(f"Running {args.config}...")
    config = json.load(open(args.config, "r"))

    # parse configs
    gpus = config.pop("gpus", "auto")
    total_chunks = config.pop("total_chunks", 128)
    # Handle kwargs from config file and CLI
    kwargs = {}
    # First, load kwargs from config file
    if "kwargs" in config:
        kwargs.update(config["kwargs"])
        print(f"Loaded kwargs from config: {config['kwargs']}")
        config.pop("kwargs", None)  # Remove from config to avoid duplication

    # Then, override with CLI kwargs if provided
    if args.kwargs:
        try:
            cli_kwargs = json.loads(args.kwargs)
            kwargs.update(cli_kwargs)
            print(f"Added/overrode kwargs from CLI: {cli_kwargs}")
        except json.JSONDecodeError as e:
            print(f"Error parsing CLI kwargs JSON: {e}")
            return 1
    if kwargs:
        print(f"Final kwargs: {kwargs}")

    pm = ProcessManager(config)
    if gpus == "auto":
        print("Using automatic GPU detection.")
        available_gpus = get_available_gpus()
        if available_gpus:
            gpus = available_gpus
            print(f"Detected GPUs: {', '.join(map(str, gpus))}")
        else:
            print("No suitable GPUs found, falling back to CPU.")
            raise ValueError("No available GPUs found.")
    elif isinstance(gpus, list):
        if len(gpus) == 0:
            print("No suitable GPUs found, falling back to CPU.")
            raise ValueError("No available GPUs found.")
        print(f"Using specified GPUs: {', '.join(map(str, gpus))}")
    else:
        raise ValueError(f"Invalid GPU configuration: {gpus}")
    
    pm.run(gpu_ids=gpus, total_chunks=total_chunks, kwargs=kwargs)
    return 0


def cmd_save_config(args: argparse.Namespace) -> int:
    """Handle the save_config command."""
    if args.verbose:
        print("Saving default configuration to ./config.json")
    
    # get default configuration from src
    default_config_path = os.path.join(os.path.dirname(__file__), "default_config.json")
    default_config = json.load(open(default_config_path, "r"))

    # update script path if provided
    if args.script:
        default_config["script_path"] = args.script

    with open("./config.json", "w") as config_file:
        json.dump(default_config, config_file, indent=4)
    print("Default configuration saved.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the benchmark command."""
    from .gpu_benchmark import run_gpu_benchmark
    
    # Parse GPU IDs
    if args.gpus.lower() == "auto":
        gpu_ids = None  # Auto-detect
    else:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        except ValueError:
            print(f"❌ Invalid GPU IDs format: {args.gpus}")
            print("📋 Use comma-separated integers (e.g., '0,1,2') or 'auto'")
            return 1
    
    if args.verbose:
        print(f"🚀 Starting GPU benchmark...")
        if gpu_ids:
            print(f"🎯 Target GPUs: {gpu_ids}")
        else:
            print(f"🔍 Auto-detecting GPUs...")
        print(f"⏱️  Test duration: {args.test_duration}s")
        print(f"📊 Test interval: {args.interval}s")
    
    try:
        run_gpu_benchmark(
            gpu_ids=gpu_ids,
            test_duration=args.test_duration,
            test_interval=args.interval,
        )
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


def cmd_list_gpus(args: argparse.Namespace) -> int:
    """Handle the list_gpus command."""
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        print("No GPUs found or nvidia-smi not available.")
        return 0
    
    print(f"Found {len(available_gpus)} available GPU(s):")
    
    if args.detailed:
        for gpu_id in available_gpus:
            gpu_info = get_gpu_info(gpu_id)
            if gpu_info:
                print(f"\n  🎮 GPU {gpu_id}: {gpu_info['name']}")
                print(f"    💾 Memory: {gpu_info['memory_free']}MB free / {gpu_info['memory_total']}MB total ({(gpu_info['memory_total'] - gpu_info['memory_free']) / gpu_info['memory_total'] * 100:.1f}% used)")
                print(f"    ⚡ GPU Utilization: {gpu_info['utilization_gpu']}%")
                print(f"    🧠 Memory Utilization: {gpu_info['utilization_memory']}%")
                print(f"    🌡️  Temperature: {gpu_info['temperature']}°C")
                print(f"    🔋 Power: {gpu_info['power_draw']:.1f}W / {gpu_info['power_limit']:.1f}W ({gpu_info['power_draw'] / gpu_info['power_limit'] * 100:.1f}%)")
                
                if gpu_info['processes']:
                    print(f"    👥 Running Processes ({len(gpu_info['processes'])}):")
                    for process in gpu_info['processes']:
                        username_str = f" (user: {process['username']})" if process['username'] else ""
                        print(f"      • PID {process['pid']}: {process['process_name']} ({process['used_memory']}MB){username_str}")
                else:
                    print(f"    👥 Running Processes: None")
            else:
                print(f"  GPU {gpu_id}: Could not get detailed info")
    else:
        print(f"  GPU IDs: {', '.join(map(str, available_gpus))}")
    
    return 0


def get_process_username(pid: int) -> Optional[str]:
    """
    Get the username of a process given its PID.
    
    Args:
        pid (int): Process ID
        
    Returns:
        Optional[str]: Username or None if not found
    """
    try:
        # Get process owner UID from /proc/PID/status
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])  # Real UID is the second field
                    return pwd.getpwuid(uid).pw_name
    except (FileNotFoundError, PermissionError, KeyError, IndexError, ValueError):
        pass
    return None


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.verbose:
        print(f"octorun {version}")
        print(f"Command: {args.command}")
    
    # Dispatch to command handler
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "save_config":
        return cmd_save_config(args)
    elif args.command == "list_gpus":
        return cmd_list_gpus(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


def cli_main() -> None:
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    cli_main()
