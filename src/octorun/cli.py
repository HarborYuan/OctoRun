"""Command-line interface for octorun."""

import argparse
import json
import os
import sys
import subprocess
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
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index,name,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        line = result.stdout.strip()
        if line:
            parts = line.split(', ')
            return {
                'id': int(parts[0]),
                'name': parts[1],
                'memory_free': int(parts[2]),
                'memory_total': int(parts[3]),
                'utilization': int(parts[4])
            }
    
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
        help="Run a task or script",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    save_config_parser = subparsers.add_parser(
        "save_config",
        help="Save default configuration to ./config.json",
    )
    save_config_parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Script to run",
    )
    
    # List GPUs command
    list_gpus_parser = subparsers.add_parser(
        "list_gpus",
        help="List available GPUs",
    )
    list_gpus_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed GPU information",
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the run command."""
    # Here you would implement the actual run logic
    print(f"Running {args.config}...")
    config = json.load(open(args.config, "r"))
    pm = ProcessManager(config)
    gpus = config.get("gpu_ids", "auto")
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
    
    total_chunks = config.get("total_chunks", 128)
    pm.run(gpu_ids=gpus, total_chunks=total_chunks, kwargs={})
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
                print(f"  GPU {gpu_id}: {gpu_info['name']}")
                print(f"    Memory: {gpu_info['memory_free']}MB free / {gpu_info['memory_total']}MB total")
                print(f"    Utilization: {gpu_info['utilization']}%")
            else:
                print(f"  GPU {gpu_id}: Could not get detailed info")
    else:
        print(f"  GPU IDs: {', '.join(map(str, available_gpus))}")
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.verbose:
        print(f"octorun v0.1.0")
        print(f"Command: {args.command}")
    
    # Dispatch to command handler
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "save_config":
        return cmd_save_config(args)
    elif args.command == "list_gpus":
        return cmd_list_gpus(args)
    else:
        parser.print_help()
        return 1


def cli_main() -> None:
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    cli_main()
