---
# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the OctoRun project.

## Core Concept

OctoRun is a Python-based command-line tool for distributing and running Python scripts across multiple GPUs in parallel. It simplifies the process of managing computational work by:

1.  **Chunking**: Dividing a large task into smaller, manageable chunks.
2.  **GPU Assignment**: Assigning each chunk to an available GPU.
3.  **Parallel Execution**: Running chunks simultaneously across multiple GPUs.
4.  **Monitoring**: Tracking the progress of each chunk and handling failures.
5.  **Lock-based System**: Using a file-based locking mechanism to prevent duplicate processing of chunks, which requires a shared file system for all participating machines.

## Required Script Arguments

For a user's script to be compatible with OctoRun, it **must** accept the following three command-line arguments:

-   `--gpu_id`: The ID of the GPU device to use (integer). For CPU-based tasks, this can serve as a `local_rank` for process identification.
-   `--chunk_id`: The ID of the current chunk being processed (integer).
-   `--total_chunks`: The total number of chunks the work has been divided into (integer).

## Typical User Workflow

1.  **Script Preparation**: The user prepares a Python script that can be parallelized.
2.  **Configuration**: The user generates a configuration file using `octorun save_config --script your_script.py`.
3.  **Execution**: The user runs the script with `octorun run --config config.json`.
4.  **Monitoring**: The user monitors the execution by checking the log files and GPU usage.
5.  **Multi-machine**: For distributed execution across multiple machines, the user must ensure that all machines have access to the same shared file system.

## Common Tasks for the Assistant

-   **Script Modification**: Help users adapt their existing scripts to accept the required OctoRun arguments (`--gpu_id`, `--chunk_id`, `--total_chunks`).
-   **Configuration**: Assist users in creating and customizing their `config.json` files.
-   **Kwargs Usage**: Help users pass custom keyword arguments to their scripts through the configuration file or the command line.
-   **Error Debugging**: Analyze log files to help users troubleshoot execution issues.
-   **Performance Optimization**: Provide suggestions on how to optimize chunk count and GPU allocation for better performance.
-   **Multi-machine Setup**: Advise users on how to set up a shared file system for distributed execution.

## Important Notes

-   For GPU-based programs, the user's script must handle GPU device selection (e.g., using `torch.cuda.set_device(args.gpu_id)`).
-   For CPU-based programs, the `gpu_id` argument can be used as a `local_rank` to identify parallel processes.
-   Chunk processing should be independent and deterministic, meaning that the same `chunk_id` should always produce the same result.
-   The user's script should use the `chunk_id` to determine which portion of the data to process.
-   Always verify that the required arguments are correctly implemented in the user's script.
-   OctoRun does not manage the `CUDA_VISIBLE_DEVICES` environment variable; the user's script is responsible for selecting the correct GPU device using the provided `gpu_id`.
---
