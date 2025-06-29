#!/usr/bin/env python3
"""
Dummy script for testing ProcessManager.
This script simulates a chunk processing job that can succeed or fail based on arguments.
"""

import argparse
import time
import sys
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Dummy script for testing ProcessManager")
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID to process')
    parser.add_argument('--total_chunks', type=int, required=True, help='Total number of chunks')
    parser.add_argument('--fail_rate', type=float, default=0.0, help='Probability of failure (0.0 to 1.0)')
    parser.add_argument('--sleep_time', type=float, default=0.1, help='Time to sleep (simulating work)')
    parser.add_argument('--fail_chunk_ids', type=str, default='', help='Comma-separated chunk IDs that should fail')
    
    args = parser.parse_args()
    
    print(f"Starting processing: GPU {args.gpu_id}, Chunk {args.chunk_id}/{args.total_chunks}")
    print(f"PID: {os.getpid()}")
    
    # Parse fail_chunk_ids
    fail_chunk_ids = set()
    if args.fail_chunk_ids:
        try:
            fail_chunk_ids = set(map(int, args.fail_chunk_ids.split(',')))
        except ValueError:
            print(f"Invalid fail_chunk_ids: {args.fail_chunk_ids}")
    
    # Simulate some work
    time.sleep(args.sleep_time)
    
    # Determine if this chunk should fail
    should_fail = False
    
    if args.chunk_id in fail_chunk_ids:
        should_fail = True
        print(f"Chunk {args.chunk_id} configured to fail")
    elif args.fail_rate > 0:
        if random.random() < args.fail_rate:
            should_fail = True
            print(f"Chunk {args.chunk_id} randomly failing (fail_rate={args.fail_rate})")
    
    if should_fail:
        print(f"ERROR: Chunk {args.chunk_id} failed!")
        sys.exit(1)
    else:
        print(f"SUCCESS: Chunk {args.chunk_id} completed successfully on GPU {args.gpu_id}")
        sys.exit(0)


if __name__ == "__main__":
    main()
