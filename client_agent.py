#!/usr/bin/env python3
"""
CTS Experiment Client Agent

This script runs inside a controlled container environment. It is responsible for
the core task of an experiment run: decompressing a given image layer
with a specific algorithm and measuring its performance.

It is invoked by the experiment orchestrator with specific parameters.

Workflow:
1. Parses command-line arguments: path to the compressed layer and the algorithm to use.
2. Reads the entire compressed file into memory to simulate network download.
3. Records baseline resource usage (CPU, memory).
4. Decompresses the data, measuring the wall-clock time with high precision.
5. Records final resource usage.
6. Calculates performance metrics (e.g., decompression time, CPU usage delta).
7. Prints a single line of JSON to standard output with the results. This is
   the communication channel back to the orchestrator.
"""
import argparse
import gzip
import json
import os
import sys
import time
import psutil

# Conditional imports for compression libraries
try:
    import zstandard as zstd
except ImportError:
    zstd = None
try:
    import lz4.frame as lz4
except ImportError:
    lz4 = None

# --- Performance Measurement ---

def get_process_resources():
    """Gets the current process's CPU and memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "cpu_times": process.cpu_times(),
        "memory_rss_bytes": mem_info.rss,
    }

def decompress_data(data: bytes, method: str) -> (bytes, float):
    """
    Decompresses data using the specified method and measures the time taken.

    Args:
        data: The compressed byte string.
        method: The decompression algorithm ('gzip', 'zstd', 'lz4').

    Returns:
        A tuple of (decompressed_data, time_taken_seconds).
        Returns (None, -1.0) on failure.
    """
    decompressed = None
    
    # --- Decompression Logic ---
    # The core of the performance measurement.
    # We use time.perf_counter() for high-precision wall-clock time.
    start_time = time.perf_counter()
    try:
        if method == 'gzip':
            decompressed = gzip.decompress(data)
        elif method == 'zstd':
            if not zstd:
                raise RuntimeError("zstandard library not installed")
            # zstd.decompress() is highly optimized
            decompressed = zstd.decompress(data)
        elif method == 'lz4':
            if not lz4:
                raise RuntimeError("lz4 library not installed")
            # lz4.decompress() is also highly optimized
            decompressed = lz4.decompress(data)
        else:
            raise ValueError(f"Unsupported decompression method: {method}")
    except Exception:
        # If decompression fails, we return an error state.
        end_time = time.perf_counter()
        return None, end_time - start_time
        
    end_time = time.perf_counter()

    time_taken = end_time - start_time
    return decompressed, time_taken


# --- Main Execution ---

def main():
    """Main function to run the client agent."""
    parser = argparse.ArgumentParser(
        description="CTS Experiment Client Agent to measure decompression performance."
    )
    parser.add_argument(
        "image_layer_path",
        type=str,
        help="Path to the compressed image layer file inside the container.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=['gzip', 'zstd', 'lz4', 'uncompressed'],
        help="The decompression algorithm to use."
    )

    args = parser.parse_args()

    # Initialize result object
    result = {
        "status": "ABNORMAL",
        "error": None,
        "decompression_time": 0.0,
        "compressed_size_bytes": 0,
        "uncompressed_size_bytes": 0,
        "cpu_user_time_delta": 0.0,
        "memory_rss_bytes_peak": 0,
    }

    # --- 1. Read Data ---
    try:
        with open(args.image_layer_path, 'rb') as f:
            compressed_data = f.read()
        result["compressed_size_bytes"] = len(compressed_data)
    except FileNotFoundError:
        result["error"] = f"File not found: {args.image_layer_path}"
        print(json.dumps(result))
        sys.exit(1)
    except Exception as e:
        result["error"] = f"Failed to read file: {str(e)}"
        print(json.dumps(result))
        sys.exit(1)

    # --- 2. Decompress and Measure ---
    # Record resources before the heavy work
    res_before = get_process_resources()
    
    decompressed_data, time_taken = decompress_data(compressed_data, args.method)

    # Record resources after the heavy work
    res_after = get_process_resources()
    result["decompression_time"] = time_taken

    if decompressed_data is None:
        result["error"] = f"Decompression failed for method '{args.method}'"
    else:
        result["status"] = "SUCCESS"
        result["uncompressed_size_bytes"] = len(decompressed_data)

    # --- 3. Calculate Resource Usage Deltas ---
    # CPU time is more reliable than trying to measure % CPU usage over a short period.
    cpu_delta = res_after["cpu_times"].user - res_before["cpu_times"].user
    result["cpu_user_time_delta"] = cpu_delta if cpu_delta > 0 else 0.0

    # For memory, we just record the peak resident set size (RSS).
    result["memory_rss_bytes_peak"] = res_after["memory_rss_bytes"]

    # --- 4. Output JSON Result ---
    # This single line is parsed by the orchestrator.
    print(json.dumps(result))


if __name__ == "__main__":
    # Ensure all required libraries are available
    if zstd is None:
        sys.stderr.write("Warning: 'zstandard' library not found. 'zstd' method will fail.\n")
    if lz4 is None:
        sys.stderr.write("Warning: 'lz4' library not found. 'lz4' method will fail.\n")
    
    main()