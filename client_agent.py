#!/usr/bin/env python3
"""
CTS Experiment Client Agent

This script runs inside a controlled container environment. It is responsible for
the core task of an experiment run: decompressing a given image layer
with a specific algorithm and measuring its performance.
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
try:
    import brotli
except ImportError:
    brotli = None

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
    """
    decompressed = None
    
    # --- Decompression Logic ---
    start_time = time.perf_counter()
    try:
        if method == 'gzip':
            decompressed = gzip.decompress(data)
        elif method == 'zstd':
            if not zstd:
                raise RuntimeError("zstandard library not installed")
            decompressed = zstd.decompress(data)
        elif method == 'lz4':
            if not lz4:
                raise RuntimeError("lz4 library not installed")
            decompressed = lz4.decompress(data)
        elif method == 'brotli':
            if not brotli:
                raise RuntimeError("brotli library not installed")
            decompressed = brotli.decompress(data)
        elif method == 'uncompressed':
            # For baseline comparison, we just copy the data
            decompressed = data
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
        # === 关键修改：加入了 'brotli' ===
        choices=['gzip', 'zstd', 'lz4', 'brotli', 'uncompressed'],
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
    res_before = get_process_resources()
    
    decompressed_data, time_taken = decompress_data(compressed_data, args.method)

    res_after = get_process_resources()
    result["decompression_time"] = time_taken

    if decompressed_data is None:
        result["error"] = f"Decompression failed for method '{args.method}'"
    else:
        result["status"] = "SUCCESS"
        result["uncompressed_size_bytes"] = len(decompressed_data)

    # --- 3. Calculate Resource Usage Deltas ---
    cpu_delta = res_after["cpu_times"].user - res_before["cpu_times"].user
    result["cpu_user_time_delta"] = cpu_delta if cpu_delta > 0 else 0.0
    result["memory_rss_bytes_peak"] = res_after["memory_rss_bytes"]

    # --- 4. Output JSON Result ---
    print(json.dumps(result))


if __name__ == "__main__":
    # Ensure all required libraries are available
    if zstd is None:
        sys.stderr.write("Warning: 'zstandard' library not found. 'zstd' method will fail.\n")
    if lz4 is None:
        sys.stderr.write("Warning: 'lz4' library not found. 'lz4' method will fail.\n")
    if brotli is None:
        sys.stderr.write("Warning: 'brotli' library not found. 'brotli' method will fail.\n")
    
    main()