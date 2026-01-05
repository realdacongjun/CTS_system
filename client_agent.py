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

def get_process_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "cpu_times": process.cpu_times(),
        "memory_rss_bytes": mem_info.rss,
    }

def decompress_data(data: bytes, method: str):
    start_time = time.perf_counter()
    try:
        if method == 'gzip':
            decompressed = gzip.decompress(data)
        elif method == 'zstd':
            if not zstd: raise RuntimeError("zstandard library not installed")
            decompressed = zstd.decompress(data)
        elif method == 'lz4':
            if not lz4: raise RuntimeError("lz4 library not installed")
            decompressed = lz4.decompress(data)
        elif method == 'brotli':
            if not brotli: raise RuntimeError("brotli library not installed")
            decompressed = brotli.decompress(data)
        elif method == 'uncompressed':
            decompressed = data
        else:
            raise ValueError(f"Unsupported decompression method: {method}")
    except Exception as e:
        return None, 0
        
    end_time = time.perf_counter()
    return decompressed, end_time - start_time

def main():
    parser = argparse.ArgumentParser()
    # 适配 run_matrix.py 的调用方式 (它是直接传文件名，没有 --file 前缀)
    parser.add_argument("image_layer_path", type=str)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    # 读取文件
    try:
        with open(args.image_layer_path, 'rb') as f:
            compressed_data = f.read()
    except Exception as e:
        # 如果读取失败，返回空JSON让脚本报错
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # 测量
    res_before = get_process_resources()
    decompressed_data, time_taken = decompress_data(compressed_data, args.method)
    res_after = get_process_resources()

    # 计算 CPU (简单计算 user time 增量)
    cpu_delta = res_after["cpu_times"].user - res_before["cpu_times"].user
    
    # === 关键修改：统一输出格式 ===
    # 必须把 Key 改成 run_matrix.py 里的名字
    result = {
        "status": "SUCCESS" if decompressed_data else "FAILED",
        "decomp_time": time_taken,              # 改名: decompression_time -> decomp_time
        "download_time": 0.0,                   # 补充字段
        "total_time": time_taken,               # 补充字段
        "cpu_usage": cpu_delta,                 # 改名: cpu_user_time_delta -> cpu_usage
        "mem_usage": res_after["memory_rss_bytes"], # 改名
        "compressed_size": len(compressed_data),    # 改名
        "original_size": len(decompressed_data) if decompressed_data else 0,
        "bandwidth_measured": 0.0
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()