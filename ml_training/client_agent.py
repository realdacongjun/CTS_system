
import time
import argparse
import json
import os
import subprocess
import shutil
import sys
import urllib.request 

def run_command(cmd):
    # 【升级点】使用 perf_counter 获取纳秒级精度
    start = time.perf_counter()
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    end = time.perf_counter()
    return end - start

def download_file(url, save_path):
    # 【升级点】同上，保证下载时间也更精准
    start = time.perf_counter()
    chunk_size = 1024 * 1024 
    try:
        with urllib.request.urlopen(url) as response:
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk: break
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")
    return time.perf_counter() - start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Target file URL")
    parser.add_argument("--method", required=True, help="Compression method")
    args = parser.parse_args()

    filename = args.url.split('/')[-1]
    local_compressed_path = f"/tmp/{filename}"
    output_dir = "/tmp/output_data"
    result_file = "/tmp/result.json"
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    result = {
        "status": "FAILED",
        "download_time": 0, "decomp_time": 0, "total_time": 0,
        "cpu_usage": 0, "mem_usage": 0
    }

    try:
        dl_time = download_file(args.url, local_compressed_path)
        result["download_time"] = dl_time

        cmd = ""
        if args.method.startswith('gzip'):
            cmd = f"tar -xzf {local_compressed_path} -C {output_dir}"
        elif args.method.startswith('brotli'):
            cmd = f"brotli -d {local_compressed_path} -o /tmp/temp.tar && tar -xf /tmp/temp.tar -C {output_dir}"
        elif args.method.startswith('zstd'):
            cmd = f"tar -I zstd -xf {local_compressed_path} -C {output_dir}"
        elif args.method.startswith('lz4'):
            cmd = f"lz4 -d {local_compressed_path} -c | tar -xf - -C {output_dir}"
        else:
            cmd = f"tar -xf {local_compressed_path} -C {output_dir}"

        decomp_time = run_command(cmd)
        
        # 【兜底策略】如果 perf_counter 依然测出 0 (极罕见)，强制给一个极小值
        if decomp_time == 0:
            decomp_time = 0.000001 # 1微秒
            
        result["decomp_time"] = decomp_time
        result["total_time"] = dl_time + decomp_time
        result["status"] = "SUCCESS"

    except Exception as e:
        result["error"] = str(e)
    
    finally:
        if os.path.exists(local_compressed_path): os.remove(local_compressed_path)
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        if os.path.exists("/tmp/temp.tar"): os.remove("/tmp/temp.tar")

    with open(result_file, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
