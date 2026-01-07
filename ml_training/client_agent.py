import time
import argparse
import json
import os
import subprocess
import shutil
import sys
# 使用标准库 urllib，避免容器里没装 requests 的尴尬
import urllib.request 

def run_command(cmd):
    """运行 shell 命令并返回耗时"""
    start = time.time()
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    return time.time() - start

def download_file(url, save_path):
    """从 Server 下载文件，返回下载耗时"""
    start = time.time()
    # 缓冲区大小设置为 1MB，模拟真实的大文件传输
    chunk_size = 1024 * 1024 
    
    try:
        with urllib.request.urlopen(url) as response:
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk: break
                    f.write(chunk)
    except Exception as e:
        # 如果下载失败，抛出异常给主程序捕获
        raise RuntimeError(f"Download failed: {str(e)}")
        
    return time.time() - start

def main():
    parser = argparse.ArgumentParser()
    # 这里接收的是 URL 而不是本地路径了
    parser.add_argument("url", help="Target file URL (e.g., http://server:8000/file.tar.gz)")
    parser.add_argument("--method", required=True, help="Compression method")
    args = parser.parse_args()

    # 1. 准备路径
    filename = args.url.split('/')[-1]
    local_compressed_path = f"/tmp/{filename}"
    output_dir = "/tmp/output_data"
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    result = {
        "status": "FAILED",
        "download_time": 0,
        "decomp_time": 0,
        "total_time": 0,
        "cpu_usage": 0, # 这里先简化，回头可以用 psutil 加回来
        "mem_usage": 0
    }

    try:
        # === 阶段 1: 真实网络下载 (传输层) ===
        # print(f"⬇️ Downloading from {args.url}...")
        dl_time = download_file(args.url, local_compressed_path)
        result["download_time"] = dl_time

        # === 阶段 2: 解压 (计算层) ===
        # print(f"📦 Decompressing {args.method}...")
        cmd = ""
        if args.method == 'gzip':
            cmd = f"tar -xzf {local_compressed_path} -C {output_dir}"
        elif args.method == 'brotli':
            # brotli 需要先解压成 tar 再解包，或者管道
            cmd = f"brotli -d {local_compressed_path} -o /tmp/temp.tar && tar -xf /tmp/temp.tar -C {output_dir}"
        elif args.method == 'zstd':
            cmd = f"tar -I zstd -xf {local_compressed_path} -C {output_dir}"
        elif 'lz4' in args.method:
            cmd = f"lz4 -d {local_compressed_path} -c | tar -xf - -C {output_dir}"
        else:
            # 默认尝试直接 tar
            cmd = f"tar -xf {local_compressed_path} -C {output_dir}"

        decomp_time = run_command(cmd)
        result["decomp_time"] = decomp_time
        result["total_time"] = dl_time + decomp_time
        result["status"] = "SUCCESS"

    except Exception as e:
        result["error"] = str(e)
        # print(f"Error: {e}", file=sys.stderr)
    
    finally:
        # 清理垃圾，防止容器炸硬盘
        if os.path.exists(local_compressed_path): os.remove(local_compressed_path)
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        # 这里的 /tmp/temp.tar 是 brotli 可能产生的中间文件
        if os.path.exists("/tmp/temp.tar"): os.remove("/tmp/temp.tar")

    # 输出 JSON 供宿主机捕获
    print(json.dumps(result))

if __name__ == "__main__":
    main()