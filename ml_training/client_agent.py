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
    # 注意：对于管道命令 (a | b)，shell=True 是必须的
    # check=True 会在命令返回非0状态码时抛出异常，这很好
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter()
    return end - start

def download_file(url, save_path):
    start = time.perf_counter()
    chunk_size = 1024 * 1024 * 4 # 稍微加大 buffer 到 4MB 提升大文件写入效率
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
    temp_tar_path = "/tmp/temp.tar"  # 专门定义中间文件路径
    
    # 1. 确保环境绝对干净（防止 Directory not empty 错误）
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 2. 预先清理可能残留的中间文件（关键！防止 Brotli 报错 File exists）
    if os.path.exists(temp_tar_path): os.remove(temp_tar_path)

    result = {
        "status": "FAILED",
        "download_time": 0, "decomp_time": 0, "total_time": 0,
        "cpu_usage": 0, "mem_usage": 0
    }

    try:
        # --- 阶段一：下载 ---
        dl_time = download_file(args.url, local_compressed_path)
        result["download_time"] = dl_time

        # --- 阶段二：解压 (核心修复区) ---
        cmd = ""
        
        if args.method.startswith('gzip'):
            # Gzip 比较老实，tar -xzf 通常没问题
            cmd = f"tar -xzf {local_compressed_path} -C {output_dir}"
            
        elif args.method.startswith('brotli'):
            # 【修复】增加 -f 参数，强制覆盖 /tmp/temp.tar
            # 如果不加 -f，一旦 temp.tar 存在，brotli 就会秒退，导致 decomp_time=0
            cmd = f"brotli -d -f {local_compressed_path} -o {temp_tar_path} && tar -xf {temp_tar_path} -C {output_dir}"
            
        elif args.method.startswith('zstd'):
            # 【修复】弃用 `tar -I zstd`，改用管道模式 `zstd -d -c -f | tar`
            # 理由：1. 确保能传 -f 参数 (Force overwrite)
            #       2. 避免 tar 版本差异导致的参数解析错误
            cmd = f"zstd -d -c -f {local_compressed_path} | tar -xf - -C {output_dir}"
            
        elif args.method.startswith('lz4'):
            # 【修复】增加 -f 参数
            cmd = f"lz4 -d -f {local_compressed_path} -c | tar -xf - -C {output_dir}"
            
        else:
            # 兜底
            cmd = f"tar -xf {local_compressed_path} -C {output_dir}"

        # 执行解压并计时
        decomp_time = run_command(cmd)
        
        # 【兜底策略】防止计时器精度问题（虽然用了 perf_counter 应该不会 0 了）
        if decomp_time < 0.000001:
            decomp_time = 0.000001
            
        result["decomp_time"] = decomp_time
        result["total_time"] = dl_time + decomp_time
        result["status"] = "SUCCESS"

    except Exception as e:
        result["error"] = str(e)
        # 如果是 subprocess 报错，打印出具体的命令，方便排查
        if isinstance(e, subprocess.CalledProcessError):
             print(f"Command failed: {cmd}", file=sys.stderr)
    
    finally:
        # --- 阶段三：强力清理 (防止磁盘爆炸) ---
        # 1. 删除下载的压缩包
        if os.path.exists(local_compressed_path): 
            try: os.remove(local_compressed_path)
            except: pass
            
        # 2. 删除解压后的目录 (数万个小文件很占 inode)
        if os.path.exists(output_dir): 
            try: shutil.rmtree(output_dir)
            except: pass
            
        # 3. 删除中间产物 (Brotli 的遗留物)
        if os.path.exists(temp_tar_path): 
            try: os.remove(temp_tar_path)
            except: pass

    with open(result_file, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()