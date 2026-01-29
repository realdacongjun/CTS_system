# client_task.py
import time
import threading
import urllib.request
import argparse
import json
import sys
import subprocess

# 自动安装依赖
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

def download_range(url, start, end, buffer_size):
    """
    模拟分片下载
    :param buffer_size: 每次 IO 读取的字节数 (影响 CPU 上下文切换频率)
    """
    headers = {"Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            while True:
                # 这里显式控制读取粒度，模拟不同的 Chunk Size 策略
                chunk = response.read(buffer_size)
                if not chunk: break
    except Exception:
        pass

def run_task(url, threads, file_size_mb, buffer_mb):
    total_bytes = int(file_size_mb * 1024 * 1024)
    part_size = total_bytes // threads
    # 将 MB 转为 Bytes，最小 1KB，防止设为 0
    buffer_bytes = max(1024, int(buffer_mb * 1024 * 1024))
    
    p = psutil.Process()
    # 简单的 CPU 预热
    p.cpu_percent(interval=None)
    
    time_start = time.time()
    # 记录初始累计 CPU 时间 (比 percent 更准)
    cpu_time_start = p.cpu_times().user + p.cpu_times().system

    workers = []
    for i in range(threads):
        start = i * part_size
        end = total_bytes - 1 if i == threads - 1 else (start + part_size - 1)
        
        t = threading.Thread(target=download_range, args=(url, start, end, buffer_bytes))
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

    duration = time.time() - time_start
    cpu_time_end = p.cpu_times().user + p.cpu_times().system
    
    # 计算平均 CPU 使用率 (逻辑核)
    # CPU% = (Total CPU Time / Wall Clock Time) * 100
    # 注意：如果双核跑满，这里可能算出 200%。为了归一化到 0-100%，除以核心数可能更好，
    # 但为了体现总 Cost，保留总占用率更有物理意义。
    cpu_percent_total = ((cpu_time_end - cpu_time_start) / duration) * 100
    
    throughput_mbps = (total_bytes * 8) / (duration * 1_000_000)

    result = {
        "duration": duration,
        "cpu_avg": cpu_percent_total,
        "throughput_mbps": throughput_mbps
    }
    print(json.dumps(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--size", type=int, default=50) 
    parser.add_argument("--buffer", type=float, default=1.0) # 新增：缓冲区大小(MB)
    args = parser.parse_args()

    run_task(args.url, args.threads, args.size, args.buffer)