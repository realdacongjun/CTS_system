import time
import argparse
import json
import socket
import threading
import urllib.request
import urllib.error
import struct
import os
import uuid

# ==========================================
# Cgroup 监控 (保持不变)
# ==========================================

class CgroupMonitor:
    def __init__(self):
        self.cgroup_path = self._detect_cgroup()
        self.start_metrics = self._read()

    def _detect_cgroup(self):
        # 尝试 v2
        if os.path.exists("/sys/fs/cgroup/cpu.stat"):
            return "/sys/fs/cgroup/cpu.stat"
        # 尝试 v1
        if os.path.exists("/sys/fs/cgroup/cpu,cpuacct/cpu.stat"):
            return "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"
        return None

    def _read(self):
        metrics = {"usage_usec": 0, "nr_throttled": 0, "throttled_usec": 0}
        if not self.cgroup_path:
            return metrics
        try:
            with open(self.cgroup_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key, value = parts[0], parts[1]
                        if key in metrics:
                            metrics[key] = int(value)
        except:
            pass
        return metrics

    def diff(self):
        end = self._read()
        return {k: end[k] - self.start_metrics[k] for k in end}

# ==========================================
# 网络工具 (增强健壮性)
# ==========================================

def get_kernel_rtt(sock):
    """
    尝试从内核获取 RTT。
    失败返回 -1。
    """
    try:
        TCP_INFO = 11
        # 不同内核版本 buffer 大小不同，尝试 128/256/512
        for size in [128, 256, 512]:
            try:
                raw = sock.getsockopt(socket.SOL_TCP, TCP_INFO, size)
                # 尝试在 offset 32 (u32 tcpi_rtt)
                if len(raw) >= 36:
                    rtt_us = struct.unpack_from("I", raw, 32)[0]
                    if 0 < rtt_us < 10000000:  # 0-10s 范围合理
                        return rtt_us / 1000.0
            except:
                continue
    except:
        pass
    return -1.0

class DownloaderThread(threading.Thread):
    def __init__(self, url, start_byte, end_byte, buffer_size, results, index):
        super().__init__()
        self.url = url
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.buffer_size = buffer_size
        self.results = results
        self.index = index
        self.bytes_downloaded = 0
        
    def run(self):
        headers = {
            "Range": f"bytes={self.start_byte}-{self.end_byte}",
            "User-Agent": "CTS-Client/1.0"
        }
        
        # 重试机制 (弱网场景必需)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                # 设置超时 (连接, 读取)
                with urllib.request.urlopen(req, timeout=(10, 60)) as resp:
                    # 尝试获取 RTT (仅第一个线程)
                    if self.index == 0 and 'rtt' not in self.results:
                        try:
                            sock = resp.fp.raw._sock
                            if hasattr(sock, 'getsockopt'):
                                rtt = get_kernel_rtt(sock)
                                if rtt > 0:
                                    self.results['rtt'] = rtt
                        except:
                            pass
                    
                    # 读取数据
                    while True:
                        chunk = resp.read(self.buffer_size)
                        if not chunk:
                            break
                        self.bytes_downloaded += len(chunk)
                        
                # 成功退出重试
                return
                
            except (urllib.error.URLError, socket.timeout) as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # 指数退避
                else:
                    # 记录错误但继续 (部分失败处理)
                    self.results[f'error_t{self.index}'] = str(e)

def run_experiment(url, threads, file_size_mb, buffer_mb):
    """
    执行下载实验
    
    Returns:
        dict: 包含 throughput, rtt, cpu_stats 的字典
    """
    total_bytes = int(file_size_mb * 1024 * 1024)
    buffer_size = max(4096, int(buffer_mb * 1024 * 1024))
    
    # 防缓存
    if '?' in url:
        url = f"{url}&_t={uuid.uuid4().hex[:8]}"
    else:
        url = f"{url}?_t={uuid.uuid4().hex[:8]}"
    
    # 启动监控
    monitor = CgroupMonitor()
    
    # 计算每个线程的 Range
    chunk_size = total_bytes // threads
    results = {}
    thread_list = []
    
    start_time = time.time()
    
    for i in range(threads):
        s = i * chunk_size
        e = s + chunk_size - 1 if i < threads - 1 else total_bytes - 1
        t = DownloaderThread(url, s, e, buffer_size, results, i)
        t.start()
        thread_list.append(t)
    
    for t in thread_list:
        t.join()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 统计
    actual_bytes = sum([t.bytes_downloaded for t in thread_list])
    throughput_mbps = (actual_bytes * 8) / (duration * 1_000_000)
    
    # CPU 统计
    cgroup_diff = monitor.diff()
    cpu_seconds = cgroup_diff["usage_usec"] / 1_000_000
    cpu_cores = cpu_seconds / duration if duration > 0 else 0
    
    # 如果没有内核 RTT，用应用层估算 (TTFB - 处理时间)
    rtt = results.get('rtt', -1.0)
    
    output = {
        "duration": round(duration, 3),
        "throughput_mbps": round(throughput_mbps, 2),
        "bytes_downloaded": actual_bytes,
        "cpu_cores_used": round(cpu_cores, 2),
        "cpu_throttle_ratio": round(cgroup_diff["throttled_usec"] / 1_000_000 / duration, 4) if duration > 0 else 0,
        "nr_throttled": cgroup_diff["nr_throttled"],
        "rtt_ms": round(rtt, 2) if rtt > 0 else -1.0,
        "threads": threads,
        "buffer_mb": buffer_mb
    }
    
    # 关键：单行 JSON 输出给 Orchestrator 解析
    print(json.dumps(output))
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CTS Pareto Client')
    parser.add_argument("--url", required=True, help='Target URL (HTTP)')
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--size", type=int, default=100, help='File size in MB')
    parser.add_argument("--buffer", type=float, default=1.0, help='Buffer size in MB')
    args = parser.parse_args()

    try:
        run_experiment(args.url, args.threads, args.size, args.buffer)
    except Exception as e:
        # 错误也以 JSON 输出，方便 Orchestrator 解析
        print(json.dumps({"error": str(e), "throughput_mbps": 0, "rtt_ms": -1}))
        exit(1)