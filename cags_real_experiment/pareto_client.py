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
import sys

# ==========================================
# Cgroup 监控
# ==========================================

class CgroupMonitor:
    def __init__(self):
        self.cgroup_path = self._detect_cgroup()
        self.start_metrics = self._read()

    def _detect_cgroup(self):
        if os.path.exists("/sys/fs/cgroup/cpu.stat"):
            return "/sys/fs/cgroup/cpu.stat"
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
# 网络工具
# ==========================================

def get_kernel_rtt(sock):
    """尝试从内核获取 RTT，失败返回 -1"""
    try:
        TCP_INFO = getattr(socket, 'TCP_INFO', 11)
        # 尝试不同 buffer 大小
        for size in [128, 256, 512, 1024]:
            try:
                raw = sock.getsockopt(socket.SOL_TCP, TCP_INFO, size)
                if len(raw) >= 36:
                    # tcpi_rtt 在 offset 28 (u32)
                    rtt_us = struct.unpack_from("I", raw, 28)[0]
                    if 0 < rtt_us < 60000000:  # 0-60s 合理
                        return rtt_us / 1000.0
            except:
                continue
    except Exception as e:
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
        self.error_msg = None
        
    def run(self):
        headers = {
            "Range": f"bytes={self.start_byte}-{self.end_byte}",
            "User-Agent": "CTS-Client/1.0"
        }
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                
                # ✅ 关键修复：timeout 使用单个值而非元组，避免 "got type tuple" 错误
                # 如果必须用分别的连接和读取超时，使用单个保守值
                timeout_val = 60  # 总超时 60 秒
                
                with urllib.request.urlopen(req, timeout=timeout_val) as resp:
                    # 检查状态码：206 (Partial Content) 或 200 (OK，虽然不高效)
                    status = resp.getcode()
                    if status not in (200, 206):
                        raise urllib.error.HTTPError(url, status, f"Unexpected status {status}", resp.headers, None)
                    
                    # 仅在 206 时测试 Range，200 意味着服务器忽略 Range，返回全部内容
                    if status == 200 and self.start_byte > 0:
                        # 服务器不支持 Range，但我们要求的是部分内容，这是一个问题
                        print(f"Warning: Server returned 200 instead of 206 for Range request", file=sys.stderr)
                    
                    # 获取 RTT（仅第一个线程，且仅在成功连接后）
                    if self.index == 0:
                        try:
                            # ✅ 关键修复：安全获取 socket，处理可能的 wrapper
                            sock = None
                            if hasattr(resp, 'fp') and hasattr(resp.fp, '_sock'):
                                raw_sock = resp.fp._sock
                                if hasattr(raw_sock, 'getsockopt'):
                                    sock = raw_sock
                                elif hasattr(raw_sock, '_sock'):  # SSL wrapper
                                    sock = raw_sock._sock
                            
                            if sock:
                                rtt = get_kernel_rtt(sock)
                                if rtt > 0:
                                    self.results['rtt'] = rtt
                        except Exception as e:
                            pass  # RTT 获取失败不重要
                    
                    # 读取数据
                    while True:
                        chunk = resp.read(self.buffer_size)
                        if not chunk:
                            break
                        self.bytes_downloaded += len(chunk)
                
                # 成功完成，退出重试循环
                return
                
            except (urllib.error.URLError, socket.timeout, urllib.error.HTTPError) as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # 指数退避
                else:
                    self.error_msg = last_error
                    print(f"Thread {self.index} failed after {max_retries} attempts: {last_error}", file=sys.stderr)
            except TypeError as e:
                # 捕获 "got type tuple" 或其他类型错误
                last_error = f"TypeError: {str(e)}"
                print(f"Thread {self.index} TypeError: {e}", file=sys.stderr)
                self.error_msg = last_error
                return  # 不重试类型错误
            except Exception as e:
                last_error = f"Unexpected: {type(e).__name__}: {str(e)}"
                print(f"Thread {self.index} unexpected error: {e}", file=sys.stderr)
                self.error_msg = last_error
                return

def run_experiment(url, threads, file_size_mb, buffer_mb):
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
    
    # 等待所有线程完成
    for t in thread_list:
        t.join()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 统计
    actual_bytes = sum([t.bytes_downloaded for t in thread_list])
    
    # 收集错误信息
    errors = [t.error_msg for t in thread_list if t.error_msg]
    if errors:
        results['errors'] = errors[:3]  # 只保留前3个错误
    
    throughput_mbps = (actual_bytes * 8) / (duration * 1_000_000) if duration > 0 else 0
    
    # CPU 统计
    cgroup_diff = monitor.diff()
    cpu_seconds = cgroup_diff["usage_usec"] / 1_000_000
    cpu_cores = cpu_seconds / duration if duration > 0 else 0
    throttle_ratio = (cgroup_diff["throttled_usec"] / 1_000_000 / duration) if duration > 0 else 0
    
    rtt = results.get('rtt', -1.0)
    
    output = {
        "duration": round(duration, 3),
        "throughput_mbps": round(throughput_mbps, 2),
        "bytes_downloaded": actual_bytes,
        "cpu_cores_used": round(cpu_cores, 2),
        "cpu_throttle_ratio": round(throttle_ratio, 4),
        "nr_throttled": cgroup_diff["nr_throttled"],
        "rtt_ms": round(rtt, 2) if rtt > 0 else -1.0,
        "threads": threads,
        "buffer_mb": buffer_mb,
        "errors": errors if errors else None
    }
    
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
        result = run_experiment(args.url, args.threads, args.size, args.buffer)
    except Exception as e:
        import traceback
        error_output = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "throughput_mbps": 0,
            "bytes_downloaded": 0,
            "rtt_ms": -1
        }
        print(json.dumps(error_output))
        exit(1)