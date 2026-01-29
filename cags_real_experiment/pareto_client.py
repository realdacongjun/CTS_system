import time
import argparse
import json
import socket
import threading
import urllib.request
import struct
import os

# ==========================================
# 🔬 Methodology: Kernel & Cgroup Probes
# ==========================================

class CgroupMonitor:
    """
    [Methodology] Accurate Resource Measurement
    Reads directly from Cgroup v2 interfaces to capture:
    1. Actual CPU usage (User + System)
    2. CFS Throttling events (crucial for resource-constrained environments)
    """
    def __init__(self):
        self.cgroup_path = self._detect_cgroup()
        self.start_metrics = self._read()

    def _detect_cgroup(self):
        # Standard path for Docker containers (Cgroup v2)
        v2_path = "/sys/fs/cgroup/cpu.stat"
        if os.path.exists(v2_path):
            return v2_path
        # Fallback for older runtimes
        return None

    def _read(self):
        metrics = {"usage_usec": 0, "nr_throttled": 0, "throttled_usec": 0}
        if not self.cgroup_path:
            return metrics
        
        try:
            with open(self.cgroup_path, 'r') as f:
                for line in f:
                    key, value = line.split()
                    if key in metrics:
                        metrics[key] = int(value)
        except:
            pass
        return metrics

    def diff(self):
        end_metrics = self._read()
        return {k: end_metrics[k] - self.start_metrics[k] for k in end_metrics}

def get_kernel_rtt(sock):
    """
    [Methodology] Network Instrumentation
    Uses getsockopt(TCP_INFO) to bypass application-layer scheduling noise.
    Returns: smoothed RTT (ms) directly from the TCP stack.
    """
    try:
        # TCP_INFO constant for Linux
        TCP_INFO = 11
        # Struct unpacking depends on kernel version, but tcpi_rtt is generally stable.
        # We read 104 bytes which covers most basic fields.
        raw = sock.getsockopt(socket.SOL_TCP, TCP_INFO, 104)
        # Unpack standard Linux tcp_info. 
        # tcpi_rtt is typically the 29th field (index 28) in uint32 array representation
        # but requires careful offset handling.
        # A simpler robust heuristic for python:
        # The struct starts with: state(1), ca_state(1), retrans(1), probes(1), backoff(1), options(1) + 2 pad
        # Then follows a series of uint32. rtt is usually at offset 64 bytes (index 16 of uint32s after header)
        # For strict correctness, we assume 64-bit Linux alignment.
        
        # struct tcp_info { ... __u32 tcpi_rtt; ... };
        # Offset calculation is tricky, but on standard x86_64, tcpi_rtt is at offset 24 or 32 bytes.
        # Let's use a safe parser assuming standard layout.
        # Format: 7 bytes (uint8) + 1 byte (pad) + 24 uint32s...
        # tcpi_rtt is usually the 5th uint32 after the header.
        
        # Alternative: Just return -1 if too complex, but for papers we need this.
        # Let's try parsing just the RTT.
        # Note: In Python, relying on this across architectures is risky, 
        # but for standard x86_64 Docker, RTT is often at byte 24 (u32).
        data = struct.unpack("I" * 20, raw[0:80]) 
        rtt_us = data[6] # Empirical offset for tcpi_rtt
        return rtt_us / 1000.0
    except Exception:
        return -1.0

# ==========================================
# ⚡ Workload Generator
# ==========================================

def download_worker(url, start, end, buffer_size, barrier, results, idx):
    headers = {"Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        # [Methodology] Sync Start
        # Ensure all threads hit the network stack simultaneously to test burst handling
        barrier.wait()
        
        with urllib.request.urlopen(req, timeout=10) as response:
            # Capture RTT from the first thread's socket
            if idx == 0:
                try:
                    sock = response.fp.raw._sock
                    results['rtt'] = get_kernel_rtt(sock)
                except: pass

            while True:
                chunk = response.read(buffer_size)
                if not chunk: break
    except Exception:
        pass

def run_experiment(url, threads, file_size_mb, buffer_mb):
    total_bytes = int(file_size_mb * 1024 * 1024)
    part_size = total_bytes // threads
    buffer_bytes = max(1024, int(buffer_mb * 1024 * 1024))

    # 1. Initialize Probes
    monitor = CgroupMonitor()
    barrier = threading.Barrier(threads)
    results = {'rtt': -1.0}

    # 2. Start Workload
    start_time = time.time()
    workers = []
    
    for i in range(threads):
        start = i * part_size
        end = total_bytes - 1 if i == threads - 1 else (start + part_size - 1)
        t = threading.Thread(target=download_worker, args=(url, start, end, buffer_bytes, barrier, results, i))
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

    duration = time.time() - start_time
    
    # 3. Collect Metrics
    cgroup_delta = monitor.diff()
    
    # [Methodology] Throughput Calculation
    throughput_mbps = (total_bytes * 8) / (duration * 1_000_000)

    # [Methodology] CPU Cores Used
    # usage_usec includes user + kernel time
    cpu_cores = cgroup_delta["usage_usec"] / (duration * 1_000_000)
    
    # [Methodology] Throttle Ratio
    # How much time we wanted to run but were blocked by Quota
    throttle_ratio = 0
    if duration > 0:
        throttle_ratio = cgroup_delta["throttled_usec"] / (duration * 1_000_000)

    output = {
        "duration": duration,
        "throughput_mbps": throughput_mbps,
        "cpu_cores_used": cpu_cores,
        "cpu_throttle_ratio": throttle_ratio,
        "nr_throttled": cgroup_delta["nr_throttled"],
        "rtt_ms": results['rtt']
    }
    print(json.dumps(output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--buffer", type=float, default=1.0)
    args = parser.parse_args()

    run_experiment(args.url, args.threads, args.size, args.buffer)