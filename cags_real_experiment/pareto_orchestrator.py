#!/usr/bin/env python3
"""
CTS Pareto Optimization Orchestrator - Production Grade with Multi-Scale Sampling
物理正确性：netnsid 定位 + Quota-Aware CPU + 隔离 IFB + 10/100/300MB 分层采样 + 动态超时
修复：TC 限速验证 + Nginx 零拷贝禁用
"""
import docker
import subprocess
import time
import os
import re
import json
import pandas as pd
import itertools
from datetime import datetime
import threading
import numpy as np
from contextlib import contextmanager
from typing import List, Dict, Any
import socket
import struct
import glob
import concurrent.futures

# ==============================
# 1. 配置区
# ==============================
NETWORK_NAME = "cts_exp_net"
SERVER_IMAGE = "nginx:alpine"
CLIENT_IMAGE = "python:3.9-slim"
DATA_FILE = "/tmp/cts_test_file_300mb.dat"

NETWORK_SCENARIOS = [
    {"name": "IoT_Weak", "bw": "2mbit", "delay": "400ms", "loss": "5%", "mbps": 2},
    {"name": "Edge_Normal", "bw": "20mbit", "delay": "100ms", "loss": "1%", "mbps": 20},
    {"name": "Cloud_Fast", "bw": "1000mbit", "delay": "5ms", "loss": "0%", "mbps": 1000}
]

# ==============================
# 2. 系统级工具
# ==============================

def sh(cmd, check=False, timeout=10):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, 
                              text=True, timeout=timeout)
        return result.stdout.strip()
    except:
        return ""

def nuclear_cleanup_safe():
    """安全清理：只清理实验相关接口，不碰全局 conntrack"""
    try:
        for iface in os.listdir('/sys/class/net/'):
            if iface in ['lo', 'eth0', 'ens160', 'ens33']:
                continue
            if 'docker' in iface or 'veth' in iface or iface.startswith('br-'):
                sh(f"tc qdisc del dev {iface} root 2>/dev/null", check=False)
                sh(f"tc qdisc del dev {iface} ingress 2>/dev/null", check=False)
        
        # 清理所有 ifb
        ifb_list = sh("ip -o link show type ifb 2>/dev/null | awk -F': ' '{print $2}'", check=False)
        for ifb in ifb_list.split('\n'):
            if ifb.strip():
                name = ifb.strip().split('@')[0]
                sh(f"tc qdisc del dev {name} root 2>/dev/null", check=False)
                sh(f"ip link set {name} down 2>/dev/null", check=False)
                sh(f"ip link del {name} 2>/dev/null", check=False)
        time.sleep(0.1)
    except:
        pass

def prepare_test_file(max_size_mb=300):
    """生成最大测试文件（所有实验共用，通过Range读取不同部分）"""
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) < max_size_mb * 1024 * 1024:
        print(f"📦 生成 {max_size_mb}MB 测试文件...")
        sh(f"dd if=/dev/urandom of={DATA_FILE} bs=1M count={max_size_mb} status=none")
    return DATA_FILE

# ==============================
# 3. VETH 定位（内核原教旨）
# ==============================

def get_veth_kernel_native(container_id, timeout=60):
    """
    稳定获取 veth：等待 netns 就绪 -> 等待 eth0 UP -> 解析 ifindex -> MAC fallback
    修复：Docker 异步网络创建导致的 /proc/<pid>/ns/net 不存在问题
    """
    start = time.time()
    client = docker.from_env()
    
    # =================== 阶段 0: 获取 container 对象和 PID ===================
    while time.time() - start < timeout:
        try:
            container = client.containers.get(container_id)
            info = container.attrs
            
            if not info['State']['Running']:
                if info['State']['ExitCode'] != 0:
                    logs = container.logs(tail=50).decode('utf-8', errors='ignore')
                    raise RuntimeError(f"Container exited ({info['State']['ExitCode']}). Logs: {logs}")
                time.sleep(0.2)
                continue
            
            pid = info['State']['Pid']
            if pid and pid != 0:
                break
                
        except docker.errors.NotFound:
            raise RuntimeError(f"Container {container_id} not found")
        except Exception as e:
            if "exited" in str(e):
                raise
            time.sleep(0.2)
    else:
        raise RuntimeError("Timeout: Container did not start")

    print(f"   [DEBUG] Container {container_id[:12]} PID: {pid}")

    # =================== 阶段 1: 关键修复 - 等待 netns 文件出现 ===================
    netns_path = f"/proc/{pid}/ns/net"
    ns_start = time.time()
    
    while time.time() - ns_start < timeout:
        if os.path.exists(netns_path):
            print(f"   [DEBUG] Netns ready: {netns_path}")
            break
        print(f"   [WAIT] Netns not ready yet: {netns_path}...")
        time.sleep(0.5)
    else:
        # 最后检查：如果仍不存在，检查容器是否还在运行
        container.reload()
        if not container.attrs['State']['Running']:
            raise RuntimeError(f"Container died while waiting for netns. Exit code: {container.attrs['State']['ExitCode']}")
        raise RuntimeError(f"Timeout: {netns_path} not created after {timeout}s")

    # =================== 阶段 2: 等待 eth0 出现并 UP ===================
    eth0_start = time.time()
    peer_ifindex = None
    
    while time.time() - eth0_start < timeout:
        try:
            # 双重检查 netns 仍存在（防止容器突然退出）
            if not os.path.exists(netns_path):
                raise RuntimeError("Netns disappeared during check")
            
            # 使用 nsenter 检查 eth0 状态
            link_output = sh(f"nsenter -t {pid} -n ip link show eth0 2>&1", check=False)
            
            # eth0 还不存在
            if "does not exist" in link_output:
                print(f"   [WAIT] eth0 not created yet...")
                time.sleep(0.5)
                continue
            
            # eth0 存在但未 UP（网络配置中）
            if "state UP" not in link_output:
                print(f"   [WAIT] eth0 exists but not UP: {link_output[:60].strip()}...")
                time.sleep(0.3)
                continue
            
            # eth0 UP - 提取 peer ifindex
            match = re.search(r'eth0@if(\d+)', link_output)
            if match:
                peer_ifindex = match.group(1)
                print(f"   [DEBUG] eth0 UP, peer ifindex: {peer_ifindex}")
                break
            else:
                # 老内核可能没有 @if 格式，尝试备选解析
                print(f"   [WARN] Could not parse eth0@ifXXXX from: {link_output[:100]}")
                time.sleep(0.5)
                
        except Exception as e:
            print(f"   [WARN] nsenter check failed: {e}")
            time.sleep(0.5)
    
    if not peer_ifindex:
        print("   [WARN] eth0 timeout, attempting MAC fallback...")
        return get_veth_by_mac(container_id, pid, timeout=15)

    # =================== 阶段 3: 在宿主机查找 veth ===================
    try:
        # 方法 A: 通过 ip link 直接查找（最快）
        result = sh(f"ip -o link show | grep '^{peer_ifindex}:' | head -1", check=False)
        if result:
            # 格式: "1044: vethXXXXX@if1043: <BROADCAST,MULTICAST,UP,LOWER_UP>..."
            match = re.match(r'\d+:\s+([^\s:@]+)', result)
            # ✅ 关键修复1：使用 startswith 确保是 veth，不是包含 veth 的其他字符串
            if match and match.group(1).startswith('veth'):
                veth_name = match.group(1)
                print(f"   [OK] Found veth via ip link: {veth_name}")
                return veth_name
        
        # 方法 B: 扫描 /sys/class/net（更可靠，但慢）
        for iface in os.listdir('/sys/class/net/'):
            if not iface.startswith('veth'):
                continue
            try:
                with open(f'/sys/class/net/{iface}/iflink', 'r') as f:
                    if f.read().strip() == peer_ifindex:
                        print(f"   [OK] Found veth via sysfs: {iface}")
                        return iface
            except:
                continue
        
        # 方法 C: 通过 bridge fdb + MAC（最后手段）
        print("   [WARN] ifindex methods failed, trying bridge fdb...")
        mac = sh(f"docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.MacAddress}}}}{{{{end}}}}' {container_id}").lower().strip()
        if mac:
            for _ in range(10):
                fdb = sh(f"bridge fdb show | grep -i '{mac}' | grep 'veth' | head -1", check=False)
                if fdb:
                    parts = fdb.split()
                    for i, p in enumerate(parts):
                        if p == 'dev' and i+1 < len(parts):
                            veth_name = parts[i+1]
                            # ✅ 关键修复2：确保提取的是 veth 接口，不是其他内容
                            if veth_name.startswith('veth'):
                                print(f"   [OK] Found veth via bridge fdb: {veth_name}")
                                return veth_name
                time.sleep(0.2)
                
    except Exception as e:
        print(f"   [ERROR] Find veth failed: {e}")
    
    raise RuntimeError(f"All veth detection methods failed for {container_id[:12]}")


def get_veth_by_mac(container_id, pid, timeout=10):
    """
    备选方案：通过 MAC 地址在 bridge fdb 中查找
    """
    try:
        # 获取容器 MAC
        mac = sh(f"docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.MacAddress}}}}{{{{end}}}}' {container_id}").lower().strip()
        print(f"   [DEBUG] MAC fallback: {mac}")
        
        if not mac:
            raise RuntimeError("Cannot get MAC")
        
        for _ in range(timeout * 2):  # 20 次尝试
            try:
                # 在 bridge fdb 中查找
                fdb = sh(f"bridge fdb show | grep -i '{mac}' | head -1", check=False)
                if fdb and 'veth' in fdb:
                    parts = fdb.split()
                    for i, p in enumerate(parts):
                        if p == 'dev' and i+1 < len(parts):
                            candidate = parts[i+1]
                            # ✅ 关键修复3：确保返回的是 veth 接口
                            if candidate.startswith('veth'):
                                print(f"   [OK] Found veth by MAC: {candidate}")
                                return candidate
            except:
                pass
            time.sleep(0.5)
            
    except Exception as e:
        print(f"   [ERROR] MAC fallback failed: {e}")
    
    raise RuntimeError("All veth detection methods failed")

# ==============================
# 4. TC 配置（完全隔离 IFB）- 带验证
# ==============================

def setup_isolated_tc(veth, bw, delay, loss, run_id):
    """
    [✅ 终极修复版] HTB 硬限速 + Netem 模拟 + 关闭 TSO/GSO
    确保带宽限制真正生效，而不是仅配置规则
    """


    """
    [融合版] 1. 关闭 Offload 2. 使用 HTB+Netem
    """
    ifb_name = f"ifb_{run_id}_{int(time.time()*1000)%1000}"
    
    # 1. 清理
    sh(f"tc qdisc del dev {veth} root 2>/dev/null", check=False)
    sh(f"tc qdisc del dev {veth} ingress 2>/dev/null", check=False)
    
    # 2. ✅ 关闭 Offload（关键步骤）
    # 先检查 ethtool 是否存在
    if sh("which ethtool"):
        sh(f"ethtool -K {veth} tso off gso off gro off", check=False)
        print(f"   [DEBUG] Offload disabled on {veth}")
    else:
        print(f"   [WARN] ethtool not found! TSO/GSO may bypass TC limits")
        # 备选方案：尝试用 ip route 限速（最后手段）
    
    # 3. 准备 IFB
    sh(f"modprobe ifb numifbs=100", check=False)
    sh(f"ip link add {ifb_name} type ifb", check=False)
    sh(f"ip link set {ifb_name} up", check=False)
    
    if sh("which ethtool"):
        sh(f"ethtool -K {ifb_name} tso off gso off gro off", check=False)

    # ==============================================================
    # 方向 A: Server -> Client (下载流，实验的主要方向)
    # 路径: Nginx (Server) -> eth0 -> VETH(Ingress) -> IFB -> 限速
    # ==============================================================
    
    # 将 VETH 的入站流量镜像到 IFB
    sh(f"tc qdisc add dev {veth} handle ffff: ingress 2>/dev/null || tc qdisc add dev {veth} ingress")
    sh(f"tc filter add dev {veth} parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev {ifb_name}")
    
    # 在 IFB 上应用 HTB (硬限速) + Netem (延迟/丢包)
    # Root HTB
    sh(f"tc qdisc add dev {ifb_name} root handle 1: htb default 1")
    # 限速类（burst 设大一些避免突发）
    sh(f"tc class add dev {ifb_name} parent 1: classid 1:1 htb rate {bw} burst 32k")
    # Netem 子类处理延迟和丢包
    sh(f"tc qdisc add dev {ifb_name} parent 1:1 handle 10: netem delay {delay} loss {loss} limit 10000")

    # ==============================================================
    # 方向 B: Client -> Server (ACK 包，如果不限速 ACK 会飞回，影响 TCP 行为)
    # 路径: Client -> VETH(Egress) -> Server (Nginx)
    # ==============================================================
    
    # 直接在 VETH 的出站方向应用 HTB + Netem
    sh(f"tc qdisc add dev {veth} root handle 1: htb default 1")
    sh(f"tc class add dev {veth} parent 1: classid 1:1 htb rate {bw} burst 32k")
    sh(f"tc qdisc add dev {veth} parent 1:1 handle 10: netem delay {delay} loss {loss} limit 10000")
    
    # 验证配置
    print(f"   [DEBUG] TC Setup Complete:")
    print(f"      VETH {veth}: Egress HTB(rate={bw})")
    print(f"      IFB  {ifb_name}: Ingress HTB(rate={bw}) + Netem(delay={delay}, loss={loss})")
    
    return ifb_name

def reset_isolated_tc(veth, ifb_name):
    if veth:
        sh(f"tc qdisc del dev {veth} root 2>/dev/null", check=False)
        sh(f"tc qdisc del dev {veth} ingress 2>/dev/null", check=False)
    if ifb_name:
        sh(f"tc qdisc del dev {ifb_name} root 2>/dev/null", check=False)
        sh(f"ip link set {ifb_name} down 2>/dev/null", check=False)
        sh(f"ip link del {ifb_name} 2>/dev/null", check=False)

def get_tc_stats(veth, ifb_name):
    stats = {}
    try:
        if veth:
            stats['veth'] = sh(f"tc -s qdisc show dev {veth}", check=False)
        if ifb_name:
            stats['ifb'] = sh(f"tc -s qdisc show dev {ifb_name}", check=False)
    except:
        pass
    return stats

# ==============================
# 5. 物理正确的 CPU 监控
# ==============================

class PhysicalCPUMonitor:
    def __init__(self, container, nano_cpus_quota):
        self.container = container
        self.quota_cores = nano_cpus_quota / 1e9
        self.host_cores = os.cpu_count()
        self.prev = None
        self.data = []
        self.running = False
        self._df_result = None
        self.start_ns = 0
        self.end_ns = 0
        
    def _read_total_ns(self):
        """直接获取 Cgroup 原始累计值 (纳秒)"""
        try:
            stats = self.container.stats(stream=False)
            return stats['cpu_stats']['cpu_usage']['total_usage']
        except:
            return 0

    def sample(self):
        try:
            stats = self.container.stats(stream=False)
            cgroup_stats = stats.get('cpu_stats', {})
            cpu_usage = cgroup_stats.get('cpu_usage', {}).get('total_usage', 0)
            system_usage = cgroup_stats.get('system_cpu_usage', 0)
            throttling = cgroup_stats.get('throttling_data', {})
            
            if self.prev:
                cpu_delta = cpu_usage - self.prev['cpu_usage']
                sys_delta = system_usage - self.prev['system_usage']
                
                if sys_delta > 0:
                    cpu_percent = (cpu_delta / sys_delta) * self.host_cores / self.quota_cores * 100
                    cpu_percent = min(cpu_percent, self.quota_cores * 100)
                else:
                    cpu_percent = 0.0
                
                throttle_ratio = throttling.get('throttled_periods', 0) / max(throttling.get('periods', 0), 1)
                
                self.data.append({
                    'timestamp': time.time(),
                    'cpu_percent': round(cpu_percent, 2),
                    'throttle_ratio': round(throttle_ratio, 4),
                    'throttled_periods': throttling.get('throttled_periods', 0)
                })
            
            self.prev = {'cpu_usage': cpu_usage, 'system_usage': system_usage}
        except:
            pass
    
    def start(self):
        self.start_ns = self._read_total_ns()
        self.running = True
        def loop():
            while self.running:
                self.sample()
                time.sleep(0.1)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        if self._df_result is not None: 
            return self._df_result
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.end_ns = self._read_total_ns()
        self._df_result = pd.DataFrame(self.data)
        return self._df_result

    def get_total_cpu_seconds(self):
        if self.end_ns and self.start_ns and self.end_ns > self.start_ns:
            return (self.end_ns - self.start_ns) / 1e9
        return 0.000001

@contextmanager
def physical_monitor(container, nano_cpus_quota):
    mon = PhysicalCPUMonitor(container, nano_cpus_quota)
    mon.start()
    try:
        yield mon
    finally:
        df = mon.stop()
        if not df.empty:
            ts = int(time.time())
            df.to_csv(f"micro_{container.id[:12]}_{ts}.csv", index=False)

# ==============================
# 6. 网络稳态检测（SYN-only）
# ==============================

def wait_for_network_steady_syn_only(server_ip, port=80, timeout=10):
    samples = []
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            t0 = time.perf_counter()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((server_ip, port))
            t1 = time.perf_counter()
            
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            sock.close()
            
            samples.append((t1 - t0) * 1000)
            
            if len(samples) >= 3:
                mean_rtt = np.mean(samples[-3:])
                if mean_rtt > 0 and np.std(samples[-3:]) / mean_rtt < 0.3:
                    return True
        except:
            pass
        time.sleep(0.3)
    return False

# ==============================
# 7. 分层采样实验生成器
# ==============================

from typing import List, Dict, Any

def get_adjusted_file_size(net_name, base_size):
    """
    根据网络带宽调整文件大小，避免慢网场景超时：
    - IoT_Weak (2mbit): 最大 10MB
    - Edge_Normal (20mbit): 最大 50MB
    - Cloud_Fast (1gbit): 保持 100MB
    """
    if "IoT" in net_name or "Weak" in net_name:
        return min(base_size, 10)
    elif "Edge" in net_name:
        return min(base_size, 50)
    else:
        return base_size

# def generate_hierarchical_experiments(NETWORK_SCENARIOS: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     experiments = []
#     print("🎯 Generating Hierarchical Experiments...")

#     for net in NETWORK_SCENARIOS:
#         adj_size = get_adjusted_file_size(net['name'], 100)

#         # ==========================
#         # 1️⃣ Baseline Anchor
#         # ==========================
#         experiments.append({
#             "network_scenarios": {
#                 "name": f"{net['name']}_BASELINE", 
#                 "bw": "unlimited", 
#                 "delay": "0ms", 
#                 "loss": "0%"
#             },
#             "cpu_quota": 1.0,
#             "threads": 4,
#             "chunk_size": 1024*1024,
#             "file_size_mb": adj_size,
#             "priority": 1,
#             "nano_cpus": int(1e9),
#             "exp_type": "anchor_baseline",
#             "bandwidth_mbps": net.get('mbps', 1000)
#         })

#         # ==========================
#         # 2️⃣ Anchor 全因子实验
#         # ==========================
#         thread_list = [1, 2, 4] if "IoT" in net['name'] else [1, 2, 4, 8, 16]

#         for cpu in [0.5, 1.0, 2.0]:
#             for t in thread_list:
#                 for c in [256*1024, 1024*1024, 4*1024*1024]:
#                     experiments.append({
#                         "network_scenarios": net,
#                         "cpu_quota": cpu,
#                         "threads": t,
#                         "chunk_size": c,
#                         "file_size_mb": adj_size,
#                         "exp_type": "anchor",
#                         "nano_cpus": int(cpu * 1e9),
#                         "priority": 1,
#                         "bandwidth_mbps": net.get('mbps', 1000)
#                     })

#         # ==========================
#         # 3️⃣ Probe Small 极端点
#         # IoT 和 Cloud / 弱网/高速网小文件测试
#         # ==========================
#         for net_probe in [NETWORK_SCENARIOS[0], NETWORK_SCENARIOS[2]]:
#             for cpu in [0.5, 2.0]:
#                 for t in [1, 16]:
#                     for c in [256*1024, 1024*1024]:
#                         experiments.append({
#                             "network_scenarios": net_probe,
#                             "cpu_quota": cpu,
#                             "threads": t,
#                             "chunk_size": c,
#                             "file_size_mb": 10,
#                             "exp_type": "probe_small",
#                             "nano_cpus": int(cpu * 1e9),
#                             "priority": 2,
#                             "bandwidth_mbps": net_probe.get('mbps', 1000)
#                         })

#         # ==========================
#         # 4️⃣ Probe Large 极端点
#         # 排除 IoT, 测大文件+高并发
#         # ==========================
#         for net_probe in [NETWORK_SCENARIOS[1], NETWORK_SCENARIOS[2]]:
#             for cpu in [0.5, 1.0, 2.0]:
#                 for t in [4, 8, 16]:
#                     for c in [1024*1024, 4*1024*1024]:
#                         experiments.append({
#                             "network_scenarios": net_probe,
#                             "cpu_quota": cpu,
#                             "threads": t,
#                             "chunk_size": c,
#                             "file_size_mb": 300,
#                             "exp_type": "probe_large",
#                             "nano_cpus": int(cpu * 1e9),
#                             "priority": 3,
#                             "bandwidth_mbps": net_probe.get('mbps', 1000)
#                         })

#     # ==========================
#     # 5️⃣ 排序，保证 priority 先行
#     # ==========================
#     experiments.sort(key=lambda x: x['priority'])
#     print(f"✅ Total Experiments Generated: {len(experiments)}")
#     return experiments
def generate_hierarchical_experiments(NETWORK_SCENARIOS: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    生成层次化实验配置
    核心原则：
    1. 同一场景同一张Pareto图必须使用相同文件大小
    2. IoT专注拉开CPU差距（因为网络是瓶颈）
    3. Edge/Cloud按文件大小分层，不混用
    """
    experiments = []
    print("🎯 Generating Hierarchical Experiments (Unified File Size per Plot)...")

    for net in NETWORK_SCENARIOS:
        # 基础大小限制 (IoT=10, Edge=50, Cloud=100)
        adj_size = get_adjusted_file_size(net['name'], 100)

        # ==========================
        # 1️⃣ Anchor 全因子扫描（每个场景的基础）
        # ==========================
        thread_list = [1, 2, 4] if "IoT" in net['name'] else [1, 2, 4, 8, 16]
        
        for cpu in [0.5, 1.0, 2.0]:
            for t in thread_list:
                for c in [256*1024, 1024*1024, 4*1024*1024]:
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": adj_size,
                        "exp_type": "anchor",
                        "nano_cpus": int(cpu * 1e9),
                        "priority": 1,
                        "bandwidth_mbps": net.get('mbps', 1000)
                    })

        # ==========================
        # 2️⃣ IoT 专项：低CPU配置（试图拉开成本差距）
        # ==========================
        if "IoT" in net['name']:
            print(f"   + Injecting Low-CPU experiments for {net['name']}")
            # 用极低CPU配额试图产生更低的成本
            for cpu in [0.25, 0.125]:  
                for t in [1, 2]:
                    for c in [64*1024, 256*1024]:  # 小分片更适合弱网
                        experiments.append({
                            "network_scenarios": net,
                            "cpu_quota": cpu,
                            "threads": t,
                            "chunk_size": c,
                            "file_size_mb": 10,  # 保持10MB避免超时
                            "exp_type": "iot_low_cpu",
                            "nano_cpus": max(int(cpu * 1e9), 100000000),  # 最小0.1核
                            "priority": 2,
                            "bandwidth_mbps": net.get('mbps', 2)
                        })

        # ==========================
        # 3️⃣ Probe Small（10MB，所有场景）
        # 目的：建立"小文件基准"，跨场景可比
        # ==========================
        for cpu in [0.5, 2.0]:
            for t in [1, 16]:
                for c in [256*1024, 1024*1024]:
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 10,
                        "exp_type": "probe_small",
                        "nano_cpus": int(cpu * 1e9),
                        "priority": 2,
                        "bandwidth_mbps": net.get('mbps', 1000)
                    })

        # ==========================
        # 4️⃣ Edge/Cloud 大文件实验（分层设计）
        # 原则：同一场景的Pareto图只用一种文件大小
        # ==========================
        if "IoT" not in net['name']:
            
            # 确定该场景的大文件配置
            if "Edge" in net['name']:
                # Edge: 50MB（标准）+ 300MB（极端）
                large_sizes = [50, 300]
            else:  # Cloud
                # Cloud: 100MB（标准）+ 300MB（极端）
                large_sizes = [100, 300]
            
            for file_size in large_sizes:
                # 4.1 标准大文件配置（类似原probe_large）
                thread_list_large = [4, 8, 16] if file_size == 300 else [2, 4, 8, 16]
                for cpu in [0.5, 1.0, 2.0]:
                    for t in thread_list_large:
                        for c in [1024*1024, 4*1024*1024]:
                            experiments.append({
                                "network_scenarios": net,
                                "cpu_quota": cpu,
                                "threads": t,
                                "chunk_size": c,
                                "file_size_mb": file_size,
                                "exp_type": "probe_large",
                                "nano_cpus": int(cpu * 1e9),
                                "priority": 3,
                                "bandwidth_mbps": net.get('mbps', 1000)
                            })
                
                # 4.2 Pareto平滑采样（填补空隙，仅针对标准大小）
                # 300MB的不做平滑（时间成本太高），只做50MB/100MB
                if file_size == adj_size:
                    print(f"   + Injecting Pareto Smoothing for {net['name']} ({file_size}MB)")
                    # 使用标准分数步长，避免cgroup调度问题
                    smooth_cpus = [0.75, 1.25, 1.5]  # 3/4, 5/4, 3/2核
                    smooth_threads = [3, 5, 6]  # 填补1-2-4-8-16的空隙
                    
                    for cpu in smooth_cpus:
                        for t in smooth_threads:
                            experiments.append({
                                "network_scenarios": net,
                                "cpu_quota": cpu,
                                "threads": t,
                                "chunk_size": 1024*1024,  # 标准1MB分片
                                "file_size_mb": file_size,
                                "exp_type": "pareto_smooth",
                                "nano_cpus": int(cpu * 1e9),
                                "priority": 4,  # 低优先级
                                "bandwidth_mbps": net.get('mbps', 1000)
                            })

    # ==========================
    # 5️⃣ 去重与排序
    # ==========================
    unique_experiments = []
    seen = set()
    
    for exp in experiments:
        # 生成唯一指纹：(场景, CPU, 线程, 分片, 文件大小)
        sig = (
            exp['network_scenarios']['name'], 
            exp['cpu_quota'], 
            exp['threads'], 
            exp['chunk_size'], 
            exp['file_size_mb']
        )
        if sig not in seen:
            seen.add(sig)
            unique_experiments.append(exp)
        else:
            # 记录去重信息
            print(f"   [DEDUP] Skipped duplicate: {sig}")

    # 按优先级排序
    unique_experiments.sort(key=lambda x: x['priority'])
    
    # 打印统计信息
    print(f"\n📊 Experiment Distribution:")
    for exp_type in ['anchor', 'iot_low_cpu', 'probe_small', 'probe_large', 'pareto_smooth']:
        count = len([e for e in unique_experiments if e['exp_type'] == exp_type])
        if count > 0:
            print(f"   {exp_type:20s}: {count:3d}")
    
    print(f"\n✅ Total Unique Experiments: {len(unique_experiments)}")
    return unique_experiments
# ==============================
# 超时计算函数
# ==============================
def calculate_timeout(file_size_mb, bandwidth_mbps, threads=1):
    if bandwidth_mbps <= 0: bandwidth_mbps = 1000
    base_time = (file_size_mb * 8) / bandwidth_mbps

    # 弱网 15x, 强网 5x
    multiplier = 15 if bandwidth_mbps <= 5 else 5

    timeout = max(60, min(base_time * multiplier, 3600))
    print(f"[DEBUG] Timeout Calc: {file_size_mb}MB @ {bandwidth_mbps}Mbps x{multiplier} -> Limit {int(timeout)}s")
    return int(timeout)

# ==============================
# 8. 单次实验执行
# ==============================

def exec_with_timeout(container, command, timeout_sec):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(container.exec_run, command)
        try:
            result = future.result(timeout=timeout_sec)
            return result.exit_code, result.output
        except concurrent.futures.TimeoutError:
            print(f"   ❌ Client timeout ({timeout_sec}s)")
            try:
                container.kill()
            except:
                pass
            return -1, b"TIMEOUT"
        except Exception as e:
            print(f"   ❌ Client error: {e}")
            return -1, b"ERROR"


def run_single_experiment(client, config, run_id):
    net_cfg = config["network_scenarios"]
    exp_type = config["exp_type"]
    file_size = config["file_size_mb"]
    is_baseline = "baseline" in exp_type or config.get("is_baseline", False)
    bandwidth_mbps = config.get("bandwidth_mbps", 1000)
    
    type_marker = {"anchor_baseline": "📏", "anchor": "⚓", 
                   "probe_small": "🧪", "probe_large": "🔬"}.get(exp_type, "○")
    
    print(f"[{run_id:03d}] {type_marker} {net_cfg['name']:15s} | "
          f"F:{file_size}MB | CPU:{config['cpu_quota']:.1f} | T:{config['threads']:2d}")
    
    nuclear_cleanup_safe()
    
    server_c = None
    client_c = None
    veth = None
    ifb_name = None

    try:
        # 1. Server - ✅ 关键修复：禁用 sendfile 确保流量经过 TC
        short_id = f"{run_id}_{int(time.time()*1000)%10000}"
        
        # sendfile off 强制 Nginx 使用常规 read/write，确保经过 TC netem
        nginx_conf = """events {
    worker_connections 1024;
}
http {
    sendfile off;
    tcp_nopush off;
    tcp_nodelay on;
    client_max_body_size 500M;
    proxy_read_timeout 600s;
    send_timeout 600s;
    server {
        listen 80;
        root /usr/share/nginx/html;
        location / {
            add_header Accept-Ranges bytes;
            add_header Cache-Control no-cache;
        }
    }
}"""
        
        with open("/tmp/nginx.conf", "w") as f:
            f.write(nginx_conf)
        
        server_c = client.containers.run(
            SERVER_IMAGE, name=f"srv_{short_id}", detach=True, network=NETWORK_NAME,
            volumes={DATA_FILE: {"bind": "/usr/share/nginx/html/data.bin", "mode": "ro"},
                     "/tmp/nginx.conf": {"bind": "/etc/nginx/nginx.conf", "mode": "ro"}},
            command="nginx -g 'daemon off;'"
        )
        
        # 2. VETH
        veth = get_veth_kernel_native(server_c.id)
        print(f"   🌐 {veth}")
        
        # 3. TC
        if not is_baseline:
            try:
                ifb_name = setup_isolated_tc(veth, net_cfg['bw'], net_cfg['delay'], net_cfg['loss'], run_id)
            except RuntimeError as e:
                print(f"   [ERROR] TC setup failed: {e}")
                return None
        else:
            ifb_name = None
        
        # 4. Server IP
        server_inspect = client.api.inspect_container(server_c.id)
        networks = server_inspect["NetworkSettings"]["Networks"]
        if NETWORK_NAME not in networks:
            raise RuntimeError(f"Container not in {NETWORK_NAME}")
        server_ip = networks[NETWORK_NAME]["IPAddress"]
        
        # 5. Client
        script_path = os.path.join(os.path.dirname(__file__), "pareto_client.py")
        client_c = client.containers.run(
            CLIENT_IMAGE, name=f"cli_{short_id}", detach=True, network=NETWORK_NAME,
            nano_cpus=config["nano_cpus"], mem_limit="512m",
            volumes={script_path: {"bind": "/app/client.py", "mode": "ro"}},
            command="sleep 3600"
        )
        
        # 6. Execute
        with physical_monitor(client_c, config["nano_cpus"]) as mon:
            chunk_mb = config["chunk_size"] / (1024*1024)
            cmd = (f"python3 /app/client.py --url http://{server_ip}/data.bin "
                   f"--threads {config['threads']} --size {file_size} --buffer {chunk_mb}")
            
            t0 = time.perf_counter()
            timeout_val = calculate_timeout(file_size, bandwidth_mbps)
            exit_code, output = exec_with_timeout(client_c, cmd, timeout_val)
            duration = time.perf_counter() - t0
            
            output_str = output.decode("utf-8", errors="ignore")
            
            client_res = {}
            for line in reversed(output_str.strip().split("\n")):
                if line.startswith("{") and line.endswith("}"):
                    try: 
                        client_res = json.loads(line)
                        break
                    except: 
                        pass
            
            df_micro = mon.stop()
        
        if exit_code not in [0, 2]:
            print(f"   ❌ Client failed: {exit_code}, output: {output_str[:100]}")
            return None
        
        # 7. Stats
        total_cpu_s = mon.get_total_cpu_seconds()
        thr = client_res.get("throughput_mbps", 0)
        bytes_downloaded = client_res.get("bytes_downloaded", 0)
        
        # ✅ 关键验证：检查实际吞吐量是否符合 TC 限制（允许 20% 误差）
        if not is_baseline and thr > 0:
            expected_max = bandwidth_mbps * 1.2  # 允许 20% burst
            if thr > expected_max:
                print(f"   [WARN] TC 可能未生效! 期望 <{expected_max:.1f}Mbps, 实际 {thr:.1f}Mbps")
        
        efficiency = file_size / total_cpu_s if total_cpu_s > 1e-6 else 0
        
        result = {
            "run_id": run_id,
            "exp_type": exp_type,
            "file_size_mb": file_size,
            "scenario": net_cfg["name"],
            "cpu_quota": config["cpu_quota"],
            "threads": config["threads"],
            "chunk_kb": config["chunk_size"]//1024,
            "duration_s": round(duration, 3),
            "throughput_mbps": round(thr, 2),
            "cost_cpu_seconds": round(total_cpu_s, 6),
            "efficiency_mb_per_cpus": round(efficiency, 2),
            "bytes_downloaded": bytes_downloaded,
            "exit_code": exit_code
        }
        
        status = "📏 BASELINE" if is_baseline else "✅"
        print(f"   {status} Thr:{thr:6.1f}Mbps | Cost:{total_cpu_s:.4f}s | Time:{duration:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"   ❌ {str(e)[:80]}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        if veth or ifb_name:
            reset_isolated_tc(veth, ifb_name)
        if client_c:
            client_c.remove(force=True)
        if server_c:
            server_c.remove(force=True)
        nuclear_cleanup_safe()

# ==============================
# 9. 主程序
# ==============================

def main():
    if os.geteuid() != 0:
        print("❌ Must run as root")
        exit(1)
    
    if not sh("which nsenter"):
        print("❌ Need util-linux (nsenter)")
        exit(1)
    
    client = docker.from_env()
    
    try:
        client.networks.create(NETWORK_NAME, driver="bridge")
    except:
        pass
    
    prepare_test_file(300)
    # ✅ 正确代码 (把全局变量传进去)
    experiments = generate_hierarchical_experiments(NETWORK_SCENARIOS)
    
    print(f"\n📊 实验设计: {len(experiments)} 次实验")
    print("=" * 70)
    
    output_csv = f"pareto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results = []
    
    for i, cfg in enumerate(experiments):
        res = run_single_experiment(client, cfg, i+1)
        if res:
            results.append(res)
            if len(results) % 5 == 0:
                pd.DataFrame(results[-5:]).to_csv(output_csv, mode="a", 
                                                  header=(len(results)<=5), index=False)
        
        if (i+1) % 10 == 0:
            print(f"\n📈 Progress: {i+1}/{len(experiments)}, Success: {len(results)}\n")
    
    if results:
        pd.DataFrame(results).to_csv(output_csv, mode="a", header=False, index=False)
    
    print(f"\n✅ Completed: {len(results)}/{len(experiments)}")

if __name__ == "__main__":
    main()
    