#!/usr/bin/env python3
"""
CTS Pareto Optimization Orchestrator - Production Grade with Multi-Scale Sampling
物理正确性：netnsid 定位 + Quota-Aware CPU + 隔离 IFB + 10/100/300MB 分层采样
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

# ==============================
# 1. 配置区
# ==============================
NETWORK_NAME = "cts_exp_net"
SERVER_IMAGE = "nginx:alpine"
CLIENT_IMAGE = "python:3.9-slim"
DATA_FILE = "/tmp/cts_test_file_300mb.dat"  # 生成最大300MB，通过Range读取不同部分

NETWORK_SCENARIOS = [
    {"name": "IoT_Weak", "bw": "2mbit", "delay": "400ms", "loss": "5%"},
    {"name": "Edge_Normal", "bw": "20mbit", "delay": "100ms", "loss": "1%"},
    {"name": "Cloud_Fast", "bw": "1000mbit", "delay": "5ms", "loss": "0%"}
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


def get_veth_kernel_native(container_id, timeout=20):
    """
    修复版：使用 ip link 和 ethtool 标准工具，避免 sysfs 路径差异
    """
    start = time.time()
    
    # 获取容器 PID
    pid = None
    while time.time() - start < timeout:
        try:
            pid = sh(f"docker inspect -f '{{{{.State.Pid}}}}' {container_id}")
            if pid and pid != '0':
                # 验证容器内网络已就绪（eth0 存在且 UP）
                eth0_state = sh(f"nsenter -t {pid} -n ip link show eth0 2>/dev/null | grep 'state UP'", check=False)
                if eth0_state:
                    break
        except:
            pass
        time.sleep(0.3)
    
    if not pid:
        raise RuntimeError("Container PID or network not ready")
    
    # 方法1：通过 ethtool 在容器内查看 peer ifindex（最可靠）
    for attempt in range(10):
        try:
            # 在容器内执行 ethtool -S eth0，查找 peer_ifindex
            peer_output = sh(f"nsenter -t {pid} -n ethtool -S eth0 2>/dev/null | grep peer_ifindex")
            if peer_output:
                # 解析 peer_ifindex 的值
                peer_idx = peer_output.split(':')[-1].strip()
                
                # 在宿主机查找该 ifindex 对应的接口名
                veth_name = sh(f"ip -o link show | grep '^[{peer_idx}]:' | awk -F': ' '{{print $2}}' | cut -d'@' -f1")
                if veth_name and veth_name.startswith("veth"):
                    return veth_name
        except:
            pass
        time.sleep(0.5)
    
    # 方法2：通过 IP 地址反查（备选）
    try:
        # 获取容器 IP
        container_ip = sh(f"docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' {container_id}")
        
        # 在宿主机上通过 arp 或 bridge fdb 查找
        for _ in range(5):
            # 尝试从 bridge fdb 获取
            fdb_output = sh(f"bridge fdb show | grep '{container_ip}' | head -1", check=False)
            if fdb_output:
                parts = fdb_output.split()
                if 'dev' in parts:
                    idx = parts.index('dev')
                    candidate = parts[idx + 1]
                    if 'veth' in candidate:
                        return candidate
            
            # 或者通过邻居表
            neigh = sh(f"ip neigh show | grep '{container_ip}' | head -1", check=False)
            if neigh:
                # 解析出接口名
                match = re.search(r'dev\s+(\S+)', neigh)
                if match and 'veth' in match.group(1):
                    return match.group(1)
            
            time.sleep(0.5)
    except:
        pass
    
    # 方法3：暴力扫描（最后手段）
    try:
        # 获取所有 veth 接口，逐个检查 iflink 是否指向容器的 eth0 ifindex
        container_eth0_idx = sh(f"nsenter -t {pid} -n cat /sys/class/net/eth0/ifindex 2>/dev/null")
        
        veth_list = sh("ip -o link show type veth 2>/dev/null | awk -F': ' '{print $2}' | cut -d'@' -f1")
        for veth in veth_list.split():
            veth = veth.strip()
            if not veth:
                continue
            try:
                peer_iflink = sh(f"cat /sys/class/net/{veth}/iflink 2>/dev/null")
                if peer_iflink == container_eth0_idx:
                    return veth
            except:
                continue
    except:
        pass
    
    raise RuntimeError(f"Cannot locate veth for {container_id[:12]} after all methods")



# ==============================
# 4. TC 配置（完全隔离 IFB）
# ==============================

def setup_isolated_tc(veth, bw, delay, loss, run_id):
    """每次实验使用独立命名的 ifb 设备"""
    ifb_name = f"ifb_{run_id}_{int(time.time()*1000)%1000}"
    
    # 清理旧规则
    sh(f"tc qdisc del dev {veth} root 2>/dev/null", check=False)
    sh(f"tc qdisc del dev {veth} ingress 2>/dev/null", check=False)
    
    # 创建独立 ifb
    sh(f"modprobe ifb numifbs=100", check=False)
    sh(f"ip link add {ifb_name} type ifb", check=False)
    sh(f"ip link set {ifb_name} up", check=False)
    
    # Egress (Server -> Client)
    sh(f"tc qdisc add dev {veth} root netem delay {delay} loss {loss} rate {bw}")
    
    # Ingress (Client -> Server) via IFB
    sh(f"tc qdisc add dev {veth} ingress")
    sh(f"tc filter add dev {veth} parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev {ifb_name}")
    sh(f"tc qdisc add dev {ifb_name} root netem delay {delay} loss {loss} rate {bw}")
    
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
    """获取 tc 统计（验证实际丢包、延迟）"""
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
# 5. 物理正确的 CPU 监控（已修正公式）
# ==============================

class PhysicalCPUMonitor:
    def __init__(self, container, nano_cpus_quota):
        self.container = container
        self.quota_cores = nano_cpus_quota / 1e9
        self.host_cores = os.cpu_count()
        self.prev = None
        self.data = []
        self.running = False
        self._df_result = None  # 缓存避免重复 stop
        
    def sample(self):
        try:
            stats = self.container.stats(stream=False)
            cgroup_stats = stats.get('cpu_stats', {})
            
            cpu_usage = cgroup_stats.get('cpu_usage', {}).get('total_usage', 0)
            system_usage = cgroup_stats.get('system_cpu_usage', 0)
            
            throttling = cgroup_stats.get('throttling_data', {})
            periods = throttling.get('periods', 0)
            throttled_periods = throttling.get('throttled_periods', 0)
            
            if self.prev:
                cpu_delta = cpu_usage - self.prev['cpu_usage']
                sys_delta = system_usage - self.prev['system_usage']
                
                if sys_delta > 0:
                    # 物理正确公式：相对于配额的使用率
                    cpu_percent = (cpu_delta / sys_delta) * self.host_cores / self.quota_cores * 100
                    cpu_percent = min(cpu_percent, 100.0)
                else:
                    cpu_percent = 0
                
                throttle_ratio = throttled_periods / max(periods, 1)
                
                self.data.append({
                    'timestamp': time.time(),
                    'cpu_percent': round(cpu_percent, 2),
                    'throttle_ratio': round(throttle_ratio, 4),
                    'throttled_periods': throttled_periods
                })
            
            self.prev = {
                'cpu_usage': cpu_usage,
                'system_usage': system_usage
            }
        except:
            pass
    
    def start(self):
        self.running = True
        def loop():
            while self.running:
                self.sample()
                time.sleep(0.5)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """返回 DataFrame，重复调用返回缓存"""
        if self._df_result is not None:
            return self._df_result
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self._df_result = pd.DataFrame(self.data)
        return self._df_result

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
            
            # 立即 RST 避免 TIME_WAIT
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
# 7. 分层采样实验生成器（恢复 10/100/300MB）
# ==============================

def generate_hierarchical_experiments() -> List[Dict[str, Any]]:
    """
    分层采样策略：
    - 10MB (Probe Small): 稀疏采样，验证小文件适应性
    - 100MB (Anchor): 全因子采样，核心帕累托前沿
    - 300MB (Probe Large): 稀疏采样，验证长时间稳定性（跳过IoT_Weak避免20分钟/次）
    """
    experiments = []
    
    # ==============================
    # Layer 1: Anchor (100MB) - 全因子
    # 3网络 × 3CPU × 5线程 × 3块大小 = 135次
    # ==============================
    print("🎯 Layer 1: Anchor experiments (100MB, full-factorial)")
    for net in NETWORK_SCENARIOS:
        # 100MB 基线（无TC）
        experiments.append({
            "network_scenarios": {"name": f"{net['name']}_BASELINE", "bw": "unlimited", "delay": "0ms", "loss": "0%"},
            "cpu_quota": 1.0, "threads": 4, "chunk_size": 1024*1024, "file_size_mb": 100,
            "exp_type": "anchor_baseline", "nano_cpus": int(1e9), "priority": 1
        })
        
        # 100MB 全因子实验
        for cpu in [0.5, 1.0, 2.0]:
            for t in [1, 2, 4, 8, 16]:
                for c in [256*1024, 1024*1024, 4*1024*1024]:
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 100,
                        "exp_type": "anchor",
                        "nano_cpus": int(cpu * 1e9),
                        "priority": 1
                    })
        
        # 100MB 验证基线
        experiments.append({
            "network_scenarios": {"name": f"{net['name']}_VERIFY", "bw": "unlimited", "delay": "0ms", "loss": "0%"},
            "cpu_quota": 1.0, "threads": 4, "chunk_size": 1024*1024, "file_size_mb": 100,
            "exp_type": "anchor_verify", "nano_cpus": int(1e9), "priority": 1
        })
    
    # ==============================
    # Layer 2: Probe Small (10MB) - 稀疏采样
    # 仅边界条件：Weak/Fast × 低/高CPU × 极端线程 × 小Chunk
    # 2网络 × 2CPU × 2线程 × 2块大小 = 16次
    # ==============================
    print("🧪 Layer 2: Probe small (10MB, sparse)")
    probe_small_nets = [NETWORK_SCENARIOS[0], NETWORK_SCENARIOS[2]]  # IoT_Weak, Cloud_Fast
    for net in probe_small_nets:
        for cpu in [0.5, 2.0]:  # 仅边界CPU
            for t in [1, 16]:   # 单线程 vs 激进多线程
                for c in [256*1024, 1024*1024]:  # 小文件不用4MB chunk
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 10,
                        "exp_type": "probe_small",
                        "nano_cpus": int(cpu * 1e9),
                        "priority": 2
                    })
    
    # ==============================
    # Layer 3: Probe Large (300MB) - 稀疏采样
    # 排除 IoT_Weak（避免 20分钟/次），仅 Edge/Cloud
    # 2网络 × 3CPU × 3线程 × 2块大小 = 36次
    # ==============================
    print("🔬 Layer 3: Probe large (300MB, sparse, skip IoT_Weak)")
    probe_large_nets = [NETWORK_SCENARIOS[1], NETWORK_SCENARIOS[2]]  # Edge_Normal, Cloud_Fast
    for net in probe_large_nets:
        for cpu in [0.5, 1.0, 2.0]:
            for t in [4, 8, 16]:  # 仅中高线程（低线程大文件无风险）
                for c in [1024*1024, 4*1024*1024]:  # 大文件用大 chunk
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 300,
                        "exp_type": "probe_large",
                        "nano_cpus": int(cpu * 1e9),
                        "priority": 3
                    })
    
    # 按优先级排序（Anchor先跑）
    experiments.sort(key=lambda x: x['priority'])
    
    # 统计
    counts = {
        'anchor': len([e for e in experiments if e['exp_type'] == 'anchor']),
        'anchor_baseline': len([e for e in experiments if 'baseline' in e['exp_type']]),
        'probe_small': len([e for e in experiments if e['exp_type'] == 'probe_small']),
        'probe_large': len([e for e in experiments if e['exp_type'] == 'probe_large'])
    }
    total = len(experiments)
    
    print(f"\n📊 分层实验设计:")
    print(f"   ⚓ Anchor (100MB):      {counts['anchor']:3d} 次 (全因子) + {counts['anchor_baseline']:2d} 基线")
    print(f"   🧪 Probe Small (10MB):  {counts['probe_small']:3d} 次 (稀疏)")
    print(f"   🔬 Probe Large (300MB): {counts['probe_large']:3d} 次 (稀疏, 无IoT_Weak)")
    print(f"   ─────────────────────────")
    print(f"   总计: {total} 次实验")
    print(f"   预估时间: ~{total * 25 / 60:.1f} 小时 (按25秒/次)")
    
    return experiments

# ==============================
# 8. 单次实验执行
# ==============================

def run_single_experiment(client, config, run_id):
    net_cfg = config["network_scenarios"]
    exp_type = config["exp_type"]
    file_size = config["file_size_mb"]
    is_baseline = "baseline" in exp_type or config.get("is_baseline", False)
    
    type_marker = {"anchor_baseline": "📏", "anchor": "⚓", "anchor_verify": "✓", 
                   "probe_small": "🧪", "probe_large": "🔬"}.get(exp_type, "○")
    
    print(f"[{run_id:03d}] {type_marker} {net_cfg['name']:15s} | "
          f"F:{file_size}MB | CPU:{config['cpu_quota']:.1f} | T:{config['threads']:2d} | "
          f"C:{config['chunk_size']//1024}KB")
    
    nuclear_cleanup_safe()
    
    server_c = None
    client_c = None
    veth = None
    ifb_name = None
    
    try:
        # 1. Server
        short_id = f"{run_id}_{int(time.time()*1000)%10000}"
        nginx_conf = """events{worker_connections 1024;}http{sendfile on;tcp_nopush on;client_max_body_size 500M;proxy_read_timeout 600s;send_timeout 600s;server{listen 80;root /usr/share/nginx/html;location/{add_header Accept-Ranges bytes;add_header Cache-Control no-cache;}}}"""
        
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
        
        # 3. TC 配置（基线不加 TC）
        if not is_baseline:
            ifb_name = setup_isolated_tc(veth, net_cfg['bw'], net_cfg['delay'], net_cfg['loss'], run_id)
        else:
            ifb_name = None
            sh(f"tc qdisc del dev {veth} root 2>/dev/null", check=False)
        
        # 4. 网络稳态
        server_ip = client.api.inspect_container(server_c.id)["NetworkSettings"]["Networks"][NETWORK_NAME]["IPAddress"]
        wait_for_network_steady_syn_only(server_ip)
        
        # 5. Client
        script_path = os.path.join(os.path.dirname(__file__), "pareto_client.py")
        client_c = client.containers.run(
            CLIENT_IMAGE, name=f"cli_{short_id}", detach=True, network=NETWORK_NAME,
            nano_cpus=config["nano_cpus"], mem_limit="512m",
            volumes={script_path: {"bind": "/app/client.py", "mode": "ro"}},
            command="sleep 3600"
        )
        
        # 6. 执行（物理监控）
        with physical_monitor(client_c, config["nano_cpus"]) as mon:
            chunk_mb = config["chunk_size"] / (1024*1024)
            cmd = (f"python3 /app/client.py --url http://{server_ip}/data.bin "
                   f"--threads {config['threads']} --size {file_size} --buffer {chunk_mb}")
            
            t0 = time.perf_counter()
            exit_code, output = client_c.exec_run(cmd, timeout=600 if file_size == 300 else 300)
            duration = time.perf_counter() - t0
            
            # 解析 JSON
            client_res = {}
            for line in reversed(output.decode("utf-8", errors="ignore").strip().split("\n")):
                if line.startswith("{") and line.endswith("}"):
                    try: client_res = json.loads(line); break
                    except: pass
            
            df_micro = mon.stop()  # 通过缓存避免重复
        
        if exit_code not in [0, 2]:
            print(f"   ❌ Client failed: {exit_code}")
            return None
        
        avg_cpu = df_micro["cpu_percent"].mean() if not df_micro.empty else 0
        max_throttle = df_micro["throttle_ratio"].max() if not df_micro.empty else 0
        thr = client_res.get("throughput_mbps", 0)
        
        tc_stats = get_tc_stats(veth, ifb_name) if not is_baseline else {}
        
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
            "avg_cpu_pct": round(avg_cpu, 2),
            "max_throttle_ratio": round(max_throttle, 4),
            "kernel_rtt_ms": round(client_res.get("rtt_ms", -1), 2),
            "tc_stats": json.dumps(tc_stats)[:500] if tc_stats else "",
            "exit_code": exit_code
        }
        
        status = "📏 BASELINE" if is_baseline else "✅"
        print(f"   {status} Thr:{result['throughput_mbps']:6.1f}Mbps | CPU:{result['avg_cpu_pct']:5.1f}%")
        
        if not is_baseline and veth and ifb_name:
            reset_isolated_tc(veth, ifb_name)
            veth, ifb_name = None, None
            
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
    experiments = generate_hierarchical_experiments()
    
    # 同优先级内混洗（避免时间漂移）
    import random
    random.seed(42)
    for p in [1, 2, 3]:
        group = [e for e in experiments if e['priority'] == p]
        random.shuffle(group)
        idx = [i for i, e in enumerate(experiments) if e['priority'] == p]
        for i, exp in zip(idx, group):
            experiments[i] = exp
    
    output_csv = f"pareto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results = []
    
    print("\n" + "=" * 70)
    
    for i, cfg in enumerate(experiments):
        res = run_single_experiment(client, cfg, i+1)
        if res:
            results.append(res)
            if len(results) % 5 == 0:
                pd.DataFrame(results[-5:]).to_csv(output_csv, mode="a", 
                                                  header=(i<5), index=False)
                os.sync()
        
        if (i+1) % 50 == 0:
            print(f"\n📈 Progress: {i+1}/{len(experiments)}\n")
    
    if results:
        pd.DataFrame(results).to_csv(output_csv, mode="a", header=False, index=False)
    
    print(f"\n✅ Completed: {len(results)}/{len(experiments)}")
    
    # 多尺度分析
    try:
        df = pd.DataFrame(results)
        print("\n📊 Multi-Scale Analysis:")
        for fsize in [10, 100, 300]:
            sub = df[df["file_size_mb"] == fsize]
            if not sub.empty:
                valid = sub[sub["throughput_mbps"] > 0]
                if not valid.empty:
                    best = valid.loc[valid["throughput_mbps"].idxmax()]
                    print(f"  {fsize}MB: Max {best['throughput_mbps']:.1f} Mbps "
                          f"(T={best['threads']}, C={best['chunk_kb']}KB, {best['scenario']})")
    except Exception as e:
        print(f"分析错误: {e}")

if __name__ == "__main__":
    main()