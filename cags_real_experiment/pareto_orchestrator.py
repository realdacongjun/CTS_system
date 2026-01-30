#!/usr/bin/env python3
"""
CTS Pareto Optimization Orchestrator
物理正确性：Host VETH TC + Bidirectional IFB + Cgroup HiL Monitor
分层采样：100MB(全因子) + 10MB/300MB(稀疏探测)
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
import random

# ==============================
# 配置区 - 分层采样策略
# ==============================
NETWORK_NAME = "cts_exp_net"
SERVER_IMAGE = "nginx:alpine"
CLIENT_IMAGE = "python:3.9-slim"
DATA_FILE = "/tmp/cts_test_file_100mb.dat"

# 网络场景定义
NETWORK_SCENARIOS = [
    {"name": "IoT_Weak", "bw": "2mbit", "delay": "400ms", "loss": "5%"},
    {"name": "Edge_Normal", "bw": "20mbit", "delay": "100ms", "loss": "1%"},
    {"name": "Cloud_Fast", "bw": "1000mbit", "delay": "5ms", "loss": "0%"}
]

def generate_hierarchical_experiments() -> List[Dict[str, Any]]:
    """
    分层采样策略：
    1. Anchor (100MB): 全因子 - 展示核心帕累托前沿
    2. Probe Small (10MB): 稀疏 - 验证小文件适应性
    3. Probe Large (300MB): 稀疏 - 验证长时间稳定性 (跳过IoT_Weak避免20分钟/次)
    """
    experiments = []
    
    # ==============================
    # Layer 1: Anchor (100MB) - 全因子
    # 3×3×5×3 = 135 次
    # ==============================
    print("🎯 Layer 1: Anchor experiments (100MB, full-factorial)")
    for net in NETWORK_SCENARIOS:
        for cpu in [0.5, 1.0, 2.0]:
            for t in [1, 2, 4, 8, 16]:
                for c in [256*1024, 1024*1024, 4*1024*1024]:
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 100,
                        "exp_type": "anchor",  # 元数据：用于后续分析分层
                        "priority": 1  # 优先运行
                    })
    
    # ==============================
    # Layer 2: Probe Small (10MB) - 稀疏
    # 仅边界条件：Weak/Fast × 低/高CPU × 极端线程 × 小Chunk
    # 2×2×2×2 = 16 次
    # ==============================
    print("🧪 Layer 2: Probe small (10MB, sparse)")
    probe_small_nets = [NETWORK_SCENARIOS[0], NETWORK_SCENARIOS[2]]  # IoT_Weak, Cloud_Fast
    for net in probe_small_nets:
        for cpu in [0.5, 2.0]:  # 仅边界
            for t in [1, 16]:   # 极端：单线程 vs 激进多线程
                for c in [256*1024, 1024*1024]:  # 小文件不用4MB chunk
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 10,
                        "exp_type": "probe_small",
                        "priority": 2
                    })
    
    # ==============================
    # Layer 3: Probe Large (300MB) - 稀疏
    # 排除 IoT_Weak (避免 20分钟/次)，仅 Edge/Cloud
    # 2×3×3×2 = 36 次
    # ==============================
    print("🔬 Layer 3: Probe large (300MB, sparse, skip IoT_Weak)")
    probe_large_nets = [NETWORK_SCENARIOS[1], NETWORK_SCENARIOS[2]]  # Edge, Cloud
    for net in probe_large_nets:
        for cpu in [0.5, 1.0, 2.0]:
            for t in [4, 8, 16]:  # 仅中高线程（低线程在大文件下无风险）
                for c in [1024*1024, 4*1024*1024]:  # 大文件用大 chunk
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 300,
                        "exp_type": "probe_large",
                        "priority": 3
                    })
    
    # 按优先级排序（Anchor先跑，确保核心数据优先获取）
    experiments.sort(key=lambda x: x['priority'])
    
    total = len(experiments)
    print(f"📊 Total experiments: {total} (Anchor: 135, Probe small: 16, Probe large: 36)")
    print(f"⏱️  Estimated time: ~{total * 25 / 60:.1f} hours (assuming 25s avg per exp)")
    
    return experiments

# [保留所有原有工具函数：sh, get_veth, prepare_test_file, reset_tc, setup_bidirectional_tc, get_ground_truth_rtt...]
# [保留 HiLMonitor 类...]
# [保留 run_single_experiment 函数...]
# 以下仅为简洁展示，实际应包含之前完整的函数实现

def sh(cmd, check=True):
    if check:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    else:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def get_veth(container_id):
    try:
        pid = sh(f"docker inspect -f '{{{{.State.Pid}}}}' {container_id}")
        iflink = sh(f"docker exec {container_id} cat /sys/class/net/eth0/iflink")
        veth = sh(f"ip -o link | awk -F': ' '/^{iflink}:/{{print $2}}' | awk -F'@' '{{print $1}}'")
        return veth
    except Exception as e:
        raise RuntimeError(f"无法获取 veth: {e}")

def prepare_test_file(size_mb):
    # 根据最大需求生成文件（300MB）
    max_size = 300
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) < max_size * 1024 * 1024:
        print(f"📦 生成 {max_size}MB 测试文件...")
        sh(f"dd if=/dev/urandom of={DATA_FILE} bs=1M count={max_size} status=none")
    return DATA_FILE

def reset_tc(veth):
    if veth:
        sh(f"tc qdisc del dev {veth} root 2>/dev/null || true", check=False)
        sh(f"tc qdisc del dev {veth} ingress 2>/dev/null || true", check=False)
    sh("tc qdisc del dev ifb0 root 2>/dev/null || true", check=False)
    sh("ip link set ifb0 down 2>/dev/null || true", check=False)
    sh("ip link del ifb0 2>/dev/null || true", check=False)

def setup_bidirectional_tc(veth, bw, delay, loss):
    reset_tc(veth)
    sh("modprobe ifb 2>/dev/null || true", check=False)
    sh("ip link add ifb0 type ifb 2>/dev/null || true", check=False)
    sh("ip link set ifb0 up", check=False)
    
    # Egress
    sh(f"tc qdisc add dev {veth} root netem delay {delay} loss {loss} rate {bw}")
    # Ingress via IFB
    sh(f"tc qdisc add dev {veth} ingress")
    sh(f"tc filter add dev {veth} parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0")
    sh(f"tc qdisc add dev ifb0 root netem delay {delay} loss {loss} rate {bw}")

def get_ground_truth_rtt(delay_str):
    return int(delay_str.replace('ms', '')) * 1.5

class HiLMonitor:
    def __init__(self, container):
        self.container = container
        self.prev_stats = None
        self.data = []
        self.running = False
        self.cgroup_path = self._find_cgroup_path(container.id)
        
    def _find_cgroup_path(self, cid):
        paths = [
            f"/sys/fs/cgroup/cpu/docker/{cid}/cpu.stat",
            f"/sys/fs/cgroup/cpu,cpuacct/docker/{cid}/cpu.stat",
            f"/sys/fs/cgroup/docker/{cid}/cpu.stat",
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def _read_cgroup(self):
        metrics = {"usage_usec": 0, "nr_throttled": 0, "throttled_usec": 0}
        if self.cgroup_path:
            try:
                with open(self.cgroup_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2 and parts[0] in metrics:
                            metrics[parts[0]] = int(parts[1])
            except:
                pass
        return metrics
    
    def sample(self):
        try:
            stats = self.container.stats(stream=False)
            cgroup_now = self._read_cgroup()
            cpu_total = stats["cpu_stats"]["cpu_usage"]["total_usage"]
            system_total = stats["cpu_stats"]["system_cpu_usage"]
            cpus = stats["cpu_stats"].get("online_cpus", 1)
            
            if self.prev_stats:
                cpu_delta = cpu_total - self.prev_stats["cpu_total"]
                sys_delta = system_total - self.prev_stats["system_total"]
                cpu_percent = (cpu_delta / sys_delta) * cpus * 100 if sys_delta > 0 else 0
                self.data.append({
                    "timestamp": time.time(),
                    "cpu_percent": round(cpu_percent, 2),
                    "throttle_count": max(0, cgroup_now["nr_throttled"] - self.prev_stats["cgroup"]["nr_throttled"]),
                    "throttle_time_ms": (cgroup_now["throttled_usec"] - self.prev_stats["cgroup"]["throttled_usec"]) / 1000
                })
            self.prev_stats = {"cpu_total": cpu_total, "system_total": system_total, "cgroup": cgroup_now, "cpus": cpus}
        except:
            pass
    
    def start(self):
        self.running = True
        def loop():
            while self.running:
                self.sample()
                time.sleep(0.2)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        return pd.DataFrame(self.data)

@contextmanager
def managed_monitor(container):
    monitor = HiLMonitor(container)
    monitor.start()
    try:
        yield monitor
    finally:
        df = monitor.stop()
        if not df.empty:
            df.to_csv(f"micro_{container.id[:12]}_{int(time.time())}.csv", index=False)

def wait_for_steady_state(container, timeout=15):
    samples = []
    for _ in range(timeout * 2):
        try:
            stats = container.stats(stream=False)
            cpu = stats["cpu_stats"]["cpu_usage"]["total_usage"]
            samples.append(cpu)
            if len(samples) > 5:
                recent = samples[-5:]
                if np.mean(recent) > 0 and np.std(recent) / np.mean(recent) < 0.05:
                    return True
        except:
            pass
        time.sleep(0.5)
    return False

def run_single_experiment(client, config, run_id):
    """单次实验执行（与之前相同，保留物理正确性）"""
    net_cfg = config["network_scenarios"]
    exp_type = config.get("exp_type", "anchor")
    file_size = config["file_size_mb"]
    
    # 显示实验类型标记
    type_marker = {"anchor": "⚓", "probe_small": "🧪", "probe_large": "🔬"}.get(exp_type, "○")
    print(f"[{run_id:03d}] {type_marker} {net_cfg['name']:12s} | "
          f"F:{file_size}MB | CPU:{config['cpu_quota']:.1f} | "
          f"T:{config['threads']:2d} | C:{config['chunk_size']//1024}KB")
    
    # [后续实现与之前提供的代码完全相同：启动 Nginx -> TC 配置 -> Client -> 监控 -> 清理]
    # 为简洁省略，实际应粘贴之前验证过的完整实现
    server_c = None
    client_c = None
    veth = None
    
    try:
        # Nginx 配置（支持 Range 请求）
        nginx_conf = """events{worker_connections 1024;}http{sendfile on;tcp_nopush on;client_max_body_size 500M;proxy_read_timeout 600s;send_timeout 600s;server{listen 80;root /usr/share/nginx/html;location/{add_header Accept-Ranges bytes;add_header Cache-Control no-cache;}}}"""
        with open("/tmp/nginx.conf", "w") as f:
            f.write(nginx_conf)
            
        server_c = client.containers.run(
            SERVER_IMAGE,
            name=f"srv_{run_id}_{int(time.time()*1000)%10000}",
            detach=True,
            network=NETWORK_NAME,
            volumes={
                DATA_FILE: {"bind": "/usr/share/nginx/html/data.bin", "mode": "ro"},
                "/tmp/nginx.conf": {"bind": "/etc/nginx/nginx.conf", "mode": "ro"}
            },
            command="nginx -g 'daemon off;'"
        )
        time.sleep(0.5)
        
        # TC 配置
        veth = get_veth(server_c.id)
        setup_bidirectional_tc(veth, net_cfg["bw"], net_cfg["delay"], net_cfg["loss"])
        estimated_rtt = get_ground_truth_rtt(net_cfg["delay"])
        
        # Client
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto_client.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"缺少客户端脚本: {script_path}")
        
        client_c = client.containers.run(
            CLIENT_IMAGE,
            name=f"cli_{run_id}_{int(time.time()*1000)%10000}",
            detach=True,
            network=NETWORK_NAME,
            nano_cpus=int(config["cpu_quota"] * 1e9),
            volumes={script_path: {"bind": "/app/client.py", "mode": "ro"}},
            command="sleep 3600"
        )
        
        if not wait_for_steady_state(client_c):
            print("   ⚠️ 未达稳态，继续执行...")
        
        # 执行与监控
        with managed_monitor(client_c) as mon:
            server_ip = client.api.inspect_container(server_c.id)["NetworkSettings"]["Networks"][NETWORK_NAME]["IPAddress"]
            chunk_mb = config["chunk_size"] / (1024*1024)
            cmd = f"python3 /app/client.py --url http://{server_ip}/data.bin --threads {config['threads']} --size {file_size} --buffer {chunk_mb}"
            
            t0 = time.perf_counter()
            exit_code, output = client_c.exec_run(cmd)
            duration = time.perf_counter() - t0
            
            client_res = {}
            for line in reversed(output.decode("utf-8", errors="ignore").strip().split("\n")):
                if line.startswith("{") and line.endswith("}"):
                    try:
                        client_res = json.loads(line)
                        break
                    except:
                        pass
        
        if exit_code not in [0, 2]:
            print(f"   ❌ Client 失败: {exit_code}")
            return None
        
        df_micro = mon.stop() if hasattr(mon, 'data') else pd.DataFrame()
        avg_cpu = df_micro["cpu_percent"].mean() if not df_micro.empty else 0
        total_throttle = int(df_micro["throttle_count"].sum()) if not df_micro.empty else 0
        thr = client_res.get("throughput_mbps", 0)
        
        result = {
            "run_id": run_id,
            "exp_type": exp_type,
            "file_size_mb": file_size,
            "scenario": net_cfg["name"],
            "bw_mbit": int(net_cfg["bw"].replace("mbit","")),
            "delay_ms": int(net_cfg["delay"].replace("ms","")),
            "cpu_quota": config["cpu_quota"],
            "threads": config["threads"],
            "chunk_kb": config["chunk_size"]//1024,
            "duration_s": round(duration, 3),
            "throughput_mbps": round(thr, 2),
            "avg_cpu_pct": round(avg_cpu, 2),
            "total_throttle_events": total_throttle,
            "cpu_efficiency": round(thr/max(avg_cpu, 0.1), 2),
            "kernel_rtt_ms": round(client_res.get("rtt_ms", -1), 2),
            "estimated_rtt_ms": round(estimated_rtt, 1)
        }
        
        print(f"   ✅ Thr:{result['throughput_mbps']:6.1f}Mbps | CPU:{result['avg_cpu_pct']:5.1f}% | Eff:{result['cpu_efficiency']:5.1f}")
        return result
        
    except Exception as e:
        print(f"   ❌ 错误: {str(e)[:80]}")
        return None
        
    finally:
        if veth:
            reset_tc(veth)
        if client_c:
            try: client_c.remove(force=True)
            except: pass
        if server_c:
            try: server_c.remove(force=True)
            except: pass

def generate_hierarchical_experiments() -> List[Dict[str, Any]]:
    """
    分层采样策略生成器
    返回: 实验配置列表，按优先级排序 (Anchor优先)
    """
    experiments = []
    
    # ==============================
    # Layer 1: Anchor (100MB) - 全因子实验
    # 3网络 × 3CPU × 5线程 × 3块大小 = 135次
    # ==============================
    for net in NETWORK_SCENARIOS:
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
                        "priority": 1
                    })
    
    # ==============================
    # Layer 2: Probe Small (10MB) - 稀疏采样
    # 仅测边界条件，验证Risk Barrier在小文件下的适应性
    # 2网络(Weak/Fast) × 2CPU(0.5/2.0) × 2线程(1/16) × 2块大小(256K/1M) = 16次
    # ==============================
    probe_small_nets = [NETWORK_SCENARIOS[0], NETWORK_SCENARIOS[2]]  # IoT_Weak, Cloud_Fast
    for net in probe_small_nets:
        for cpu in [0.5, 2.0]:  # 仅边界CPU
            for t in [1, 16]:   # 单线程vs激进多线程
                for c in [256*1024, 1024*1024]:  # 小文件不用4MB chunk
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 10,
                        "exp_type": "probe_small",
                        "priority": 2
                    })
    
    # ==============================
    # Layer 3: Probe Large (300MB) - 稀疏采样  
    # 验证长时间传输稳定性，跳过IoT_Weak(避免20分钟/次)
    # 2网络(Edge/Cloud) × 3CPU × 3线程(4/8/16) × 2块大小(1M/4M) = 36次
    # ==============================
    probe_large_nets = [NETWORK_SCENARIOS[1], NETWORK_SCENARIOS[2]]  # Edge_Normal, Cloud_Fast
    for net in probe_large_nets:
        for cpu in [0.5, 1.0, 2.0]:
            for t in [4, 8, 16]:  # 仅中高线程(低线程大文件无风险)
                for c in [1024*1024, 4*1024*1024]:  # 大文件用大chunk
                    experiments.append({
                        "network_scenarios": net,
                        "cpu_quota": cpu,
                        "threads": t,
                        "chunk_size": c,
                        "file_size_mb": 300,
                        "exp_type": "probe_large",
                        "priority": 3
                    })
    
    # 按优先级排序，确保Anchor数据优先获取
    experiments.sort(key=lambda x: x['priority'])
    
    # 统计信息
    counts = {
        'anchor': len([e for e in experiments if e['exp_type'] == 'anchor']),
        'probe_small': len([e for e in experiments if e['exp_type'] == 'probe_small']),
        'probe_large': len([e for e in experiments if e['exp_type'] == 'probe_large'])
    }
    total = len(experiments)
    
    print(f"📊 分层实验设计:")
    print(f"   ⚓ Anchor (100MB):     {counts['anchor']:3d} 次 (全因子)")
    print(f"   🧪 Probe Small (10MB): {counts['probe_small']:3d} 次 (稀疏)")
    print(f"   🔬 Probe Large (300MB):{counts['probe_large']:3d} 次 (稀疏, 无IoT_Weak)")
    print(f"   ─────────────────────────")
    print(f"   总计: {total} 次实验")
    print(f"   预估时间: ~{total * 20 / 3600:.1f} 小时 (按20秒/次)")
    
    return experiments

# ==============================
# 主程序
# ==============================


def main():
    if os.geteuid() != 0:
        print("❌ 需 root 权限运行 TC")
        exit(1)
    
    client = docker.from_env()
    
    # 网络准备
    try:
        net = client.networks.create(NETWORK_NAME, driver="bridge")
        print(f"🌐 创建网络: {NETWORK_NAME}")
    except:
        net = client.networks.get(NETWORK_NAME)
        print(f"🌐 使用现有网络: {NETWORK_NAME}")
        # 清理残留
        for c in client.containers.list(all=True):
            if NETWORK_NAME in c.attrs.get("NetworkSettings", {}).get("Networks", {}):
                try: c.remove(force=True)
                except: pass
    
    # 生成测试文件（最大300MB）
    prepare_test_file(300)
    
    # 生成分层实验队列
    experiments = generate_hierarchical_experiments()
    
    # 混洗同优先级实验（避免时间漂移）
    random.seed(42)  # 可重复
    # 按优先级分组混洗
    for p in [1, 2, 3]:
        group = [e for e in experiments if e['priority'] == p]
        random.shuffle(group)
        # 放回原位保持优先级顺序
        idx = [i for i, e in enumerate(experiments) if e['priority'] == p]
        for i, exp in zip(idx, group):
            experiments[i] = exp
    
    print("=" * 70)
    output_csv = f"pareto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results = []
    
    for i, cfg in enumerate(experiments):
        res = run_single_experiment(client, cfg, i+1)
        if res:
            results.append(res)
            pd.DataFrame([res]).to_csv(output_csv, mode="a", header=(i==0), index=False)
        
        # 每 50 个实验显示进度
        if (i+1) % 50 == 0:
            anchor_done = len([r for r in results if r.get('exp_type') == 'anchor'])
            print(f"\n📈 进度: {i+1}/{len(experiments)} | 已获取 Anchor 数据: {anchor_done}/135\n")
    
    print(f"\n✅ 完成: {len(results)}/{len(experiments)} | 数据: {output_csv}")
    
    # 快速帕累托预览
    if results:
        df = pd.DataFrame(results)
        print("\n📊 快速分析:")
        for fsize in [10, 100, 300]:
            sub = df[df['file_size_mb'] == fsize]
            if not sub.empty:
                best = sub.loc[sub['throughput_mbps'].idxmax()]
                print(f"  {fsize}MB: 最佳吞吐 {best['throughput_mbps']:.1f} Mbps "
                      f"(Threads={best['threads']}, {best['scenario']})")

if __name__ == "__main__":
    main()