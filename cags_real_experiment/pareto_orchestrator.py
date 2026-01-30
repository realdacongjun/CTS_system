#!/usr/bin/env python3
"""
CTS Pareto Optimization Orchestrator
物理正确性：Host VETH TC + Bidirectional IFB + Cgroup HiL Monitor
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

# ==============================
# 配置区
# ==============================
NETWORK_NAME = "cts_exp_net"
SERVER_IMAGE = "nginx:alpine"
CLIENT_IMAGE = "python:3.9-slim"  # ✅ Debian-based, glibc 兼容
DATA_FILE = "/tmp/cts_test_file_100mb.dat"

PARAMS = {
    "network_scenarios": [
        {"name": "IoT_Weak", "bw": "2mbit", "delay": "400ms", "loss": "5%"},
        {"name": "Edge_Normal", "bw": "20mbit", "delay": "100ms", "loss": "1%"},
        {"name": "Cloud_Fast", "bw": "1000mbit", "delay": "5ms", "loss": "0%"}
    ],
    "cpu_quota": [0.5, 1.0, 2.0],
    "threads": [1, 2, 4, 8, 16],
    "chunk_size": [256*1024, 1024*1024, 4*1024*1024],
    "file_size_mb": [100]
}

# ==============================
# 工具函数
# ==============================

def sh(cmd, check=True):
    if check:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    else:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def get_veth(container_id):
    """解析容器在宿主机侧的 veth 接口名"""
    try:
        pid = sh(f"docker inspect -f '{{{{.State.Pid}}}}' {container_id}")
        iflink = sh(f"docker exec {container_id} cat /sys/class/net/eth0/iflink")
        veth = sh(f"ip -o link | awk -F': ' '/^{iflink}:/{{print $2}}' | awk -F'@' '{{print $1}}'")
        return veth
    except Exception as e:
        raise RuntimeError(f"无法获取 veth: {e}")

def prepare_test_file(size_mb):
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) < size_mb * 1024 * 1024:
        print(f"📦 生成 {size_mb}MB 测试文件...")
        sh(f"dd if=/dev/urandom of={DATA_FILE} bs=1M count={size_mb} status=none")
    return DATA_FILE

def reset_tc(veth):
    """彻底清理 TC 和 IFB (防止僵尸接口)"""
    if veth:
        sh(f"tc qdisc del dev {veth} root 2>/dev/null || true", check=False)
        sh(f"tc qdisc del dev {veth} ingress 2>/dev/null || true", check=False)
    # 清理 IFB (即使 veth 未知也要执行)
    sh("tc qdisc del dev ifb0 root 2>/dev/null || true", check=False)
    sh("ip link set ifb0 down 2>/dev/null || true", check=False)
    sh("ip link del ifb0 2>/dev/null || true", check=False)

def setup_bidirectional_tc(veth, bw, delay, loss):
    """双向弱网：Egress(veth) + Ingress(IFB)"""
    reset_tc(veth)  # 先清理再创建
    
    sh("modprobe ifb 2>/dev/null || true", check=False)
    sh("ip link add ifb0 type ifb 2>/dev/null || true", check=False)
    sh("ip link set ifb0 up", check=False)
    
    # Egress: Server -> Client (下载方向)
    sh(f"tc qdisc add dev {veth} root netem delay {delay} loss {loss} rate {bw}")
    
    # Ingress: Client -> Server (ACK方向) 通过 IFB 重定向
    sh(f"tc qdisc add dev {veth} ingress")
    sh(f"tc filter add dev {veth} parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0")
    sh(f"tc qdisc add dev ifb0 root netem delay {delay} loss {loss} rate {bw}")

def get_ground_truth_rtt(delay_str):
    """理论 RTT (含 Docker Bridge 不对称修正 1.5x)"""
    return int(delay_str.replace('ms', '')) * 1.5

# ==============================
# HiL 监控器
# ==============================

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
            
            self.prev_stats = {
                "cpu_total": cpu_total,
                "system_total": system_total,
                "cgroup": cgroup_now,
                "cpus": cpus
            }
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
    """等待 CPU 波动 < 5%"""
    samples = []
    for _ in range(timeout * 2):
        try:
            stats = container.stats(stream=False)
            cpu = stats["cpu_stats"]["cpu_usage"]["total_usage"]
            samples.append(cpu)
            if len(samples) > 5:
                recent = samples[-5:]
                if np.std(recent) / np.mean(recent) < 0.05:
                    return True
        except:
            pass
        time.sleep(0.5)
    return False

# ==============================
# 单次实验
# ==============================

def run_single_experiment(client, config, run_id):
    net_cfg = config["network_scenarios"]
    print(f"[{run_id:03d}] {net_cfg['name']:12s} | CPU:{config['cpu_quota']:.1f} | T:{config['threads']:2d} | Chunk:{config['chunk_size']//1024}KB")
    
    server_c = None
    client_c = None
    veth = None
    
    try:
        # 1. Server (Nginx)
        nginx_conf = """events{worker_connections 1024;}http{sendfile on;tcp_nopush on;server{listen 80;root /usr/share/nginx/html;location / {add_header Accept-Ranges bytes;add_header Cache-Control no-cache;}}}"""
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
        
        # 2. TC 配置 (Host VETH)
        veth = get_veth(server_c.id)
        setup_bidirectional_tc(veth, net_cfg["bw"], net_cfg["delay"], net_cfg["loss"])
        estimated_rtt = get_ground_truth_rtt(net_cfg["delay"])
        
        # 3. Client (零依赖，无安装命令)
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto_client_fixed.py")
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
        
        # ✅ 注意：此处无任何 pip/apt 安装命令，脚本纯标准库运行
        
        if not wait_for_steady_state(client_c):
            print("   ⚠️ 未达稳态，继续执行...")
        
        # 4. 执行与监控
        with managed_monitor(client_c) as mon:
            server_ip = client.api.inspect_container(server_c.id)["NetworkSettings"]["Networks"][NETWORK_NAME]["IPAddress"]
            chunk_mb = config["chunk_size"] / (1024*1024)
            cmd = f"python3 /app/client.py --url http://{server_ip}/data.bin --threads {config['threads']} --size {config['file_size_mb']} --buffer {chunk_mb}"
            
            t0 = time.perf_counter()
            exit_code, output = client_c.exec_run(cmd)
            duration = time.perf_counter() - t0
            
            # 解析 JSON
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
        # 严格清理 (防止僵尸 IFB)
        if veth:
            reset_tc(veth)
        if client_c:
            try: client_c.remove(force=True)
            except: pass
        if server_c:
            try: server_c.remove(force=True)
            except: pass

# ==============================
# 主程序
# ==============================

def main():
    if os.geteuid() != 0:
        print("❌ 需 root 权限运行 TC")
        exit(1)
    
    client = docker.from_env()
    
    # 清理遗留网络
    try:
        net = client.networks.create(NETWORK_NAME, driver="bridge")
    except:
        net = client.networks.get(NETWORK_NAME)
        # 尝试清理旧容器
        for c in client.containers.list(all=True):
            if NETWORK_NAME in c.attrs.get("NetworkSettings", {}).get("Networks", {}):
                try: c.remove(force=True)
                except: pass
    
    # 生成实验队列
    keys = PARAMS.keys()
    experiments = [dict(zip(keys, combo)) for combo in itertools.product(*[PARAMS[k] for k in keys])]
    
    print("=" * 70)
    print(f"🚀 CTS Pareto 实验套件 | 总配置数: {len(experiments)}")
    print("=" * 70)
    
    results = []
    output_csv = f"pareto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    for i, cfg in enumerate(experiments):
        res = run_single_experiment(client, cfg, i+1)
        if res:
            results.append(res)
            pd.DataFrame([res]).to_csv(output_csv, mode="a", header=(i==0), index=False)
    
    print(f"\n✅ 完成: {len(results)}/{len(experiments)} | 数据: {output_csv}")

if __name__ == "__main__":
    main()