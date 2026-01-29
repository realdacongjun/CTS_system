# 文件名: pareto_orchestrator.py
import docker
import time
import csv
import json
import os
import numpy as np

# ================= 🔬 实验配置区 (Thesis Configuration) =================
IMAGE_NAME = "python:3.9-slim"
NETWORK_NAME = "cts_pareto_net"
DATA_FILE = "pareto_data_final.csv"
# 确保这里引用的是客户端脚本
CLIENT_SCRIPT_PATH = os.path.abspath("pareto_client.py")

# 变量 1: 资源限制 (Sidecar模式 vs 独立容器模式)
CPU_QUOTAS = ["1", "1,2"]

# 变量 2: 网络环境 [带宽Mbps, 延迟ms, 丢包%]
SCENARIOS = {
    "Weak":   [5,   400, 1.0],
    "Edge":   [20,  50,  0.1],
    "Cloud":  [100, 10,  0.0]
}

# 变量 3: 并发线程数
THREADS = [1, 2, 4, 8, 16]

# 变量 4: 读写缓冲区大小 (MB)
CHUNKS = [0.1, 1.0, 4.0] 

# 固定参数
FILE_SIZE_MB = 100
REPEAT_COUNT = 3

client = docker.from_env()

def setup_infra():
    """初始化隔离网络"""
    try:
        n = client.networks.get(NETWORK_NAME)
        n.remove()
    except: pass
    return client.networks.create(NETWORK_NAME, driver="bridge")

def start_server(net):
    """启动服务端容器"""
    print(f"🛠️  启动 Server (Core 0, File: {FILE_SIZE_MB}MB)...")
    cmd = f"sh -c 'dd if=/dev/urandom of=data.bin bs=1M count={FILE_SIZE_MB} && python -m http.server 80'"
    return client.containers.run(
        IMAGE_NAME, name="cts_server", command=cmd, detach=True, remove=True,
        cap_add=["NET_ADMIN"], network=NETWORK_NAME, cpuset_cpus="0"
    )

def set_server_tc(container, bw, delay, loss):
    """配置 TC 网络环境"""
    # 安装 iproute2
    container.exec_run("apt-get update")
    container.exec_run("apt-get install -y iproute2")
    
    iface = "eth0" # 默认网卡
    
    # 修复点：移除了 check=False
    container.exec_run(f"tc qdisc del dev {iface} root")
    
    burst = "32kbit"
    # TBF 控制带宽
    container.exec_run(f"tc qdisc add dev {iface} root handle 1: tbf rate {bw}mbit burst {burst} limit 100mb")
    # Netem 控制延迟和丢包
    container.exec_run(f"tc qdisc add dev {iface} parent 1:1 handle 10: netem delay {delay}ms loss {loss}%")

def run_client(net, threads, chunk, cpuset):
    """启动客户端容器运行任务"""
    try:
        volumes = {CLIENT_SCRIPT_PATH: {'bind': '/app/run.py', 'mode': 'ro'}}
        cmd = f"python /app/run.py --url http://cts_server:80/data.bin --threads {threads} --size {FILE_SIZE_MB} --buffer {chunk}"
        
        container = client.containers.run(
            IMAGE_NAME, name="cts_client", command=cmd, detach=True, volumes=volumes,
            network=NETWORK_NAME, cpuset_cpus=cpuset, working_dir="/app"
        )
        result = container.wait()
        logs = container.logs().decode('utf-8')
        container.remove()
        
        # 解析 JSON 输出
        for line in logs.strip().split('\n'):
            if line.startswith('{') and 'throughput_mbps' in line:
                return json.loads(line)
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🚀 开始帕累托实验 (Pareto Orchestrator - Fixed)...")
    
    headers = ['Cores', 'Scenario', 'Threads', 'ChunkSize', 
               'Duration_Mean', 'Duration_Std',
               'TP_Mean', 'TP_Std', 
               'CPU_Mean', 'CPU_Std', 
               'Cost_Mean']
    
    with open(DATA_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(headers)

    net = setup_infra()
    server = start_server(net)
    # 给 Server 多一点时间初始化环境
    time.sleep(10) 

    try:
        for cores in CPU_QUOTAS:
            for scene, params in SCENARIOS.items():
                print(f"\n>>> Scene: {scene} | Cores: {cores}")
                set_server_tc(server, *params)
                time.sleep(1)
                
                for threads in THREADS:
                    for chunk in CHUNKS:
                        print(f"   T={threads}, C={chunk}MB ... ", end="", flush=True)
                        
                        raw_data = []
                        for _ in range(REPEAT_COUNT):
                            res = run_client(net, threads, chunk, cores)
                            if res: raw_data.append(res)
                        
                        if not raw_data:
                            print("Fail")
                            continue
                        
                        durs = [r['duration'] for r in raw_data]
                        tps  = [r['throughput_mbps'] for r in raw_data]
                        cpus = [r['cpu_avg'] for r in raw_data]
                        
                        avg_dur, std_dur = np.mean(durs), np.std(durs)
                        avg_tp,  std_tp  = np.mean(tps),  np.std(tps)
                        avg_cpu, std_cpu = np.mean(cpus), np.std(cpus)
                        avg_cost = avg_cpu * avg_dur
                        
                        row = [cores, scene, threads, chunk, 
                               f"{avg_dur:.4f}", f"{std_dur:.4f}",
                               f"{avg_tp:.2f}",  f"{std_tp:.2f}",
                               f"{avg_cpu:.2f}", f"{std_cpu:.2f}",
                               f"{avg_cost:.2f}"]
                               
                        with open(DATA_FILE, 'a', newline='') as f:
                            csv.writer(f).writerow(row)
                        
                        print(f"✅ TP={avg_tp:.1f}±{std_tp:.1f}, Cost={avg_cost:.1f}")

    finally:
        print("\n🧹 清理环境...")
        try:
            server.stop()
        except: pass
        try:
            net.remove()
        except: pass
        print(f"💾 完成! 数据已保存至: {DATA_FILE}")

if __name__ == "__main__":
    main()