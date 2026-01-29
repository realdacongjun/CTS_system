# run_experiment_final.py
import docker
import time
import csv
import json
import os
import numpy as np

# ================= 🔬 实验配置区 =================
IMAGE_NAME = "python:3.9-slim"
NETWORK_NAME = "cts_pareto_gold"
DATA_FILE = "pareto_data_gold.csv"
CLIENT_SCRIPT_PATH = os.path.abspath("client_task.py")

# 变量 1: 核心数 (资源限制)
CPU_QUOTAS = ["1", "1,2"]

# 变量 2: 网络环境
SCENARIOS = {
    "Weak":   [5,   400, 1.0],
    "Edge":   [20,  50,  0.1],
    "Cloud":  [100, 10,  0.0]
}

# 变量 3: 并发线程数
THREADS = [1, 2, 4, 8, 16]

# 变量 4: 读写缓冲区大小 (MB) -> 对应你的 Chunk Size 创新点
# 小 Chunk 导致高 Syscall (CPU高)，大 Chunk 内存占用高但 CPU 低
CHUNKS = [0.1, 1.0, 4.0] 

# 固定参数
FILE_SIZE_MB = 100
REPEAT_COUNT = 3  # 重复次数

client = docker.from_env()

def setup_infra():
    try:
        n = client.networks.get(NETWORK_NAME)
        n.remove()
    except: pass
    return client.networks.create(NETWORK_NAME, driver="bridge")

def start_server(net):
    print(f"🛠️  启动 Server (Core 0)...")
    cmd = f"sh -c 'dd if=/dev/urandom of=data.bin bs=1M count={FILE_SIZE_MB} && python -m http.server 80'"
    return client.containers.run(
        IMAGE_NAME, name="cts_server", command=cmd, detach=True, remove=True,
        cap_add=["NET_ADMIN"], network=NETWORK_NAME, cpuset_cpus="0"
    )

def set_server_tc(container, bw, delay, loss):
    container.exec_run("apt-get update && apt-get install -y iproute2")
    # 简单粗暴探测 eth0，绝大多数容器环境都适用
    iface = "eth0" 
    container.exec_run(f"tc qdisc del dev {iface} root", check=False)
    burst = "32kbit"
    container.exec_run(f"tc qdisc add dev {iface} root handle 1: tbf rate {bw}mbit burst {burst} limit 100mb")
    container.exec_run(f"tc qdisc add dev {iface} parent 1:1 handle 10: netem delay {delay}ms loss {loss}%")

def run_client(net, threads, chunk, cpuset):
    try:
        volumes = {CLIENT_SCRIPT_PATH: {'bind': '/app/run.py', 'mode': 'ro'}}
        # 传入 buffer 参数
        cmd = f"python /app/run.py --url http://cts_server:80/data.bin --threads {threads} --size {FILE_SIZE_MB} --buffer {chunk}"
        
        container = client.containers.run(
            IMAGE_NAME, name="cts_client", command=cmd, detach=True, volumes=volumes,
            network=NETWORK_NAME, cpuset_cpus=cpuset, working_dir="/app"
        )
        result = container.wait()
        logs = container.logs().decode('utf-8')
        container.remove()
        
        for line in logs.strip().split('\n'):
            if line.startswith('{') and 'throughput_mbps' in line:
                return json.loads(line)
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🚀 开始帕累托实验 (Thesis Gold Version)...")
    
    # CSV 表头增加 std (标准差)
    headers = ['Cores', 'Scenario', 'Threads', 'ChunkSize', 
               'Duration_Mean', 'Duration_Std',
               'TP_Mean', 'TP_Std', 
               'CPU_Mean', 'CPU_Std', 
               'Cost_Mean']
    
    with open(DATA_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(headers)

    net = setup_infra()
    server = start_server(net)
    time.sleep(5)

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
                        
                        # 提取数据数组
                        durs = [r['duration'] for r in raw_data]
                        tps  = [r['throughput_mbps'] for r in raw_data]
                        cpus = [r['cpu_avg'] for r in raw_data]
                        
                        # 计算统计量
                        avg_dur, std_dur = np.mean(durs), np.std(durs)
                        avg_tp,  std_tp  = np.mean(tps),  np.std(tps)
                        avg_cpu, std_cpu = np.mean(cpus), np.std(cpus)
                        
                        # Cost = CPU占用率 * 时间 (资源消耗积分)
                        avg_cost = avg_cpu * avg_dur
                        
                        # 写入 CSV
                        row = [cores, scene, threads, chunk, 
                               f"{avg_dur:.4f}", f"{std_dur:.4f}",
                               f"{avg_tp:.2f}",  f"{std_tp:.2f}",
                               f"{avg_cpu:.2f}", f"{std_cpu:.2f}",
                               f"{avg_cost:.2f}"]
                               
                        with open(DATA_FILE, 'a', newline='') as f:
                            csv.writer(f).writerow(row)
                        
                        print(f"✅ TP={avg_tp:.1f}±{std_tp:.1f}, Cost={avg_cost:.1f}")

    finally:
        server.stop()
        net.remove()
        print(f"\n💾 完成! 数据已保存至: {DATA_FILE}")

if __name__ == "__main__":
    main()