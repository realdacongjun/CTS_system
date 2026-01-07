import random
import shlex

# === 1. 目标镜像列表 ===
TARGET_IMAGES = [
    'quay.io/centos/centos:stream9', 'fedora:latest', 'ubuntu:latest',
    'mongo:latest', 'mysql:latest', 'postgres:latest',
    'rust:latest', 'ruby:latest', 'python:latest',
    'nginx:latest', 'httpd:latest', 
    'rabbitmq:latest', 'wordpress:latest', 'nextcloud:latest',
    'gradle:latest', 'node:latest'
]

# === 2. 压缩算法配置 (修复点：全部改为列表格式) ===
# 格式: [cmd, arg1, arg2...]
# 注意：run_matrix.py 会自动在前面加上 tar -I 和后面加上 -cf ...
COMPRESSION_METHODS = {
    'gzip-1':     ["gzip", "-1"],
    'gzip-6':     ["gzip", "-6"],
    'gzip-9':     ["gzip", "-9"],
    
    'zstd-1':     ["zstd", "-1", "--force"],
    'zstd-3':     ["zstd", "-3", "--force"],
    'zstd-6':     ["zstd", "-6", "--force"],
    
    'lz4-fast':   ["lz4", "-1", "--force"],
    'lz4-medium': ["lz4", "-3", "--force"],
    'lz4-slow':   ["lz4", "-9", "--force"],
    
    'brotli-1':   ["brotli", "-1", "--force"]
}

# === 3. 实验设置 ===
REPETITIONS = 1 
DB_PATH = "experiment_results.db"
TEMP_DIR = "/tmp/cts_experiment_data"
CLIENT_IMAGE = "cts_client_image:latest" 

# === 4. Profile 生成 (带安全熔断) ===
CLIENT_PROFILES = {}

# 固定场景
FIXED_PROFILES = {
    'C1': {'cpu': 0.2, 'mem': '4g', 'bw': '2mbit',   'delay': '100ms', 'desc': '极低性能 (IoT)'},
    'C2': {'cpu': 0.5, 'mem': '4g', 'bw': '20mbit',  'delay': '50ms',  'desc': '低性能 (Edge)'},
    'C3': {'cpu': 1.0, 'mem': '4g', 'bw': '50mbit',  'delay': '20ms',  'desc': '中等性能'},
    'C4': {'cpu': 1.5, 'mem': '4g', 'bw': '100mbit', 'delay': '10ms',  'desc': '中高性能'},
    'C5': {'cpu': 2.0, 'mem': '4g', 'bw': '200mbit', 'delay': '5ms',   'desc': '高性能'},
    'C6': {'cpu': 4.0, 'mem': '4g', 'bw': '500mbit', 'delay': '1ms',   'desc': '顶级性能 (DataCenter)'},
}
CLIENT_PROFILES.update(FIXED_PROFILES)

# 随机场景
NUM_RANDOM_SAMPLES = 80 
random.seed(2026) 

MAX_CPU_LIMIT = 6.0      
MAX_MEM_LIMIT_MB = 10240 

for i in range(NUM_RANDOM_SAMPLES):
    profile_name = f"Train_Rand_{i:03d}"
    cpu = round(random.uniform(0.2, MAX_CPU_LIMIT), 1)
    mem_mb = min(max(2048, int(cpu * 1536)), MAX_MEM_LIMIT_MB)
    mem = f"{mem_mb}m"
    
    r = random.random()
    if r < 0.4: bw_val = random.randint(1, 20)
    elif r < 0.7: bw_val = random.randint(20, 100)
    else: bw_val = random.randint(100, 1000)
    bw = f"{bw_val}mbit"
    
    delay_val = random.randint(5, 400)
    delay = f"{delay_val}ms"
    
    CLIENT_PROFILES[profile_name] = {
        "cpu": cpu, "mem": mem, "bw": bw, "delay": delay, "desc": "Random_Train"
    }

if __name__ == "__main__":
    print(f"✅ 配置已加载: {len(CLIENT_PROFILES)} 组场景")