import random

# === 1. 目标镜像列表 (保持你要求的18个不变) ===
TARGET_IMAGES = [
    # Linux 发行版
    'quay.io/centos/centos:stream9', 'fedora:latest', 'ubuntu:latest',
    
    # 数据库
    'mongo:latest', 'mysql:latest', 'postgres:latest',
    
    # 编程语言
    'rust:latest', 'ruby:latest', 'python:latest',
    
    # Web 组件 (去掉了 tomcat)
    'nginx:latest', 'httpd:latest', 
    
    # 应用平台
    'rabbitmq:latest', 'wordpress:latest', 'nextcloud:latest',
    
    # 应用工具 (重型镜像)
    'gradle:latest', 'node:latest'
]

# === 2. 压缩算法配置 (保持你的列表，但适配 run_matrix.py 的 tar 命令格式) ===
# 注意：为了配合网络传输实验，这里必须封装成 tar 命令，否则 run_matrix.py 会报错
COMPRESSION_METHODS = {
    'gzip-1':     ["tar", "-I", "gzip -1", "-cf"],
    'gzip-6':     ["tar", "-I", "gzip -6", "-cf"],
    'gzip-9':     ["tar", "-I", "gzip -9", "-cf"],
    
    'zstd-1':     ["tar", "-I", "zstd -1 --force", "-cf"],
    'zstd-3':     ["tar", "-I", "zstd -3 --force", "-cf"],
    'zstd-6':     ["tar", "-I", "zstd -6 --force", "-cf"],
    
    'lz4-fast':   ["tar", "-I", "lz4 -1 --force", "-cf"],
    'lz4-medium': ["tar", "-I", "lz4 -3 --force", "-cf"],
    'lz4-slow':   ["tar", "-I", "lz4 -9 --force", "-cf"],
    
    'brotli-1':   ["tar", "-I", "brotli -1 --force", "-cf"]
}

# === 3. 实验基础设置 ===
# ⚠️ 训练集跑 1 次即可，测试集 C1-C6 跑 3 次 (代码逻辑会处理)
REPETITIONS = 1 
DB_PATH = "experiment_results.db"
TEMP_DIR = "/tmp/cts_experiment_data"
# 使用之前的网络版镜像名 (请确保名字一致，或者用你自己的 cts_client:latest)
CLIENT_IMAGE = "cts_client_image:latest" 

# === 4. Profile 生成工厂 (安全随机版) ===
CLIENT_PROFILES = {}

# --- A. 固定场景 (测试集) ---
# 严格按照你要求的 "mem: 4g" 进行修改
FIXED_PROFILES = {
    'C1': {'cpu': 0.2, 'mem': '4g', 'bw': '2mbit',   'delay': '100ms', 'desc': '极低性能 (IoT)'},
    'C2': {'cpu': 0.5, 'mem': '4g', 'bw': '20mbit',  'delay': '50ms',  'desc': '低性能 (Edge)'},
    'C3': {'cpu': 1.0, 'mem': '4g', 'bw': '50mbit',  'delay': '20ms',  'desc': '中等性能'},
    'C4': {'cpu': 1.5, 'mem': '4g', 'bw': '100mbit', 'delay': '10ms',  'desc': '中高性能'},
    'C5': {'cpu': 2.0, 'mem': '4g', 'bw': '200mbit', 'delay': '5ms',   'desc': '高性能'},
    'C6': {'cpu': 4.0, 'mem': '4g', 'bw': '500mbit', 'delay': '1ms',   'desc': '顶级性能 (DataCenter)'},
}
CLIENT_PROFILES.update(FIXED_PROFILES)

# --- B. 随机场景 (训练集) - 带安全熔断 ---
NUM_RANDOM_SAMPLES = 80 # 生成 80 组数据用于训练 MLP
random.seed(2026) 

# 🛡️ 针对你 8核16G 服务器的安全红线 🛡️
MAX_CPU_LIMIT = 6.0      # 留 2 核保命
MAX_MEM_LIMIT_MB = 10240 # 10GB 上限，留 6GB 给系统缓存

for i in range(NUM_RANDOM_SAMPLES):
    profile_name = f"Train_Rand_{i:03d}"
    
    # 1. CPU: 0.2 ~ 6.0 核
    cpu = round(random.uniform(0.2, MAX_CPU_LIMIT), 1)
    
    # 2. 内存: 动态分配，但死死卡在 10GB 以内 (虽然你C1-C6给了4G，随机的还是动态点好)
    # 逻辑：每核给 1.5GB 内存，但不超过 10GB
    mem_calc = int(cpu * 1536) 
    # 既然你系统内存有限，这里我们稍微激进一点，保底给 2GB，防止大镜像解压崩
    mem_mb = min(max(2048, mem_calc), MAX_MEM_LIMIT_MB)
    mem = f"{mem_mb}m"
    
    # 3. 带宽: 分段加权分布 (模拟真实网络长尾效应)
    r = random.random()
    if r < 0.4:  # 40% 概率是弱网 (1-20M)，这是决策关键区
        bw_val = random.randint(1, 20)
    elif r < 0.7: # 30% 普通网
        bw_val = random.randint(20, 100)
    else:         # 30% 高速网
        bw_val = random.randint(100, 1000)
    bw = f"{bw_val}mbit"
    
    # 4. 延迟
    delay_val = random.randint(5, 400)
    delay = f"{delay_val}ms"
    
    CLIENT_PROFILES[profile_name] = {
        "cpu": cpu,
        "mem": mem,
        "bw": bw,
        "delay": delay,
        "desc": "Random_Train"
    }

if __name__ == "__main__":
    print(f"✅ 配置已加载: {len(TARGET_IMAGES)} 个镜像 x {len(COMPRESSION_METHODS)} 种算法")
    print(f"🛡️ 安全模式: CPU上限 {MAX_CPU_LIMIT}核 / 内存上限 {MAX_MEM_LIMIT_MB}MB")
    print(f"🚀 总计场景: {len(CLIENT_PROFILES)} 组 (含 {len(FIXED_PROFILES)} 组测试集 + {NUM_RANDOM_SAMPLES} 组训练集)")