# config.py

# === 修改前 (会报错) ===
# 'bw': '2m' ... 'bw': '500m'

# === 修改后 (正确写法) ===
# config.py

CLIENT_PROFILES = {
    # === 低配组：模拟弱网、弱算力设备 (如 IoT、边缘节点) ===
    'C1': {
        'cpu': 0.2,      # 极弱 CPU：只有 20% 的算力，解压 Gzip 会非常慢
        'mem': '4g',     # 【关键修改】给足 4GB，保证跑 Rust/Mongo 不死机
        'bw': '2mbit',   # 极低带宽：模型会学到这里必须用高压缩率算法 (Brotli)
        'delay': '100ms',# 高延迟
        'desc': '极低性能 (IoT)'
    },
    'C2': {
        'cpu': 0.5,      # 弱 CPU
        'mem': '4g',     # 【关键修改】4GB
        'bw': '20mbit',  # 低带宽
        'delay': '50ms',
        'desc': '低性能 (Edge)'
    },

    # === 中配组：模拟普通服务器 ===
    'C3': {
        'cpu': 1.0,      # 单核 CPU
        'mem': '4g',     # 【关键修改】4GB
        'bw': '50mbit',  # 中等带宽
        'delay': '20ms',
        'desc': '中等性能'
    },
    'C4': {
        'cpu': 1.5,      # 1.5核
        'mem': '4g',     # 【关键修改】4GB
        'bw': '100mbit', # 百兆带宽
        'delay': '10ms',
        'desc': '中高性能'
    },

    # === 高配组：模拟高性能计算中心/内网 ===
    'C5': {
        'cpu': 2.0,      # 双核强力
        'mem': '4g',     # 【关键修改】4GB
        'bw': '200mbit', # 高带宽
        'delay': '5ms',
        'desc': '高性能'
    },
    'C6': {
        'cpu': 4.0,      # 4核火力全开：模型会学到这里可以用 LZ4 秒解压
        'mem': '4g',     # 【关键修改】4GB
        'bw': '500mbit', # 超高带宽：传输不是瓶颈，解压才是
        'delay': '1ms',
        'desc': '顶级性能 (DataCenter)'
    },
}

# 确保其他配置保持不变...
# === 目标镜像列表 (18个) ===
TARGET_IMAGES = [
    # Linux 发行版
    'centos:latest', 'fedora:latest', 'ubuntu:latest',
    # 数据库
    'mongo:latest', 'mysql:latest', 'postgres:latest',
    # 编程语言
    'rust:latest', 'ruby:latest', 'python:latest',
    # Web 组件
    'nginx:latest', 'httpd:latest', 'tomcat:latest',
    # 应用平台
    'rabbitmq:latest', 'wordpress:latest', 'nextcloud:latest',
    # 应用工具
    'gradle:latest', 'node:latest'
]

# === 压缩算法配置 ===
# 格式: (算法名称, 对应命令参数)
COMPRESSION_METHODS = {
    'gzip-1':     ['gzip', '-1'],
    'gzip-6':     ['gzip', '-6'],
    'gzip-9':     ['gzip', '-9'],
    'zstd-1':     ['zstd', '-1', '--force'],
    'zstd-3':     ['zstd', '-3', '--force'],
    'zstd-6':     ['zstd', '-6', '--force'],
    'lz4-fast':   ['lz4', '-1', '--force'],
    'lz4-medium': ['lz4', '-3', '--force'],
    'lz4-slow':   ['lz4', '-9', '--force'],
    'brotli-1':   ['brotli', '-1', '--force']
}

# === 实验设置 ===
REPETITIONS = 3  # 每个组合重复次数
DB_PATH = "experiment_results.db"
TEMP_DIR = "/tmp/cts_experiment_data"
CLIENT_IMAGE = "cts_client:latest" # 你的客户端Agent镜像名
REGISTRY_URL = "172.17.0.1:5000" # 你的私有仓库地址(如果有)