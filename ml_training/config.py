# config.py

# === 修改前 (会报错) ===
# 'bw': '2m' ... 'bw': '500m'

# === 修改后 (正确写法) ===
CLIENT_PROFILES = {
    'C1': {'cpu': 0.2, 'mem': '512m', 'bw': '2mbit',   'delay': '100ms', 'desc': '极低性能'},
    'C2': {'cpu': 0.5, 'mem': '1g',   'bw': '20mbit',  'delay': '50ms',  'desc': '低性能'},
    'C3': {'cpu': 1.0, 'mem': '2g',   'bw': '5mbit',   'delay': '20ms',  'desc': '中等性能'},
    'C4': {'cpu': 1.5, 'mem': '2g',   'bw': '50mbit',  'delay': '10ms',  'desc': '中高性能'},
    'C5': {'cpu': 0.8, 'mem': '2g',   'bw': '100mbit', 'delay': '5ms',   'desc': '高性能'},
    'C6': {'cpu': 4.0, 'mem': '4g',   'bw': '500mbit', 'delay': '1ms',   'desc': '顶级性能'},
}
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
    'gradle:latest', 'logstash:latest', 'node:latest'
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