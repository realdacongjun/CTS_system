#!/bin/bash

# CTS闭环系统实验环境初始化脚本
# 需要root权限执行

echo "🚀 开始初始化CTS实验环境..."

# 检查root权限
if [ "$EUID" -ne 0 ]; then
  echo "❌ 请以root权限执行此脚本: sudo $0"
  exit 1
fi

# 更新系统包
echo "🔄 更新系统包..."
apt-get update && apt-get upgrade -y

# 安装必要依赖
echo "📦 安装Python依赖..."
pip install torch pandas numpy matplotlib seaborn scipy scikit-learn psutil pyyaml

# 安装网络工具
echo "🌐 安装网络工具..."
apt-get install -y iproute2 iputils-ping net-tools

# 检查Docker安装
if ! command -v docker &> /dev/null; then
    echo "🐳 安装Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker $SUDO_USER
fi

# 启动Docker服务
systemctl start docker
systemctl enable docker

# 创建必要的目录结构
echo "📁 创建目录结构..."
mkdir -p ./configs
mkdir -p ./data/raw
mkdir -p ./data/processed
mkdir -p ./figures
mkdir -p ./logs

# 设置权限
chmod -R 755 ./data
chmod -R 755 ./figures
chmod -R 755 ./logs

echo "✅ 环境初始化完成！"
echo "💡 请确保已配置Harbor私有仓库，并修改configs/global_config.yaml中的相关参数"