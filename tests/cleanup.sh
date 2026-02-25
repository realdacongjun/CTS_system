#!/bin/bash
echo "🧹 开始清理 CTS 实验环境..."

# 1. 杀进程
echo "   停止实验进程..."
sudo pkill -9 -f run_experiment.py 2>/dev/null

# 2. 清理 Docker
echo "   清理 Docker 容器和网络..."
docker rm -f cts-client-worker 2>/dev/null
docker network rm cts-net 2>/dev/null

# 3. 清理文件
echo "   清理日志和结果..."
cd /root/CTS_system/tests
rm -rf logs/
# rm -rf results/  # 这行注释掉，防止误删数据，想删再解开

echo "✅ 清理完成！"