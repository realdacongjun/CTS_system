#!/bin/bash
set -e  # 遇到错误立即退出

# ==========================================
# CTS 实验镜像准备脚本 (完整版)
# ==========================================

echo "🐳 CTS 实验镜像准备脚本"
echo "================================"

# --------------------------
# 1. 配置区域 (请修改这里)
# --------------------------

# CSV 文件路径
CSV_FILE="./data/image_features_database.csv"

# Harbor 配置
HARBOR_REGISTRY="192.168.1.100/cts-images"  # 改成你的 Harbor IP/项目
HARBOR_USERNAME="admin"
HARBOR_PASSWORD="Harbor12345"

# 镜像选择模式
# - "first3": 自动取 CSV 前3个镜像
# - "manual": 使用下面 MANUAL_IMAGES 定义的列表
SELECT_MODE="first3" 

# 手动指定镜像列表 (当 SELECT_MODE="manual" 时生效)
MANUAL_IMAGES=(
    "ubuntu:latest"
    "nginx:latest"
    "mysql:latest"
)

# --------------------------
# 2. 核心：细粒度压缩算法列表
# (来自你的 run_matrix.py)
# --------------------------
COMPRESSION_ALGORITHMS=(
    "gzip-1"
    "gzip-6"
    "gzip-9"
    "zstd-1"
    "zstd-3"
    "zstd-6"
    "lz4-fast"
    "lz4-medium"
    "lz4-slow"
    "brotli-1"
    "uncompressed" # 基线
)

# --------------------------
# 3. 环境检查
# --------------------------

# 检查 CSV 文件
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ 错误: 找不到 CSV 文件: $CSV_FILE"
    echo "   请确保文件位于: ./data/image_features_database.csv"
    exit 1
fi

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker 命令"
    exit 1
fi

# --------------------------
# 4. 登录 Harbor
# --------------------------
echo ""
echo "🔐 正在登录 Harbor: $HARBOR_REGISTRY..."
docker login $HARBOR_REGISTRY -u $HARBOR_USERNAME -p $HARBOR_PASSWORD || {
    echo "❌ Harbor 登录失败，请检查用户名/密码/网络"
    exit 1
}

# --------------------------
# 5. 读取镜像列表
# --------------------------
echo ""
echo "📖 正在读取镜像列表..."
IMAGE_LIST=()

if [ "$SELECT_MODE" = "first3" ]; then
    echo "   模式: 自动读取 CSV 前3个镜像..."
    # 使用 awk 读取第一列，跳过表头，处理引号
    while IFS=, read -r full_image_name _; do
        # 跳过空行和表头
        if [ -z "$full_image_name" ] || [ "$full_image_name" = "image_name" ]; then
            continue
        fi
        # 去掉可能的双引号
        full_image_name=$(echo "$full_image_name" | tr -d '"')
        IMAGE_LIST+=("$full_image_name")
        echo "   -> 发现镜像: $full_image_name"
        # 只取前3个
        if [ ${#IMAGE_LIST[@]} -ge 3 ]; then
            break
        fi
    done < "$CSV_FILE"
else
    echo "   模式: 使用手动指定列表..."
    IMAGE_LIST=("${MANUAL_IMAGES[@]}")
fi

if [ ${#IMAGE_LIST[@]} -eq 0 ]; then
    echo "❌ 错误: 没有找到任何镜像"
    exit 1
fi

echo ""
echo "✅ 待处理镜像 (${#IMAGE_LIST[@]} 个):"
for img in "${IMAGE_LIST[@]}"; do echo "   - $img"; done

# --------------------------
# 6. 主处理循环
# --------------------------
echo ""
echo "🚀 开始处理 (共 ${#COMPRESSION_ALGORITHMS[@]} 种变体)..."

for source_image in "${IMAGE_LIST[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "处理源镜像: $source_image"
    echo "--------------------------------------------------"

    # 1. 拉取源镜像 (如果本地没有)
    echo "   [1/3] 检查/拉取源镜像..."
    if ! docker inspect "$source_image" > /dev/null 2>&1; then
        echo "   本地未找到，正在拉取..."
        docker pull "$source_image" || {
            echo "   ⚠️  警告: 拉取失败，跳过此镜像 (可能是私有镜像或网络问题)"
            continue
        }
    else
        echo "   本地已存在，跳过拉取"
    fi

    # 2. 解析镜像名 (生成简洁的目标名)
    # 处理像 "quay.io/centos/centos:stream9" 这样的复杂名字
    echo "   [2/3] 解析镜像名称..."
    
    # 提取最后一段 (如 "centos:stream9")
    if [[ "$source_image" == *"/"* ]]; then
        simple_tag_part=$(echo "$source_image" | awk -F'/' '{print $NF}')
    else
        simple_tag_part="$source_image"
    fi
    
    # 分离 name 和 tag
    target_repo_name=$(echo "$simple_tag_part" | cut -d':' -f1)
    original_tag=$(echo "$simple_tag_part" | cut -d':' -f2)
    
    # 如果没有 tag，默认 latest
    if [ -z "$original_tag" ] || [ "$original_tag" = "$target_repo_name" ]; then
        original_tag="latest"
    fi

    echo "   源镜像解析: Repo=$target_repo_name, Tag=$original_tag"

    # 3. 遍历所有压缩算法，打标并推送
    echo "   [3/3] 开始打标推送 (${#COMPRESSION_ALGORITHMS[@]} 个)..."
    
    for algo_suffix in "${COMPRESSION_ALGORITHMS[@]}"; do
        # 构造目标镜像名
        # 格式: HARBOR_REGISTRY/RepoName:OriginalTag-AlgoSuffix
        target_image="${HARBOR_REGISTRY}/${target_repo_name}:${original_tag}-${algo_suffix}"
        
        echo -n "   处理 $algo_suffix... "
        
        # Tag
        docker tag "$source_image" "$target_image"
        
        # Push
        docker push "$target_image" > /dev/null 2>&1 || {
            echo "❌ 推送失败"
            continue
        }
        
        echo "✅"
    done
done

echo ""
echo "=================================================="
echo "🎉 所有镜像准备完成！"
echo "=================================================="
echo ""
echo "📝 生成的镜像格式示例:"
echo "   $HARBOR_REGISTRY/ubuntu:latest-gzip-6"
echo "   $HARBOR_REGISTRY/nginx:latest-zstd-3"
echo ""
echo "💡 下一步: 请确保你的 DockerExecutor 能正确拼接 '-algo' 后缀"