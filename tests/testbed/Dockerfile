FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /cts

# 合并所有 apt-get 安装，只更新一次索引
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    sudo \
    curl \
    iputils-ping \
    zstd \
    gzip \
    lz4 \
    brotli \
    pigz \
    python3 \
    python3-pip \
    python3-dev \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 强制覆盖软链接，避免冲突
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 🔧 核心修复1：添加 psutil 依赖（解决 ModuleNotFoundError）
# 🔧 核心修复2：锁定 pyyaml=6.0.1（支持 ensure_ascii 参数，解决 dump 报错）
RUN pip install --no-cache-dir --upgrade pip==22.0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    requests==2.31.0 \
    pandas==1.5.3 \
    numpy==1.24.4 \
    pyyaml==6.0.1 \          
    psutil==5.9.5 \          
    torch==1.13.1 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    aiohttp==3.8.6 \
    typing-extensions==4.5.0

# 🔧 核心修复3：tc命令免密（确保普通用户也能执行网络配置）
RUN echo "ctsuser ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers && \
    # 修复 /cts 目录权限（彻底解决日志写入权限问题）
    mkdir -p /cts/logs /cts/results /cts/experiments && \
    chown -R ctsuser:ctsuser /cts && \
    chmod -R 755 /cts

# 用户配置
RUN useradd -m ctsuser 
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/cts/src:/cts:/cts/testbed:$PYTHONPATH  
CMD ["sleep", "infinity"]