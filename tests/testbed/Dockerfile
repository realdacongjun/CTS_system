FROM ubuntu:22.04

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

# 🔧 步骤1：先创建用户（关键！解决 chown 无效用户问题）
RUN useradd -m ctsuser

# 🔧 步骤2：配置权限（此时 ctsuser 已存在）
RUN echo "ctsuser ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers && \
    # 提前创建目录并赋予权限
    mkdir -p /cts/logs /cts/results /cts/experiments /cts/configs /cts/src && \
    chown -R ctsuser:ctsuser /cts && \
    chmod -R 755 /cts


# 🔧 优化 pip 下载参数，解决超时问题
RUN pip install --no-cache-dir --upgrade pip==22.0.4 -i https://mirrors.aliyun.com/pypi/simple/ && \
    # 先单独下载 NumPy（大文件优先，增加超时和重试）
    pip install --no-cache-dir --default-timeout=1000 --retries=5 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    numpy==2.0.2 && \
    # 再装其他依赖
    pip install --no-cache-dir --default-timeout=1000 --retries=5 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    requests==2.31.0 \
    pandas==1.5.3 \
    pyyaml==6.0.1 \
    psutil==5.9.5 \         
    torch==1.13.1 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    aiohttp==3.8.6 \
    typing-extensions==4.5.0

# 切换到普通用户
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/cts/src:/cts:/cts/testbed:$PYTHONPATH

CMD ["sleep", "infinity"]


