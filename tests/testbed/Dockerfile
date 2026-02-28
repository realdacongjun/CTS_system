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

# 🔧 核心修复：指定兼容 Python 3.8 的依赖版本
RUN pip install --no-cache-dir --upgrade pip==22.0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    requests==2.31.0 \
    pandas==1.5.3 \
    numpy==1.24.4 \
    pyyaml==6.0.1 \
    torch==1.13.1 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    aiohttp==3.8.6 \
    typing-extensions==4.5.0

# tc命令免密
RUN echo "ALL ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers

# 用户配置
RUN useradd -m ctsuser && chown -R ctsuser:ctsuser /cts
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/cts/src:/cts:$PYTHONPATH

CMD ["sleep", "infinity"]