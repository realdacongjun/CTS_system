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

# 🔧 修复：使用 -sf 强制覆盖已存在的文件
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 国内专属优化：清华源 + CPU版torch
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    requests \
    pandas \
    numpy \
    pyyaml \
    torch==2.0.1 \
    scipy \
    scikit-learn \
    aiohttp

# tc命令免密
RUN echo "ALL ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers

# 用户配置
RUN useradd -m ctsuser && chown -R ctsuser:ctsuser /cts
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/cts/src:/cts:$PYTHONPATH

CMD ["sleep", "infinity"]