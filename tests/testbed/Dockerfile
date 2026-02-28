FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /cts

# 🔧 优化1：合并所有 apt-get 安装，只更新一次索引
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

# 创建软链接
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 🔧 优化2：使用国内 PyPI 源 + 只安装 CPU 版 torch + 移除不需要的 torchvision
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

# 配置 sudoers
RUN echo "ALL ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers

# 用户配置
RUN useradd -m ctsuser && chown -R ctsuser:ctsuser /cts
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/cts/src:/cts:$PYTHONPATH

CMD ["sleep", "infinity"]