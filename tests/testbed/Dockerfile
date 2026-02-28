# testbed/Dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /cts

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    sudo \
    curl iputils-ping \
    zstd gzip lz4 brotli \
    python3 python3-pip python3-dev \
    docker.io \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# 确保在 Dockerfile 中添加了这些
RUN apt-get update && apt-get install -y --no-install-recommends \
    pigz `# 并行 gzip` \
    zstd `# zstd 算法` \
    lz4  `# lz4 算法` \
    brotli `# brotli 算法`

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 🔧 修复：移除不存在的包 (python-logging, perf-counter)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    requests \
    pandas numpy \
    pyyaml \
    torch torchvision \
    scipy scikit-learn \
    aiohttp

RUN echo "ALL ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers

RUN useradd -m ctsuser && chown -R ctsuser:ctsuser /cts
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
# 🔧 修复：添加 src 到 PYTHONPATH
ENV PYTHONPATH=/cts/src:/cts:$PYTHONPATH

CMD ["sleep", "infinity"]