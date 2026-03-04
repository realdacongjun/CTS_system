# # 基础镜像：Ubuntu 22.04（Python 3.10，支撑 NumPy 2.0.2）
# FROM ubuntu:22.04

# ENV DEBIAN_FRONTEND=noninteractive
# WORKDIR /cts

# # 1. 安装系统依赖（无修改）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     iproute2 \
#     sudo \
#     curl \
#     iputils-ping \
#     zstd \
#     gzip \
#     lz4 \
#     brotli \
#     pigz \
#     python3 \
#     python3-pip \
#     python3-dev \
#     docker.io \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # 2. 软链接 Python/pip（无修改）
# RUN ln -sf /usr/bin/python3 /usr/bin/python && \
#     ln -sf /usr/bin/pip3 /usr/bin/pip

# # 3. 创建用户+权限配置（无修改）
# RUN useradd -m ctsuser && \
#     echo "ctsuser ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers && \
#     mkdir -p /cts/logs /cts/results /cts/experiments /cts/configs /cts/src && \
#     chown -R ctsuser:ctsuser /cts && \
#     chmod -R 755 /cts

# # 4. 安装 Python 依赖（全量兼容 NumPy 2.0.2）
# RUN pip install --no-cache-dir --upgrade pip==24.0 -i https://mirrors.aliyun.com/pypi/simple/ && \
#     pip install --no-cache-dir --default-timeout=1000 --retries=5 \
#     -i https://mirrors.aliyun.com/pypi/simple/ \
#     numpy==2.0.2 && \
#     pip install --no-cache-dir --default-timeout=1000 --retries=5 \
#     -i https://mirrors.aliyun.com/pypi/simple/ \
#     --extra-index-url https://download.pytorch.org/whl/cpu \
#     requests==2.31.0 \
#     pyyaml==6.0.1 \
#     psutil==5.9.5 \
#     pandas==2.1.4 \
#     scipy==1.11.4 \
#     scikit-learn==1.3.2 \
#     torch==2.0.1+cpu \
#     aiohttp==3.9.1 \
#     typing-extensions==4.8.0 \
#     matplotlib

# # 5. 切换用户+环境变量（无修改）
# USER ctsuser

# ENV PYTHONIOENCODING=utf-8
# ENV LANG=C.UTF-8
# ENV PYTHONPATH=/cts/src:/cts:/cts/testbed:$PYTHONPATH

# CMD ["sleep", "infinity"]

# 基础镜像：Ubuntu 22.04（Python 3.10）
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /cts

# 1️⃣ 安装系统依赖
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

# 2️⃣ 软链接 python / pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 3️⃣ 创建用户 + 目录权限
RUN useradd -m ctsuser && \
    echo "ctsuser ALL=(ALL) NOPASSWD: /usr/sbin/tc" >> /etc/sudoers && \
    mkdir -p /cts/logs /cts/results /cts/experiments /cts/configs /cts/src && \
    chown -R ctsuser:ctsuser /cts && \
    chmod -R 755 /cts

# 4️⃣ 升级 pip（使用清华源）
RUN pip install --no-cache-dir --upgrade pip \
    -i https://pypi.tuna.tsinghua.edu.cn/simple


RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.12.0 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    psutil==5.9.5 \
    aiohttp==3.9.1 \
    typing-extensions==4.8.0 \
    torch==2.2.2+cpu
# 6️⃣ 切换用户
USER ctsuser

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/cts/src:/cts:/cts/testbed:$PYTHONPATH

CMD ["sleep", "infinity"]