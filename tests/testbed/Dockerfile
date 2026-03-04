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

# 基础镜像：锁定Ubuntu 22.04哈希值（杜绝tag漂移），支撑Python 3.10 + NumPy 2.0.2
FROM ubuntu:22.04@sha256:2b7412e6465c3c7fc5bb21d3e6f1917c167358449fecac8176c6e496e5c1f05f2

# 核心环境变量：解决时区、交互模式、编码、pip配置等基础问题
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=5 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# 工作目录（提前创建，确保Root权限）
WORKDIR /cts

# 1. 安装系统依赖（精简+稳定+Root适配）
# 添加时区/字体依赖（解决matplotlib报错），保留docker.io（若需容器内运行Docker）
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
    tzdata \
    fonts-dejavu \
    # 软链接Python/pip（Root下全局生效）
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    # 清理缓存，大幅减少镜像体积（Root下清理更彻底）
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 2. 创建业务目录（Root权限，确保读写）
RUN mkdir -p /cts/logs /cts/results /cts/experiments /cts/configs /cts/src \
    && chmod -R 755 /cts  # 保证目录全局可访问，Root下读写无限制

# 3. 安装Python依赖（全量兼容NumPy 2.0.2，Root下全局安装）
# 合并pip命令提升效率，指定所有库精确版本（适配NumPy 2.0.2）
RUN pip install --no-cache-dir --upgrade pip==24.0 \
    && pip install --no-cache-dir \
    numpy==2.0.2 \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    psutil==5.9.5 \
    pandas==2.1.4 \
    scipy==1.14.0 \          
    scikit-learn==1.5.1 \    
    torch==2.3.1+cpu \       
    aiohttp==3.9.1 \
    typing-extensions==4.8.0 \
    matplotlib==3.9.2 \     
    # 清理pip缓存（Root下清理用户级+系统级缓存）
    && rm -rf ~/.cache/pip /root/.cache/pip

# 完善Python路径（Root下全局生效）
ENV PYTHONPATH=/cts/src:/cts:/cts/testbed:$PYTHONPATH \
    PATH=/root/.local/bin:$PATH

# 容器保活（Root下更稳定的保活方式）
CMD ["tail", "-f", "/dev/null"]

# 可维护性标签（建议添加）
LABEL maintainer="your-name@example.com" \
      version="1.0.0" \
      description="CTS实验环境 | Root执行 | Ubuntu 22.04 + Python 3.10 + NumPy 2.0.2"