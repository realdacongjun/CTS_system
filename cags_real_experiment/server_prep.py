#!/usr/bin/env python3
"""
server_prep_real.py (Thesis Final Version)
功能：从真实 Docker 镜像提取 Layer 数据，构建全矩阵实验数据集。
适用：验证 CTS 系统在 Text, Mixed, Binary, OS 四种类型下的泛化性能。
"""

import os
import subprocess
import sys
import tarfile
import shutil
import time

# ==========================================
# 🧪 论文级测试集配置 (Test Set Configuration)
# ==========================================
# 这里的镜像必须与训练集互斥，以证明泛化性
REAL_IMAGES = {
    # 1. 文本密集型 (Text-Heavy) -> 对应 IoT 弱网场景
    # 选用 Perl: 包含大量 .pl 脚本和文档，与 Python/Node 结构相似但内容不同
    'generalized_text.tar': 'perl:latest',
    
    # 2. 混合型 (Mixed-Content) -> 对应 Edge 边缘场景
    # 选用 HAProxy: 典型的 C 语言编写的网络工具，含二进制、配置和文档
    'generalized_mixed.tar': 'haproxy:latest',
    
    # 3. 二进制密集型 (Binary-Heavy) -> 对应 Cloud 强网场景
    # 选用 Redis: 内存数据库，数据结构紧凑，极难压缩
    'generalized_binary.tar': 'redis:latest',
    
    # 4. 操作系统层 (OS-Base) -> 验证对小文件的处理
    # 选用 Alpine: 云原生基座，Musl Libc 架构，区别于 Ubuntu/CentOS
    'generalized_os.tar': 'alpine:latest'
}

# Nginx 默认托管目录
NGINX_ROOT = "/usr/share/nginx/html"

def check_tools():
    """检查必要的系统工具"""
    required = ['docker', 'gzip', 'brotli', 'lz4', 'zstd']
    missing = [t for t in required if shutil.which(t) is None]
    if missing:
        print(f"❌ 错误: 缺少工具: {', '.join(missing)}")
        print("请运行: sudo yum install -y docker gzip brotli lz4 zstd")
        sys.exit(1)

def extract_largest_layer(image_name, output_tar):
    """
    拉取镜像 -> 导出 tar -> 提取最大的 layer.tar
    """
    print(f"\n🐳 正在处理真实镜像: {image_name} ...")
    
    # 1. Docker Pull
    print(f"   -> Pulling {image_name} (可能需要几秒钟)...")
    try:
        subprocess.run(['docker', 'pull', image_name], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"   ❌ Pull 失败: 请检查网络或手动运行 'docker pull {image_name}'")
        sys.exit(1)
    
    # 2. Docker Save
    temp_save_path = f"/tmp/temp_{int(time.time())}.tar"
    print(f"   -> Saving image to {temp_save_path}...")
    subprocess.run(['docker', 'save', '-o', temp_save_path, image_name], check=True)
    
    # 3. 寻找最大的 Layer
    print("   -> 提取最大的 Layer 层...")
    max_size = 0
    largest_layer_member = None
    
    try:
        with tarfile.open(temp_save_path) as tar:
            for member in tar.getmembers():
                # Docker save 的 tar 包结构中，layer 都在子目录下且以 .tar 结尾
                if member.name.endswith('.tar') and '/' in member.name:
                    if member.size > max_size:
                        max_size = member.size
                        largest_layer_member = member
            
            if largest_layer_member:
                f = tar.extractfile(largest_layer_member)
                with open(output_tar, 'wb') as out:
                    shutil.copyfileobj(f, out)
                size_mb = max_size / 1024 / 1024
                print(f"   ✅ 提取成功: {os.path.basename(output_tar)} ({size_mb:.2f} MB)")
            else:
                print("   ❌ 错误: 未在镜像中找到 Layer 文件！")
    except Exception as e:
        print(f"   ❌ 提取过程出错: {e}")
    finally:
        # 清理巨大的临时文件
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)

def compress_file(input_file, output_file, method):
    """生成不同算法的压缩副本"""
    if not os.path.exists(input_file):
        return

    print(f"   -> Compressing to {method}...")
    try:
        if method == 'gzip':
            # -6: 默认均衡
            subprocess.run(['gzip', '-c', '-6', input_file], stdout=open(output_file, 'wb'), check=True)
        elif method == 'brotli':
            # -q 9: 高压缩比 (IoT场景关键)，虽然慢点但值得
            subprocess.run(['brotli', '-q', '9', '-f', '-o', output_file, input_file], check=True)
        elif method == 'lz4':
            # 默认极速 (Cloud场景关键)
            subprocess.run(['lz4', '-f', input_file, output_file], check=True)
        elif method == 'zstd':
            # -3: 均衡模式 (Edge场景关键)
            subprocess.run(['zstd', '-3', '-f', input_file, '-o', output_file], check=True)
    except subprocess.CalledProcessError:
        print(f"   ❌ 压缩失败: {method}")

def main():
    # 权限检查
    if os.geteuid() != 0:
        print("❌ 权限不足: 请使用 sudo 运行此脚本")
        sys.exit(1)
        
    check_tools()
    
    # 确保 Nginx 目录存在
    if not os.path.exists(NGINX_ROOT):
        os.makedirs(NGINX_ROOT, exist_ok=True)

    print("♻️  正在清理旧的实验数据...")
    subprocess.run(f"rm -f {NGINX_ROOT}/generalized*", shell=True)

    # 主循环：提取 + 压缩
    for target_filename, docker_image in REAL_IMAGES.items():
        full_path = os.path.join(NGINX_ROOT, target_filename)
        
        # 1. 提取真实 Layer
        extract_largest_layer(docker_image, full_path)
        
        # 2. 生成 4 种压缩副本
        methods = [
            ('.tar.gz', 'gzip'), 
            ('.tar.br', 'brotli'), 
            ('.tar.lz4', 'lz4'),
            ('.tar.zst', 'zstd')
        ]
        
        for ext, method in methods:
            out_path = full_path.replace('.tar', ext)
            compress_file(full_path, out_path, method)

    print("\n" + "="*50)
    print("✅ 服务端数据准备完毕！(Thesis Ready)")
    print("="*50)
    
    # 列出最终文件，供用户核对
    subprocess.run(f"ls -lh {NGINX_ROOT}/generalized*", shell=True)

if __name__ == '__main__':
    main()