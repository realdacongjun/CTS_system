#!/usr/bin/env python3
"""
server_prep_real.py (Fixed Version)
修复：增强了 Layer 提取逻辑，不再依赖特定的目录结构，而是直接寻找体积最大的文件。
"""

import os
import subprocess
import sys
import tarfile
import shutil
import time

# ==========================================
# 🧪 论文级测试集配置
# ==========================================
REAL_IMAGES = {
    'generalized_text.tar': 'perl:latest',
    'generalized_mixed.tar': 'haproxy:latest',
    'generalized_binary.tar': 'redis:latest',
    'generalized_os.tar': 'alpine:latest'
}

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
    鲁棒性增强版：寻找 tar 包里体积最大的文件作为 Layer
    """
    print(f"\n🐳 正在处理真实镜像: {image_name} ...")
    
    # 1. Docker Pull
    print(f"   -> Pulling {image_name}...")
    try:
        subprocess.run(['docker', 'pull', image_name], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"   ❌ Pull 失败: 请检查网络")
        sys.exit(1)
    
    # 2. Docker Save
    temp_save_path = f"/tmp/temp_{int(time.time())}.tar"
    print(f"   -> Saving image to {temp_save_path}...")
    subprocess.run(['docker', 'save', '-o', temp_save_path, image_name], check=True)
    
    # 3. 寻找最大的文件 (不再依赖文件名必须叫 layer.tar)
    print("   -> 正在扫描包内文件...")
    max_size = 0
    largest_member = None
    
    try:
        with tarfile.open(temp_save_path) as tar:
            all_members = tar.getmembers()
            
            # debug: 打印一下包里有什么，方便排错
            # print(f"      (DEBUG: 包内包含 {len(all_members)} 个文件)")
            
            for member in all_members:
                # 必须是普通文件(isFile)，不能是目录
                if member.isfile():
                    if member.size > max_size:
                        max_size = member.size
                        largest_member = member
            
            if largest_member and max_size > 1024 * 10: # 至少大于 10KB
                f = tar.extractfile(largest_member)
                with open(output_tar, 'wb') as out:
                    shutil.copyfileobj(f, out)
                
                size_mb = max_size / 1024 / 1024
                print(f"   ✅ 提取成功! 找到最大层: {largest_member.name}")
                print(f"      大小: {size_mb:.2f} MB -> 保存为: {os.path.basename(output_tar)}")
            else:
                print("   ❌ 错误: 未在镜像中找到有效的数据文件！")
                print("   📦 包内文件列表 (Debug):")
                for m in all_members:
                    print(f"      - {m.name} ({m.size} bytes)")
    except Exception as e:
        print(f"   ❌ 提取过程出错: {e}")
    finally:
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)

def compress_file(input_file, output_file, method):
    """生成不同算法的压缩副本"""
    if not os.path.exists(input_file):
        return

    print(f"   -> Compressing to {method}...")
    try:
        if method == 'gzip':
            subprocess.run(['gzip', '-c', '-6', input_file], stdout=open(output_file, 'wb'), check=True)
        elif method == 'brotli':
            # 这里的 -q 9 对于大文件可能比较慢，如果你觉得太慢可以改成 -q 5
            subprocess.run(['brotli', '-q', '9', '-f', '-o', output_file, input_file], check=True)
        elif method == 'lz4':
            subprocess.run(['lz4', '-f', input_file, output_file], check=True)
        elif method == 'zstd':
            subprocess.run(['zstd', '-3', '-f', input_file, '-o', output_file], check=True)
    except subprocess.CalledProcessError:
        print(f"   ❌ 压缩失败: {method}")

def main():
    if os.geteuid() != 0:
        print("❌ 权限不足: 请使用 sudo 运行此脚本")
        sys.exit(1)
        
    check_tools()· 
    
    if not os.path.exists(NGINX_ROOT):
        os.makedirs(NGINX_ROOT, exist_ok=True)

    print("♻️  正在清理旧的实验数据...")
    subprocess.run(f"rm -f {NGINX_ROOT}/generalized*", shell=True)

    for target_filename, docker_image in REAL_IMAGES.items():
        full_path = os.path.join(NGINX_ROOT, target_filename)
        extract_largest_layer(docker_image, full_path)
        
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
    subprocess.run(f"ls -lh {NGINX_ROOT}/generalized*", shell=True)

if __name__ == '__main__':
    main()