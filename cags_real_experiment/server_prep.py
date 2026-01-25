#!/usr/bin/env python3
"""
server_prep.py - 在服务器上准备端到端实验所需的测试数据
"""

import os
import subprocess
import sys
import random
import string


def generate_dummy_tar(base_filename, size_mb=100, entropy_type='binary'):
    """
    生成指定大小的虚拟tar文件
    entropy_type: 'text' 表示低熵数据，'binary' 表示高熵数据
    """
    print(f"Generating {base_filename} ({size_mb}MB) with {entropy_type} entropy...")
    
    # 生成文件内容
    if entropy_type == 'text':
        # 低熵：生成类似文本的内容
        content = ''.join(random.choices(string.ascii_lowercase + ' \n\t', k=1024)).encode() * (size_mb * 1024)
    else:
        # 高熵：生成随机字节
        content = os.urandom(size_mb * 1024 * 1024)
    
    # 写入文件
    with open(base_filename, 'wb') as f:
        f.write(content)
    
    print(f"Created {base_filename}")


def compress_file(input_file, output_file, method):
    """
    使用指定方法压缩文件
    """
    print(f"Compressing {input_file} -> {output_file} using {method}...")
    
    if method == 'gzip':
        cmd = ['gzip', '-c', input_file]
        with open(output_file, 'wb') as f_out:
            subprocess.run(cmd, stdout=f_out, check=True)
    elif method == 'brotli':
        cmd = ['brotli', '-q', '11', '-o', output_file, input_file]
        subprocess.run(cmd, check=True)
    elif method == 'lz4':
        cmd = ['lz4', '-f', input_file, output_file]
        subprocess.run(cmd, check=True)
    elif method == 'zstd':
        # -3 是默认级别，最均衡；-f 是强制覆盖
        cmd = ['zstd', '-3', '-f', input_file, '-o', output_file]
        subprocess.run(cmd, check=True)
    
    print(f"Created {output_file}")


def main():
    nginx_dir = '/usr/share/nginx/html'
    
    # 检查是否具有写权限
    if not os.access(nginx_dir, os.W_OK):
        print(f"Error: Cannot write to {nginx_dir}. Are you running as root?")
        sys.exit(1)
    
    # 文件配置
    files_to_create = [
        ('generalized_text.tar', 'text'),
        ('generalized_binary.tar', 'binary')
    ]
    
    for base_name, entropy_type in files_to_create:
        full_path = os.path.join(nginx_dir, base_name)
        
        # 检查基础文件是否存在
        if not os.path.exists(full_path):
            # 生成基础文件
            temp_file = f'/tmp/{base_name}'
            generate_dummy_tar(temp_file, size_mb=100, entropy_type=entropy_type)
            
            # 移动到nginx目录
            os.rename(temp_file, full_path)
        
        # 生成压缩版本
        compress_methods = [
            ('.tar.gz', 'gzip'),
            ('.tar.br', 'brotli'),
            ('.tar.lz4', 'lz4'),
            ('.tar.zst', 'zstd')  # 新增 Zstd 支持
        ]
        
        for ext, method in compress_methods:
            compressed_file = os.path.join(nginx_dir, base_name.replace('.tar', ext))
            if not os.path.exists(compressed_file):
                compress_file(full_path, compressed_file, method)
    
    print("All test files prepared successfully!")


if __name__ == '__main__':
    main()