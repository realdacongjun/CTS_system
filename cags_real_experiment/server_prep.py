#!/usr/bin/env python3
"""
server_prep.py (V2.0) - 生成具有【真实压缩特性】的测试数据
"""

import os
import subprocess
import sys
import random
import string
import time

def generate_realistic_text(filepath, target_mb=100):
    """
    生成模拟服务器日志的文本文件
    特点：有规律的结构 + 随机的内容，压缩率通常在 10:1 到 20:1 之间
    """
    print(f"📄 Generating Realistic Text (Logs): {filepath} ...")
    
    # 定义日志模板
    log_levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
    components = ['AuthService', 'PaymentGate', 'UserDB', 'Frontend']
    messages = [
        "Connection timed out while reaching upstream",
        "User login successful for session_id",
        "Database query took longer than expected",
        "Invalid token provided in header",
        "Cache miss for key user_profile",
        "Garbage collection started",
        "Request received from IP 192.168.1.X"
    ]
    
    with open(filepath, 'w') as f:
        current_size = 0
        target_bytes = target_mb * 1024 * 1024
        
        # 批量写入以提高性能
        buffer = []
        while current_size < target_bytes:
            # 构造一行日志
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            level = random.choice(log_levels)
            comp = random.choice(components)
            msg = random.choice(messages)
            rand_id = random.randint(10000, 99999)
            
            line = f"[{ts}] {level} [{comp}] {msg} - ID:{rand_id}\n"
            buffer.append(line)
            
            if len(buffer) > 1000:
                chunk = "".join(buffer)
                f.write(chunk)
                current_size += len(chunk.encode('utf-8'))
                buffer = []
        
        # 写入剩余buffer
        if buffer:
            f.write("".join(buffer))

def generate_semi_compressible_binary(filepath, target_mb=100):
    """
    生成半可压缩的二进制文件
    原理：混合随机数据和重复数据块，模拟真实的二进制程序/库文件
    压缩率预期：2:1 到 3:1
    """
    print(f"💿 Generating Semi-Compressible Binary: {filepath} ...")
    
    with open(filepath, 'wb') as f:
        target_bytes = target_mb * 1024 * 1024
        current_size = 0
        
        # 生成一个 1MB 的随机块
        random_block = os.urandom(1024 * 1024)
        
        # 循环写入这个块（这样就有重复模式，利于LZ4/Gzip压缩），但每隔一段加点噪音
        while current_size < target_bytes:
            # 写入重复块 (可压缩部分)
            f.write(random_block)
            current_size += len(random_block)
            
            # 写入一点纯随机噪音 (防止压缩率过高)
            noise = os.urandom(1024 * 100) # 100KB noise
            f.write(noise)
            current_size += len(noise)

def compress_file(input_file, output_file, method):
    print(f"   -> Compressing to {method}...")
    if method == 'gzip':
        # -6 是默认均衡模式
        subprocess.run(['gzip', '-c', '-6', input_file], stdout=open(output_file, 'wb'), check=True)
    elif method == 'brotli':
        # -q 5 稍微降低一点质量以加快生成速度，但依然比gzip强
        subprocess.run(['brotli', '-q', '5', '-o', output_file, input_file], check=True)
    elif method == 'lz4':
        subprocess.run(['lz4', '-f', input_file, output_file], check=True)
    elif method == 'zstd':
        subprocess.run(['zstd', '-3', '-f', input_file, '-o', output_file], check=True)

def main():
    nginx_dir = '/usr/share/nginx/html'
    if not os.access(nginx_dir, os.W_OK):
        print("❌ Need root permission (sudo)")
        sys.exit(1)

    # 1. 生成 Text (100MB)
    text_tar = os.path.join(nginx_dir, 'generalized_text.tar')
    generate_realistic_text(text_tar, 100)
    
    # 2. 生成 Binary (100MB)
    bin_tar = os.path.join(nginx_dir, 'generalized_binary.tar')
    generate_semi_compressible_binary(bin_tar, 100)

    # 3. 压缩副本
    files = [text_tar, bin_tar]
    methods = [
        ('.tar.gz', 'gzip'), 
        ('.tar.br', 'brotli'), 
        ('.tar.lz4', 'lz4'),
        ('.tar.zst', 'zstd')
    ]

    for f in files:
        print(f"\nProcessing {os.path.basename(f)}...")
        for ext, method in methods:
            out = f.replace('.tar', ext)
            compress_file(f, out, method)
            
    print("\n✅ Data Generation Complete!")
    subprocess.run(f"ls -lh {nginx_dir}/generalized*", shell=True)

if __name__ == '__main__':
    main()