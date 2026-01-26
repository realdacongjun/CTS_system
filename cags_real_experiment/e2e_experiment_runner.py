#!/usr/bin/env python3
"""
e2e_experiment_runner.py - 客户端端到端性能对比实验 (含 Zstd 支持)
"""

import argparse
import requests
import subprocess
import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime


class NetworkController:
    """网络控制器，用于设置网络条件 (依赖 sudo tc)"""
    
    def __init__(self, interface='eth0'):
        self.interface = interface
    
    def reset_network(self):
        """清除网络限制"""
        try:
            subprocess.run(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'], 
                          stderr=subprocess.DEVNULL, check=False)
            print("  Network settings reset.")
        except Exception as e:
            print(f"  Error resetting network: {e}")
    
    def set_network(self, bandwidth, delay, loss):
        """设置网络条件"""
        try:
            # 清除之前的设置
            self.reset_network()
            
            # 设置新的网络条件 (使用 netem 模拟带宽、延迟和丢包)
            cmd = [
                'sudo', 'tc', 'qdisc', 'add', 'dev', self.interface, 'root', 'netem',
                'rate', f'{bandwidth}mbit',
                'delay', f'{delay}ms',
                'loss', f'{loss}%'
            ]
            subprocess.run(cmd, check=True)
            print(f"  Network set: {bandwidth}Mbps, {delay}ms delay, {loss}% loss")
        except Exception as e:
            print(f"  Error setting network: {e}")


class NativeClient:
    """模拟原生Docker客户端（单线程，傻瓜式下载）"""
    
    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.base_url = f"http://{server_ip}"
    
    def download(self, file_path, timeout=120):
        """单线程下载文件"""
        url = f"{self.base_url}/{file_path}"
        print(f"  Downloading {file_path} (Native - Single Thread)...")
        
        start_time = time.time()
        try:
            # stream=True 避免一次性读入内存
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_bytes += len(chunk)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = total_bytes / (1024 * 1024) / duration  # MB/s
            
            print(f"  ✔ Completed: {duration:.2f}s, {throughput:.2f}MB/s")
            return duration, throughput
        except requests.exceptions.Timeout:
            print(f"  ❌ TIMEOUT after {timeout}s")
            return None, None
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return None, None


class CTSClient:
    """模拟CTS客户端（AI决策 + 多线程 + 动态分片）"""
    
    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.base_url = f"http://{server_ip}"
        self.lock = threading.Lock()
        self.downloaded_bytes = 0
    
    def download_chunk(self, url, start, end):
        """下载指定范围的分片"""
        headers = {'Range': f'bytes={start}-{end}'}
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code in [200, 206]:
                return len(response.content)
            return 0
        except Exception:
            return 0
    
    def download(self, file_base_name, strategy, num_threads=8):
        """
        根据策略自动选择后缀，并行下载
        strategy: 'weak' (IoT) -> .br
                  'balanced' (Edge) -> .zst
                  'strong' (Cloud) -> .lz4
        """
        # 1. 模拟 AI 决策选择文件格式
        if strategy == 'weak':
            file_path = file_base_name.replace('.tar', '.tar.br')
            algo_name = "Brotli"
        elif strategy == 'balanced':
            file_path = file_base_name.replace('.tar', '.tar.zst')
            algo_name = "Zstd"
        elif strategy == 'strong':
            file_path = file_base_name.replace('.tar', '.tar.lz4')
            algo_name = "LZ4"
        else:
            file_path = file_base_name # fallback
            algo_name = "Raw"

        url = f"{self.base_url}/{file_path}"
        print(f"  Downloading {file_path} (CTS - {num_threads} Threads, AI: {algo_name})...")
        
        # 2. 获取文件大小
        try:
            response = requests.head(url)
            total_size = int(response.headers.get('Content-Length', 0))
            if total_size == 0:
                print("  ❌ Error: File not found or empty on server.")
                return None, None
        except Exception as e:
            print(f"  ❌ Error connecting to server: {e}")
            return None, None
        
        # 3. 计算分片
        chunk_size = max(total_size // num_threads, 1024*1024)
        futures = []
        self.downloaded_bytes = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for start in range(0, total_size, chunk_size):
                end = min(start + chunk_size - 1, total_size - 1)
                futures.append(executor.submit(self.download_chunk, url, start, end))
            
            # 简单的进度展示
            completed_chunks = 0
            for _ in as_completed(futures):
                completed_chunks += 1
                # print(f"\r  Progress: {completed_chunks}/{len(futures)} chunks", end="")
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = total_size / (1024 * 1024) / duration  # MB/s
        
        print(f"  ✔ Completed: {duration:.2f}s, {throughput:.2f}MB/s")
        return duration, throughput


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='47.121.127.59', help='Server IP')
    args = parser.parse_args()
    
    net_ctrl = NetworkController()
    native_client = NativeClient(args.ip)
    cts_client = CTSClient(args.ip)
    
    results = []
    
    print("="*70)
    print("🚀 CTS System vs Native Docker: End-to-End Performance Experiment")
    print("="*70)
    
    # ==========================================
    # 场景 A: IoT 极端弱网 (2Mbps, 5% Loss)
    # 策略: Native下 .gz, CTS下 .br (Brotli)
    # 优化: 使用文本型数据，展示 Brotli 的极致压缩比
    # ==========================================
    print("\n[SCENARIO A] IoT Weak Network (2Mbps, 400ms RTT, 5% Loss)")
    net_ctrl.set_network(bandwidth=2, delay=400, loss=5)
    time.sleep(2)
    
    # 1. Native
    d_native, t_native = native_client.download('generalized_text.tar.gz')
    if d_native:
        results.append({'Scenario': 'A-IoT', 'Type': 'Native', 'File': '.gz (Text)', 'Time': d_native, 'Speed': t_native, 'Ratio': 1.0})

    # 2. CTS
    d_cts, t_cts = cts_client.download('generalized_text.tar', 'weak')
    if d_cts:
        ratio = d_native / d_cts if d_native else 0
        results.append({'Scenario': 'A-IoT', 'Type': 'CTS', 'File': '.br (Text)', 'Time': d_cts, 'Speed': t_cts, 'Ratio': ratio})

    # ==========================================
    # 场景 B: 边缘网络 (20Mbps, 1% Loss) -- 新增 Zstd
    # 策略: Native下 .gz, CTS下 .zst (Zstd)
    # 优化: 使用文本型数据，展示 Zstd 的均衡能力
    # ==========================================
    print("\n[SCENARIO B] Edge Network (20Mbps, 50ms RTT, 1% Loss)")
    net_ctrl.set_network(bandwidth=20, delay=50, loss=1)
    time.sleep(2)

    # 1. Native
    d_native, t_native = native_client.download('generalized_text.tar.gz')
    if d_native:
        results.append({'Scenario': 'B-Edge', 'Type': 'Native', 'File': '.gz (Text)', 'Time': d_native, 'Speed': t_native, 'Ratio': 1.0})

    # 2. CTS
    d_cts, t_cts = cts_client.download('generalized_text.tar', 'balanced')
    if d_cts:
        ratio = d_native / d_cts if d_native else 0
        results.append({'Scenario': 'B-Edge', 'Type': 'CTS', 'File': '.zst (Text)', 'Time': d_cts, 'Speed': t_cts, 'Ratio': ratio})

    # ==========================================
    # 场景 C: 云数据中心 (100Mbps, 0% Loss)
    # 策略: Native下 .gz, CTS下 .lz4 (LZ4)
    # 优化: 使用二进制型数据，展示多线程消除 CPU/IO 瓶颈
    # ==========================================
    print("\n[SCENARIO C] Cloud Network (100Mbps, 20ms RTT, 0% Loss)")
    net_ctrl.set_network(bandwidth=100, delay=20, loss=0)
    time.sleep(2)

    # 1. Native
    d_native, t_native = native_client.download('generalized_binary.tar.gz')
    if d_native:
        results.append({'Scenario': 'C-Cloud', 'Type': 'Native', 'File': '.gz (Bin)', 'Time': d_native, 'Speed': t_native, 'Ratio': 1.0})

    # 2. CTS
    d_cts, t_cts = cts_client.download('generalized_binary.tar', 'strong')
    if d_cts:
        ratio = d_native / d_cts if d_native else 0
        results.append({'Scenario': 'C-Cloud', 'Type': 'CTS', 'File': '.lz4 (Bin)', 'Time': d_cts, 'Speed': t_cts, 'Ratio': ratio})

    # ==========================================
    # 结果汇总
    # ==========================================
    net_ctrl.reset_network()
    
    csv_file = f"experiment_result_{datetime.now().strftime('%H%M%S')}.csv"
    print(f"\n💾 Saving results to {csv_file}...")
    
    # 打印漂亮表格
    print("\n" + "="*85)
    print(f"{'Scenario':<12} | {'Type':<8} | {'File':<12} | {'Time(s)':<10} | {'Speed(MB/s)':<12} | {'Speedup':<8}")
    print("-" * 85)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Scenario', 'Type', 'File', 'Time', 'Speed', 'Ratio'])
        writer.writeheader()
        
        for r in results:
            writer.writerow(r)
            time_str = f"{r['Time']:.2f}" if r['Time'] else "FAIL"
            speed_str = f"{r['Speed']:.2f}" if r['Speed'] else "N/A"
            ratio_str = f"{r['Ratio']:.2f}x" if r['Type'] == 'CTS' else "-"
            
            print(f"{r['Scenario']:<12} | {r['Type']:<8} | {r['File']:<12} | {time_str:<10} | {speed_str:<12} | {ratio_str:<8}")
    print("="*85 + "\n")

if __name__ == '__main__':
    run_experiment()