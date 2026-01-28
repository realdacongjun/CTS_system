#!/usr/bin/env python3
"""
grid_search_thesis.py - 毕业设计补充实验：传输层参数敏感性分析
目的：证明 "对于特定的压缩算法和网络环境，存在一个局部最优的并发线程数"。
输出：sensitivity_data.csv -> 用于绘制论文中的参数分析图
"""

import requests
import time
import csv
import subprocess
import statistics
import os
from concurrent.futures import ThreadPoolExecutor

# ================= 🔧 配置区域 =================
# 1. 你的服务端 IP
SERVER_IP = "47.121.127.59"  
BASE_URL = f"http://{SERVER_IP}"

# 2. 测试用的基准文件 (混合型镜像最具代表性)
# 请确保服务端目录下存在 .br, .zst, .lz4 后缀的该文件
TEST_FILE_NAME = "generalized_mixed.tar"

# 3. 定义关键测试路径 (剪枝策略：只测 AI 会选的组合)
TEST_CASES = [
    {
        'id': 'Scenario_A_IoT',
        'desc': 'Weak Network (2Mbps, 400ms, 5%)',
        'net_config': {'bw': 2, 'delay': 400, 'loss': 5},
        'fixed_algo': '.br',   # 在弱网下，AI 必定选 Brotli
        'threads_scope': [1, 2, 4, 8] # 扫描范围
    },
    {
        'id': 'Scenario_B_Edge',
        'desc': 'Edge Network (20Mbps, 50ms, 1%)',
        'net_config': {'bw': 20, 'delay': 50, 'loss': 1},
        'fixed_algo': '.zst',  # 在边缘网下，AI 必定选 Zstd
        'threads_scope': [1, 2, 4, 8, 16]
    },
    {
        'id': 'Scenario_C_Cloud',
        'desc': 'Cloud Network (100Mbps, 20ms, 0%)',
        'net_config': {'bw': 100, 'delay': 20, 'loss': 0},
        'fixed_algo': '.lz4',  # 在强网下，AI 必定选 LZ4
        'threads_scope': [1, 2, 4, 8, 16]
    }
]
# ==============================================

class GridSearcher:
    def set_network(self, config):
        """利用 tc 设置网络环境"""
        # 先清除旧规则
        subprocess.run(['sudo', 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'], stderr=subprocess.DEVNULL)
        # 添加新规则
        cmd = [
            'sudo', 'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'netem',
            'rate', f"{config['bw']}mbit", 
            'delay', f"{config['delay']}ms", 
            'loss', f"{config['loss']}%"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"   ⚡ [Network Set] {config['bw']}Mbps, {config['delay']}ms, {config['loss']}% Loss")
            time.sleep(2) # 等待规则生效
        except Exception as e:
            print(f"   ❌ Network Error: {e}")

    def reset_network(self):
        """重置网络"""
        subprocess.run(['sudo', 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'], stderr=subprocess.DEVNULL)

    def probe_speed(self, url, concurrency):
        """执行一次测速 (不写磁盘)"""
        try:
            # 1. 获取文件大小
            head = requests.head(url, timeout=5)
            total_size = int(head.headers.get('Content-Length', 0))
            if total_size == 0: return None

            # 2. 【核心逻辑一致性】
            # 必须使用与 CTSClient 一致的 "Total/N" 分片策略
            # 这样测出来的数据才能支撑主实验
            chunk_size = max(total_size // concurrency, 1024*1024)

            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for s in range(0, total_size, chunk_size):
                    e = min(s + chunk_size - 1, total_size - 1)
                    # 只下载数据流，丢弃到内存/空处，纯测传输性能
                    futures.append(executor.submit(
                        requests.get, url, 
                        headers={'Range': f'bytes={s}-{e}'}, 
                        timeout=30
                    ))
                
                # 等待所有分片完成
                for f in futures:
                    resp = f.result()
                    if resp.status_code not in [200, 206]:
                        return None
            
            duration = time.time() - start_time
            return duration
        except Exception as e:
            # print(f"Probe failed: {e}")
            return None

def run():
    searcher = GridSearcher()
    results = []
    output_file = "thesis_sensitivity_data.csv"

    print("="*60)
    print("🔬 Thesis Experiment: Conditional Parameter Sensitivity Analysis")
    print("   (验证不同场景下的最优线程数)")
    print("="*60)

    try:
        for case in TEST_CASES:
            print(f"\n📂 Context: {case['desc']}")
            print(f"   ℹ️  Fixed Algorithm: {case['fixed_algo']} (AI Decision)")
            
            # 1. 设置环境
            searcher.set_network(case['net_config'])
            target_url = f"{BASE_URL}/{TEST_FILE_NAME}{case['fixed_algo']}"
            
            best_time = 9999
            best_thread = -1

            # 2. 扫描参数
            for n in case['threads_scope']:
                print(f"   -> Testing Threads = {n} ... ", end='', flush=True)
                
                # 为了数据准确，每个点测 2 次取平均
                samples = []
                for _ in range(2):
                    t = searcher.probe_speed(target_url, n)
                    if t: samples.append(t)
                
                if samples:
                    avg_time = statistics.mean(samples)
                    print(f"{avg_time:.2f}s")
                    
                    results.append({
                        'Scenario': case['id'],
                        'Algorithm': case['fixed_algo'],
                        'Threads': n,
                        'Time_s': round(avg_time, 2)
                    })
                    
                    if avg_time < best_time:
                        best_time = avg_time
                        best_thread = n
                else:
                    print("Failed (Timeout/Error)")
                    results.append({
                        'Scenario': case['id'],
                        'Algorithm': case['fixed_algo'],
                        'Threads': n,
                        'Time_s': 'Failed'
                    })

            print(f"   🏆 Optimal Concurrency: {best_thread}")

    finally:
        searcher.reset_network()
        print("\n⚡ Network Reset.")

    # 保存数据
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Scenario', 'Algorithm', 'Threads', 'Time_s'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Data saved to {output_file}")
        print("   -> Now use this CSV to plot the 'Sensitivity Analysis' graph.")
    except Exception as e:
        print(f"❌ Save failed: {e}")

if __name__ == '__main__':
    run()