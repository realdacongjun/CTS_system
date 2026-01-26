#!/usr/bin/env python3
"""
e2e_runner_thesis.py - 毕业设计专用：全矩阵 + 3次重复 + 统计分析
"""

import argparse
import requests
import subprocess
import time
import csv
import os
import statistics  # 用于计算平均值和标准差
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =================配置区域=================
# 重复次数：学术实验通常建议 3次 或 5次
REPEAT_COUNT = 3 

# 四大测试镜像
TEST_IMAGES = [
    {'name': 'Perl (Text)',    'file': 'generalized_text.tar'},
    {'name': 'HAProxy (Mix)',  'file': 'generalized_mixed.tar'},
    {'name': 'Redis (Bin)',    'file': 'generalized_binary.tar'},
    {'name': 'Alpine (OS)',    'file': 'generalized_os.tar'}
]

# 三大网络场景
SCENARIOS = [
    {'name': 'A-IoT',   'bw': 2,   'delay': 400, 'loss': 5, 'strategy': 'weak'},
    {'name': 'B-Edge',  'bw': 20,  'delay': 50,  'loss': 1, 'strategy': 'balanced'},
    {'name': 'C-Cloud', 'bw': 100, 'delay': 20,  'loss': 0, 'strategy': 'strong'}
]
# =========================================

class NetworkController:
    def __init__(self, interface='eth0'):
        self.interface = interface
    
    def set_network(self, bw, delay, loss):
        # 先清除旧规则
        subprocess.run(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'], 
                      stderr=subprocess.DEVNULL, check=False)
        # 设置新规则
        cmd = [
            'sudo', 'tc', 'qdisc', 'add', 'dev', self.interface, 'root', 'netem',
            'rate', f'{bw}mbit', 'delay', f'{delay}ms', 'loss', f'{loss}%'
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"  ⚡ [Network] Set to {bw}Mbps, {delay}ms, {loss}% loss")
        except Exception as e:
            print(f"  ❌ [Network] Error: {e}")

    def reset(self):
        subprocess.run(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'], 
                      stderr=subprocess.DEVNULL, check=False)

class NativeClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def download(self, filename):
        target = f"{filename}.gz"
        url = f"{self.base_url}/{target}"
        start = time.time()
        try:
            # 缩短超时时间到 300s，避免弱网下卡太久
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            size = 0
            for chunk in resp.iter_content(8192): size += len(chunk)
            dur = time.time() - start
            return dur
        except Exception as e:
            return None

class CTSClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def download_chunk(self, url, start, end):
        try:
            h = {'Range': f'bytes={start}-{end}'}
            r = requests.get(url, headers=h, timeout=30)
            return len(r.content)
        except: return 0

    def download(self, filename, strategy):
        suffix = '.lz4'
        if strategy == 'weak': suffix = '.br'
        elif strategy == 'balanced': suffix = '.zst'
        
        target = f"{filename}{suffix}"
        url = f"{self.base_url}/{target}"
        
        try:
            head = requests.head(url, timeout=10)
            total = int(head.headers.get('Content-Length', 0))
        except: return None

        pool_size = 8
        chunk_size = max(total // pool_size, 1024*1024)
        futures = []
        start_t = time.time()
        
        with ThreadPoolExecutor(pool_size) as ex:
            for s in range(0, total, chunk_size):
                e = min(s + chunk_size - 1, total - 1)
                futures.append(ex.submit(self.download_chunk, url, s, e))
            for f in as_completed(futures): pass
            
        dur = time.time() - start_t
        return dur

def get_stats(data_list):
    """计算平均值和标准差"""
    if not data_list: return 0, 0
    if len(data_list) == 1: return data_list[0], 0
    return statistics.mean(data_list), statistics.stdev(data_list)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True, help="Server IP")
    args = parser.parse_args()

    net = NetworkController()
    native = NativeClient(f"http://{args.ip}")
    cts = CTSClient(f"http://{args.ip}")
    
    results = []
    
    print("="*60)
    print(f"🎓 Thesis Experiment: Full Matrix x {REPEAT_COUNT} Repeats")
    print("="*60)
    print("⚠️  Estimated time: 1.5 - 2 Hours. Do not close terminal.\n")

    try:
        for scen in SCENARIOS:
            print(f"\n🌍 [SCENARIO: {scen['name']}]")
            net.set_network(scen['bw'], scen['delay'], scen['loss'])
            time.sleep(2)
            
            for img in TEST_IMAGES:
                print(f"\n  📦 Image: {img['name']}")
                
                # --- Native Loop ---
                nat_times = []
                for i in range(REPEAT_COUNT):
                    print(f"     Running Native ({i+1}/{REPEAT_COUNT})... ", end='', flush=True)
                    t = native.download(img['file'])
                    if t: 
                        nat_times.append(t)
                        print(f"Done ({t:.2f}s)")
                    else:
                        print("Failed")
                
                # --- CTS Loop ---
                cts_times = []
                for i in range(REPEAT_COUNT):
                    print(f"     Running CTS    ({i+1}/{REPEAT_COUNT})... ", end='', flush=True)
                    t = cts.download(img['file'], scen['strategy'])
                    if t:
                        cts_times.append(t)
                        print(f"Done ({t:.2f}s)")
                    else:
                        print("Failed")

                # --- 统计与记录 ---
                avg_nat, std_nat = get_stats(nat_times)
                avg_cts, std_cts = get_stats(cts_times)
                
                # 只有当两个都有数据时才记录加速比
                if avg_nat > 0 and avg_cts > 0:
                    speedup = avg_nat / avg_cts
                    results.append({
                        'Scenario': scen['name'],
                        'Image': img['name'],
                        'Strategy': scen['strategy'],
                        'Native_Mean': f"{avg_nat:.2f}",
                        'Native_Std': f"{std_nat:.2f}",
                        'CTS_Mean': f"{avg_cts:.2f}",
                        'CTS_Std': f"{std_cts:.2f}",
                        'Speedup': f"{speedup:.2f}"
                    })
                    print(f"  📊 Result: Native={avg_nat:.2f}s ±{std_nat:.2f}, CTS={avg_cts:.2f}s ±{std_cts:.2f} => {speedup:.2f}x Speedup")

    finally:
        net.reset()
        print("\n⚡ Network Reset.")

    # 保存结果
    csv_file = f"thesis_results_final_{datetime.now().strftime('%d_%H%M')}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Scenario', 'Image', 'Strategy', 
            'Native_Mean', 'Native_Std', 
            'CTS_Mean', 'CTS_Std', 
            'Speedup'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Experiment Complete! Data saved to {csv_file}")

if __name__ == '__main__':
    run()