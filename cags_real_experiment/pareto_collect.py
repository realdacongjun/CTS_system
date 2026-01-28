#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thesis_system_suite.py - 硕士论文全链路系统实验套件 (Final Defense Version)
========================================================================
架构设计：
1. [Measurement] 进程级资源隔离采集，支持真并发与调度并发对比。
2. [Physics]     自动计算能效成本密度 (Cost Index) 与链路饱和度 (Utilization)。
3. [Constraint]  实施可靠性约束筛选 (Reliability Filter, Fail_Rate < 5%)。
4. [Optimization]执行严格的非支配排序 (Non-dominated Sorting) 构建帕累托前沿。
5. [Vis]         生成带误差条 (Error Bars) 的科研级图表。

依赖:
  pip3 install requests psutil pandas seaborn matplotlib
运行:
  sudo python3 thesis_system_suite.py
"""

import os
import csv
import time
import subprocess
import statistics
import threading
import requests
import psutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

# ================= 🔧 全局配置 (Configuration) =================
SERVER_IP = "47.121.127.59"
BASE_URL = f"http://{SERVER_IP}"
OUTPUT_CSV = "thesis_final_dataset.csv"
PLOT_DIR = "./thesis_figures"
REPEATS = 5  # 统计学显著性要求 (n=5)

# 实验变量 1: 网络环境 (定义物理带宽上限)
NETWORKS = [
    {"id": "Weak",  "bw": 2,   "delay": 400, "loss": 5}, # 关注生存率
    {"id": "Edge",  "bw": 20,  "delay": 50,  "loss": 1}, # 关注能效
    {"id": "Cloud", "bw": 100, "delay": 20,  "loss": 0}  # 关注吞吐
]

# 实验变量 2: 目标文件
FILES = [
    {"id": "Small_OS",   "name": "generalized_os.tar.br"},
    {"id": "Med_Mixed",  "name": "generalized_mixed.tar.zst"},
    {"id": "Large_Text", "name": "generalized_text.tar.lz4"}
]

# 实验变量 3: 控制参数
THREADS_LIST = [1, 2, 4, 8, 16]
CHUNK_MB_LIST = [-1, 1, 4, 16] # -1 = Pure Concurrency (Total/N)

# 硬件常量
CPU_CORES = psutil.cpu_count(logical=True)

# ================= 模块 1: 科研级测量探针 =================

class ProcessMonitor:
    """进程级资源监控 (Isolation)"""
    def __init__(self):
        self.running = False
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
        self.mem_samples = []
        self._thread = None

    def start(self):
        self.running = True
        self.cpu_samples = []
        self.mem_samples = []
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread: self._thread.join()
        if not self.cpu_samples: return 0.0, 0.0
        
        # Trimmed Mean (去极值平均)
        n = len(self.cpu_samples)
        if n > 5:
            sorted_cpu = sorted(self.cpu_samples)
            trim = int(n * 0.1)
            avg_cpu = statistics.mean(sorted_cpu[trim : n-trim])
        else:
            avg_cpu = statistics.mean(self.cpu_samples)
        
        avg_mem = statistics.mean(self.mem_samples)
        return avg_cpu, avg_mem

    def _monitor_loop(self):
        while self.running:
            try:
                # 10Hz 高频采样
                self.cpu_samples.append(self.process.cpu_percent(interval=0.1))
                self.mem_samples.append(self.process.memory_info().rss / 1024**2)
            except: break

class NetworkManager:
    @staticmethod
    def set(config):
        subprocess.run("tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)
        cmd = f"tc qdisc add dev eth0 root netem rate {config['bw']}mbit delay {config['delay']}ms loss {config['loss']}%"
        subprocess.run(cmd, shell=True, check=True)
        time.sleep(2)

    @staticmethod
    def reset():
        subprocess.run("tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)

class ScientificDownloader:
    def __init__(self):
        self.sess = requests.Session()
        # 禁用重试，为了精确捕捉 Failure Rate
        self.sess.mount('http://', HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0))
        self.monitor = ProcessMonitor()

    def run(self, url, threads, chunk_mb, net_bw):
        try:
            head = self.sess.head(url, timeout=5)
            total = int(head.headers.get('Content-Length', 0))
            if total == 0: return self._fail("Meta_Zero")
        except: return self._fail("Meta_Timeout")

        # 严格并发模型构建
        ranges = []
        if chunk_mb == -1:
            part = total // threads
            for i in range(threads):
                s = i * part
                e = (i + 1) * part - 1 if i < threads - 1 else total - 1
                ranges.append((s, e))
        else:
            size = int(chunk_mb * 1024**2)
            for s in range(0, total, size):
                ranges.append((s, min(s + size - 1, total - 1)))

        # 执行
        self.monitor.start()
        start = time.time()
        err = "Success"
        try:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                futures = {ex.submit(self._fetch, url, s, e): (s,e) for s,e in ranges}
                for f in as_completed(futures):
                    try: f.result()
                    except requests.exceptions.Timeout: 
                        err = "ERR_Timeout"; ex.shutdown(wait=False, cancel_futures=True); break
                    except requests.exceptions.ConnectionError:
                        err = "ERR_Conn"; ex.shutdown(wait=False, cancel_futures=True); break
                    except Exception: 
                        err = "ERR_Other"; ex.shutdown(wait=False, cancel_futures=True); break
        except: err = "ERR_Sys"
        
        dur = time.time() - start
        raw_cpu, mem = self.monitor.stop()

        # 核心物理指标计算
        size_mb = total / 1024**2
        if err == "Success":
            tp = size_mb / dur
            # Utilization: 相对物理链路的饱和度
            util = min(100.0, (tp / (net_bw / 8.0)) * 100)
            # Norm CPU: 单核归一化负载
            norm_cpu = raw_cpu / CPU_CORES
            # Cost Index: 物理能效密度 (CPU积分时间 / 数据量)
            # Cost = (Normalized_CPU * Duration) / FileSize
            cost = (norm_cpu * dur) / size_mb 
        else:
            tp = util = norm_cpu = cost = 0

        return {
            "Time_s": dur, "Throughput_MBps": tp, "Link_Util_Pct": util,
            "CPU_Norm_Pct": norm_cpu, "Cost_Index": cost, "Mem_RSS_MB": mem, "Error": err
        }

    def _fetch(self, url, s, e):
        r = self.sess.get(url, headers={'Range': f'bytes={s}-{e}'}, timeout=20)
        r.raise_for_status(); _ = r.content

    def _fail(self, err):
        return {"Time_s":None, "Throughput_MBps":0, "Link_Util_Pct":0, 
                "CPU_Norm_Pct":0, "Cost_Index":0, "Mem_RSS_MB":0, "Error":err}

# ================= 模块 2: 严谨数据分析与绘图 =================

def get_pareto_frontier(df):
    """非支配排序算法 (Non-dominated Sorting)"""
    # 目标: Cost 越小越好 (-Cost 越大越好), TP 越大越好
    points = df[['Cost_Mean', 'TP_Mean']].values
    is_pareto = [True] * len(points)
    
    for i, A in enumerate(points):
        for j, B in enumerate(points):
            if i == j: continue
            # B 支配 A: Cost_B <= Cost_A 且 TP_B >= TP_A (且至少一个严格更优)
            if (B[0] <= A[0] and B[1] >= A[1]) and (B[0] < A[0] or B[1] > A[1]):
                is_pareto[i] = False
                break
    return df[is_pareto].sort_values('Cost_Mean')

def run_analysis():
    print("\n" + "="*60)
    print("📊 Phase 2: Scientific Analysis & Visualization")
    print("="*60)
    
    if not os.path.exists(OUTPUT_CSV): return
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

    df = pd.read_csv(OUTPUT_CSV)
    
    # 1. 统计聚合 (含 StdDev 用于误差条)
    grouped = df.groupby(['Network', 'FileID', 'Threads'])
    summary = grouped.agg({
        'Throughput_MBps': ['mean', 'std'],
        'Cost_Index': ['mean', 'std'],
        'Link_Util_Pct': 'mean',
        'Time_s': ['mean', 'std'], # Time std is Jitter
        'Error': lambda x: (x != 'Success').mean()
    }).reset_index()
    
    summary.columns = ['Network', 'FileID', 'Threads', 
                       'TP_Mean', 'TP_Std', 'Cost_Mean', 'Cost_Std',
                       'Util_Mean', 'Time_Mean', 'Time_Jitter', 'Fail_Rate']
    summary['Survival_Rate'] = 1 - summary['Fail_Rate']

    sns.set(style="whitegrid", context="paper", font_scale=1.4)

    # --- 图 1: 可靠性约束下的 Pareto 前沿 (Cloud/Text) ---
    subset = summary[(summary['Network']=='Cloud') & (summary['FileID']=='Large_Text')].copy()
    if not subset.empty:
        plt.figure(figsize=(10, 7))
        
        # 可行域筛选 (Reliability Constraint)
        reliable = subset[subset['Fail_Rate'] <= 0.05]
        unreliable = subset[subset['Fail_Rate'] > 0.05]
        
        # 帕累托筛选
        frontier = get_pareto_frontier(reliable)
        
        # 绘图
        plt.errorbar(reliable['Cost_Mean'], reliable['TP_Mean'], yerr=reliable['TP_Std'], 
                     fmt='o', color='gray', alpha=0.4, label='Feasible Solutions')
        
        if not unreliable.empty:
            plt.scatter(unreliable['Cost_Mean'], unreliable['TP_Mean'], 
                        marker='x', color='red', s=80, label='Infeasible (>5% Fail)')

        plt.plot(frontier['Cost_Mean'], frontier['TP_Mean'], 'b--', linewidth=2, label='Pareto Frontier')
        plt.errorbar(frontier['Cost_Mean'], frontier['TP_Mean'], yerr=frontier['TP_Std'], 
                     fmt='s', color='blue', markersize=8)

        for i, row in frontier.iterrows():
            plt.text(row['Cost_Mean'], row['TP_Mean'] + row['TP_Std'] + 0.2, 
                     f" {int(row['Threads'])}T", fontsize=10, fontweight='bold', color='blue')

        plt.xlabel(r"Energy Cost Index ($CPU \cdot s / MB$)")
        plt.ylabel("Throughput (MB/s)")
        plt.title("Pareto Efficiency Frontier (Cloud Network)")
        plt.legend()
        plt.savefig(f"{PLOT_DIR}/Fig_Pareto_Strict.png", dpi=300)
        print(f"✅ Generated: {PLOT_DIR}/Fig_Pareto_Strict.png")

    # --- 图 2: 风险势垒生存曲线 (Weak/Text) ---
    weak_data = summary[(summary['Network']=='Weak')].copy()
    if not weak_data.empty:
        plt.figure(figsize=(8,6))
        sns.lineplot(data=weak_data, x='Threads', y='Survival_Rate', hue='FileID', 
                     marker='s', linewidth=3)
        plt.axhline(0.95, color='r', linestyle='--', label='95% Reliability Constraint')
        plt.ylim(0, 1.1)
        plt.ylabel("Survival Probability")
        plt.title("Risk Barrier Analysis (Weak Network)")
        plt.savefig(f"{PLOT_DIR}/Fig_Survival_Strict.png", dpi=300)
        print(f"✅ Generated: {PLOT_DIR}/Fig_Survival_Strict.png")

    # --- 图 3: 链路饱和度分析 ---
    plt.figure(figsize=(8,6))
    sns.barplot(data=summary, x='Threads', y='Util_Mean', hue='Network')
    plt.ylabel("Link Saturation Ratio (%)")
    plt.ylim(0, 105)
    plt.title("Bandwidth Utilization Analysis")
    plt.savefig(f"{PLOT_DIR}/Fig_Util_Saturation.png", dpi=300)
    print(f"✅ Generated: {PLOT_DIR}/Fig_Util_Saturation.png")

# ================= 主控制流 =================

def main():
    if os.geteuid() != 0:
        print("❌ Sudo required for TC network control.")
        return
    
    plt.switch_backend('Agg') # 适配无头服务器
    
    # 1. 采集阶段
    print(f"🚀 Phase 1: Measurement (Repeats={REPEATS})")
    dl = ScientificDownloader()
    
    # 初始化 CSV
    headers = ['Network', 'FileID', 'Mode', 'Threads', 'Chunk_MB', 'Trial',
               'Time_s', 'Throughput_MBps', 'Link_Util_Pct', 
               'CPU_Norm_Pct', 'Cost_Index', 'Mem_RSS_MB', 'Error']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()
        
    try:
        for net in NETWORKS:
            print(f"\n🌍 Network: {net['id']} (Cap={net['bw']} Mbps)")
            NetworkManager.set(net)
            
            for f_info in FILES:
                url = f"{BASE_URL}/{f_info['name']}"
                print(f"   📂 {f_info['id']}")
                
                # Baseline (Native)
                for i in range(REPEATS):
                    res = dl.run(url, 1, -1, net['bw'])
                    res.update({'Network':net['id'], 'FileID':f_info['id'], 'Mode':'Baseline', 
                                'Threads':1, 'Chunk_MB':-1, 'Trial':i+1})
                    with open(OUTPUT_CSV, 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=headers).writerow(res)

                # CTS Space
                for th in THREADS_LIST:
                    for ch in CHUNK_MB_LIST:
                        # 剪枝逻辑
                        if f_info['id'] == 'Small_OS' and (th > 4 or (ch != -1 and ch > 1)): continue
                        
                        for i in range(REPEATS):
                            res = dl.run(url, th, ch, net['bw'])
                            res.update({'Network':net['id'], 'FileID':f_info['id'], 'Mode':'CTS', 
                                        'Threads':th, 'Chunk_MB':ch, 'Trial':i+1})
                            
                            log = f"{res['Throughput_MBps']:.1f}MB/s" if res['Error']=="Success" else res['Error']
                            print(f"\r      T={th:<2} C={ch:<2} -> {log}      ", end="", flush=True)
                            
                            with open(OUTPUT_CSV, 'a', newline='') as f:
                                csv.DictWriter(f, fieldnames=headers).writerow(res)
                        print("")
    finally:
        NetworkManager.reset()
    
    # 2. 分析阶段
    run_analysis()
    print("\n✨ Thesis Experiment Suite Completed Successfully.")

if __name__ == "__main__":
    main()