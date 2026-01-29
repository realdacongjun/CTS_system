#!/usr/bin/env python3
"""
e2e_validation.py - 端到端系统增益验证 (The Proof of Victory)
=========================================================
功能：
1. 模拟真实客户端，向服务端 API 发起协商请求。
2. 对比 "CTS 自适应传输" vs "传统基准 (Baseline)"。
3. 输出最终的性能提升百分比。
"""

import requests
import time
import threading
import psutil
import json
from concurrent.futures import ThreadPoolExecutor

# ================= 配置 =================
# 填入你 2核2G 服务端的公网 IP
SERVER_IP = "47.121.127.59" 
CONTROLLER_API = f"http://{SERVER_IP}:5000/negotiate"
FILE_URL_BASE = f"http://{SERVER_IP}"

# 要测试的文件 (必须是服务端真实存在的)
TARGET_FILE = "generalized_mixed.tar.zst" 

# 模拟的网络环境 (发送给控制器看，用于触发不同策略)
# 你可以修改这里来测试 Weak / Edge / Cloud 不同场景
CURRENT_ENV = {
    "bandwidth_mbps": 2,    # 模拟弱网带宽
    "loss_rate": 6,         # 模拟高丢包 (触发风险势垒!)
    "rtt_ms": 200
}

def download_worker(url, start, end, results, index):
    headers = {'Range': f'bytes={start}-{end}'}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        results[index] = len(r.content)
    except:
        results[index] = 0

def run_transfer(mode, threads, chunk_mb, file_name):
    """执行传输任务的核心函数"""
    print(f"\n🚀 [{mode}] 启动传输...")
    print(f"   配置: 线程={threads}, 分片={chunk_mb if chunk_mb > 0 else 'Auto'}MB")
    
    url = f"{FILE_URL_BASE}/{file_name}"
    
    # 1. 获取文件大小
    try:
        head = requests.head(url, timeout=5)
        total_size = int(head.headers.get('Content-Length', 0))
    except:
        print("❌ 无法连接文件服务器")
        return 0, 0

    # 2. 规划分片
    ranges = []
    if chunk_mb == -1: # 纯并发模式
        part = total_size // threads
        for i in range(threads):
            s = i * part
            e = (i + 1) * part - 1 if i < threads - 1 else total_size - 1
            ranges.append((s, e))
    else: # 固定分片模式
        size = int(chunk_mb * 1024 * 1024)
        for s in range(0, total_size, size):
            ranges.append((s, min(s + size - 1, total_size - 1)))

    # 3. 开始下载计时
    start_time = time.time()
    results = [0] * len(ranges)
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for i, (s, e) in enumerate(ranges):
            futures.append(executor.submit(download_worker, url, s, e, results, i))
        
        for f in futures:
            f.result()

    duration = time.time() - start_time
    total_bytes = sum(results)
    
    # 4. 计算结果
    if total_bytes < total_size:
        print(f"   ❌ 传输失败 (丢包/超时)")
        return 0, float('inf')
    
    tp = (total_bytes / 1024 / 1024) / duration
    print(f"   ✅ 完成! 耗时: {duration:.2f}s, 吞吐: {tp:.2f} MB/s")
    return tp, duration

def main():
    print("="*60)
    print("🏆 CTS System End-to-End Validation")
    print("="*60)

    # --- 步骤 1: 运行 Baseline (模拟 Docker 默认行为) ---
    # 通常 Docker 也是并发下载，但没有智能调度，我们假设它默认 3 线程
    tp_base, time_base = run_transfer("Baseline (Default)", threads=3, chunk_mb=-1, file_name=TARGET_FILE)

    # --- 步骤 2: 运行 CTS (智能协商) ---
    print(f"\n🤖 正在向控制器 ({CONTROLLER_API}) 请求策略...")
    try:
        # 发起协商
        resp = requests.post(CONTROLLER_API, json={
            "client_info": CURRENT_ENV,
            "image_info": {"file_id": "Med_Mixed"}
        }, timeout=5)
        decision = resp.json()['decision']
        
        # 解析策略
        smart_threads = decision['threads']
        smart_chunk = decision['chunk_size_mb']
        is_barrier = decision['meta']['barrier_triggered']
        scenario = decision['meta']['scenario']
        
        print(f"   💡 控制器响应: 场景=[{scenario}]")
        if is_barrier:
            print("   🛡️  [风险势垒已触发] -> 强制降级以保活!")
        
        # 执行智能传输
        tp_cts, time_cts = run_transfer("CTS (Smart)", smart_threads, smart_chunk, TARGET_FILE)
        
    except Exception as e:
        print(f"❌ 协商失败: {e}")
        return

    # --- 步骤 3: 计算增益 ---
    print("\n" + "="*60)
    print("📊 最终战报 (Final Report)")
    print("="*60)
    
    if tp_cts > 0 and tp_base > 0:
        gain = ((tp_cts - tp_base) / tp_base) * 100
        print(f"Baseline 吞吐: {tp_base:.2f} MB/s")
        print(f"CTS      吞吐: {tp_cts:.2f} MB/s")
        print(f"🚀 性能提升: {gain:+.2f}%")
        
        if gain > 0:
            print("\n✅ 结论: CTS 系统有效提升了传输效率！")
        else:
            print("\n⚠️ 结论: 需检查策略表或网络环境。")
    else:
        print("❌ 实验未完成 (存在失败传输)")

if __name__ == "__main__":
    main()