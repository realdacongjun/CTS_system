import requests
import time
import json
import os
import sys
import csv
from datetime import datetime
# 导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_sensor import RealSensor
from real_downloader import RealDownloader
from cags_scheduler import CAGSCorrectionLayer


def get_client_environment(target_url):
    """
    获取客户端环境信息
    """
    print("[Client] 正在感知本地环境...")
    sensor = RealSensor(target_url)
    profile = sensor.get_full_client_profile()
    
    # 构造客户端信息
    client_info = {
        'bandwidth_mbps': profile['network_profile']['bandwidth_mbps'],
        'rtt_ms': profile['network_profile']['rtt_ms'],
        'cpu_load': profile['current_cpu_load'],
        'memory_gb': profile['system_info']['total_memory_gb'],
        'connection_stability': profile['network_profile']['connection_stability']
    }
    
    print(f"[Client] 环境感知完成: {client_info}")
    return client_info, profile


def request_server_strategy(server_url, client_info, image_info, server_info):
    """
    向服务端请求下载策略
    """
    print("[Client] 正在向服务端请求AI决策...")
    
    payload = {
        'client_info': client_info,
        'image_info': image_info,
        'server_info': server_info
    }
    
    try:
        response = requests.post(
            f"{server_url}/negotiate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            strategy = response.json()
            print(f"[Client] 服务端AI决策完成: {strategy['strategy']}")
            return strategy
        else:
            print(f"[Client] 服务端API调用失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[Client] 连接服务端失败: {e}")
        return None


def main():
    """
    主程序流程：
    1. 感知本地环境
    2. 向服务端请求AI决策
    3. 初始化AIMD修正层
    4. 执行下载
    """
    
    # 配置参数
    SERVER_URL = "http://192.168.1.100:5000"  # 服务端地址，请根据实际环境修改
    TARGET_URL = "http://192.168.1.100:80/download.bin"  # 目标下载文件
    OUTPUT_FILE = "downloaded_file.bin"  # 本地保存路径
    
    # 图像信息（可以根据实际镜像调整）
    IMAGE_INFO = {
        'size_mb': 1024.0,  # 1GB镜像
        'avg_layer_entropy': 0.7,
        'text_ratio': 0.2,
        'layer_count': 5,
        'zero_ratio': 0.1
    }
    
    # 服务端信息
    SERVER_INFO = {
        'download_url': TARGET_URL
    }
    
    print("="*50)
    print("🚀 CAGS 客户端启动")
    print("="*50)
    
    # 第一步：感知本地环境
    client_info, profile = get_client_environment(TARGET_URL)
    
    # 第二步：请求服务端AI决策
    strategy = request_server_strategy(SERVER_URL, client_info, IMAGE_INFO, SERVER_INFO)
    
    if strategy is None:
        print("[Client] ⚠️  服务端不可达，使用默认策略进行下载...")
        # Fallback 策略
        strategy = {
            'target_url': TARGET_URL,
            'strategy': {
                'initial_chunk_size': 2 * 1024 * 1024,  # 2MB
                'concurrency': 3
            },
            'meta_info': {
                'predicted_time_s': 0,
                'uncertainty': 0.1,
                'cost': 1.0
            }
        }
    
    # 获取下载参数
    download_url = strategy['target_url']
    chunk_size = strategy['strategy']['initial_chunk_size']
    concurrency = strategy['strategy']['concurrency']
    
# 第五步：执行下载
    print("[Client] 开始执行下载...")
    downloader = RealDownloader(download_url, file_size, OUTPUT_FILE)
    
    # === 修正开始：在外部计时 ===
    start_time = time.time() 
    success = downloader.download_with_chunks(chunk_size, concurrency, correction, micro_log_file)
    total_time = time.time() - start_time
    # === 修正结束 ===

    # 记录宏观实验数据
    summary_file = "experiment_summary.csv"
    # 第三步：获取文件大小
    try:
        response = requests.head(download_url)
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size == 0:
            print("[Client] ⚠️  无法获取文件大小，尝试使用Range请求获取")
            response = requests.get(download_url, headers={'Range': 'bytes=0-0'}, timeout=5)
            if response.status_code == 206:
                content_range = response.headers.get('Content-Range', '')
                if content_range:
                    file_size = int(content_range.split('/')[-1])
    except:
        print("[Client] ⚠️  获取文件大小失败，默认使用1GB")
        file_size = 1024 * 1024 * 1024  # 1GB
    
    # 第四步：初始化AIMD修正层
    correction = CAGSCorrectionLayer(initial_chunk_size=chunk_size)
    
    # 为微观数据记录创建日志文件
    micro_log_file = f"micro_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 第五步：执行下载
    print("[Client] 开始执行下载...")
    downloader = RealDownloader(download_url, file_size, OUTPUT_FILE)
    success, total_time = downloader.download_with_chunks(chunk_size, concurrency, correction, micro_log_file)
    
    # 记录宏观实验数据
    summary_file = "experiment_summary.csv"
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        if not file_exists:
            writer.writerow([
                "Timestamp", "Mode", "RTT_ms", "BW_Mbps", "CPU_Load", 
                "AI_Uncertainty", "Init_Chunk_MB", "Concurrency", 
                "Total_Time_s", "Avg_Speed_MBps", "Success"
            ])
        
        # 提取数据
        bw = client_info['bandwidth_mbps']
        rtt = client_info['rtt_ms']
        cpu_load = client_info['cpu_load']
        uncert = strategy['meta_info']['uncertainty'] if strategy else 0
        init_chunk_mb = chunk_size / (1024*1024)
        avg_speed = (file_size / (1024*1024)) / total_time if total_time > 0 else 0
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CAGS",  # 模式
            f"{rtt:.0f}",  # RTT
            f"{bw:.2f}",  # 带宽
            f"{cpu_load:.3f}",  # CPU负载
            f"{uncert:.4f}",  # AI不确定性
            f"{init_chunk_mb:.2f}",  # 初始块大小
            concurrency,  # 并发数
            f"{total_time:.2f}",  # 总耗时
            f"{avg_speed:.2f}",  # 平均速度
            "TRUE" if success else "FALSE"  # 成功标志
        ])
    
    print(f"[Client] 📝 实验数据已记录至 {summary_file}")
    print(f"[Client] 📊 微观数据已记录至 {micro_log_file}")
    
    if success:
        print("[Client] ✅ 下载成功完成!")
    else:
        print("[Client] ❌ 下载过程中出现问题!")


if __name__ == "__main__":
    main()