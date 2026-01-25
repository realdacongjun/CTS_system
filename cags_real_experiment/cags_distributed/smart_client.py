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
    获取客户端环境信息 (已增强鲁棒性)
    """
    print("[Client] 正在感知本地环境...")
    sensor = RealSensor(target_url)
    profile = sensor.get_full_client_profile()
    
    # [关键修复] 使用 .get() 安全获取嵌套字典，防止 Sensor 探测失败导致 Key 丢失
    net_profile = profile.get('network_profile', {})
    sys_info = profile.get('system_info', {})

    # [关键修复] 构造客户端信息，提供保守默认值
    client_info = {
        'bandwidth_mbps': net_profile.get('bandwidth_mbps', 10.0), # 默认 10M
        'rtt_ms': net_profile.get('rtt_ms', 100.0),                # 默认 100ms
        'cpu_load': profile.get('current_cpu_load', 0.5),          # 默认 50%
        'memory_gb': sys_info.get('total_memory_gb', 2.0),         # 默认 2G
        'connection_stability': net_profile.get('connection_stability', 1.0)
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
            
            # 显示压缩算法预测时间排序
            if 'meta_info' in strategy and 'top_algorithms' in strategy['meta_info']:
                print("[Client] 压缩算法预测时间排序 (前5):")
                for i, (algo, time_pred) in enumerate(strategy['meta_info']['top_algorithms']):
                    print(f"  {i+1}. {algo}: {time_pred:.2f}s")
            
            return strategy
        else:
            print(f"[Client] 服务端API调用失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[Client] 连接服务端失败: {e}")
        return None


def record_experiment_summary(success, total_time, client_info, strategy, chunk_size, concurrency, output_file):
    """
    记录实验摘要数据到CSV文件
    """
    summary_file = "experiment_summary.csv"
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写表头
        if not file_exists:
            writer.writerow([
                "Timestamp", "Mode", "BW_Mbps", "RTT_ms", "CPU_Load", "Memory_GB",
                "Uncertainty", "Init_Chunk_MB", "Concurrency", "Total_Time_s", "Avg_Speed_MB_s", "Success",
                "Top_Algo_1", "Top_Algo_2", "Top_Algo_3"
            ])
        
        # 提取数据 (使用 .get 安全提取)
        bw = client_info.get('bandwidth_mbps', 0)
        rtt = client_info.get('rtt_ms', 0)
        cpu_load = client_info.get('cpu_load', 0)
        memory_gb = client_info.get('memory_gb', 0)
        
        uncert = 0
        if strategy and 'meta_info' in strategy:
             uncert = strategy['meta_info'].get('uncertainty', 0)

        init_chunk_mb = chunk_size / (1024*1024)
        
        # 获取顶级算法
        top_algo_1 = ""
        top_algo_2 = ""
        top_algo_3 = ""
        if strategy and 'meta_info' in strategy and 'top_algorithms' in strategy['meta_info']:
            algos = strategy['meta_info']['top_algorithms']
            if len(algos) > 0:
                top_algo_1 = f"{algos[0][0]}({algos[0][1]:.2f}s)"
            if len(algos) > 1:
                top_algo_2 = f"{algos[1][0]}({algos[1][1]:.2f}s)"
            if len(algos) > 2:
                top_algo_3 = f"{algos[2][0]}({algos[2][1]:.2f}s)"
        
        # 计算平均速度
        try:
            file_size = os.path.getsize(output_file)
            avg_speed = (file_size / (1024*1024)) / total_time if total_time > 0 else 0
        except:
            avg_speed = 0
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
            "CAGS",  # Mode
            f"{bw:.2f}",  # BW_Mbps
            f"{rtt:.0f}",  # RTT_ms
            f"{cpu_load:.2f}",  # CPU_Load
            f"{memory_gb:.1f}",  # Memory_GB
            f"{uncert:.4f}",  # Uncertainty
            f"{init_chunk_mb:.2f}",  # Init_Chunk_MB
            concurrency,  # Concurrency
            f"{total_time:.2f}",  # Total_Time_s
            f"{avg_speed:.2f}",  # Avg_Speed_MB_s
            "TRUE" if success else "FALSE",  # Success
            top_algo_1,  # Top_Algo_1
            top_algo_2,  # Top_Algo_2
            top_algo_3   # Top_Algo_3
        ])
    
    print(f"[Client] 📝 实验数据已记录至 {summary_file}")


def main():
    """
    主程序流程
    """
    # [修改点1] 配置参数：允许通过环境变量或命令行参数配置IP
    server_ip = os.getenv('SERVER_IP', '47.121.137.243')  # 可通过环境变量配置
    SERVER_URL = f"http://{server_ip}:5000" 
    TARGET_URL = f"http://{server_ip}/real_test.bin" 
    OUTPUT_FILE = "downloaded_file.bin" 
    
    # 图像信息
    IMAGE_INFO = {
        'size_mb': 1024.0,  # 1GB镜像
        'avg_layer_entropy': 0.7,
        'text_ratio': 0.2,
        'layer_count': 5,
        'zero_ratio': 0.1
    }
    
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
                'cost': 1.0,
                'top_algorithms': [('gzip-6', 10.0), ('zstd-3', 12.0)]
            }
        }
    
    # 获取下载参数
    download_url = strategy['target_url']
    chunk_size = strategy['strategy']['initial_chunk_size']
    concurrency = strategy['strategy']['concurrency']
    
    print(f"[Client] 开始下载: {download_url}")
    print(f"[Client] 初始策略: 块大小 {chunk_size/(1024*1024):.2f}MB, 并发数 {concurrency}")
    
    # 第三步：获取文件大小
    try:
        # [修改点2] 增加 timeout 防止弱网卡死
        response = requests.head(download_url, timeout=10)
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size == 0:
            print("[Client] ⚠️  无法获取文件大小，尝试使用Range请求获取")
            response = requests.get(download_url, headers={'Range': 'bytes=0-0'}, timeout=10)
            if response.status_code == 206:
                content_range = response.headers.get('Content-Range', '')
                if content_range:
                    file_size = int(content_range.split('/')[-1])
    except requests.exceptions.RequestException as e:
        print(f"[Client] ⚠️  网络请求失败: {e}")
        print("[Client] ⚠️  使用默认文件大小 1GB")
        file_size = 1024 * 1024 * 1024  # 1GB
    except Exception as e:
        print(f"[Client] ⚠️  获取文件大小异常: {e}")
        print("[Client] ⚠️  使用默认文件大小 1GB")
        file_size = 1024 * 1024 * 1024  # 1GB
    
    # 第四步：初始化AIMD修正层
    correction = CAGSCorrectionLayer(initial_chunk_size=chunk_size)
    
    # 第五步：执行下载
    print("[Client] 开始执行下载...")
    downloader = RealDownloader(download_url, file_size, OUTPUT_FILE)
    
    # 生成微观数据日志文件名
    micro_log_file = f"microscopic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    success, total_time = downloader.download_with_chunks(chunk_size, concurrency, correction, log_file=micro_log_file)
    
    if success:
        print("[Client] ✅ 下载成功完成!")
    else:
        print("[Client] ❌ 下载过程中出现问题!")
    
    # 记录实验摘要数据
    record_experiment_summary(success, total_time, client_info, strategy, chunk_size, concurrency, OUTPUT_FILE)


if __name__ == "__main__":
    main()