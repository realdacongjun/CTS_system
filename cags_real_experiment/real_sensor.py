import requests
import time
import socket
from urllib.parse import urlparse
import psutil
import cpuinfo
import threading
from typing import Dict, Any

class RealSensor:
    """
    真实环境感知类 (修复版)
    功能：测量RTT、带宽、CPU负载等参数
    """
    
    def __init__(self, url):
        self.url = url
        self.parsed_url = urlparse(url)
        
    def measure_rtt(self):
        """
        通过HEAD请求测量RTT
        """
        try:
            start_time = time.time()
            # 增加 timeout 到 5秒，防止弱网直接报错
            response = requests.head(self.url, timeout=5)
            end_time = time.time()
            rtt = (end_time - start_time) * 1000  # 转换为毫秒
            return rtt if rtt > 0 else 50.0
        except Exception as e:
            print(f"[Sensor] RTT测量超时或失败: {e}")
            return 100.0  # 失败时返回保守值 100ms
    
    def estimate_bandwidth(self, sample_size=200 * 1024):  # 增加到 200KB 提高准确度
        """
        通过下载小段数据估算初始带宽
        """
        try:
            headers = {'Range': f'bytes=0-{sample_size-1}'}
            start_time = time.time()
            # 增加 timeout 到 10秒
            response = requests.get(self.url, headers=headers, timeout=10)
            end_time = time.time()
            
            if response.status_code == 206:  # Partial Content
                data_size = len(response.content)
                duration = end_time - start_time
                if duration > 0:
                    bandwidth_mbps = (data_size * 8) / (duration * 1024 * 1024)
                    return bandwidth_mbps
        except Exception as e:
            print(f"[Sensor] 带宽估算超时或失败: {e}")
        
        return 5.0  # 失败时返回保守值 5Mbps
    
    def probe_system_info(self):
        """
        探测系统静态信息
        """
        try:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_name = cpu_info.get('brand_raw', 'Unknown')
            # 内存信息
            mem_info = psutil.virtual_memory()
            total_memory_gb = round(mem_info.total / (1024**3), 2)
            free_memory_gb = round(mem_info.available / (1024**3), 2)
            
            return {
                "cpu_name": cpu_name,
                "cpu_cores": psutil.cpu_count(logical=False),
                "total_memory_gb": total_memory_gb,
                "free_memory_gb": free_memory_gb
            }
        except Exception:
            return {
                "cpu_name": "Unknown", 
                "total_memory_gb": 2.0,
                "free_memory_gb": 1.0
            }
    
    def probe_cpu_performance(self):
        """
        简单的CPU基准测试
        """
        try:
            start_time = time.time()
            # 减少循环次数，避免在感知阶段卡太久 (100万 -> 50万)
            sum(i*i for i in range(500000))
            end_time = time.time()
            duration = end_time - start_time
            if duration == 0: duration = 0.001
            return int(100 / duration) # 简化的评分
        except:
            return 100
    
    def get_network_profile(self):
        rtt = self.measure_rtt()
        bandwidth = self.estimate_bandwidth()
        return {
            'rtt_ms': rtt,
            'bandwidth_mbps': bandwidth,
            'connection_stability': max(0.1, min(1.0, 1.0 - (rtt / 1000)))
        }
    
    def get_full_client_profile(self):
        """
        【关键修复】确保返回字典包含 current_cpu_load
        """
        network_profile = self.get_network_profile()
        system_info = self.probe_system_info()
        cpu_score = self.probe_cpu_performance()
        
        # 🔥 获取当前 CPU 负载 (采样 0.5s)
        current_cpu_load = psutil.cpu_percent(interval=0.5)
        
        # 内存
        mem = psutil.virtual_memory()
        mem_free_gb = round(mem.available / (1024**3), 2)

        return {
            "cpu_score": cpu_score,
            "system_info": system_info,
            "network_profile": network_profile,
            
            # 🔥 [修复点] 必须显式包含这几个 key，否则 client 会报错
            "current_cpu_load": current_cpu_load,
            "mem_free_gb": mem_free_gb,
            
            "estimated_decompression_speed": max(20, network_profile['bandwidth_mbps'] * 5),
            "latency_requirement": 400
        }

if __name__ == "__main__":
    # 简单的自测逻辑
    url = "http://47.121.137.243/real_test.bin"
    print("正在测试 Sensor...")
    s = RealSensor(url)
    p = s.get_full_client_profile()
    print(f"✅ 测试通过! CPU Load: {p['current_cpu_load']}%")