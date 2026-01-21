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
    真实环境感知类
    功能：测量RTT、带宽等网络参数
    参考了client_probe.py和network_profiler.py的设计模式
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
            response = requests.head(self.url, timeout=5)
            end_time = time.time()
            rtt = (end_time - start_time) * 1000  # 转换为毫秒
            return rtt if rtt > 0 else 50.0  # 默认值50ms
        except Exception as e:
            print(f"RTT测量失败: {e}")
            return 50.0  # 默认RTT 50ms
    
    def estimate_bandwidth(self, sample_size=100 * 1024):  # 100KB
        """
        通过下载小段数据估算初始带宽
        """
        try:
            headers = {'Range': f'bytes=0-{sample_size-1}'}
            start_time = time.time()
            response = requests.get(self.url, headers=headers, timeout=10)
            end_time = time.time()
            
            if response.status_code == 206:  # Partial Content
                data_size = len(response.content)
                duration = end_time - start_time
                if duration > 0:
                    bandwidth_mbps = (data_size * 8) / (duration * 1024 * 1024)  # Mbps
                    return bandwidth_mbps
        except Exception as e:
            print(f"带宽估算失败: {e}")
        
        return 1.0  # 默认1Mbps
    
    def probe_system_info(self):
        """
        探测系统信息，参考client_probe.py的实现
        """
        try:
            # 获取CPU信息
            cpu_info = cpuinfo.get_cpu_info()
            cpu_name = cpu_info.get('brand_raw', 'Unknown')
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            
            # 获取内存信息
            mem_info = psutil.virtual_memory()
            total_memory_gb = round(mem_info.total / (1024**3), 2)
            
            return {
                "cpu_name": cpu_name,
                "cpu_cores": cpu_cores,
                "cpu_threads": cpu_threads,
                "total_memory_gb": total_memory_gb
            }
        except Exception as e:
            print(f"系统信息探测失败: {e}")
            return {
                "cpu_name": "Unknown",
                "cpu_cores": psutil.cpu_count(logical=False) or 1,
                "cpu_threads": psutil.cpu_count(logical=True) or 1,
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
    
    def probe_cpu_performance(self):
        """
        探测CPU性能，参考client_probe.py的实现
        """
        def cpu_benchmark():
            # 执行一定数量的数学运算
            start_time = time.time()
            result = 0
            for i in range(1000000):
                result += i * i
            end_time = time.time()
            return end_time - start_time
        
        # 多次测试取平均值
        total_time = 0
        for _ in range(3):
            total_time += cpu_benchmark()
        
        avg_time = total_time / 3
        
        # 转换为评分（时间越短评分越高）
        # 基准：1秒完成测试得1000分
        cpu_score = int(1000 / avg_time)
        return cpu_score
    
    def get_network_profile(self):
        """
        获取网络概况，参考network_profiler.py的设计
        """
        rtt = self.measure_rtt()
        bandwidth = self.estimate_bandwidth()
        
        # 基于RTT和带宽估算连接稳定性（参考network_profiler.py的逻辑）
        stability = max(0.1, min(1.0, 1.0 - (rtt / 500)))  # RTT越低稳定性越高
        
        # 基于带宽估算丢包率（简化模型）
        packet_loss_rate = max(0.001, min(0.1, 0.1 / (bandwidth + 1)))  # 带宽越高丢包率越低
        
        return {
            'rtt_ms': rtt,
            'bandwidth_mbps': bandwidth,
            'host': self.parsed_url.hostname,
            'connection_stability': stability,
            'packet_loss_rate': packet_loss_rate
        }
    
    def get_full_client_profile(self):
        """
        获取完整的客户端配置文件，结合网络和系统信息
        """
        network_profile = self.get_network_profile()
        system_info = self.probe_system_info()
        cpu_score = self.probe_cpu_performance()
        
        # 基于网络质量估算解压性能（参考network_profiler.py的逻辑）
        # 假设网络较好的节点通常计算能力也较强
        estimated_decompression_speed = max(20, network_profile['bandwidth_mbps'] * 5)
        
        return {
            "cpu_score": cpu_score,
            "system_info": system_info,
            "network_profile": network_profile,
            "estimated_decompression_speed": estimated_decompression_speed,
            "latency_requirement": 400
        }


def test_sensor():
    """
    测试传感器功能
    """
    url = "http://47.120.14.12/real_test.bin"
    sensor = RealSensor(url)
    
    profile = sensor.get_full_client_profile()
    print(f"完整客户端配置: {profile}")


if __name__ == "__main__":
    test_sensor()