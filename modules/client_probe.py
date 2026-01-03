"""
需求端探针（Client Probe）
目的：让服务器知道"这个客户端适合什么压缩格式"

探测内容：
- CPU性能
- 网络带宽
- 解压性能
- 启动延迟要求
"""


import time
import hashlib
import gzip
import zlib
import threading
import psutil
import cpuinfo
import requests
from io import BytesIO


class ClientProbe:
    """客户端探针类"""
    
    def __init__(self):
        pass
    
    def probe_cpu(self):
        """
        探测CPU性能
        返回CPU评分
        """
        # 使用CPU密集型任务进行测试
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
    
    def probe_bandwidth(self):
        """
        探测网络带宽
        返回带宽值(Mbps)
        """
        try:
            # 使用一个较大的公开文件进行测速
            test_url = "http://speedtest.tele2.net/1MB.zip"
            start_time = time.time()
            
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                data = response.content
                end_time = time.time()
                
                # 计算带宽 (Mbps)
                duration = end_time - start_time
                size_bits = len(data) * 8
                bandwidth_mbps = (size_bits / duration) / 1_000_000
                
                return round(bandwidth_mbps, 2)
            else:
                # 回退到模拟数据
                return 68
        except Exception as e:
            print(f"带宽测试失败: {e}")
            # 回退到模拟数据
            return 68
    
    def probe_decompression_speed(self):
        """
        探测解压性能
        返回解压速度评分
        """
        # 创建测试数据
        test_data = b"x" * 1024 * 1024  # 1MB测试数据
        
        # 测试gzip解压速度
        gzip_start = time.time()
        gzip_data = gzip.compress(test_data)
        gzip_compression_time = time.time() - gzip_start
        
        gzip_decompress_start = time.time()
        gzip.decompress(gzip_data)
        gzip_decompress_time = time.time() - gzip_decompress_start
        
        # 测试zlib解压速度
        zlib_start = time.time()
        zlib_data = zlib.compress(test_data)
        zlib_compression_time = time.time() - zlib_start
        
        zlib_decompress_start = time.time()
        zlib.decompress(zlib_data)
        zlib_decompress_time = time.time() - zlib_decompress_start
        
        # 计算解压速度 (MB/s)
        gzip_speed = 1 / gzip_decompress_time if gzip_decompress_time > 0 else 100
        zlib_speed = 1 / zlib_decompress_time if zlib_decompress_time > 0 else 100
        
        # 返回平均速度
        return int((gzip_speed + zlib_speed) / 2)
    
    def probe_system_info(self):
        """
        探测系统信息
        返回系统硬件信息
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
    
    def get_client_profile(self):
        """
        获取完整的客户端配置文件
        """
        return {
            "cpu_score": self.probe_cpu(),
            "bandwidth_mbps": self.probe_bandwidth(),
            "decompression_speed": self.probe_decompression_speed(),
            "system_info": self.probe_system_info(),
            "latency_requirement": 400  # 默认延迟要求
        }


def main():
    """测试客户端探针"""
    probe = ClientProbe()
    profile = probe.get_client_profile()
    print("客户端探针结果:")
    for key, value in profile.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()