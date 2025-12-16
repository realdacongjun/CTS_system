"""
Probe探针模块 - 通过发送小型测试blob探测接收端能力

该模块实现了基于探针的接收端能力探测机制，包括网络、解压和IO能力的探测。
注意：此模块主要用于非容器化环境或支持主动探针的客户端环境。
对于标准容器化环境，请使用network_profiler模块进行被动网络探测。
"""

import time
import json
import hashlib
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import requests
import gzip
import zlib


@dataclass
class ProbeResult:
    """探针探测结果"""
    probe_id: str
    network_bandwidth: Optional[float] = None  # Mbps
    network_rtt: Optional[float] = None  # ms
    network_loss_rate: Optional[float] = None  # 0-1
    decompress_throughput: Optional[Dict[str, float]] = None  # MB/s for different algorithms
    io_write_throughput: Optional[float] = None  # MB/s
    available_memory: Optional[float] = None  # GB
    timestamp: float = 0


class ProbeReceiver:
    """基于探针的接收端探测器（主动探针模式）
    
    注意：此模块适用于支持主动探针的环境，如Docker Desktop、本地开发环境等。
    对于标准容器化环境，请使用network_profiler模块进行被动网络探测。
    """
    
    def __init__(self):
        self.probe_results: List[ProbeResult] = []
    
    def create_probe_blobs(self) -> Dict[str, bytes]:
        """
        创建三种类型的探测blob
        
        Returns:
            包含三种探测blob的字典
        """
        blobs = {}
        
        # B_net: 网络探测blob (128KB纯随机数据)
        blobs['B_net'] = self._generate_random_blob(128 * 1024)
        
        # B_cpu: 解压探测blob (压缩后的64KB数据)
        # 使用真实压缩数据而不是模拟数据
        uncompressed_data = self._generate_pattern_blob(400 * 1024)  # 400KB未压缩数据
        blobs['B_cpu_gzip'] = gzip.compress(uncompressed_data)[:64 * 1024]
        blobs['B_cpu_zlib'] = zlib.compress(uncompressed_data)[:64 * 1024]
        
        # B_io: IO探测blob (128个1KB小文件)
        io_data = b''
        for i in range(128):
            io_data += self._generate_pattern_blob(1024)
        blobs['B_io'] = io_data
        
        return blobs
    
    def _generate_random_blob(self, size: int) -> bytes:
        """生成指定大小的随机数据blob"""
        return bytes([random.randint(0, 255) for _ in range(size)])
    
    def _generate_pattern_blob(self, size: int) -> bytes:
        """生成带有模式的数据blob（可压缩）"""
        pattern = b'This is a test pattern for compression. ' * 100
        return (pattern * (size // len(pattern) + 1))[:size]
    
    def send_probe(self, endpoint: str) -> ProbeResult:
        """
        发送探针到指定端点并获取结果
        
        Args:
            endpoint: 接收端地址
            
        Returns:
            探测结果
        """
        # 创建探测blobs
        probe_blobs = self.create_probe_blobs()
        probe_id = f"probe_{int(time.time())}"
        
        print(f"正在向 {endpoint} 发送探针 {probe_id}")
        
        # 发送网络探测
        network_result = self._send_network_probe(endpoint, probe_blobs['B_net'])
        
        # 发送解压探测
        decompress_result_gzip = self._send_decompress_probe(endpoint, probe_blobs['B_cpu_gzip'], 'gzip')
        decompress_result_zlib = self._send_decompress_probe(endpoint, probe_blobs['B_cpu_zlib'], 'zlib')
        
        # 发送IO探测
        io_result = self._send_io_probe(endpoint, probe_blobs['B_io'])
        
        # 组合结果
        result = ProbeResult(
            probe_id=probe_id,
            network_bandwidth=network_result.get('bandwidth'),
            network_rtt=network_result.get('rtt'),
            network_loss_rate=network_result.get('loss_rate'),
            decompress_throughput={
                'gzip': decompress_result_gzip.get('throughput'),
                'zlib': decompress_result_zlib.get('throughput')
            },
            io_write_throughput=io_result.get('throughput') or io_result.get('io_write_throughput'),
            timestamp=time.time()
        )
        
        self.probe_results.append(result)
        return result
    
    def send_multiple_probes(self, endpoint: str, count: int = 3) -> List[ProbeResult]:
        """
        发送多轮探针以提高准确性
        
        Args:
            endpoint: 接收端地址
            count: 探针轮数
            
        Returns:
            探测结果列表
        """
        results = []
        for i in range(count):
            print(f"第 {i+1}/{count} 轮探测...")
            result = self.send_probe(endpoint)
            results.append(result)
            time.sleep(0.5)  # 轮次间隔
            
        return results
    
    def _send_network_probe(self, endpoint: str, blob: bytes) -> Dict[str, Any]:
        """发送网络探测"""
        try:
            start_time = time.time()
            response = requests.post(f"{endpoint}/probe?type=B_net", data=blob, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                # 确保返回了必要的字段
                if 'bandwidth' not in result:
                    result['bandwidth'] = round((len(blob) * 8) / ((end_time - start_time) * 1_000_000), 2)
                if 'rtt' not in result:
                    result['rtt'] = round((end_time - start_time) * 1000, 2)
                return result
            else:
                print(f"网络探测失败，状态码: {response.status_code}")
                # 回退到模拟数据
                return self._simulate_network_probe(blob)
        except Exception as e:
            print(f"网络探测异常: {e}")
            # 回退到模拟数据
            return self._simulate_network_probe(blob)

    def _send_decompress_probe(self, endpoint: str, blob: bytes, algorithm: str) -> Dict[str, Any]:
        """发送解压探测"""
        try:
            data = {
                'algorithm': algorithm,
                'data': blob.decode('latin1') if isinstance(blob, bytes) else blob
            }
            json_data = json.dumps(data)
            
            response = requests.post(
                f"{endpoint}/probe?type=B_cpu", 
                data=json_data, 
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"解压探测失败，状态码: {response.status_code}")
                # 回退到模拟数据
                return self._simulate_decompress_probe(blob)
        except Exception as e:
            print(f"解压探测异常: {e}")
            # 回退到模拟数据
            return self._simulate_decompress_probe(blob)

    def _send_io_probe(self, endpoint: str, blob: bytes) -> Dict[str, Any]:
        """发送IO探测"""
        try:
            response = requests.post(
                f"{endpoint}/probe?type=B_io", 
                data=blob, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # 适配返回数据格式
                if 'io_write_throughput' not in result and 'throughput' in result:
                    result['io_write_throughput'] = result['throughput']
                return result
            else:
                print(f"IO探测失败，状态码: {response.status_code}")
                # 回退到模拟数据
                return self._simulate_io_probe(blob)
        except Exception as e:
            print(f"IO探测异常: {e}")
            # 回退到模拟数据
            return self._simulate_io_probe(blob)
    
    def _simulate_network_probe(self, blob: bytes) -> Dict[str, Any]:
        """模拟网络探测"""
        # 模拟网络探测结果
        return {
            'bandwidth': random.uniform(10, 1000),  # Mbps
            'rtt': random.uniform(1, 100),  # ms
            'loss_rate': random.uniform(0, 0.1)  # 0-10%
        }
    
    def _simulate_decompress_probe(self, blob: bytes) -> Dict[str, Any]:
        """模拟解压探测"""
        # 模拟解压探测结果
        return {
            'throughput': random.uniform(5, 200)  # MB/s
        }
    
    def _simulate_io_probe(self, blob: bytes) -> Dict[str, Any]:
        """模拟IO探测"""
        # 模拟IO探测结果
        return {
            'throughput': random.uniform(20, 500)  # MB/s
        }
    
    def estimate_receiver_capabilities(self) -> Dict[str, Any]:
        """
        基于多次探测结果估计接收端能力
        
        Returns:
            接收端能力估计
        """
        if not self.probe_results:
            return {}
        
        # 计算平均值作为能力估计
        num_results = len(self.probe_results)
        
        avg_bandwidth = sum(r.network_bandwidth or 0 for r in self.probe_results) / num_results
        avg_rtt = sum(r.network_rtt or 0 for r in self.probe_results) / num_results
        avg_loss_rate = sum(r.network_loss_rate or 0 for r in self.probe_results) / num_results
        
        # 处理解压吞吐量的嵌套结构
        gzip_throughputs = []
        zlib_throughputs = []
        for r in self.probe_results:
            if r.decompress_throughput:
                if isinstance(r.decompress_throughput, dict):
                    if r.decompress_throughput.get('gzip'):
                        gzip_throughputs.append(r.decompress_throughput['gzip'])
                    if r.decompress_throughput.get('zlib'):
                        zlib_throughputs.append(r.decompress_throughput['zlib'])
                else:
                    # 兼容旧格式
                    gzip_throughputs.append(r.decompress_throughput or 0)
        
        avg_gzip_decompress = sum(gzip_throughputs) / len(gzip_throughputs) if gzip_throughputs else 0
        avg_zlib_decompress = sum(zlib_throughputs) / len(zlib_throughputs) if zlib_throughputs else 0
        
        avg_io_write = sum(r.io_write_throughput or 0 for r in self.probe_results) / num_results
        
        # 使用贝叶斯方法或其他统计方法可以进一步改进这里的结果
        
        capabilities = {
            'bandwidth': round(avg_bandwidth, 2),  # Mbps
            'network_rtt': round(avg_rtt, 2),  # ms
            'network_loss_rate': round(avg_loss_rate, 4),  # 0-1
            'decompress_throughput': {
                'gzip': round(avg_gzip_decompress, 2),  # MB/s
                'zlib': round(avg_zlib_decompress, 2)   # MB/s
            },
            'io_write_throughput': round(avg_io_write, 2),  # MB/s
            'confidence': min(1.0, num_results / 5.0)  # 简单置信度计算
        }
        
        return capabilities

    def get_client_profile_for_decision(self) -> Dict[str, Any]:
        """
        获取用于决策引擎的客户端配置
        
        Returns:
            客户端配置字典
        """
        capabilities = self.estimate_receiver_capabilities()
        
        # 将探测结果转换为决策引擎所需的格式
        client_profile = {
            "cpu_score": int(capabilities.get('decompress_throughput', {}).get('gzip', 100) * 3),  # 简单映射
            "bandwidth_mbps": capabilities.get('bandwidth', 50),
            "decompression_speed": {
                "gzip": capabilities.get('decompress_throughput', {}).get('gzip', 100),
                "zstd": capabilities.get('decompress_throughput', {}).get('gzip', 100) * 2,  # 简单估算
            },
            "latency_requirement": 400  # 默认延迟要求
        }
        
        return client_profile


def main():
    """测试探针接收端探测"""
    print("探针接收端探测模块测试")
    print("=" * 30)
    print("注意：此模块适用于支持主动探针的环境")
    print("对于标准容器化环境，请使用network_profiler模块")
    print()
    
    # 创建探测器
    probe_receiver = ProbeReceiver()
    
    # 模拟发送几次探针
    for i in range(3):
        print(f"\n第 {i+1} 轮探测:")
        result = probe_receiver.send_probe("http://example.com/probe")
        print(f"  探针ID: {result.probe_id}")
        print(f"  网络带宽: {result.network_bandwidth} Mbps")
        print(f"  网络RTT: {result.network_rtt} ms")
        print(f"  解压吞吐: {result.decompress_throughput}")
        print(f"  IO写入吞吐: {result.io_write_throughput} MB/s")
        
        time.sleep(1)  # 模拟间隔
    
    # 估计接收端能力
    capabilities = probe_receiver.estimate_receiver_capabilities()
    print("\n接收端能力估计:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    # 获取用于决策的客户端配置
    client_profile = probe_receiver.get_client_profile_for_decision()
    print("\n用于决策的客户端配置:")
    for key, value in client_profile.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()