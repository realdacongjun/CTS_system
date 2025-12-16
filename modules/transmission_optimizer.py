"""
智能传输层（Transmission Optimizer）
包含访问预测、差分传输、自适应传输策略三部分
"""


import hashlib
import os
from collections import deque
import time


class PullPredictor:
    """访问预测器"""
    
    def __init__(self, window_size=24):
        """
        初始化访问预测器
        
        Args:
            window_size: 时间窗口大小（小时）
        """
        self.window_size = window_size
        self.pull_history = {}  # 镜像拉取历史记录
        self.alpha = 0.3  # 指数平滑参数
    
    def record_pull(self, image_name):
        """
        记录镜像拉取事件
        
        Args:
            image_name: 镜像名称
        """
        now = time.time()
        if image_name not in self.pull_history:
            self.pull_history[image_name] = deque()
        
        self.pull_history[image_name].append(now)
        
        # 清理过期记录（超过时间窗口的记录）
        self._cleanup_old_records(image_name, now)
    
    def _cleanup_old_records(self, image_name, current_time):
        """清理过期记录"""
        if image_name in self.pull_history:
            expiration_time = current_time - (self.window_size * 3600)
            while (self.pull_history[image_name] and 
                   self.pull_history[image_name][0] < expiration_time):
                self.pull_history[image_name].popleft()
    
    def predict_popularity(self, image_name):
        """
        预测镜像热度
        
        Args:
            image_name: 镜像名称
            
        Returns:
            popularity: 热度值（0-1）
        """
        if image_name not in self.pull_history:
            return 0.0
        
        # 计算近期拉取频率
        recent_pulls = len(self.pull_history[image_name])
        
        # 简单的指数平滑预测
        # 这里简化处理，实际应用中可以更复杂
        if recent_pulls > 0:
            # 基于拉取次数和时间分布计算热度
            return min(1.0, recent_pulls / 10.0)
        else:
            return 0.0


class DeltaEngine:
    """差分传输引擎"""
    
    def __init__(self, block_size=4096):
        """
        初始化差分传输引擎
        
        Args:
            block_size: 块大小
        """
        self.block_size = block_size
    
    def compute_rolling_hash(self, data):
        """
        计算滚动哈希（简化版 Adler-32）
        
        Args:
            data: 字节数据
            
        Returns:
            hash_value: 哈希值
        """
        MOD_ADLER = 65521
        a = 1
        b = 0
        
        for byte in data:
            a = (a + byte) % MOD_ADLER
            b = (b + a) % MOD_ADLER
            
        return (b << 16) | a
    
    def generate_delta(self, old_data, new_data):
        """
        生成差分数据
        
        Args:
            old_data: 旧数据
            new_data: 新数据
            
        Returns:
            delta_info: 差分信息
        """
        # 简化的差分算法
        # 实际应用中可以使用更复杂的rsync算法
        
        old_blocks = {}
        new_blocks = {}
        
        # 分割为块并计算哈希
        for i in range(0, len(old_data), self.block_size):
            block = old_data[i:i+self.block_size]
            hash_val = self.compute_rolling_hash(block)
            old_blocks[hash_val] = block
        
        for i in range(0, len(new_data), self.block_size):
            block = new_data[i:i+self.block_size]
            hash_val = self.compute_rolling_hash(block)
            new_blocks[hash_val] = block
        
        # 计算重复块和新块
        reuse_blocks = 0
        new_block_count = 0
        delta_size = 0
        
        for hash_val, block in new_blocks.items():
            if hash_val in old_blocks:
                reuse_blocks += 1
            else:
                new_block_count += 1
                delta_size += len(block)
        
        return {
            "total_delta_size": delta_size / 1024.0 / 1024.0,  # MB
            "reuse_blocks": reuse_blocks,
            "new_blocks": new_block_count
        }


class AdaptiveTransmissionStrategy:
    """自适应传输策略"""
    
    def __init__(self):
        pass
    
    def select_transmission_method(self, client_profile, delta_info, compressed_size):
        """
        选择传输方法
        
        Args:
            client_profile: 客户端配置
            delta_info: 差分信息
            compressed_size: 压缩后大小
            
        Returns:
            method: 传输方法
            reason: 选择原因
        """
        bandwidth = client_profile.get("bandwidth_mbps", 50)
        delta_size = delta_info.get("total_delta_size", compressed_size)
        
        # 如果差分数据很小，选择差分传输
        if delta_size < compressed_size * 0.5:
            return "delta", f"差分传输更优 (delta: {delta_size:.1f}MB < full: {compressed_size:.1f}MB)"
        
        # 如果带宽很低，选择预先压缩好的格式
        if bandwidth < 20:
            return "pre_compressed", "低带宽环境下使用预压缩数据"
        
        # 默认使用完整传输
        return "full", "默认完整传输"


class TransmissionOptimizer:
    """传输优化器主类"""
    
    def __init__(self):
        self.predictor = PullPredictor()
        self.delta_engine = DeltaEngine()
        self.strategy = AdaptiveTransmissionStrategy()
    
    def optimize_transmission(self, image_name, client_profile, image_data, cached_data=None):
        """
        优化传输过程
        
        Args:
            image_name: 镜像名称
            client_profile: 客户端配置
            image_data: 镜像数据
            cached_data: 缓存数据（用于差分）
            
        Returns:
            optimization_result: 优化结果
        """
        # 1. 记录拉取事件并预测热度
        self.predictor.record_pull(image_name)
        predicted_popularity = self.predictor.predict_popularity(image_name)
        
        # 2. 如果有缓存数据，计算差分
        delta_info = None
        if cached_data:
            try:
                # 简化处理，实际应用中需要读取文件内容
                delta_info = self.delta_engine.generate_delta(cached_data, image_data)
            except Exception as e:
                print(f"差分计算失败: {e}")
                delta_info = {
                    "total_delta_size": len(image_data) / 1024.0 / 1024.0,
                    "reuse_blocks": 0,
                    "new_blocks": len(image_data) // 4096 + 1
                }
        else:
            # 没有缓存数据，需要完整传输
            delta_info = {
                "total_delta_size": len(image_data) / 1024.0 / 1024.0,
                "reuse_blocks": 0,
                "new_blocks": len(image_data) // 4096 + 1
            }
        
        # 3. 选择传输策略
        transmission_method, reason = self.strategy.select_transmission_method(
            client_profile, delta_info, len(image_data) / 1024.0 / 1024.0)
        
        return {
            "predicted_popularity": predicted_popularity,
            "delta_info": delta_info,
            "transmission_method": transmission_method,
            "strategy_reason": reason,
            "estimated_transmission_size": delta_info["total_delta_size"] if transmission_method == "delta" else len(image_data) / 1024.0 / 1024.0
        }


def main():
    """测试传输优化器"""
    optimizer = TransmissionOptimizer()
    
    # 模拟数据
    image_name = "nginx:latest"
    client_profile = {
        "bandwidth_mbps": 68,
        "decompression_speed": 280,
        "cpu_score": 3100
    }
    
    # 模拟镜像数据
    image_data = b"x" * 1024 * 1024  # 1MB数据
    cached_data = b"x" * 512 * 1024 + b"y" * 512 * 1024  # 512KB相同 + 512KB不同
    
    result = optimizer.optimize_transmission(image_name, client_profile, image_data, cached_data)
    
    print("传输优化结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()