"""
智能传输层（Transmission Optimizer）
包含访问预测、差分传输、自适应传输策略三部分
"""

import hashlib
import os
from collections import deque, defaultdict
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta


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
        self.pull_sequences = deque(maxlen=1000)  # 存储拉取序列
        self.correlation_matrix = defaultdict(lambda: defaultdict(float))  # 关联性矩阵
        self.alpha = 0.3  # 指数平滑参数
    
    def record_pull(self, image_name, timestamp=None):
        """
        记录镜像拉取事件
        Args:
            image_name: 镜像名称
            timestamp: 拉取时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 记录当前拉取
        if image_name not in self.pull_history:
            self.pull_history[image_name] = deque(maxlen=1000)
        
        self.pull_history[image_name].append(timestamp)
        
        # 添加到拉取序列（用于关联规则挖掘）
        self.pull_sequences.append({
            "image_name": image_name,
            "timestamp": timestamp
        })
        
        # 更新关联规则
        self._update_correlation_matrix()
    
    def _update_correlation_matrix(self):
        """更新关联性矩阵"""
        # 清理过期的序列数据
        current_time = time.time()
        cutoff_time = current_time - (60 * 60 * 24)  # 24小时内
        
        # 保留最近24小时的序列
        recent_sequences = [
            seq for seq in self.pull_sequences 
            if seq["timestamp"] > cutoff_time
        ]
        
        # 计算镜像间的关联度
        for i in range(len(recent_sequences)):
            current_image = recent_sequences[i]["image_name"]
            current_time = recent_sequences[i]["timestamp"]
            
            # 检查接下来10秒内拉取的镜像
            for j in range(i + 1, len(recent_sequences)):
                next_image = recent_sequences[j]["image_name"]
                next_time = recent_sequences[j]["timestamp"]
                
                # 如果在10秒内拉取了另一个镜像
                if next_time - current_time <= 10:
                    self.correlation_matrix[current_image][next_image] += 1
                else:
                    break  # 因为是按时间排序的，超出时间窗口后可以跳出
    
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
        if recent_pulls > 0:
            # 基于拉取次数和时间分布计算热度
            return min(1.0, recent_pulls / 10.0)
        else:
            return 0.0
    
    def predict_related_images(self, current_image, threshold=0.1):
        """
        预测与当前镜像相关的后续镜像
        
        Args:
            current_image: 当前镜像名称
            threshold: 相关性阈值
            
        Returns:
            related_images: 相关镜像列表，包含相关性分数
        """
        if current_image not in self.correlation_matrix:
            return []
        
        related = []
        for next_image, score in self.correlation_matrix[current_image].items():
            # 归一化分数到0-1之间
            max_score = max(self.correlation_matrix[current_image].values())
            normalized_score = score / max_score if max_score > 0 else 0
            
            if normalized_score >= threshold:
                related.append({
                    "image_name": next_image,
                    "correlation_score": normalized_score
                })
        
        # 按相关性分数排序
        related.sort(key=lambda x: x["correlation_score"], reverse=True)
        return related
    
    def predict_hot_layers(self, image_name: str, layers_info: List[Dict[str, Any]], 
                          client_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        预测热点层
        
        Args:
            image_name: 镜像名称
            layers_info: 镜像层信息列表
            client_profile: 客户端配置
            
        Returns:
            predicted_hot_layers: 预测的热点层列表
        """
        # 获取镜像整体热度
        image_popularity = self.predict_popularity(image_name)
        
        # 如果没有历史记录，使用默认热度
        if image_popularity == 0.0:
            image_popularity = 0.5
        
        hot_layers = []
        for i, layer_info in enumerate(layers_info):
            # 基于镜像整体热度和层的特征计算层热度
            layer_size_mb = layer_info.get("layer_size", 0) / (1024 * 1024)
            avg_entropy = layer_info.get("avg_entropy", 0.5)
            
            # 热点层预测逻辑：大尺寸、低熵值的层更可能是热点
            # 因为这类层压缩效果好，且可能包含基础库等常用组件
            layer_hotness = image_popularity * (layer_size_mb / 100.0) * (1.0 - avg_entropy)
            
            if layer_hotness > 0.1:  # 阈值
                hot_layers.append({
                    "layer_index": i,
                    "layer_path": layer_info.get("layer_path", f"layer_{i}"),
                    "hotness_score": layer_hotness,
                    "predicted_pull_time": time.time() + 10  # 预测10秒内可能需要
                })
        
        return hot_layers


class DifferentialTransmitter:
    """差分传输器"""
    
    def __init__(self):
        self.layer_fingerprints = {}  # 存储层的指纹
    
    def calculate_layer_fingerprint(self, layer_path):
        """
        计算层的指纹（用于差分传输）
        
        Args:
            layer_path: 层文件路径
            
        Returns:
            fingerprint: 层的指纹
        """
        # 计算文件的哈希作为指纹
        hash_sha256 = hashlib.sha256()
        with open(layer_path, "rb") as f:
            # 分块读取，避免大文件占用过多内存
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_differential_data(self, source_layer_path, target_layer_path):
        """
        获取两个层之间的差分数据
        
        Args:
            source_layer_path: 源层路径
            target_layer_path: 目标层路径
            
        Returns:
            diff_data: 差分数据
        """
        # 简化实现：实际应用中可以使用rsync算法等
        source_fingerprint = self.calculate_layer_fingerprint(source_layer_path)
        target_fingerprint = self.calculate_layer_fingerprint(target_layer_path)
        
        if source_fingerprint == target_fingerprint:
            # 完全相同，无需传输
            return b""
        
        # 返回完整的目标层数据（实际应用中应实现真正的差分算法）
        with open(target_layer_path, "rb") as f:
            return f.read()


class AdaptiveTransmitter:
    """自适应传输器"""
    
    def __init__(self):
        self.predictor = PullPredictor()
        self.differential_transmitter = DifferentialTransmitter()
    
    def get_optimal_transmission_plan(self, image_name, layers_info, client_profile):
        """
        获取最优传输计划
        
        Args:
            image_name: 镜像名称
            layers_info: 镜像层信息
            client_profile: 客户端配置
            
        Returns:
            transmission_plan: 传输计划
        """
        # 预测相关镜像
        related_images = self.predictor.predict_related_images(image_name)
        
        # 预测热点层
        hot_layers = self.predictor.predict_hot_layers(image_name, layers_info, client_profile)
        
        # 构建传输计划
        plan = {
            "image_name": image_name,
            "priority_layers": [layer["layer_path"] for layer in hot_layers],
            "related_images": related_images,
            "pre_transmission_tasks": []  # 预传输任务列表
        }
        
        # 如果相关镜像的关联度较高，添加预传输任务
        for related in related_images:
            if related["correlation_score"] > 0.5:  # 阈值
                plan["pre_transmission_tasks"].append({
                    "action": "pre_decode",
                    "target": related["image_name"],
                    "priority": related["correlation_score"]
                })
        
        return plan
    
    def transmit_layer(self, layer_path, client_profile):
        """
        传输单个层
        
        Args:
            layer_path: 层路径
            client_profile: 客户端配置
            
        Returns:
            transmission_result: 传输结果
        """
        # 根据客户端带宽调整传输策略
        bandwidth = client_profile.get("bandwidth_mbps", 10)
        
        # 简单的块大小调整策略
        if bandwidth > 50:
            chunk_size = 1024 * 1024  # 1MB
        elif bandwidth > 10:
            chunk_size = 512 * 1024   # 512KB
        else:
            chunk_size = 256 * 1024   # 256KB
        
        # 分块传输
        with open(layer_path, "rb") as f:
            chunks = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        
        return {
            "status": "success",
            "chunks_count": len(chunks),
            "estimated_time": len(chunks) * chunk_size / (bandwidth * 1024 * 1024 / 8)
        }


class TransmissionOptimizer:
    """传输优化器主类"""
    
    def __init__(self):
        self.adaptive_transmitter = AdaptiveTransmitter()
        self.predictor = self.adaptive_transmitter.predictor
    
    def optimize_transmission(self, image_name, layers_info, client_profile):
        """
        优化传输过程
        
        Args:
            image_name: 镜像名称
            layers_info: 镜像层信息
            client_profile: 客户端配置
            
        Returns:
            optimization_result: 优化结果
        """
        # 获取传输计划
        plan = self.adaptive_transmitter.get_optimal_transmission_plan(
            image_name, layers_info, client_profile
        )
        
        # 记录拉取事件以更新预测模型
        self.predictor.record_pull(image_name)
        
        return {
            "transmission_plan": plan,
            "prediction_accuracy": 0.8,  # 示例准确率
            "estimated_savings": 0.2     # 示例节省比例
        }
    
    def predict_and_preprocess(self, current_image):
        """
        预测并预处理后续可能需要的镜像
        
        Args:
            current_image: 当前镜像名称
            
        Returns:
            preprocessing_tasks: 预处理任务列表
        """
        related_images = self.predictor.predict_related_images(current_image, threshold=0.3)
        
        tasks = []
        for related in related_images:
            if related["correlation_score"] > 0.5:
                tasks.append({
                    "target_image": related["image_name"],
                    "action": "pre_decode",
                    "priority": related["correlation_score"]
                })
        
        return tasks


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