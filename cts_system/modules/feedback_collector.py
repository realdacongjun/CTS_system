"""
反馈收集器（Feedback Collector）
目的：收集客户端实际性能数据，用于验证和优化决策模型
"""

import json
import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FeedbackData:
    """反馈数据结构"""
    node_id: str
    image_id: str
    algo_used: str
    actual_transfer_time: float
    actual_decomp_time: float
    predicted_transfer_time: Optional[float] = None
    predicted_decomp_time: Optional[float] = None
    timestamp: float = time.time()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, feedback_file: str = "registry/feedback_data.json"):
        self.feedback_file = feedback_file
        self.feedback_data = []
        self._load_feedback_data()
    
    def _load_feedback_data(self):
        """从文件加载反馈数据"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedback_data = [FeedbackData(**item) for item in data]
            except Exception as e:
                print(f"加载反馈数据失败: {e}")
    
    def _save_feedback_data(self):
        """保存反馈数据到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            
            data = [asdict(feedback) for feedback in self.feedback_data]
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存反馈数据失败: {e}")
    
    def collect_feedback(self, feedback: FeedbackData):
        """
        收集反馈数据
        
        Args:
            feedback: 反馈数据
        """
        self.feedback_data.append(feedback)
        self._save_feedback_data()
        print(f"反馈数据已收集: {feedback.node_id}, {feedback.image_id}, {feedback.algo_used}")
    
    def get_feedback_for_node(self, node_id: str) -> list:
        """
        获取特定节点的反馈数据
        
        Args:
            node_id: 节点ID
            
        Returns:
            该节点的反馈数据列表
        """
        return [f for f in self.feedback_data if f.node_id == node_id]
    
    def get_feedback_for_image(self, image_id: str) -> list:
        """
        获取特定镜像的反馈数据
        
        Args:
            image_id: 镜像ID
            
        Returns:
            该镜像的反馈数据列表
        """
        return [f for f in self.feedback_data if f.image_id == image_id]
    
    def calculate_accuracy(self, node_id: str = None, image_id: str = None) -> Dict[str, float]:
        """
        计算预测准确性
        
        Args:
            node_id: 节点ID（可选）
            image_id: 镜像ID（可选）
            
        Returns:
            准确性指标
        """
        # 过滤数据
        data = self.feedback_data
        if node_id:
            data = [f for f in data if f.node_id == node_id]
        if image_id:
            data = [f for f in data if f.image_id == image_id]
        
        # 过滤掉没有预测值的数据
        data = [f for f in data if f.predicted_transfer_time and f.predicted_decomp_time]
        
        if not data:
            return {"transfer_accuracy": 0.0, "decomp_accuracy": 0.0}
        
        # 计算准确性（使用平均绝对百分比误差 MAPE 的倒数作为准确性）
        transfer_errors = []
        decomp_errors = []
        
        for feedback in data:
            if feedback.predicted_transfer_time > 0:
                transfer_error = abs(feedback.actual_transfer_time - feedback.predicted_transfer_time) / feedback.predicted_transfer_time
                transfer_errors.append(transfer_error)
            
            if feedback.predicted_decomp_time > 0:
                decomp_error = abs(feedback.actual_decomp_time - feedback.predicted_decomp_time) / feedback.predicted_decomp_time
                decomp_errors.append(decomp_error)
        
        avg_transfer_error = sum(transfer_errors) / len(transfer_errors) if transfer_errors else float('inf')
        avg_decomp_error = sum(decomp_errors) / len(decomp_errors) if decomp_errors else float('inf')
        
        # 使用 1/(1+MAPE) 作为准确性指标
        transfer_accuracy = 1 / (1 + avg_transfer_error) if avg_transfer_error != float('inf') else 0.0
        decomp_accuracy = 1 / (1 + avg_decomp_error) if avg_decomp_error != float('inf') else 0.0
        
        return {
            "transfer_accuracy": transfer_accuracy,
            "decomp_accuracy": decomp_accuracy,
            "sample_count": len(data)
        }
    
    def get_recent_feedback(self, hours: int = 24) -> list:
        """
        获取最近若干小时的反馈数据
        
        Args:
            hours: 小时数
            
        Returns:
            最近的反馈数据列表
        """
        cutoff_time = time.time() - (hours * 3600)
        return [f for f in self.feedback_data if f.timestamp >= cutoff_time]
    
    def update_confidence(self, node_id: str, actual_performance: Dict[str, float]):
        """
        更新节点的性能置信度
        
        Args:
            node_id: 节点ID
            actual_performance: 实际性能数据
        """
        # 这里可以实现更新节点置信度的逻辑
        # 例如，根据实际性能与预测性能的差异调整置信度
        pass


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()