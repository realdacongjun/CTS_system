"""
CTS Brain Module (brain.py)
负责加载双塔模型，根据感知数据做出决策
"""
import torch
import torch.nn as nn
import numpy as np
import os
import joblib

# 定义与训练时一致的模型结构 (用于加载权重)
class DualTowerFTTransformer(nn.Module):
    def __init__(self, client_dim=4, image_dim=3): # 简化版维度
        super().__init__()
        self.client_tower = nn.Sequential(nn.Linear(client_dim, 32), nn.ReLU())
        self.image_tower = nn.Sequential(nn.Linear(image_dim, 32), nn.ReLU())
        self.head = nn.Linear(64, 1)
    
    def forward(self, c, i):
        c_out = self.client_tower(c)
        i_out = self.image_tower(i)
        return self.head(torch.cat([c_out, i_out], dim=1))

class DecisionBrain:
    def __init__(self, model_path="../ml_training/models/best_model.pth"):
        self.model = None
        self.device = torch.device("cpu")
        self._load_model(model_path)
        
        self.algos = ["gzip-default", "zstd-fast", "zstd-high", "lz4"]

    def _load_model(self, path):
        if os.path.exists(path):
            try:
                # 这里假设你训练好的模型结构匹配
                self.model = DualTowerFTTransformer()
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
                print("🧠 [Brain] 双塔 AI 模型加载成功！")
            except Exception as e:
                print(f"⚠️ [Brain] 模型加载失败 ({e})，切换至规则引擎模式。")
        else:
            print("ℹ️ [Brain] 未找到模型文件，切换至规则引擎模式。")

    def make_decision(self, client_profile, image_profile):
        """
        输入: 客户端特征 + 镜像特征
        输出: 推荐的压缩算法
        """
        # 1. 规则兜底 (Rule-based Fallback)
        # 如果是极弱网 (带宽 < 5Mbps) -> 强制用高压缩 (Zstd-high)
        if client_profile['bandwidth_mbps'] < 5:
            return "zstd-high", "Rule: Weak Network"
        
        # 如果是极高熵 (已经压缩过的文件) -> 不压缩 (No-op/LZ4)
        if image_profile['avg_layer_entropy'] > 0.95:
            return "lz4", "Rule: High Entropy"

        # 2. AI 预测 (如果有模型)
        if self.model:
            # TODO: 这里需要接入真实的 scaler 和预测逻辑
            # 为了演示，我们暂时返回一个模拟的 AI 决策
            pass

        # 3. 默认逻辑 (Heuristic)
        # 你的 CPU 很强 (3502分)，带宽如果一般，倾向于 Zstd
        if client_profile['cpu_score'] > 2000:
            return "zstd-fast", "Heuristic: High CPU"
        
        return "gzip-default", "Baseline"