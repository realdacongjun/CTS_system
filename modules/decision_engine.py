import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
from typing import Dict, Any, Tuple
import threading

"""
压缩决策器（Compression Decision Engine）
目的：根据客户端和镜像的特征，选择最优的压缩算法和级别

决策公式：
总成本 = 压缩端耗时 + 传输耗时 + 客户端解压时间
"""


class CompressionDecisionEngine:
    """压缩决策引擎"""
    
    def __init__(self, strategy="ml_model", inference_timeout=0.05):  # 50ms超时
        """
        初始化压缩决策引擎
        
        Args:
            strategy: 决策策略，可选 "cost_model"（基于成本模型）或 "ml_model"（基于机器学习模型）
            inference_timeout: MLP推理超时时间（秒）
        """
        # 预定义的压缩算法选项
        self.compression_options = [
            "gzip-1", "gzip-6", "gzip-9",
            "zstd-1", "zstd-3", "zstd-5", "zstd-10", "zstd-19",
            "lz4-fast", "lz4-10", "lz4-12"
        ]
        
        self.strategy = strategy
        self.inference_timeout = inference_timeout  # 推理超时时间
        self.ml_model = None
        self.scaler = StandardScaler()
        
        # 如果使用机器学习策略，加载模型
        if self.strategy == "ml_model":
            self._load_ml_model()
    
    def _load_ml_model(self):
        """
        加载机器学习模型
        这里实现双塔MLP模型加载
        """
        model_path = "models/dual_tower_mlp_model.pkl"
        scaler_path = "models/scaler.pkl"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("双塔MLP模型加载成功")
            else:
                print("未找到预训练模型，初始化新模型")
                # 初始化模型
                self.ml_model = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42
                )
                # 由于没有训练数据，暂时回退到成本模型
                print("警告: 模型未训练，将回退到成本模型策略")
                self.strategy = "cost_model"
        except Exception as e:
            print(f"机器学习模型加载失败: {e}")
            # 回退到成本模型
            self.strategy = "cost_model"
    
    def calculate_cost(self, client_profile, image_profile, compression_method):
        """
        计算特定压缩方法的总成本
        
        Args:
            client_profile: 客户端配置
            image_profile: 镜像配置
            compression_method: 压缩方法
            
        Returns:
            cost: 总成本
            details: 成本明细
        """
        # 获取基础参数
        bandwidth_mbps = client_profile.get("bandwidth_mbps", 50)
        decompression_speed = client_profile.get("decompression_speed", {"gzip": 50, "zstd": 100}).get(
            compression_method.split('-')[0], 100
        )
        cpu_score = client_profile.get("cpu_score", 1000)
        
        total_size_mb = image_profile.get("total_size_mb", 100)
        entropy = image_profile.get("avg_layer_entropy", 0.5)
        text_ratio = image_profile.get("text_ratio", 0.2)
        binary_ratio = image_profile.get("binary_ratio", 0.7)
        
        # 根据压缩方法估算压缩比和压缩时间
        compression_ratio, compression_time_multiplier = self._get_compression_stats(compression_method, entropy, text_ratio)
        
        # 计算压缩后大小
        compressed_size_mb = total_size_mb * compression_ratio
        
        # 计算各项成本
        # 传输时间（秒）= 大小（Mb）* 8 / 带宽（Mbps）
        transmission_time = (compressed_size_mb * 8) / bandwidth_mbps
        
        # 压缩时间（与CPU性能和压缩算法相关）
        compression_time = total_size_mb * compression_time_multiplier * (1000 / cpu_score)
        
        # 解压时间（与客户端解压性能相关）
        decompression_time = (compressed_size_mb * 10) / decompression_speed  # 简化的计算模型
        
        # 总成本
        total_cost = compression_time + transmission_time + decompression_time
        
        details = {
            "compressed_size_mb": compressed_size_mb,
            "transmission_time": transmission_time,
            "compression_time": compression_time,
            "decompression_time": decompression_time
        }
        
        return total_cost, details
    
    def calculate_layer_cost(self, client_profile: dict, layer_info: dict, compression_method: str) -> tuple:
        """
        计算单个层的成本
        """
        bandwidth_mbps = client_profile.get("bandwidth_mbps", 50)
        decompression_speed = client_profile.get("decompression_speed", {"gzip": 50, "zstd": 100}).get(
            compression_method.split('-')[0], 100
        )
        cpu_score = client_profile.get("cpu_score", 1000)
        
        layer_size_mb = layer_info.get("size", 0) / (1024 * 1024)
        entropy = layer_info.get("avg_entropy", 0.5)
        text_ratio = layer_info.get("text_ratio", 0.2)
        binary_ratio = layer_info.get("binary_ratio", 0.7)
        
        # 根据压缩方法估算压缩比和压缩时间
        compression_ratio, compression_time_multiplier = self._get_compression_stats(compression_method, entropy, text_ratio)
        
        # 计算压缩后大小
        compressed_size_mb = layer_size_mb * compression_ratio
        
        # 计算各项成本
        transmission_time = (compressed_size_mb * 8) / bandwidth_mbps
        compression_time = layer_size_mb * compression_time_multiplier * (1000 / cpu_score)
        decompression_time = (compressed_size_mb * 10) / decompression_speed  # 简化的计算模型
        
        total_cost = compression_time + transmission_time + decompression_time
        
        details = {
            "compressed_size_mb": compressed_size_mb,
            "transmission_time": transmission_time,
            "compression_time": compression_time,
            "decompression_time": decompression_time
        }
        
        return total_cost, details
    
    def _get_compression_stats(self, compression_method, entropy, text_ratio):
        """
        获取压缩方法的统计信息
        """
        # 根据压缩方法和特征估算压缩比和时间倍数
        algo, level = compression_method.split('-', 1)
        
        # 基础压缩比（根据熵值调整）
        base_ratio = 1.0 - (entropy * 0.7)  # 最大压缩率为30%
        
        # 根据算法和级别调整
        if algo == "gzip":
            level_factor = {
                "1": 0.8, "6": 1.0, "9": 1.1
            }.get(level, 1.0)
            time_multiplier = {
                "1": 0.5, "6": 1.0, "9": 1.5
            }.get(level, 1.0)
        elif algo == "zstd":
            level_factor = {
                "1": 0.9, "3": 1.0, "5": 1.05, "10": 1.1, "19": 1.15
            }.get(level, 1.0)
            time_multiplier = {
                "1": 0.3, "3": 0.5, "5": 0.8, "10": 1.2, "19": 2.0
            }.get(level, 1.0)
        elif algo == "lz4":
            level_factor = {
                "fast": 0.7, "10": 0.8, "12": 0.85
            }.get(level, 0.7)
            time_multiplier = {
                "fast": 0.2, "10": 0.5, "12": 0.7
            }.get(level, 0.2)
        else:
            level_factor = 1.0
            time_multiplier = 1.0
        
        # 调整压缩比，文本文件通常有更好的压缩效果
        if text_ratio > 0.5:
            base_ratio *= 0.8  # 文本文件压缩效果更好
        
        return base_ratio * level_factor, time_multiplier
    
    def _prepare_features(self, client_profile, image_profile, method=None):
        """
        准备特征向量，将客户端画像和镜像特征组合
        """
        # 客户端特征
        client_features = [
            client_profile.get("cpu_score", 1000),
            client_profile.get("bandwidth_mbps", 50),
            client_profile.get("network_rtt", 0.0),
            client_profile.get("disk_io_speed", 0.0),
            client_profile.get("memory_size", 0),
            client_profile.get("decompression_speed", {"gzip": 50, "zstd": 100}).get("gzip", 50),
            client_profile.get("decompression_speed", {"gzip": 50, "zstd": 100}).get("zstd", 100),
            client_profile.get("decompression_speed", {"gzip": 50, "zstd": 100}).get("lz4", 100),
        ]
        
        # 镜像特征
        image_features = [
            image_profile.get("total_size_mb", 100),
            image_profile.get("avg_layer_entropy", 0.5),
            image_profile.get("text_ratio", 0.2),
            image_profile.get("binary_ratio", 0.7),
            image_profile.get("layer_count", 1),
            image_profile.get("file_type_distribution", {}).get("text", 0),
            image_profile.get("file_type_distribution", {}).get("binary", 0),
        ]
        
        # 如果提供了压缩方法，添加到特征向量
        if method:
            method_features = [1 if method == option else 0 for option in self.compression_options]
            features = client_features + image_features + method_features
        else:
            features = client_features + image_features
        
        return np.array(features).reshape(1, -1)
    
    def _predict_with_ml_model(self, client_profile, image_profile):
        """
        使用机器学习模型进行预测
        """
        # 为每个压缩选项预测成本
        predictions = {}
        
        for method in self.compression_options:
            # 准备特征
            features = self._prepare_features(client_profile, image_profile)
            # 标准化
            features_scaled = self.scaler.transform(features)
            
            # 预测成本
            cost = self.ml_model.predict(features_scaled)[0]
            predictions[method] = cost
        
        # 选择成本最低的选项
        best_method = min(predictions, key=predictions.get)
        
        return best_method, {
            "method_costs": predictions,
            "predicted_best": best_method,
            "predicted_cost": predictions[best_method]
        }
    
    def _predict_with_cost_model(self, client_profile, image_profile):
        """
        使用成本模型进行预测
        """
        method_costs = {}
        
        for method in self.compression_options:
            cost, _ = self.calculate_cost(client_profile, image_profile, method)
            method_costs[method] = cost
        
        # 找到成本最低的方法
        best_method = min(method_costs, key=method_costs.get)
        
        return best_method, {
            "predicted_cost": method_costs[best_method],
            "method_costs": method_costs,
            "strategy_used": "cost_model"
        }
    
    def make_decision(self, client_profile, image_profile):
        """
        基于客户端画像和镜像特征做出压缩策略决策
        
        Args:
            client_profile: 客户端画像
            image_profile: 镜像特征
            
        Returns:
            best_method: 最优压缩方法
            prediction_details: 预测详情
        """
        if self.strategy == "ml_model" and self.ml_model:
            # 使用双塔MLP模型进行预测
            return self._predict_with_ml_model(client_profile, image_profile)
        else:
            # 使用成本模型
            return self._predict_with_cost_model(client_profile, image_profile)
    
    def train_model(self, training_data):
        """
        训练MLP模型
        
        Args:
            training_data: 训练数据，格式为 [(client_profile, image_profile, actual_cost, method), ...]
        """
        if self.strategy != "ml_model":
            return
        
        print("开始训练双塔MLP模型...")
        
        # 准备训练数据
        X = []
        y = []
        
        for client_profile, image_profile, actual_cost, method in training_data:
            features = self._prepare_features(client_profile, image_profile).flatten()
            X.append(features)
            y.append(actual_cost)
        
        X = np.array(X)
        y = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.ml_model.fit(X_scaled, y)
        
        # 保存模型
        os.makedirs("models", exist_ok=True)
        model_path = "models/dual_tower_mlp_model.pkl"
        scaler_path = "models/scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.ml_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("模型训练完成并已保存")


def main():
    """测试压缩决策引擎"""
    # 测试传统成本模型
    print("测试传统成本模型:")
    engine_cost = CompressionDecisionEngine(strategy="cost_model")
    
    # 模拟客户端和镜像配置
    client_profile = {
        "cpu_score": 3100,
        "bandwidth_mbps": 68,
        "decompression_speed": 280,
        "latency_requirement": 400
    }
    
    image_profile = {
        "layer_count": 6,
        "total_size_mb": 742,
        "avg_layer_entropy": 0.78,
        "text_ratio": 0.18,
        "binary_ratio": 0.82,
        "compressed_file_ratio": 0.12,
        "avg_file_size_kb": 42.3,
        "median_file_size_kb": 8.1,
        "small_file_ratio": 0.67,
        "big_file_ratio": 0.15,
        "predict_popularity": 0.54
    }
    
    decision = engine_cost.make_decision(client_profile, image_profile)
    print("压缩决策结果:")
    for key, value in decision.items():
        print(f"  {key}: {value}")
    
    print("\n测试机器学习模型（模拟）:")
    # 测试机器学习模型（模拟）
    engine_ml = CompressionDecisionEngine(strategy="ml_model")
    decision_ml = engine_ml.make_decision(client_profile, image_profile)
    print("压缩决策结果:")
    for key, value in decision_ml.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()