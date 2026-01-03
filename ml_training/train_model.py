"""
决策模型训练模块
功能：训练双塔MLP模型用于压缩策略决策
输入：客户端画像、镜像特征、实际成本、使用方法
输出：训练好的MLP模型和特征缩放器
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from .config import get_client_capabilities, get_image_profiles, get_compression_config


class ModelTrainer:
    """决策模型训练器"""
    
    def __init__(self, model_save_path: str = "models", scaler_save_path: str = "models"):
        """
        初始化训练器
        
        Args:
            model_save_path: 模型保存路径
            scaler_save_path: 特征缩放器保存路径
        """
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path
        self.client_scaler = StandardScaler()
        self.image_scaler = StandardScaler()
        self.method_scaler = StandardScaler()
        
        # 从配置中获取实验设计参数
        self.client_profiles = get_client_capabilities()['profiles']
        self.image_profiles = get_image_profiles()
        self.algorithms = get_compression_config()['algorithms']
        
        # 创建保存目录
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(scaler_save_path, exist_ok=True)
    
    def _extract_client_features(self, client_profile: Dict[str, Any]) -> np.ndarray:
        """
        提取客户端特征向量
        
        Args:
            client_profile: 客户端画像
            
        Returns:
            客户端特征向量
        """
        features = [
            client_profile.get('cpu_score', 0),
            client_profile.get('bandwidth_mbps', 0),
            client_profile.get('decompression_speed', {}).get('gzip', 0),
            client_profile.get('decompression_speed', {}).get('zstd', 0),
            client_profile.get('decompression_speed', {}).get('lz4', 0),
            client_profile.get('network_rtt', 0),
            client_profile.get('disk_io_speed', 0),
            client_profile.get('memory_size', 0),
            client_profile.get('latency_requirement', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def _extract_image_features(self, image_profile: Dict[str, Any]) -> np.ndarray:
        """
        提取镜像特征向量
        
        Args:
            image_profile: 镜像特征
            
        Returns:
            镜像特征向量
        """
        features = [
            image_profile.get('total_size_mb', 0),
            image_profile.get('avg_layer_entropy', 0),
            image_profile.get('text_ratio', 0),
            image_profile.get('binary_ratio', 0),
            image_profile.get('layer_count', 0),
            image_profile.get('file_type_distribution', {}).get('text', 0),
            image_profile.get('file_type_distribution', {}).get('binary', 0),
            image_profile.get('file_type_distribution', {}).get('compressed', 0),
            image_profile.get('avg_file_size', 0),
            image_profile.get('compression_ratio_estimate', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def _encode_method(self, method: str) -> np.ndarray:
        """
        对压缩方法进行独热编码
        
        Args:
            method: 压缩方法字符串（如 "gzip-6", "zstd-3", "lz4-fast"）
            
        Returns:
            独热编码向量
        """
        # 定义支持的算法类型
        algorithms = self.algorithms
        # 获取当前方法在算法列表中的位置
        try:
            idx = algorithms.index(method)
            # 创建独热编码向量
            encoded = [0] * len(algorithms)
            encoded[idx] = 1
        except ValueError:
            # 如果方法不在列表中，使用默认方法
            default_method = 'gzip-6'
            try:
                idx = algorithms.index(default_method)
                encoded = [0] * len(algorithms)
                encoded[idx] = 1
            except ValueError:
                # 如果默认方法也不在列表中，使用全零向量
                encoded = [0] * len(algorithms)
        
        return np.array(encoded).reshape(1, -1)
    
    def prepare_features(self, training_data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练特征
        
        Args:
            training_data: 训练数据列表 [(client_profile, image_profile, actual_cost, method), ...]
            
        Returns:
            特征矩阵和目标向量
        """
        client_features = []
        image_features = []
        method_features = []
        targets = []
        
        for client_profile, image_profile, actual_cost, method in training_data:
            client_features.append(self._extract_client_features(client_profile)[0])
            image_features.append(self._extract_image_features(image_profile)[0])
            method_features.append(self._encode_method(method)[0])
            targets.append(actual_cost)
        
        # 转换为numpy数组
        client_features = np.vstack(client_features)
        image_features = np.vstack(image_features)
        method_features = np.vstack(method_features)
        targets = np.array(targets)
        
        # 标准化特征
        client_features_scaled = self.client_scaler.fit_transform(client_features)
        image_features_scaled = self.image_scaler.fit_transform(image_features)
        method_features_scaled = self.method_scaler.fit_transform(method_features)
        
        # 连接所有特征
        all_features = np.hstack([
            client_features_scaled,
            image_features_scaled,
            method_features_scaled
        ])
        
        return all_features, targets
    
    def train_model(self, training_data: List[Tuple], test_size: float = 0.2) -> Dict[str, float]:
        """
        训练MLP模型
        
        Args:
            training_data: 训练数据列表
            test_size: 测试集比例
            
        Returns:
            评估指标字典
        """
        print("开始准备训练特征...")
        features, targets = self.prepare_features(training_data)
        
        print(f"特征矩阵形状: {features.shape}")
        print(f"目标向量形状: {targets.shape}")
        print(f"训练样本数量: {len(training_data)}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 创建并训练MLP模型
        print("开始训练MLP模型...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        self.model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # 计算评估指标
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        print("模型训练完成!")
        print(f"训练集 MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
        
        return metrics
    
    def save_model(self):
        """保存训练好的模型和缩放器"""
        # 保存模型
        model_path = os.path.join(self.model_save_path, "dual_tower_mlp_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存缩放器
        client_scaler_path = os.path.join(self.scaler_save_path, "client_scaler.pkl")
        image_scaler_path = os.path.join(self.scaler_save_path, "image_scaler.pkl")
        method_scaler_path = os.path.join(self.scaler_save_path, "method_scaler.pkl")
        
        with open(client_scaler_path, 'wb') as f:
            pickle.dump(self.client_scaler, f)
        
        with open(image_scaler_path, 'wb') as f:
            pickle.dump(self.image_scaler, f)
        
        with open(method_scaler_path, 'wb') as f:
            pickle.dump(self.method_scaler, f)
        
        print(f"模型已保存至: {model_path}")
        print(f"缩放器已保存至: {client_scaler_path}, {image_scaler_path}, {method_scaler_path}")
    
    def load_training_data_from_feedback(self, feedback_dir: str = "registry") -> List[Tuple]:
        """
        从反馈数据中加载训练数据
        
        Args:
            feedback_dir: 反馈数据目录
            
        Returns:
            训练数据列表
        """
        training_data = []
        
        # 假设反馈数据存储在JSON文件中
        feedback_file = os.path.join(feedback_dir, "feedback_data.json")
        
        if not os.path.exists(feedback_file):
            print(f"警告: 找不到反馈数据文件 {feedback_file}")
            return training_data
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            for record in feedback_data:
                # 提取客户端画像
                client_profile = {
                    'cpu_score': record.get('cpu_score', 0),
                    'bandwidth_mbps': record.get('bandwidth_mbps', 0),
                    'decompression_speed': record.get('decompression_speed', {}),
                    'network_rtt': record.get('network_rtt', 0),
                    'disk_io_speed': record.get('disk_io_speed', 0),
                    'memory_size': record.get('memory_size', 0),
                    'latency_requirement': record.get('latency_requirement', 0)
                }
                
                # 提取镜像特征
                image_profile = {
                    'total_size_mb': record.get('total_size_mb', 0),
                    'avg_layer_entropy': record.get('avg_layer_entropy', 0),
                    'text_ratio': record.get('text_ratio', 0),
                    'binary_ratio': record.get('binary_ratio', 0),
                    'layer_count': record.get('layer_count', 0),
                    'file_type_distribution': record.get('file_type_distribution', {}),
                    'avg_file_size': record.get('avg_file_size', 0),
                    'compression_ratio_estimate': record.get('compression_ratio_estimate', 0)
                }
                
                # 计算实际成本
                actual_cost = (
                    record.get('actual_compress_time', 0) +
                    record.get('actual_transfer_time', 0) +
                    record.get('actual_decomp_time', 0)
                )
                
                # 获取使用的方法
                method = record.get('algo_used', 'gzip-6')
                
                training_data.append((client_profile, image_profile, actual_cost, method))
        
        except Exception as e:
            print(f"加载反馈数据时出错: {e}")
        
        return training_data


def main():
    """主函数，用于训练模型"""
    print("开始训练决策模型...")
    
    # 初始化训练器
    trainer = ModelTrainer()
    
    # 从反馈数据加载训练数据
    print("从反馈数据中加载训练数据...")
    training_data = trainer.load_training_data_from_feedback()
    
    # 如果没有从反馈中加载到数据，生成一些示例数据
    if not training_data:
        print("未找到反馈数据，生成示例训练数据...")
        training_data = generate_sample_data()
    
    print(f"加载到 {len(training_data)} 条训练数据")
    
    if len(training_data) == 0:
        print("没有训练数据，无法训练模型")
        return
    
    # 训练模型
    metrics = trainer.train_model(training_data)
    
    # 保存模型
    trainer.save_model()
    
    print("模型训练完成!")
    print(f"评估指标: {metrics}")


def generate_sample_data() -> List[Tuple]:
    """
    生成示例训练数据（用于测试）
    
    Returns:
        示例训练数据
    """
    sample_data = []
    
    # 使用配置中的实验设计参数
    from .config import CLIENT_CAPABILITIES, IMAGE_PROFILES, COMPRESSION_CONFIG
    
    # 生成实验设计中的数据
    for client_profile in CLIENT_CAPABILITIES['profiles']:
        for image_profile in IMAGE_PROFILES:
            for method in COMPRESSION_CONFIG['algorithms']:
                # 计算一个模拟的实际成本（基于特征的简单函数）
                actual_cost = (
                    image_profile['total_size_mb'] / client_profile['decompression_speed']['gzip'] +
                    image_profile['total_size_mb'] * 8 / client_profile['bandwidth_mbps'] +
                    np.random.uniform(0.1, 1.0)  # 添加一些随机性
                )
                
                sample_data.append((client_profile, image_profile, actual_cost, method))
    
    return sample_data


if __name__ == "__main__":
    main()