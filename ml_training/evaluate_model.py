"""
模型评估模块
功能：评估训练好的决策模型性能
输入：测试数据、已训练模型
输出：评估指标、预测结果分析
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
from .config import get_client_capabilities, get_image_profiles, get_compression_config


class ModelEvaluator:
    """决策模型评估器"""
    
    def __init__(self, model_path: str = "models/dual_tower_mlp_model.pkl", 
                 scaler_paths: Dict[str, str] = None):
        """
        初始化评估器
        
        Args:
            model_path: 模型文件路径
            scaler_paths: 特征缩放器文件路径字典
        """
        self.model_path = model_path
        self.scaler_paths = scaler_paths or {
            'client': 'models/client_scaler.pkl',
            'image': 'models/image_scaler.pkl',
            'method': 'models/method_scaler.pkl'
        }
        
        self.model = None
        self.client_scaler = None
        self.image_scaler = None
        self.method_scaler = None
        
        # 从配置中获取实验设计参数
        self.client_profiles = get_client_capabilities()['profiles']
        self.image_profiles = get_image_profiles()
        self.algorithms = get_compression_config()['algorithms']
        
        self._load_model()
    
    def _load_model(self):
        """加载训练好的模型和缩放器"""
        # 加载模型
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"模型已从 {self.model_path} 加载")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载缩放器
        scaler_files = [
            (self.scaler_paths['client'], 'client_scaler'),
            (self.scaler_paths['image'], 'image_scaler'),
            (self.scaler_paths['method'], 'method_scaler')
        ]
        
        for path, attr_name in scaler_files:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    setattr(self, attr_name, pickle.load(f))
                print(f"{attr_name} 已从 {path} 加载")
            else:
                raise FileNotFoundError(f"缩放器文件不存在: {path}")
    
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
        # 获取当前方法在算法列表中的位置
        try:
            idx = self.algorithms.index(method)
            # 创建独热编码向量
            encoded = [0] * len(self.algorithms)
            encoded[idx] = 1
        except ValueError:
            # 如果方法不在列表中，使用默认方法
            default_method = 'gzip-6'
            try:
                idx = self.algorithms.index(default_method)
                encoded = [0] * len(self.algorithms)
                encoded[idx] = 1
            except ValueError:
                # 如果默认方法也不在列表中，使用全零向量
                encoded = [0] * len(self.algorithms)
        
        return np.array(encoded).reshape(1, -1)
    
    def prepare_features(self, test_data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备测试特征
        
        Args:
            test_data: 测试数据列表 [(client_profile, image_profile, actual_cost, method), ...]
            
        Returns:
            特征矩阵和目标向量
        """
        client_features = []
        image_features = []
        method_features = []
        targets = []
        
        for client_profile, image_profile, actual_cost, method in test_data:
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
        client_features_scaled = self.client_scaler.transform(client_features)
        image_features_scaled = self.image_scaler.transform(image_features)
        method_features_scaled = self.method_scaler.transform(method_features)
        
        # 连接所有特征
        all_features = np.hstack([
            client_features_scaled,
            image_features_scaled,
            method_features_scaled
        ])
        
        return all_features, targets
    
    def evaluate(self, test_data: List[Tuple]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据列表
            
        Returns:
            评估指标字典
        """
        if not test_data:
            raise ValueError("测试数据不能为空")
        
        print("准备测试特征...")
        features, targets = self.prepare_features(test_data)
        
        print(f"特征矩阵形状: {features.shape}")
        print(f"目标向量形状: {targets.shape}")
        print(f"测试样本数量: {len(test_data)}")
        
        # 进行预测
        print("进行预测...")
        predictions = self.model.predict(features)
        
        # 计算评估指标
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 计算平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # 计算决策损益比
        decision_loss_ratio = self.calculate_decision_loss_ratio(test_data, predictions)
        
        # 预测最优方法
        predicted_best_methods = []
        for client_profile, image_profile, _, _ in test_data:
            pred_method, _ = self.predict_optimal_method(client_profile, image_profile)
            predicted_best_methods.append(pred_method)
        
        # 计算算法选择准确率
        method_accuracy = self.evaluate_algorithm_selection_accuracy(test_data)
        
        metrics = {
            'basic_metrics': {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'mape': mape
            },
            'decision_metrics': {
                'decision_loss_ratio': decision_loss_ratio,
                'method_accuracy': method_accuracy,
                'total_samples': len(test_data),
                'correct_method_predictions': int(method_accuracy * len(test_data) / 100) if method_accuracy >= 0 else 0
            },
            'predictions': {
                'y_true': targets.tolist(),
                'y_pred': predictions.tolist(),
                'methods_true': [method for _, _, _, method in test_data],
                'methods_predicted': predicted_best_methods
            }
        }
        
        print("模型评估完成!")
        print(f"\n=== 基本评估指标 ===")
        print(f"MSE (均方误差): {mse:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"RMSE (均方根误差): {np.sqrt(mse):.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
        
        print(f"\n=== 决策评估指标 ===")
        print(f"决策损益比: {decision_loss_ratio:.2f}%")
        print(f"算法选择准确率: {method_accuracy:.2f}%")
        print(f"总样本数: {len(test_data)}")
        
        return metrics
    
    def plot_predictions(self, test_data: List[Tuple], save_path: str = None):
        """
        绘制预测结果对比图
        
        Args:
            test_data: 测试数据列表
            save_path: 保存图片路径，如果为None则显示图片
        """
        features, targets = self.prepare_features(test_data)
        predictions = self.model.predict(features)
        
        # 创建对比图
        plt.figure(figsize=(12, 5))
        
        # 实际值vs预测值散点图
        plt.subplot(1, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('实际值 vs 预测值')
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = targets - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
    
    def predict_optimal_method(self, client_profile: Dict[str, Any], 
                              image_profile: Dict[str, Any]) -> Tuple[str, float]:
        """
        预测最优压缩方法
        
        Args:
            client_profile: 客户端画像
            image_profile: 镜像特征
            
        Returns:
            (最优方法, 预测成本)
        """
        costs = []
        
        for method in self.algorithms:
            # 准备单个样本特征
            client_features = self._extract_client_features(client_profile)
            image_features = self._extract_image_features(image_profile)
            method_features = self._encode_method(method)
            
            # 标准化特征
            client_features_scaled = self.client_scaler.transform(client_features)
            image_features_scaled = self.image_scaler.transform(image_features)
            method_features_scaled = self.method_scaler.transform(method_features)
            
            # 连接特征
            all_features = np.hstack([
                client_features_scaled,
                image_features_scaled,
                method_features_scaled
            ])
            
            # 预测成本
            cost = self.model.predict(all_features)[0]
            costs.append(cost)
        
        # 找到最小成本的索引
        min_idx = np.argmin(costs)
        optimal_method = self.algorithms[min_idx]
        optimal_cost = costs[min_idx]
        
        return optimal_method, optimal_cost
    
    def evaluate_algorithm_selection_accuracy(self, test_data: List[Tuple]) -> float:
        """
        评估算法选择准确性
        计算模型选择的算法是否与实际最优算法一致的比例
        
        Args:
            test_data: 测试数据列表
            
        Returns:
            算法选择准确率（百分比）
        """
        correct_predictions = 0
        total_predictions = len(test_data)
        
        for client_profile, image_profile, actual_cost, actual_method in test_data:
            # 预测最优方法
            predicted_method, predicted_cost = self.predict_optimal_method(client_profile, image_profile)
            
            # 检查是否选择了最优方法
            if predicted_method == actual_method:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        return accuracy
    
    def calculate_decision_loss_ratio(self, test_data: List[Tuple], predictions: np.ndarray) -> float:
        """
        计算决策损益比
        
        Args:
            test_data: 测试数据列表
            predictions: 模型预测值
            
        Returns:
            决策损益比（百分比）
        """
        total_loss = 0
        valid_samples = 0
        
        # 按客户端和镜像分组测试数据
        sample_groups = {}
        for i, (client_profile, image_profile, actual_cost, method) in enumerate(test_data):
            group_key = (str(client_profile), str(image_profile))
            if group_key not in sample_groups:
                sample_groups[group_key] = []
            sample_groups[group_key].append((i, method, actual_cost))
        
        # 对每个分组计算决策损益比
        for group_key, samples in sample_groups.items():
            if len(samples) > 1:  # 只有当同一配置有多个方法测试时才计算
                # 找到真实最优成本
                true_best_cost = min(sample[2] for sample in samples)
                true_best_method = next(sample[1] for sample in samples if sample[2] == true_best_cost)
                
                # 对每个样本计算损益比
                for idx, method, actual_cost in samples:
                    if method == true_best_method:
                        # 如果真实使用的就是最优方法，没有损失
                        loss_ratio = 0
                    else:
                        # 计算额外成本占最优成本的比例
                        loss_ratio = (actual_cost - true_best_cost) / true_best_cost * 100
                    
                    total_loss += loss_ratio
                    valid_samples += 1
        
        return total_loss / valid_samples if valid_samples > 0 else 0
    
    def plot_evaluation_results(self, metrics: Dict[str, Any], 
                              output_path: str = "ml_training/evaluation_results.png"):
        """
        绘制评估结果图表
        
        Args:
            metrics: 评估指标字典
            output_path: 输出图片路径
        """
        y_true = np.array(metrics['predictions']['y_true'])
        y_pred = np.array(metrics['predictions']['y_pred'])
        
        plt.figure(figsize=(15, 5))
        
        # 图1: 真实值 vs 预测值
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('真实值 vs 预测值')
        
        # 图2: 残差图
        plt.subplot(1, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        
        # 图3: 决策损益比分布
        plt.subplot(1, 3, 3)
        loss_ratios = []
        for i in range(min(100, len(y_true))):  # 只显示前100个样本
            if y_true[i] > 0:
                loss_ratio = (y_pred[i] - y_true[i]) / y_true[i] * 100
                loss_ratios.append(loss_ratio)
        
        plt.hist(loss_ratios, bins=20, edgecolor='black')
        plt.xlabel('相对误差 (%)')
        plt.ylabel('频次')
        plt.title('预测误差分布 (前100样本)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"评估结果图表已保存至: {output_path}")


def main():
    """主函数，用于评估模型"""
    print("开始模型评估...")
    
    try:
        # 初始化评估器
        evaluator = ModelEvaluator()
        
        # 生成测试数据
        test_data = generate_test_data()
        
        print(f"生成了 {len(test_data)} 条测试数据")
        
        if len(test_data) == 0:
            print("没有测试数据，无法评估模型")
            return
        
        # 评估模型
        results = evaluator.evaluate(test_data)
        
        # 绘制评估结果图表
        evaluator.plot_evaluation_results(results)
        
        # 除了新的图表外，也保留原来的对比图
        evaluator.plot_predictions(test_data, "ml_training/prediction_comparison.png")
        
        # 保存评估结果
        output_path = "ml_training/model_evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存至: {output_path}")
        
        # 测试最优方法预测
        print("\n测试最优方法预测...")
        sample_client = {
            'cpu_score': 3000,
            'bandwidth_mbps': 50,
            'decompression_speed': {
                'gzip': 100,
                'zstd': 150,
                'lz4': 200
            },
            'network_rtt': 20,
            'disk_io_speed': 300,
            'memory_size': 16,
            'latency_requirement': 0.5
        }
        
        sample_image = {
            'total_size_mb': 100,
            'avg_layer_entropy': 0.6,
            'text_ratio': 0.3,
            'binary_ratio': 0.7,
            'layer_count': 5,
            'file_type_distribution': {
                'text': 0.3,
                'binary': 0.6,
                'compressed': 0.1
            },
            'avg_file_size': 1024*1024,
            'compression_ratio_estimate': 0.5
        }
        
        optimal_method, optimal_cost = evaluator.predict_optimal_method(sample_client, sample_image)
        print(f"对于给定客户端和镜像，最优压缩方法是: {optimal_method}，预测成本: {optimal_cost:.4f}")
        
    except Exception as e:
        print(f"评估模型时出错: {e}")
        import traceback
        traceback.print_exc()


def generate_test_data() -> List[Tuple]:
    """
    生成测试数据
    
    Returns:
        测试数据列表
    """
    test_data = []
    
    # 使用配置中的实验设计参数
    from .config import CLIENT_CAPABILITIES, IMAGE_PROFILES, COMPRESSION_CONFIG
    
    # 生成实验设计中的数据
    for client_profile in CLIENT_CAPABILITIES['profiles'][:2]:  # 只取前2个配置以减少测试数据量
        for image_profile in IMAGE_PROFILES[:3]:  # 只取前3个镜像配置
            for method in COMPRESSION_CONFIG['algorithms'][:3]:  # 只取前3个算法
                # 计算一个模拟的实际成本（基于特征的简单函数）
                actual_cost = (
                    image_profile['total_size_mb'] / client_profile['decompression_speed']['gzip'] +
                    image_profile['total_size_mb'] * 8 / client_profile['bandwidth_mbps'] +
                    np.random.uniform(0.1, 1.0)  # 添加一些随机性
                )
                
                test_data.append((client_profile, image_profile, actual_cost, method))
    
    return test_data


if __name__ == "__main__":
    main()