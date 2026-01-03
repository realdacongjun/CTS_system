"""
数据收集和处理工具
功能：从系统各模块收集训练数据，预处理数据以供模型训练使用
输入：系统各模块的原始数据
输出：标准化的训练数据格式
"""

import os
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3
from .config import get_client_capabilities, get_image_profiles, get_compression_config, DATA_COLLECTION_CONFIG


class DataCollector:
    """数据收集器"""
    
    def __init__(self, 
                 feedback_dir: str = "registry", 
                 cache_dir: str = "cache", 
                 log_dir: str = "feedback",
                 data_dir: str = "/tmp/exp_data"):
        """
        初始化数据收集器
        
        Args:
            feedback_dir: 反馈数据目录
            cache_dir: 缓存数据目录
            log_dir: 日志数据目录
            data_dir: 实验数据目录
        """
        self.feedback_dir = feedback_dir
        self.cache_dir = cache_dir
        self.log_dir = log_dir
        self.data_dir = data_dir
        
        # 从配置中获取实验设计参数
        self.client_profiles = get_client_capabilities()['profiles']
        self.image_profiles = get_image_profiles()
        self.algorithms = get_compression_config()['algorithms']
        
        # 数据库路径
        self.db_path = os.path.join(self.data_dir, "experiment_manifest.db")
        
        # 确保目录存在
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_feedback_data(self) -> List[Dict[str, Any]]:
        """
        从反馈系统收集数据
        
        Returns:
            收集到的反馈数据列表
        """
        feedback_data = []
        
        # 从反馈数据文件中读取
        feedback_file = os.path.join(self.feedback_dir, "feedback_data.json")
        
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 标准化数据格式
                for record in raw_data:
                    standardized_record = self._standardize_feedback_record(record)
                    feedback_data.append(standardized_record)
                    
            except Exception as e:
                print(f"读取反馈数据时出错: {e}")
        
        return feedback_data
    
    def _standardize_feedback_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化反馈记录格式
        
        Args:
            record: 原始反馈记录
            
        Returns:
            标准化后的记录
        """
        standardized = {
            # 客户端画像
            'node_id': record.get('node_id', ''),
            'cpu_score': record.get('cpu_score', 0),
            'bandwidth_mbps': record.get('bandwidth_mbps', 0),
            'decompression_speed': record.get('decompression_speed', {
                'gzip': 0, 'zstd': 0, 'lz4': 0
            }),
            'network_rtt': record.get('network_rtt', 0),
            'disk_io_speed': record.get('disk_io_speed', 0),
            'memory_size': record.get('memory_size', 0),
            'latency_requirement': record.get('latency_requirement', 0),
            
            # 镜像特征
            'image_id': record.get('image_id', ''),
            'total_size_mb': record.get('total_size_mb', 0),
            'avg_layer_entropy': record.get('avg_layer_entropy', 0),
            'text_ratio': record.get('text_ratio', 0),
            'binary_ratio': record.get('binary_ratio', 0),
            'layer_count': record.get('layer_count', 0),
            'file_type_distribution': record.get('file_type_distribution', {
                'text': 0, 'binary': 0, 'compressed': 0
            }),
            'avg_file_size': record.get('avg_file_size', 0),
            'compression_ratio_estimate': record.get('compression_ratio_estimate', 0),
            
            # 实际性能数据
            'algo_used': record.get('algo_used', 'gzip-6'),
            'actual_compress_time': record.get('actual_compress_time', 0),
            'actual_transfer_time': record.get('actual_transfer_time', 0),
            'actual_decomp_time': record.get('actual_decomp_time', 0),
            'timestamp': record.get('timestamp', ''),
            'transfer_size': record.get('transfer_size', 0),
        }
        
        return standardized
    
    def collect_cache_data(self) -> List[Dict[str, Any]]:
        """
        从缓存系统收集数据
        
        Returns:
            收集到的缓存数据列表
        """
        cache_data = []
        
        # 遍历缓存目录中的文件
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    
                    # 处理缓存数据
                    for record in raw_data:
                        standardized_record = self._standardize_cache_record(record)
                        cache_data.append(standardized_record)
                        
                except Exception as e:
                    print(f"读取缓存数据文件 {filepath} 时出错: {e}")
        
        return cache_data
    
    def _standardize_cache_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化缓存记录格式
        
        Args:
            record: 原始缓存记录
            
        Returns:
            标准化后的记录
        """
        standardized = {
            'cache_key': record.get('cache_key', ''),
            'original_size': record.get('original_size', 0),
            'compressed_size': record.get('compressed_size', 0),
            'compression_ratio': record.get('compression_ratio', 0),
            'compression_method': record.get('compression_method', ''),
            'compression_time': record.get('compression_time', 0),
            'access_count': record.get('access_count', 0),
            'last_accessed': record.get('last_accessed', ''),
            'created_at': record.get('created_at', ''),
        }
        
        return standardized
    
    def collect_log_data(self) -> List[Dict[str, Any]]:
        """
        从日志系统收集数据
        
        Returns:
            收集到的日志数据列表
        """
        log_data = []
        
        # 遍历日志目录中的文件
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.log_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    
                    # 处理日志数据
                    for record in raw_data:
                        standardized_record = self._standardize_log_record(record)
                        log_data.append(standardized_record)
                        
                except Exception as e:
                    print(f"读取日志数据文件 {filepath} 时出错: {e}")
        
        return log_data
    
    def _standardize_log_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化日志记录格式
        
        Args:
            record: 原始日志记录
            
        Returns:
            标准化后的记录
        """
        standardized = {
            'log_type': record.get('log_type', ''),
            'message': record.get('message', ''),
            'timestamp': record.get('timestamp', ''),
            'node_id': record.get('node_id', ''),
            'image_id': record.get('image_id', ''),
            'operation': record.get('operation', ''),
            'duration': record.get('duration', 0),
            'size': record.get('size', 0),
            'status': record.get('status', ''),
        }
        
        return standardized
    
    def collect_experiment_data(self) -> List[Dict[str, Any]]:
        """
        从实验数据目录收集数据
        
        Returns:
            收集到的实验数据列表
        """
        experiment_data = []
        
        # 从数据库获取已完成的实验
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 获取已完成的实验记录
                cursor.execute("SELECT * FROM experiments WHERE status IN ('SUCCESS', 'ABNORMAL')")
                rows = cursor.fetchall()
                
                # 获取列名
                column_names = [description[0] for description in cursor.description]
                
                for row in rows:
                    record = dict(zip(column_names, row))
                    experiment_data.append(record)
                
                conn.close()
            except Exception as e:
                print(f"从数据库读取实验数据时出错: {e}")
        
        # 从JSON文件获取实验数据
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and f != 'aggregated_results.json']
        
        for filename in data_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    raw_data = json.load(f)
                
                # 检查是否是单个记录或记录列表
                if isinstance(raw_data, list):
                    for record in raw_data:
                        experiment_data.append(record)
                else:
                    experiment_data.append(raw_data)
            except Exception as e:
                print(f"读取实验数据文件 {filepath} 时出错: {e}")
        
        return experiment_data
    
    def generate_experimental_data(self) -> List[Tuple]:
        """
        根据实验设计参数生成系统化实验数据
        这是核心功能，用于生成覆盖所有相变点的训练数据
        
        Returns:
            训练数据列表 [(client_profile, image_profile, actual_cost, method), ...]
        """
        training_data = []
        
        print(f"开始生成实验数据...")
        print(f"客户端配置数: {len(self.client_profiles)}")
        print(f"镜像配置数: {len(self.image_profiles)}")
        print(f"算法配置数: {len(self.algorithms)}")
        print(f"总实验数: {len(self.client_profiles) * len(self.image_profiles) * len(self.algorithms)}")
        
        # 遍历所有组合
        for client_profile in self.client_profiles:
            for image_profile in self.image_profiles:
                for method in self.algorithms:
                    # 模拟实际成本计算
                    actual_cost = self._simulate_actual_cost(client_profile, image_profile, method)
                    
                    # 添加到训练数据
                    training_data.append((client_profile, image_profile, actual_cost, method))
        
        print(f"生成了 {len(training_data)} 条实验数据")
        return training_data
    
    def _simulate_actual_cost(self, client_profile: Dict[str, Any], 
                            image_profile: Dict[str, Any], 
                            method: str) -> float:
        """
        模拟计算实际成本
        这里使用一个简化的模型，实际部署时应该替换为真实的性能测试
        
        Args:
            client_profile: 客户端画像
            image_profile: 镜像特征
            method: 压缩方法
            
        Returns:
            模拟的实际成本
        """
        # 解析压缩方法
        algo_parts = method.split('-')
        algo = algo_parts[0]
        level = 1
        if len(algo_parts) > 1:
            try:
                level = int(algo_parts[1])
            except ValueError:
                level = 1
        
        # 基于算法和级别计算压缩时间
        compression_time = 0
        if algo == 'gzip':
            # Gzip压缩时间与级别成正比，与CPU性能成反比
            compression_time = (image_profile['total_size_mb'] * (1.0 - image_profile['avg_layer_entropy'] * 0.3)) / (client_profile['cpu_score'] / 1000) * (level * 0.3)
        elif algo == 'zstd':
            # Zstd压缩时间与级别成正比，但效率更高
            compression_time = (image_profile['total_size_mb'] * (1.0 - image_profile['avg_layer_entropy'] * 0.4)) / (client_profile['cpu_score'] / 1000) * (level * 0.2)
        elif algo == 'lz4':
            # LZ4压缩时间固定较低
            compression_time = (image_profile['total_size_mb'] * (1.0 - image_profile['avg_layer_entropy'] * 0.2)) / (client_profile['cpu_score'] / 1000) * 0.1
        elif algo == 'brotli':
            # Brotli压缩时间较长但压缩比高
            compression_time = (image_profile['total_size_mb'] * (1.0 - image_profile['avg_layer_entropy'] * 0.5)) / (client_profile['cpu_score'] / 1000) * (level * 0.4)
        
        # 传输时间基于带宽和压缩后的大小
        compression_ratio = 0.5  # 初始压缩比
        if algo == 'gzip':
            compression_ratio = 0.7 - (level * 0.05)
        elif algo == 'zstd':
            compression_ratio = 0.65 - (level * 0.04)
        elif algo == 'lz4':
            compression_ratio = 0.8 - (level * 0.02)
        elif algo == 'brotli':
            compression_ratio = 0.6 - (level * 0.03)
        
        # 限制压缩比在合理范围内
        compression_ratio = max(0.1, min(0.9, compression_ratio))
        
        compressed_size = image_profile['total_size_mb'] * compression_ratio
        # 考虑网络RTT对传输的影响
        transfer_time = (compressed_size * 8) / client_profile['bandwidth_mbps'] + (client_profile['network_rtt'] / 1000)
        
        # 解压时间基于解压速度
        decomp_speed = client_profile['decompression_speed'].get(algo, 100)
        if decomp_speed <= 0:
            decomp_speed = 100  # 默认值
        decomp_time = compressed_size / decomp_speed
        
        # 总成本 = 压缩时间 + 传输时间 + 解压时间
        total_cost = compression_time + transfer_time + decomp_time
        
        # 添加一些随机噪声以模拟真实环境的不确定性
        noise = np.random.normal(0, total_cost * 0.05)  # 5%的噪声
        return max(0.001, total_cost + noise)  # 确保成本为正
    
    def merge_and_filter_data(self, 
                             feedback_data: List[Dict[str, Any]], 
                             cache_data: List[Dict[str, Any]], 
                             log_data: List[Dict[str, Any]],
                             experiment_data: List[Dict[str, Any]] = None,
                             start_date: str = None,
                             end_date: str = None) -> List[Tuple]:
        """
        合并和过滤数据，生成训练数据
        
        Args:
            feedback_data: 反馈数据
            cache_data: 缓存数据
            log_data: 日志数据
            experiment_data: 实验数据
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            训练数据列表 [(client_profile, image_profile, actual_cost, method), ...]
        """
        training_data = []
        
        # 合并所有数据
        all_data = feedback_data + cache_data + log_data
        if experiment_data:
            all_data += experiment_data
        
        # 过滤日期范围
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = datetime.min
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.max
        
        # 处理合并后的数据，生成训练数据
        for record in all_data:
            # 检查时间范围
            record_time_str = record.get('timestamp', '')
            if record_time_str:
                try:
                    # 尝试解析不同的时间格式
                    if isinstance(record_time_str, (int, float)):
                        record_time = datetime.fromtimestamp(record_time_str)
                    else:
                        # 尝试ISO格式
                        record_time = datetime.fromisoformat(record_time_str.replace('Z', '+00:00'))
                    
                    if not (start_dt <= record_time <= end_dt):
                        continue
                except ValueError:
                    # 如果时间格式不正确，跳过这条记录
                    continue
            
            # 检查是否是实验数据记录
            if 'profile_id' in record and 'image_name' in record:
                # 从实验数据构建训练数据
                client_profile = self._find_client_profile_by_name(record['profile_id'])
                image_profile = self._find_image_profile_by_name(record['image_name'])
                method = record['method']
                
                # 数据清洗：过滤异常数据
                if record.get('status') == 'ABNORMAL':
                    continue  # 跳过异常状态的数据
                
                # 数据清洗：解压时间比例校验
                decomp_time = record.get('decompression_time', 0)
                if decomp_time < 0.01:  # 解压时间小于10ms
                    continue  # 跳过解压时间异常的数据
                
                actual_cost = record.get('cost_total', 0)
            else:
                # 从反馈数据构建训练数据
                client_profile = {
                    'cpu_score': record.get('cpu_score', 0),
                    'bandwidth_mbps': record.get('bandwidth_mbps', 0),
                    'decompression_speed': record.get('decompression_speed', {}),
                    'network_rtt': record.get('network_rtt', 0),
                    'disk_io_speed': record.get('disk_io_speed', 0),
                    'memory_size': record.get('memory_size', 0),
                    'latency_requirement': record.get('latency_requirement', 0)
                }
                
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
                
                method = record.get('algo_used', 'gzip-6')
            
            if client_profile and image_profile:
                training_data.append((client_profile, image_profile, actual_cost, method))
        
        return training_data
    
    def _find_client_profile_by_name(self, name: str) -> Dict[str, Any]:
        """根据名称查找客户端配置"""
        for profile in self.client_profiles:
            if profile['name'] == name:
                return profile
        return None
    
    def _find_image_profile_by_name(self, name: str) -> Dict[str, Any]:
        """根据名称查找镜像配置"""
        for profile in self.image_profiles:
            if profile['name'] == name:
                return profile
        return None
    
    def save_training_data(self, training_data: List[Tuple], 
                          filepath: str = "ml_training/training_data.json"):
        """
        保存训练数据到文件
        
        Args:
            training_data: 训练数据
            filepath: 保存文件路径
        """
        # 转换为可序列化的格式
        serializable_data = []
        for client_profile, image_profile, actual_cost, method in training_data:
            serializable_data.append({
                'client_profile': client_profile,
                'image_profile': image_profile,
                'actual_cost': actual_cost,
                'method': method
            })
        
        # 保存到文件
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"训练数据已保存至: {filepath}")
        print(f"共保存 {len(training_data)} 条训练数据")
        
        # 保存实验设计参数
        experiment_config = {
            'client_profiles': self.client_profiles,
            'image_profiles': self.image_profiles,
            'algorithms': self.algorithms,
            'total_experiments': len(training_data),
            'experimental_design': DATA_COLLECTION_CONFIG['experimental_design']
        }
        
        config_path = filepath.replace('.json', '_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, ensure_ascii=False, indent=2)
        
        print(f"实验配置已保存至: {config_path}")
    
    def load_training_data(self, filepath: str = "ml_training/training_data.json") -> List[Tuple]:
        """
        从文件加载训练数据
        
        Args:
            filepath: 训练数据文件路径
            
        Returns:
            训练数据列表
        """
        if not os.path.exists(filepath):
            print(f"训练数据文件不存在: {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                serializable_data = json.load(f)
            
            training_data = []
            for record in serializable_data:
                client_profile = record['client_profile']
                image_profile = record['image_profile']
                actual_cost = record['actual_cost']
                method = record['method']
                
                training_data.append((client_profile, image_profile, actual_cost, method))
            
            print(f"从 {filepath} 加载了 {len(training_data)} 条训练数据")
            return training_data
            
        except Exception as e:
            print(f"加载训练数据时出错: {e}")
            return []
    
    def analyze_experiment_variance(self) -> Dict[str, Any]:
        """
        分析实验数据的方差，识别异常数据
        
        Returns:
            方差分析结果
        """
        experiment_data = self.collect_experiment_data()
        
        # 按实验配置分组
        grouped_data = {}
        for record in experiment_data:
            if record.get('status') in ['SUCCESS', 'ABNORMAL'] and record.get('decompression_time', 0) >= 0.01:
                key = f"{record.get('profile_id', 'unknown')}_{record.get('image_name', 'unknown')}_{record.get('method', 'unknown')}"
                if key not in grouped_data:
                    grouped_data[key] = []
                if 'cost_total' in record:
                    grouped_data[key].append(record['cost_total'])
        
        # 计算每组的统计信息
        variance_analysis = {}
        for key, durations in grouped_data.items():
            if len(durations) >= 3:
                mean_duration = sum(durations) / len(durations)
                variance = sum((x - mean_duration) ** 2 for x in durations) / len(durations)
                std_dev = variance ** 0.5
                cv = (std_dev / mean_duration) * 100 if mean_duration > 0 else 0  # 变异系数
                
                # 如果变异系数超过15%，标记为高变异
                high_cv = cv > 15
                
                variance_analysis[key] = {
                    'count': len(durations),
                    'mean_duration': mean_duration,
                    'std_deviation': std_dev,
                    'variance': variance,
                    'cv': cv,  # 变异系数
                    'high_cv': high_cv,  # 高变异系数标记
                    'durations': durations
                }
        
        return variance_analysis


def main():
    """主函数，用于演示数据收集功能"""
    print("开始收集训练数据...")
    
    # 初始化数据收集器
    collector = DataCollector()
    
    # 收集各种数据
    print("收集反馈数据...")
    feedback_data = collector.collect_feedback_data()
    
    print("收集缓存数据...")
    cache_data = collector.collect_cache_data()
    
    print("收集日志数据...")
    log_data = collector.collect_log_data()
    
    print("收集实验数据...")
    experiment_data = collector.collect_experiment_data()
    
    print(f"收集到 {len(feedback_data)} 条反馈数据")
    print(f"收集到 {len(cache_data)} 条缓存数据")
    print(f"收集到 {len(log_data)} 条日志数据")
    print(f"收集到 {len(experiment_data)} 条实验数据")
    
    # 分析实验数据方差
    print("分析实验数据方差...")
    variance_analysis = collector.analyze_experiment_variance()
    high_cv_count = sum(1 for v in variance_analysis.values() if v['high_cv'])
    print(f"发现 {high_cv_count} 组高变异系数数据")
    
    # 合并所有数据
    print("合并所有数据...")
    all_data = collector.merge_and_filter_data(
        feedback_data, cache_data, log_data, experiment_data
    )
    
    # 保存实验数据
    collector.save_training_data(all_data)
    
    # 演示从文件加载训练数据
    loaded_data = collector.load_training_data()
    print(f"从文件加载了 {len(loaded_data)} 条训练数据")
    
    # 显示一些样本数据
    if loaded_data:
        print("\n样本数据:")
        sample = loaded_data[0]
        print(f"客户端画像: {sample[0]}")
        print(f"镜像特征: {sample[1]}")
        print(f"实际成本: {sample[2]}")
        print(f"使用方法: {sample[3]}")


if __name__ == "__main__":
    main()