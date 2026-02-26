import json
import yaml
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import platform

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"配置文件加载失败 {config_path}: {e}")
        raise

def save_json_result(data: Any, filepath: str, ensure_ascii: bool = False):
    """保存JSON结果"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)
        logging.info(f"结果已保存到: {filepath}")
    except Exception as e:
        logging.error(f"结果保存失败 {filepath}: {e}")
        raise

def load_json_result(filepath: str) -> Any:
    """加载JSON结果"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"结果加载失败 {filepath}: {e}")
        raise

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """计算基本统计量"""
    if not data or len(data) == 0:
        return {
            'count': 0, 'mean': 0.0, 'std': 0.0,
            'min': 0.0, 'max': 0.0, 'median': 0.0,
            'q25': 0.0, 'q75': 0.0, 'cv': 0.0
        }
    
    data_arr = np.array(data)
    mean_val = np.mean(data_arr)
    
    return {
        'count': len(data),
        'mean': float(mean_val),
        'std': float(np.std(data_arr)),
        'min': float(np.min(data_arr)),
        'max': float(np.max(data_arr)),
        'median': float(np.median(data_arr)),
        'q25': float(np.percentile(data_arr, 25)),
        'q75': float(np.percentile(data_arr, 75)),
        'cv': float(np.std(data_arr) / mean_val) if mean_val != 0 else 0.0
    }

def perform_paired_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """
    执行配对样本t检验 (Paired T-test)
    要求: len(group1) == len(group2)
    """
    try:
        from scipy import stats
        
        # 【优化】添加输入校验
        if len(group1) != len(group2):
            raise ValueError(f"配对t检验要求两组数据长度相同: {len(group1)} != {len(group2)}")
        
        if len(group1) < 2:
            return {'error': '样本量过少，无法进行t检验'}
        
        # 配对样本t检验
        t_stat, p_value = stats.ttest_rel(group1, group2)
        
        # 计算效应量(Cohen's d)
        diff = np.array(group1) - np.array(group2)
        std_diff = np.std(diff)
        effect_size = np.mean(diff) / std_diff if std_diff != 0 else 0
        
        return {
            'test_type': 'paired_t_test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size_cohens_d': float(effect_size),
            'significant': bool(p_value < 0.05),
            'sample_size': len(group1)
        }
    except ImportError:
        logging.warning("⚠️  scipy 未安装，无法进行t检验")
        return {'error': 'scipy not installed'}
    except Exception as e:
        logging.warning(f"t检验执行失败: {e}")
        return {'error': str(e)}

def format_timestamp() -> str:
    """生成格式化时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_result_directory(base_dir: str, experiment_name: str) -> str:
    """创建实验结果目录"""
    timestamp = format_timestamp()
    result_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法运算"""
    return numerator / denominator if denominator != 0 else default

def percentage_change(old_value: float, new_value: float) -> float:
    """
    计算百分比变化
    【优化】避免返回 inf，确保 JSON 可序列化
    """
    if old_value == 0:
        if new_value == 0:
            return 0.0
        else:
            # 旧值为0，新值非0，返回一个很大的数而不是 inf
            return 999.0 
    return ((new_value - old_value) / old_value) * 100

def merge_results(*results_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个结果字典"""
    merged = {}
    for result_dict in results_dicts:
        merged.update(result_dict)
    return merged

def filter_dataframe(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """根据条件过滤DataFrame"""
    if df.empty:
        return df
        
    filtered_df = df.copy()
    for column, condition in conditions.items():
        if column in filtered_df.columns:
            if isinstance(condition, (list, tuple)):
                filtered_df = filtered_df[filtered_df[column].isin(condition)]
            else:
                filtered_df = filtered_df[filtered_df[column] == condition]
    return filtered_df

def aggregate_metrics(df: pd.DataFrame, group_by: List[str], 
                     metrics: List[str]) -> pd.DataFrame:
    """聚合指标计算"""
    if df.empty:
        return pd.DataFrame()
        
    agg_dict = {}
    for metric in metrics:
        if metric in df.columns:
            agg_dict[metric] = ['mean', 'std', 'count']
    
    if agg_dict:
        grouped = df.groupby(group_by).agg(agg_dict)
        # 展平列名
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        return grouped.reset_index()
    else:
        return pd.DataFrame()

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    设置日志配置
    【优化】防止重复添加 Handler
    """
    # 获取 root logger
    root_logger = logging.getLogger()
    
    # 如果已经配置过，先清空现有的 handlers (防止重复)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # 设置级别
    root_logger.setLevel(level)
    
    # 定义格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件 Handler
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        cpu_count = 0
        memory_total_gb = 0.0
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': cpu_count,
        'memory_total_gb': memory_total_gb,
        'timestamp': datetime.now().isoformat()
    }

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """验证配置完整性"""
    if not config:
        logging.error("配置为空")
        return False
        
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"配置缺少必要键: {missing_keys}")
        return False
    return True

def pretty_print_dict(d: Dict[str, Any], indent: int = 0):
    """美化打印字典"""
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            pretty_print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")