"""
决策模型训练配置文件
功能：定义训练过程中的配置参数和常量
"""

import os
from typing import Dict, Any, List


# 模型配置
MODEL_CONFIG = {
    'hidden_layer_sizes': (128, 64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 1000,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,
    'learning_rate_init': 0.001,
    'alpha': 0.0001
}

# 数据路径配置
PATH_CONFIG = {
    'model_save_path': 'models',
    'scaler_save_path': 'models',
    'feedback_dir': 'registry',
    'cache_dir': 'cache',
    'log_dir': 'feedback',
    'training_data_path': 'ml_training/training_data.json'
}

# 特征配置
FEATURE_CONFIG = {
    'client_features': [
        'cpu_score',
        'bandwidth_mbps',
        'decompression_speed_gzip',
        'decompression_speed_zstd',
        'decompression_speed_lz4',
        'network_rtt',
        'disk_io_speed',
        'memory_size',
        'latency_requirement'
    ],
    'image_features': [
        'total_size_mb',
        'avg_layer_entropy',
        'text_ratio',
        'binary_ratio',
        'layer_count',
        'file_type_dist_text',
        'file_type_dist_binary',
        'file_type_dist_compressed',
        'avg_file_size',
        'compression_ratio_estimate'
    ],
    'method_features': [
        'is_gzip',
        'is_zstd',
        'is_lz4',
        'compression_level'
    ]
}

# 压缩算法配置
COMPRESSION_CONFIG = {
    'algorithms': [
        'gzip-1', 'gzip-6', 'gzip-9',
        'zstd-1', 'zstd-3', 'zstd-6',
        'lz4-fast', 'lz4-medium', 'lz4-slow',
        'brotli-1'
    ],
    'default_method': 'gzip-6'
}

# 物理仿真画像配置表
CLIENT_PROFILES = {
    "C1": {
        "desc": "极端弱端 (IoT/树莓派)",
        "cpu_limit": 0.2,       # 0.2核
        "mem_limit": "512m",    # 512MB内存
        "bw_rate": "2mbit",     # 2Mbps带宽
        "latency": "200ms",     # 200ms延迟
        "disk_read": "5mb"      # 5MB/s磁盘读取
    },
    "C2": {
        "desc": "CPU瓶颈 (老旧PC)",
        "cpu_limit": 0.5,
        "mem_limit": "1g",
        "bw_rate": "20mbit",
        "latency": "50ms",
        "disk_read": "10mb"
    },
    "C3": {
        "desc": "网络瓶颈 (4G移动端)",
        "cpu_limit": 1.0,
        "mem_limit": "2g",
        "bw_rate": "5mbit",
        "latency": "100ms",
        "disk_read": "50mb"
    },
    "C4": {
        "desc": "均衡配置 (标准云)",
        "cpu_limit": 1.5,
        "mem_limit": "2g",
        "bw_rate": "50mbit",
        "latency": "20ms",
        "disk_read": "150mb"
    },
    "C5": {
        "desc": "解压瓶颈 (高带宽弱CPU)",
        "cpu_limit": 0.8,
        "mem_limit": "2g",
        "bw_rate": "100mbit",
        "latency": "10ms",
        "disk_read": "200mb"
    },
    "C6": {
        "desc": "高端节点 (5G/数据中心)",
        "cpu_limit": 4.0,       # 释放大部分性能
        "mem_limit": "4g",
        "bw_rate": "500mbit",
        "latency": "5ms",
        "disk_read": "500mb"
    }
}

# 镜像配置 - 提供可扩展的接口，实际实验时可以灵活设置
IMAGE_PROFILES = [
    # Linux Distro 类
    {
        'name': 'centos',
        'total_size_mb': 200.0,
        'avg_layer_entropy': 0.4,
        'text_ratio': 0.6,
        'binary_ratio': 0.4,
        'layer_count': 3,
        'file_type_distribution': {'text': 0.6, 'binary': 0.4, 'compressed': 0.0},
        'avg_file_size': 4096,
        'compression_ratio_estimate': 0.35
    },
    {
        'name': 'fedora',
        'total_size_mb': 180.0,
        'avg_layer_entropy': 0.45,
        'text_ratio': 0.55,
        'binary_ratio': 0.45,
        'layer_count': 3,
        'file_type_distribution': {'text': 0.55, 'binary': 0.45, 'compressed': 0.0},
        'avg_file_size': 3584,
        'compression_ratio_estimate': 0.32
    },
    {
        'name': 'ubuntu',
        'total_size_mb': 28.0,
        'avg_layer_entropy': 0.3,
        'text_ratio': 0.7,
        'binary_ratio': 0.3,
        'layer_count': 1,
        'file_type_distribution': {'text': 0.7, 'binary': 0.3, 'compressed': 0.0},
        'avg_file_size': 2048,
        'compression_ratio_estimate': 0.25
    },
    
    # Database 类
    {
        'name': 'mongo',
        'total_size_mb': 681.0,
        'avg_layer_entropy': 0.65,
        'text_ratio': 0.15,
        'binary_ratio': 0.85,
        'layer_count': 5,
        'file_type_distribution': {'text': 0.15, 'binary': 0.85, 'compressed': 0.0},
        'avg_file_size': 10240,
        'compression_ratio_estimate': 0.42
    },
    {
        'name': 'mysql',
        'total_size_mb': 516.0,
        'avg_layer_entropy': 0.6,
        'text_ratio': 0.2,
        'binary_ratio': 0.8,
        'layer_count': 5,
        'file_type_distribution': {'text': 0.2, 'binary': 0.8, 'compressed': 0.0},
        'avg_file_size': 8192,
        'compression_ratio_estimate': 0.4
    },
    {
        'name': 'postgres',
        'total_size_mb': 314.0,
        'avg_layer_entropy': 0.55,
        'text_ratio': 0.25,
        'binary_ratio': 0.75,
        'layer_count': 4,
        'file_type_distribution': {'text': 0.25, 'binary': 0.75, 'compressed': 0.0},
        'avg_file_size': 6144,
        'compression_ratio_estimate': 0.38
    },
    
    # Language 类
    {
        'name': 'rust',
        'total_size_mb': 188.0,
        'avg_layer_entropy': 0.4,
        'text_ratio': 0.4,
        'binary_ratio': 0.6,
        'layer_count': 3,
        'file_type_distribution': {'text': 0.4, 'binary': 0.6, 'compressed': 0.0},
        'avg_file_size': 4096,
        'compression_ratio_estimate': 0.3
    },
    {
        'name': 'ruby',
        'total_size_mb': 882.0,
        'avg_layer_entropy': 0.45,
        'text_ratio': 0.65,
        'binary_ratio': 0.35,
        'layer_count': 5,
        'file_type_distribution': {'text': 0.65, 'binary': 0.35, 'compressed': 0.0},
        'avg_file_size': 3072,
        'compression_ratio_estimate': 0.32
    },
    {
        'name': 'python',
        'total_size_mb': 923.0,
        'avg_layer_entropy': 0.4,
        'text_ratio': 0.7,
        'binary_ratio': 0.3,
        'layer_count': 6,
        'file_type_distribution': {'text': 0.7, 'binary': 0.3, 'compressed': 0.0},
        'avg_file_size': 2048,
        'compression_ratio_estimate': 0.3
    },
    
    # Web Component 类
    {
        'name': 'nginx',
        'total_size_mb': 141.0,
        'avg_layer_entropy': 0.4,
        'text_ratio': 0.4,
        'binary_ratio': 0.6,
        'layer_count': 3,
        'file_type_distribution': {'text': 0.4, 'binary': 0.6, 'compressed': 0.0},
        'avg_file_size': 2048,
        'compression_ratio_estimate': 0.3
    },
    {
        'name': 'httpd',
        'total_size_mb': 148.0,
        'avg_layer_entropy': 0.45,
        'text_ratio': 0.5,
        'binary_ratio': 0.5,
        'layer_count': 3,
        'file_type_distribution': {'text': 0.5, 'binary': 0.5, 'compressed': 0.0},
        'avg_file_size': 2560,
        'compression_ratio_estimate': 0.33
    },
    {
        'name': 'tomcat',
        'total_size_mb': 198.0,
        'avg_layer_entropy': 0.5,
        'text_ratio': 0.3,
        'binary_ratio': 0.7,
        'layer_count': 4,
        'file_type_distribution': {'text': 0.3, 'binary': 0.7, 'compressed': 0.0},
        'avg_file_size': 3072,
        'compression_ratio_estimate': 0.35
    },
    
    # Application Platform 类
    {
        'name': 'rabbitmq',
        'total_size_mb': 170.0,
        'avg_layer_entropy': 0.55,
        'text_ratio': 0.25,
        'binary_ratio': 0.75,
        'layer_count': 4,
        'file_type_distribution': {'text': 0.25, 'binary': 0.75, 'compressed': 0.0},
        'avg_file_size': 4096,
        'compression_ratio_estimate': 0.38
    },
    {
        'name': 'wordpress',
        'total_size_mb': 167.0,
        'avg_layer_entropy': 0.4,
        'text_ratio': 0.65,
        'binary_ratio': 0.35,
        'layer_count': 4,
        'file_type_distribution': {'text': 0.65, 'binary': 0.35, 'compressed': 0.0},
        'avg_file_size': 2048,
        'compression_ratio_estimate': 0.3
    },
    {
        'name': 'nextcloud',
        'total_size_mb': 400.0,
        'avg_layer_entropy': 0.5,
        'text_ratio': 0.55,
        'binary_ratio': 0.45,
        'layer_count': 5,
        'file_type_distribution': {'text': 0.55, 'binary': 0.45, 'compressed': 0.0},
        'avg_file_size': 3072,
        'compression_ratio_estimate': 0.35
    },
    
    # Application Tool 类
    {
        'name': 'gradle',
        'total_size_mb': 250.0,
        'avg_layer_entropy': 0.45,
        'text_ratio': 0.4,
        'binary_ratio': 0.6,
        'layer_count': 4,
        'file_type_distribution': {'text': 0.4, 'binary': 0.6, 'compressed': 0.05},
        'avg_file_size': 3584,
        'compression_ratio_estimate': 0.33
    },
    {
        'name': 'logstash',
        'total_size_mb': 300.0,
        'avg_layer_entropy': 0.5,
        'text_ratio': 0.35,
        'binary_ratio': 0.65,
        'layer_count': 4,
        'file_type_distribution': {'text': 0.35, 'binary': 0.65, 'compressed': 0.0},
        'avg_file_size': 4096,
        'compression_ratio_estimate': 0.35
    },
    {
        'name': 'node',
        'total_size_mb': 942.0,
        'avg_layer_entropy': 0.45,
        'text_ratio': 0.6,
        'binary_ratio': 0.4,
        'layer_count': 6,
        'file_type_distribution': {'text': 0.6, 'binary': 0.4, 'compressed': 0.0},
        'avg_file_size': 3072,
        'compression_ratio_estimate': 0.32
    }
]

# 数据收集配置
DATA_COLLECTION_CONFIG = {
    'min_samples_for_training': 100,
    'required_client_fields': [
        'cpu_score', 'bandwidth_mbps', 'network_rtt'
    ],
    'required_image_fields': [
        'total_size_mb', 'avg_layer_entropy'
    ],
    'min_data_quality_threshold': 0.8,
    # 实验设计参数将在运行时通过 get_experimental_design() 函数获取
}

# 训练过程配置
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'learning_rate': 0.001
}

# 评估指标配置
EVALUATION_CONFIG = {
    'metrics': ['mse', 'mae', 'rmse', 'mape', 'r2_score'],
    'acceptable_mape_threshold': 15.0,  # 平均绝对百分比误差阈值
    'acceptable_r2_threshold': 0.7      # R² 阈值
}

# 部署配置
DEPLOYMENT_CONFIG = {
    'model_update_frequency': 'daily',  # 模型更新频率
    'min_new_samples_for_update': 50,   # 更新模型所需的最少新样本数
    'backup_model_count': 3,            # 保留的模型备份数量
    'model_validation_enabled': True    # 是否启用模型验证
}

# 系统资源配置
RESOURCE_CONFIG = {
    'max_memory_usage_mb': 1024,
    'max_training_time_minutes': 60,
    'enable_gpu_acceleration': False,
    'max_parallel_jobs': 1
}

# 日志配置
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'ml_training/training.log',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'enable_console_logging': True
}


def get_model_config() -> Dict[str, Any]:
    """获取模型配置"""
    return MODEL_CONFIG


def get_path_config() -> Dict[str, Any]:
    """获取路径配置"""
    return PATH_CONFIG


def get_feature_config() -> Dict[str, Any]:
    """获取特征配置"""
    return FEATURE_CONFIG


def get_compression_config() -> Dict[str, Any]:
    """获取压缩算法配置"""
    return COMPRESSION_CONFIG


def get_client_capabilities() -> Dict[str, Any]:
    """获取客户端能力配置"""
    profiles = []
    for profile_id, config in CLIENT_PROFILES.items():
        profiles.append({
            'name': profile_id,
            'description': config['desc'],
            'cpu_limit': config['cpu_limit'],
            'mem_limit': config['mem_limit'],
            'bw_rate': config['bw_rate'],
            'latency': config['latency'],
            'disk_read': config['disk_read']
        })
    return {'profiles': profiles}


def get_image_profiles() -> List[Dict[str, Any]]:
    """获取镜像特征配置"""
    return IMAGE_PROFILES


def get_experimental_design():
    """获取实验设计参数，避免模块加载时的循环依赖"""
    return {
        'client_profiles': get_client_capabilities()['profiles'],
        'image_profiles': IMAGE_PROFILES,
        'algorithms': COMPRESSION_CONFIG['algorithms'],
        'total_experiments': len(get_client_capabilities()['profiles']) * 
                           len(IMAGE_PROFILES) * 
                           len(COMPRESSION_CONFIG['algorithms']),  # 6 * 18 * 10 = 1080
        'replications_per_experiment': 3,
        'total_executions': len(get_client_capabilities()['profiles']) * 
                          len(IMAGE_PROFILES) * 
                          len(COMPRESSION_CONFIG['algorithms']) * 3  # 3240
    }

def get_data_collection_config() -> Dict[str, Any]:
    """获取数据收集配置"""
    config = DATA_COLLECTION_CONFIG.copy()
    config['experimental_design'] = get_experimental_design()
    return config


def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    return TRAINING_CONFIG


def get_evaluation_config() -> Dict[str, Any]:
    """获取评估配置"""
    return EVALUATION_CONFIG


def get_deployment_config() -> Dict[str, Any]:
    """获取部署配置"""
    return DEPLOYMENT_CONFIG


def get_resource_config() -> Dict[str, Any]:
    """获取资源配置"""
    return RESOURCE_CONFIG


def get_logging_config() -> Dict[str, Any]:
    """获取日志配置"""
    return LOGGING_CONFIG


def setup_directories():
    """创建必要的目录"""
    directories = [
        PATH_CONFIG['model_save_path'],
        PATH_CONFIG['scaler_save_path'],
        os.path.dirname(PATH_CONFIG['training_data_path'])
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# 初始化目录
setup_directories()