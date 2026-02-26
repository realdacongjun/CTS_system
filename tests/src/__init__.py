"""
CTS实验框架核心模块
==================

包含环境控制、模型封装、决策引擎等核心组件。
"""

__version__ = "1.0.0"
__author__ = "CTS Research Team"

# 导出主要类
from .environment import EnvironmentController
from .executor import DockerExecutor
from .model_wrapper import CFTNetWrapper
from .decision_engine import CAGSEngine

__all__ = [
    'EnvironmentController',
    'DockerExecutor', 
    'CFTNetWrapper',
    'CAGSEngine'
]