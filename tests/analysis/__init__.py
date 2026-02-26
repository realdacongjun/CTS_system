"""
CTS结果分析包
=============

包含评估指标计算和可视化生成功能。
"""

__version__ = "1.0.0"

# 导出分析模块
from . import metrics
from . import visualization

__all__ = [
    'metrics',
    'visualization'
]