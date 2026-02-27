"""
CTS实验框架核心模块
==================

包含性能预测模型 (CompactCFTNetV2) 和 自适应决策引擎 (CAGSDecisionEngine)。

使用示例:
    >>> from src.model_wrapper import CFTNetWrapper
    >>> from src.decision_engine import CAGSDecisionEngine
"""

__version__ = "1.0.0"
__author__ = "CTS Research Team"

# ==========================================
# 【推荐】不在这里自动导入类
# ==========================================
# 原因:
# 1. 避免 model_wrapper / decision_engine 在初始化时的副作用
# 2. 避免循环依赖 (Circular Import)
# 3. 让使用者明确知道类是从哪个文件来的，代码可读性更好
#
# 用法:
#   不要写: from src import CFTNetWrapper
#   请写: from src.model_wrapper import CFTNetWrapper
#
# ==========================================

# 如果你确实想支持 `from src import CFTNetWrapper`，
# 可以使用下面的懒加载方式 (Python 3.7+)，但通常没必要。

# try:
#     from .model_wrapper import CFTNetWrapper
#     from .decision_engine import CAGSDecisionEngine
# except ImportError as e:
#     # 允许在依赖未安装时只导入 __version__
#     import warnings
#     warnings.warn(f"核心模块导入失败: {e}")