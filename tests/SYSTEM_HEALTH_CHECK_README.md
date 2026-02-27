# CTS 系统健康检查使用指南

## 📋 概述

`system_health_check.py` 是一个完整的系统验证脚本，用于在运行实验前快速检查所有关键组件是否正常工作。

## 🚀 快速开始

### 方法一：直接运行 Python 脚本
```bash
cd tests/
python system_health_check.py
```

### 方法二：使用批处理脚本（Windows）
```bash
cd tests/
run_health_check.bat
```

## 📊 检查项目详解

### 1. 目录结构检查 ⭐⭐
验证必需的目录是否存在：
- `configs/` - 配置文件目录
- `src/` - 核心源代码目录
- `data/` - 数据文件目录
- `models/` - 模型文件目录
- `results/` - 结果输出目录

### 2. 配置文件检查 ⭐⭐⭐
验证配置文件的完整性和可读性：
- `configs/model_config.yaml` - 模型配置文件
- `config.yaml` 或 `configs/global_config.yaml` - 全局配置文件

### 3. 核心模块导入检查 ⭐⭐⭐
验证所有核心Python模块能否正确导入：
- `src/cts_model.py` - 核心模型文件
- `src/model_wrapper.py` - 模型包装器
- `src/decision_engine.py` - 决策引擎
- `baseline_downloader.py` - 基线下载器
- `adaptive_downloader.py` - 自适应下载器

### 4. 模型文件检查 ⭐⭐
验证机器学习模型相关文件：
- `.pth` 模型权重文件
- `.pkl` 预处理对象文件

### 5. 系统初始化测试 ⭐⭐⭐
**最重要的检查项** - 实际加载和初始化整个CTS系统：
- CFTNetWrapper 初始化
- CAGSDecisionEngine 初始化
- 模型参数验证
- 特征维度匹配检查

### 6. 代理服务器检查 ⭐
可选检查项，验证代理服务器连通性：
- `localhost:8000` 连接测试

### 7. 数据CSV检查 ⭐⭐
验证实验所需的数据文件：
- `data/image_features_database.csv`
- 必要列的存在性检查

## 🎯 检查结果解读

### ✅ 所有检查通过
```
🎉 所有核心检查通过！
下一步:
  1. 确保 proxy_server.py 正在运行
  2. 运行: python experiment_runner.py
```

### ⚠️ 部分检查失败
```
⚠️  部分检查未通过，请查看上面的错误信息。
提示:
  - '模型文件' 和 '代理服务' 失败不影响代码结构验证
  - 请优先解决 '核心模块' 和 '系统初始化' 的错误
```

## 🔧 常见问题解决

### 1. 模块导入失败
**错误信息：** `ImportError: No module named 'xxx'`

**解决方案：**
```bash
pip install -r requirements.txt
```

### 2. 配置文件缺失
**错误信息：** `缺少配置: configs/model_config.yaml`

**解决方案：**
- 确保配置文件存在于正确位置
- 检查文件名是否正确

### 3. 模型文件路径错误
**错误信息：** `模型权重缺失: xxx.pth`

**解决方案：**
- 确认模型文件实际存在
- 检查 `model_config.yaml` 中的路径配置

### 4. 系统初始化失败
**错误信息：** `初始化失败: xxx`

**解决方案：**
- 检查模型文件是否完整
- 验证PyTorch版本兼容性
- 确认所有依赖包已正确安装

## 📋 依赖包要求

运行健康检查需要以下Python包：
```bash
pip install colorama pyyaml requests pandas numpy torch
```

## 🎯 最佳实践

1. **实验前必做**：每次运行正式实验前都执行健康检查
2. **问题定位**：根据检查结果的严重等级优先解决问题
3. **环境验证**：在不同环境中部署前都要运行检查
4. **自动化集成**：可将健康检查集成到CI/CD流程中

## 📞 支持信息

如果遇到问题，请检查：
1. Python版本 >= 3.8
2. 所有依赖包已正确安装
3. 项目目录结构完整
4. 配置文件格式正确

---
*最后更新：2024年*