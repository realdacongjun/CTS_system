# CTS 系统重构版实验框架

## 📁 新架构目录结构

```
tests/
├── config.yaml              # 统一配置文件
├── experiment_runner.py     # 主实验运行器（核心）
├── baseline_downloader.py   # 基线下载器
├── adaptive_downloader.py   # 自适应下载器
├── proxy_server.py          # 代理服务器
├── prepare_images.py        # 镜像准备脚本
│
├── src/                     # 核心模型组件
│   ├── __init__.py
│   ├── cts_model.py         # CompactCFTNetV2 模型
│   ├── model_wrapper.py     # 模型包装器
│   ├── decision_engine.py   # CAGS 决策引擎
│   ├── environment.py       # 环境控制器
│   ├── executor.py          # Docker 执行器
│   └── utils.py             # 工具函数
│
├── configs/                 # 配置文件
│   ├── global_config.yaml
│   └── model_config.yaml
│
├── models/                  # 模型文件
│   ├── cts_optimized.pth
│   └── preprocessing_objects.pkl
│
├── data/                    # 数据文件
│   └── image_features_database.csv
│
└── results/                 # 实验结果
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 准备测试镜像
python prepare_images.py
```

### 2. 启动代理服务器
```bash
# 在终端1中启动代理服务器
python proxy_server.py
```

### 3. 运行实验
```bash
# 在终端2中运行实验
python experiment_runner.py
```

## 🔧 核心组件说明

### experiment_runner.py（核心）
- 整合 CTS 模型和下载决策逻辑
- 初始化 CFTNetWrapper 和 CAGSDecisionEngine
- 控制实验流程和结果收集

### src/ 目录组件
- **cts_model.py**: 包含 CompactCFTNetV2 神经网络模型
- **model_wrapper.py**: 模型加载和推理包装器
- **decision_engine.py**: CAGS 决策引擎，实现多目标优化
- **environment.py**: 环境状态采集和网络配置
- **executor.py**: Docker 容器执行管理

### 下载器模块
- **baseline_downloader.py**: 基线策略下载器
- **adaptive_downloader.py**: 自适应策略下载器（集成 CTS）

## 🎯 实验流程

1. **初始化阶段**
   - 加载 CTS 模型系统
   - 采集环境状态
   - 准备测试镜像

2. **决策阶段**
   - 调用 CAGSDecisionEngine 进行智能决策
   - 根据环境特征选择最优压缩策略
   - 考虑不确定性进行鲁棒决策

3. **执行阶段**
   - 通过自适应下载器执行下载
   - 记录性能指标和实验数据

4. **分析阶段**
   - 保存实验结果到 CSV
   - 生成性能对比报告

## ⚙️ 配置说明

### config.yaml 主要配置项：
```yaml
experiment:
  repeat_times: 3           # 实验重复次数
  results_file: "./results/experiment_results.csv"

images:                     # 测试镜像列表
  - "alpine:latest"
  - "nginx:alpine"
  - "busybox:latest"

model:                      # 模型配置
  model_path: "./models/cts_optimized.pth"
  preprocess_path: "./models/preprocessing_objects.pkl"
  device: "cpu"             # 或 "cuda"
```

## 📊 实验数据流

```
环境感知 → 特征提取 → CFT-Net预测 → CAGS决策 → 策略执行 → 结果记录
    ↓          ↓          ↓           ↓          ↓          ↓
Environment  Features   Model      Decision   Download   Results
Controller   Database   Wrapper    Engine     Executor   Storage
```

## 🛠️ 开发说明

### 添加新的压缩策略：
1. 在 `COMPRESSION_MAP` 中添加映射关系
2. 更新 `decision_engine.py` 中的策略空间
3. 修改 `adaptive_downloader.py` 的处理逻辑

### 扩展环境特征：
1. 修改 `environment.py` 的特征采集逻辑
2. 更新模型输入特征维度
3. 重新训练模型

## 📈 预期结果

实验将生成包含以下信息的 CSV 文件：
- 策略类型（baseline/cts_*）
- 压缩算法和线程数
- 下载时间
- 成功率
- 决策置信度（CTS独有）

这形成了完整的**感知→决策→执行→反馈**闭环实验系统。