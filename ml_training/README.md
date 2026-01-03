# 机器学习模型训练模块 (ml_training)

## 概述

`ml_training` 模块是 CTS_system 的机器学习组件，专门用于训练和评估决策模型。该模块实现了双塔MLP模型的训练、评估和数据收集功能，用于优化容器镜像的压缩策略选择。

## 目录结构

```
ml_training/
├── train_model.py        # 模型训练主模块
├── evaluate_model.py     # 模型评估模块
├── data_collector.py     # 数据收集和处理工具
├── exp_orchestrator.py   # 实验编排模块
├── run_matrix.py         # 实验矩阵执行脚本
├── log_parser.py         # 日志分析工具
├── config.py            # 配置文件
└── README.md            # 本说明文档
```

## 主要功能

### 1. 模型训练 (train_model.py)

- **功能**: 训练双塔MLP模型用于压缩策略决策
- **输入**: 客户端画像、镜像特征、实际成本、使用方法
- **输出**: 训练好的MLP模型和特征缩放器
- **特性**:
  - 提取客户端特征（CPU性能、带宽等）
  - 提取镜像特征（大小、熵值、文件类型分布等）
  - 对压缩方法进行独热编码
  - 支持从反馈数据加载训练样本
  - 自动保存训练模型和缩放器

### 2. 模型评估 (evaluate_model.py)

- **功能**: 评估训练好的决策模型性能
- **输入**: 测试数据、已训练模型
- **输出**: 评估指标、预测结果分析、可视化图表
- **特性**:
  - 计算多种评估指标（MSE、MAE、R²、MAPE等）
  - 绘制实际值vs预测值散点图
  - 绘制残差图
  - 预测最优压缩方法

### 3. 数据收集 (data_collector.py)

- **功能**: 从系统各模块收集训练数据
- **输入**: 系统各模块的原始数据
- **输出**: 标准化的训练数据格式
- **特性**:
  - 从反馈系统收集性能数据
  - 从缓存系统收集压缩数据
  - 从日志系统收集操作数据
  - 合并和过滤数据生成训练样本
  - 保存和加载训练数据

### 4. 实验编排 (exp_orchestrator.py)

- **功能**: 自动化控制大规模实验流程，管理容器环境和数据收集
- **输入**: 实验配置参数
- **输出**: 实验结果数据
- **特性**:
  - 环境初始化：根据配置动态生成docker run命令
  - 网络隔离与仿真：使用tc命令注入带宽限制和延迟
  - 清理机制：每次实验前清理系统缓存，确保实验准确性
  - 断点续跑：记录已完成实验，支持中断后继续
  - 资源监控：记录宿主机资源使用情况
  - 原子化写入：使用UUID命名实验结果文件，防止冲突

### 5. 实验矩阵执行 (run_matrix.py)

- **功能**: 实现三级循环实验执行（客户端配置 -> 目标镜像 -> 压缩方法 -> 重复次数）
- **输入**: 实验配置参数
- **输出**: 实验结果数据和报告
- **特性**:
  - 按照6个客户端配置 × 18个镜像 × 10种算法 × 3次重复的矩阵执行实验
  - 生成实验报告，包含成功率、平均执行时间等统计信息
  - 按客户端配置分类统计性能表现

### 6. 日志分析 (log_parser.py)

- **功能**: 实时监控实验状态，分析实验日志，检测异常情况
- **输入**: 实验日志文件
- **输出**: 分析报告和异常检测结果
- **特性**:
  - 统计实验状态（完成、失败、待处理等）
  - 检测高错误率并发出警告
  - 按实验配置分组统计性能
  - 支持实时监控和历史分析
  - 生成详细的分析报告

### 7. 配置管理 (config.py)

- **功能**: 定义训练过程中的配置参数和常量
- **特性**:
  - 模型配置（隐藏层大小、激活函数等）
  - 数据路径配置
  - 特征配置
  - 实验参数空间模板（包括物理参数映射）
  - 训练和评估配置
  - 部署和资源管理配置

## 训练数据格式

训练数据采用以下格式：

```python
training_data = [
    (client_profile, image_profile, actual_cost, method),
    ...
]
```

### client_profile（客户端画像）
- `cpu_score`: CPU性能分数
- `bandwidth_mbps`: 带宽（Mbps）
- `decompression_speed`: 解压速度字典（gzip, zstd, lz4）
- `network_rtt`: 网络往返时间
- `disk_io_speed`: 磁盘I/O速度
- `memory_size`: 内存大小
- `latency_requirement`: 延迟要求

### image_profile（镜像特征）
- `total_size_mb`: 总大小（MB）
- `avg_layer_entropy`: 平均层熵值
- `text_ratio`: 文本比例
- `binary_ratio`: 二进制比例
- `layer_count`: 层数量
- `file_type_distribution`: 文件类型分布

### actual_cost（实际成本）
- 实际测量的总成本（压缩时间 + 传输时间 + 解压时间）

### method（使用的方法）
- 实际使用的压缩算法（如 "gzip-6", "zstd-3", "lz4-fast" 等）

## 使用方法

### 1. 运行实验矩阵（自动化编排）

```bash
python -m ml_training.run_matrix
```

### 2. 训练模型

```bash
python -m ml_training.train_model
```

### 3. 评估模型

```bash
python -m ml_training.evaluate_model
```

### 4. 分析日志

```bash
python -m ml_training.log_parser
```

### 5. 收集数据

```bash
python -m ml_training.data_collector
```

## 实验设计

根据实验设计原则，系统实现了以下参数配置：

### 客户端配置 (6个)
- C1: 极端弱端（CPU-L, BW-XS, RTT-H, IO-L）
- C2: CPU瓶颈（CPU-L, BW-M, RTT-M, IO-L）
- C3: 网络瓶颈（CPU-M, BW-S, RTT-M, IO-H）
- C4: 均衡配置（CPU-M, BW-L, RTT-L, IO-H）
- C5: 解压瓶颈（CPU-H, BW-S, RTT-L, IO-H）
- C6: 高端节点（CPU-H, BW-L, RTT-L, IO-H）

### 镜像配置 (18个)
- **Linux Distro 类**：CentOS, Fedora, Ubuntu
- **Database 类**：Mongo, MySQL, Postgres
- **Language 类**：Rust, Ruby, Python
- **Web Component 类**：Nginx, Httpd, Tomcat
- **Application Platform 类**：Rabbitmq, Wordpress, Nextcloud
- **Application Tool 类**：Gradle, Logstash, Node

### 压缩算法 (10种)
- Gzip (gzip-1, gzip-6, gzip-9)
- Zstd (zstd-1, zstd-3, zstd-6)
- LZ4 (lz4-fast, lz4-medium, lz4-slow)
- Brotli (brotli-1)

### 实验规模
- 总实验次数：6客户端 × 18镜像 × 10算法 = 1080次
- 每实验重复3次取平均，总计约3240次执行
- 符合硕士级实验规模要求

## 模型架构

使用双塔MLP架构：
- 左塔：处理客户端画像特征
- 右塔：处理镜像特征
- 输出层：预测不同压缩策略的预期成本

## 评估指标

- **MSE (均方误差)**: 衡量预测值与实际值的平均平方误差
- **MAE (平均绝对误差)**: 衡量预测值与实际值的平均绝对误差
- **R² Score**: 衡量模型对数据变化的解释程度
- **MAPE (平均绝对百分比误差)**: 衡量预测误差的百分比

## 数据来源

1. **反馈闭环机制**: 从 PerformanceMonitor 模块收集实际性能数据
2. **系统日志**: 从操作日志中提取传输和压缩性能数据
3. **缓存系统**: 从缓存记录中提取压缩比和时间数据
4. **自动化实验**: 通过exp_orchestrator.py收集系统化实验数据

## 模型更新策略

- **触发条件**: 当收集到足够数量的新样本时
- **验证机制**: 训练完成后进行模型性能验证
- **备份策略**: 保留多个版本的模型以支持回滚
- **部署策略**: 自动部署验证通过的新模型