# CAGS 真实环境实验系统

## 项目概述

CAGS（Context-aware Granularity Scaling）是一个基于AI和不确定性感知的容器镜像传输调度系统，旨在解决异构网络环境下的可靠传输问题。本项目包含两个部署模式：

1. **单机版**：所有组件运行在同一台机器上，用于快速验证
2. **分布式版**：服务端（AI决策）与客户端（执行）分离，用于真实场景部署

## 文件结构

```
cags_real_experiment/
├── cts_model.py              # AI模型定义
├── cags_scheduler.py         # CAGS调度算法（战略层+修正层）
├── real_sensor.py            # 环境感知模块
├── real_downloader.py        # 真实下载执行器（含数据采集）
├── benchmark_executor.py     # 单机版主程序（含数据采集）
├── cags_distributed/         # 分布式版
│   ├── server_brain.py       # 服务端AI决策服务
│   └── smart_client.py       # 客户端主程序（含数据采集）
├── README.md                 # 本说明文档
└── PROJECT_OVERVIEW.md       # 详细项目说明文档
```

## 部署模式

### 1. 单机版（已有的实验模式）

运行三种模式的对比实验：

```bash
# CAGS自适应策略模式
python benchmark_executor.py --mode cags --url http://47.120.14.12/real_test.bin

# 静态策略模式（模拟Docker）
python benchmark_executor.py --mode static --url http://47.120.14.12/real_test.bin

# AIMD模式（动态调整）
python benchmark_executor.py --mode aimd --url http://47.120.14.12/real_test.bin
```

### 2. 分布式版（端云协同模式）

#### 服务端部署（4C8G边缘服务器）

```bash
cd cags_distributed
python server_brain.py
```

服务将运行在 `http://192.168.1.100:5000`（IP需根据实际环境修改）

#### 客户端部署（2C2G终端设备）

```bash
cd cags_distributed
python smart_client.py
```

客户端会：
1. 感知本地环境
2. 向服务端请求AI决策
3. 使用返回的策略执行下载
4. 在服务端不可达时使用默认策略

## 数据采集功能

### 微观数据采集
- **文件**: `microscopic_log_YYYYMMDD_HHMMSS.csv`
- **内容**: 每个分片的详细信息
- **字段**: [时间戳, 分片大小(KB), 瞬时速度(MB/s), 状态]

### 宏观数据采集
- **文件**: `experiment_summary.csv`
- **内容**: 实验的整体统计数据
- **字段**: [时间戳, 模式, 带宽, RTT, CPU负载, 内存, 不确定性, 初始分片, 并发数, 总耗时, 平均速度, 成功标志]

## 核心功能

### AI模型
- 使用CTSDualTowerModel进行预测
- 输出Gamma, v, Alpha, Beta四个参数
- 计算不确定性U = β / (v × (α - 1))

### 战略层
- 基于势垒函数的非凸优化
- 考虑传输时间、CPU开销和风险因素
- 支持不确定性感知的风险放大机制

### 修正层（AIMD）
- 动态调整分片大小
- 成功时加性增长，失败时乘性减少
- 连续失败才触发调整，避免过度敏感

### 环境感知
- 实时测量RTT、带宽
- 探测CPU性能、内存信息
- 估算网络稳定性、丢包率

## 技术特点

1. **特征标准化**：确保输入数据与训练分布一致
2. **容错机制**：服务端不可达时自动降级
3. **多线程下载**：基于HTTP Range Requests
4. **实时反馈**：AIMD机制动态调整策略
5. **数据完整性**：下载后验证文件大小
6. **数据采集**：详细的微观和宏观数据记录

## 实验验证

通过对比CAGS、Static和AIMD三种模式在不同网络条件下的表现，验证AI驱动的自适应传输策略的有效性，特别是在弱网环境下的优势。