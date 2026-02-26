# CTS闭环系统完整评估实验方案

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

这是一个**完全独立、可直接运行、逻辑严谨**的实验框架，专门用于验证「CFT-Net性能预测 + CAGS自适应决策」双创新点组合的闭环系统。

## 🎯 实验目标

本实验方案旨在通过五个精心设计的核心实验，全面验证CTS闭环系统的有效性：

1. **端到端性能基准** - 验证相比工业基线的性能提升
2. **协同增益消融** - 分解各组件的具体贡献
3. **动态鲁棒性** - 测试系统在复杂环境下的稳定性
4. **轻量化部署** - 验证工程可行性和资源效率
5. **长期稳定性** - 评估生产环境下的可靠性

## 📁 标准文件结构

```
cts_experiment/
├── 00_prepare_environment.sh    # 环境初始化脚本
├── 01_build_images.sh            # 多压缩版本镜像构建脚本
├── configs/                      # 实验配置文件目录
│   ├── global_config.yaml        # 全局配置
│   └── model_config.yaml         # 模型配置
├── data/                         # 数据目录
│   ├── raw/                      # 原始实验数据
│   └── processed/                # 处理后实验数据
├── models/                       # 模型文件目录
│   ├── cts_optimized_0218_2125_seed42.pth     # CFT-Net模型
│   └── preprocessing_objects_optimized.pkl    # 预处理对象
├── src/                          # 核心代码目录
│   ├── __init__.py
│   ├── environment.py            # 环境控制与特征采集
│   ├── model_wrapper.py          # CFT-Net模型封装
│   ├── decision_engine.py        # CAGS决策引擎
│   ├── executor.py               # Docker真实执行
│   └── utils.py                  # 工具函数
├── experiments/                  # 实验脚本目录
│   ├── exp1_end_to_end.py        # 实验一：端到端性能基准
│   ├── exp2_ablation.py          # 实验二：协同增益消融
│   ├── exp3_robustness.py        # 实验三：动态鲁棒性
│   ├── exp4_lightweight.py       # 实验四：轻量化部署
│   └── exp5_stability.py         # 实验五：长期稳定性
├── analysis/                     # 结果分析与可视化
│   ├── metrics.py                # 评估指标计算
│   └── visualization.py          # 论文级图表生成
├── run_all_experiments.py        # 一键执行所有实验
└── README.md                     # 本说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd cts_experiment

# 执行环境初始化（需要root权限）
sudo ./00_prepare_environment.sh

# 构建测试镜像（需要配置Harbor仓库）
./01_build_images.sh
```

### 2. 配置修改

编辑 `configs/global_config.yaml`：

```yaml
# 修改私有仓库配置
registry:
  address: "your-harbor-ip/cts-images"  # 你的Harbor地址
  username: "admin"
  password: "your-password"

# 修改网络接口
network:
  interface: "eth0"  # 你的服务器网卡名

# 根据需要调整场景参数
scenarios:
  iot_weak_network:
    bandwidth_mbps: 2    # 可根据实际环境调整
    delay_ms: 100
    loss_rate: 0.05
```

### 3. 模型配置

确认模型文件已放置在 `models/` 目录下：
- `cts_optimized_0218_2125_seed42.pth` - CFT-Net模型权重
- `preprocessing_objects_optimized.pkl` - 预处理对象

### 4. 执行实验

```bash
# 一键执行所有实验
python run_all_experiments.py
```

## 🧪 核心实验详解

### 实验一：端到端性能基准
**验证目标**：量化CTS相比工业基线的性能提升

**执行逻辑**：
- 覆盖IoT弱网、边缘网络、云数据中心三大场景
- 每场景测试3个典型镜像
- 配对对比：基线 vs CTS系统
- 统计检验：配对样本t检验(p<0.05)

**关键指标**：
- 端到端拉取耗时
- 吞吐量提升百分比
- 任务成功率
- 统计显著性

### 实验二：协同增益消融
**验证目标**：分解CFT-Net和CAGS的各自贡献

**对比变体**：
- V1_Static：纯静态基线
- V2_CFT_Only：仅CFT-Net预测
- V3_CFT_CAGS_NoUnc：关闭不确定性过滤
- V4_CTS_Full：完整CTS系统

**分析维度**：
- 各变体性能阶梯对比
- 协同增益量化分析
- 组件贡献度分解

### 实验三：动态鲁棒性
**验证目标**：验证复杂环境下的系统稳定性

**测试场景**：
- 动态网络波动：带宽/延迟随机跳变
- OOD极端弱网：训练数据外的极端场景
- 不确定性检测能力验证

**评估指标**：
- 吞吐量变异系数(CV)
- OOD检测准确率
- 性能退化程度

### 实验四：轻量化部署
**验证目标**：确认工程可行性和资源效率

**测试维度**：
- 模型参数量和内存占用
- 决策延迟分布(P50/P95/P99)
- CPU/Memory资源消耗
- 决策开销占比分析

### 实验五：长期稳定性
**验证目标**：评估生产环境可靠性

**测试方案**：
- 50次连续冷启动拉取
- 前期vs后期性能对比
- 配置一致性和异常检测
- 长期运行稳定性评估

## 📊 输出结果

### 数据文件
- `data/processed/all_raw_results.json` - 所有实验原始数据
- `data/processed/comprehensive_metrics.json` - 综合评估指标
- 各实验独立结果文件

### 可视化图表
- `figures/fig1_scenario_comparison.png` - 场景性能对比柱状图
- `figures/fig2_ablation_ladder.png` - 消融实验阶梯图
- `figures/fig3_robustness_boxplot.png` - 鲁棒性箱线图
- `figures/fig4_decision_latency.png` - 决策延迟分布图
- `figures/fig5_stability_trend.png` - 稳定性趋势图

所有图表均为300 DPI，支持中英文，可直接用于学术论文。

## 🔧 技术特点

### 完整性保障
- ✅ 端到端可执行，无需额外依赖
- ✅ 标准化实验流程和配置
- ✅ 自动化结果收集和分析
- ✅ 论文级可视化输出

### 科学性保证
- ✅ 严格的统计检验方法
- ✅ 多维度性能评估指标
- ✅ 可重现的实验设计
- ✅ 详尽的结果分析

### 工程化设计
- ✅ 模块化代码结构
- ✅ 完善的日志记录
- ✅ 异常处理和容错机制
- ✅ 资源管理和清理

## 📋 系统要求

### 硬件要求
- CPU: 4核以上
- 内存: 8GB以上
- 存储: 50GB可用空间
- 网络: 可配置网络环境的服务器

### 软件依赖
- Python 3.8+
- Docker 20.10+
- Linux系统（推荐Ubuntu 20.04+）
- Root权限（用于网络配置）

### Python依赖包
```bash
pip install torch pandas numpy matplotlib seaborn scipy scikit-learn psutil pyyaml
```

## 📝 注意事项

1. **权限要求**：网络环境配置需要root权限
2. **Docker配置**：确保Docker服务正常运行
3. **网络环境**：建议在隔离的测试环境中执行
4. **资源配置**：根据服务器配置调整实验参数
5. **时间预算**：完整实验约需2-4小时执行时间

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个实验框架。

## 📄 许可证

本项目采用MIT许可证，详情请见[LICENSE](LICENSE)文件。

## 🙏 致谢

感谢所有为CTS系统研究做出贡献的研究人员和开源社区。

---
*CTS闭环系统实验平台 - 为容器传输优化研究提供标准化评估框架*