# CTS实验框架使用详细指南

## 📋 准备工作清单

在开始实验之前，请确保完成以下准备工作：

### 1. 系统环境检查
```bash
# 检查Python版本（需要3.8+）
python --version

# 检查Docker状态
docker --version
systemctl status docker  # Linux系统

# 检查磁盘空间（建议50GB+）
df -h
```

### 2. 网络环境准备
- 确保服务器有公网访问能力
- 准备Harbor私有仓库或类似镜像仓库
- 确认网络接口名称（如eth0, ens33等）

### 3. 权限配置
```bash
# 添加当前用户到docker组（Linux）
sudo usermod -aG docker $USER

# 或者直接使用root权限运行实验
sudo su
```

## ⚙️ 配置文件详解

### 全局配置 (`configs/global_config.yaml`)

```yaml
global:
  seed: 42                    # 随机种子，确保实验可重现
  result_save_dir: "./data/processed"  # 结果保存目录
  figure_save_dir: "./figures"         # 图表保存目录
  log_dir: "./logs"           # 日志目录
  repeat_times: 5             # 每个条件重复实验次数

registry:
  address: "your-harbor.example.com/cts-images"  # 你的镜像仓库地址
  username: "admin"           # 仓库用户名
  password: "your-password"   # 仓库密码

network:
  interface: "eth0"           # 服务器网络接口名

# 测试镜像配置
images:
  - name: "alpine"           # 镜像名称
    tag: "latest"            # 镜像标签
    size_mb: 5               # 预估大小(MB)

# 实验场景配置
scenarios:
  iot_weak_network:          # IoT弱网场景
    name: "IoT弱网场景"
    bandwidth_mbps: 2        # 带宽2Mbps
    delay_ms: 100            # 延迟100ms
    loss_rate: 0.05          # 丢包率5%
    cpu_cores: 0.2           # CPU限制0.2核
    mem_limit_mb: 2048       # 内存限制2GB
```

### 模型配置 (`configs/model_config.yaml`)

```yaml
model:
  model_path: "./models/cts_optimized_0218_2125_seed42.pth"
  preprocess_path: "./models/preprocessing_objects_optimized.pkl"
  device: "cuda"              # 使用GPU，或改为"cpu"

decision_engine:
  safety_threshold: 0.5       # 安全阈值
  enable_uncertainty: true    # 启用不确定性过滤
  enable_dpc: true            # 启用DPC校准
```

## 🚀 执行流程

### 1. 环境初始化
```bash
# 给脚本执行权限
chmod +x 00_prepare_environment.sh
chmod +x 01_build_images.sh

# 执行环境初始化（需要root权限）
sudo ./00_prepare_environment.sh
```

### 2. 镜像准备
```bash
# 修改构建脚本中的仓库配置
vim 01_build_images.sh

# 执行镜像构建和推送
./01_build_images.sh
```

### 3. 配置验证
```bash
# 运行基础功能测试
python test_basic_functionality.py
```

### 4. 执行完整实验
```bash
# 一键执行所有实验
python run_all_experiments.py
```

## 📊 结果解读

### 实验一：端到端性能基准
重点关注：
- **性能提升百分比**：CTS相比基线的实际提升
- **统计显著性**：p值<0.05表示提升具有统计意义
- **场景适应性**：不同网络环境下的表现差异

### 实验二：协同增益消融
分析要点：
- **组件贡献度**：CFT-Net和CAGS各自的价值
- **协同效应**：1+1>2的整体效果
- **边际收益**：每个新增组件带来的额外价值

### 实验三：动态鲁棒性
评估维度：
- **稳定性指标**：变异系数越小越好
- **异常处理**：OOD场景下的应对能力
- **自适应能力**：面对环境变化的调整速度

### 实验四：轻量化部署
关键指标：
- **资源效率**：决策开销占总时间的比例(<1%为佳)
- **响应速度**：P99延迟应该在可接受范围内
- **部署友好性**：内存和CPU占用合理

### 实验五：长期稳定性
监控重点：
- **性能衰减**：<5%被认为是稳定的
- **配置一致性**：避免频繁的策略震荡
- **系统可靠性**：长时间运行的成功率

## 🔧 常见问题解决

### Q1: 权限不足错误
```bash
# 错误信息：Permission denied
# 解决方案：使用sudo或切换到root用户
sudo python run_all_experiments.py
```

### Q2: Docker连接失败
```bash
# 检查Docker服务状态
systemctl status docker

# 重启Docker服务
sudo systemctl restart docker

# 验证Docker权限
docker ps
```

### Q3: 网络配置失败
```bash
# 检查网络接口
ip addr show

# 手动测试tc命令
sudo tc qdisc add dev eth0 root netem rate 10mbit
sudo tc qdisc del dev eth0 root netem
```

### Q4: 模型加载失败
```bash
# 检查模型文件完整性
ls -la models/

# 验证PyTorch安装
python -c "import torch; print(torch.__version__)"
```

### Q5: 实验中途断电/中断
```bash
# 查看已完成的部分结果
ls data/processed/

# 从断点继续（需要修改run_all_experiments.py）
# 或重新开始完整实验
```

## 📈 数据分析建议

### 结果文件说明
```
data/processed/
├── all_raw_results.json          # 所有实验原始数据
├── comprehensive_metrics.json     # 综合计量指标
├── exp1_end_to_end_results.json   # 实验一结果
├── exp2_ablation_results.json     # 实验二结果
├── exp3_robustness_results.json   # 实验三结果
├── exp4_lightweight_results.json  # 实验四结果
└── exp5_stability_results.json    # 实验五结果
```

### 自定义分析
```python
import json
import pandas as pd

# 加载结果数据
with open('data/processed/all_raw_results.json', 'r') as f:
    results = json.load(f)

# 转换为DataFrame进行分析
df = pd.DataFrame(results['exp1']['scenarios']['edge_network']['baseline_results'])

# 计算统计指标
print(df['throughput_mbps'].describe())
```

## 🎨 图表定制

如果需要修改图表样式或添加新的可视化：

```python
# 修改analysis/visualization.py中的绘图函数
# 例如调整颜色主题：
sns.set_palette("Set2")  # 更改配色方案

# 调整图表尺寸：
fig, ax = plt.subplots(figsize=(12, 8))  # 更大的图表

# 修改字体大小：
plt.rcParams.update({'font.size': 14})
```

## 📝 实验报告撰写

基于实验结果，建议按以下结构撰写技术报告：

1. **引言**：研究背景和目标
2. **方法论**：实验设计和评估指标
3. **实验结果**：定量分析和图表展示
4. **讨论**：结果解释和局限性分析
5. **结论**：主要发现和未来工作

## 💡 最佳实践建议

1. **预实验**：先用少量迭代测试配置正确性
2. **监控日志**：实时查看`logs/experiment_main.log`
3. **备份数据**：定期备份重要实验结果
4. **版本控制**：使用Git管理代码和配置变更
5. **文档记录**：详细记录每次实验的配置和结果

---
*如有其他问题，请参考README.md或联系技术支持*