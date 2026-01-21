# CAGS 系统项目详细说明文档

## 项目背景与目标

CAGS（Context-aware Granularity Scaling）是一个基于AI和不确定性感知的容器镜像传输调度系统，旨在解决异构网络环境下的可靠传输问题。项目经历了从仿真验证到真实环境部署的演进过程。

## 项目架构演进

### 1. 初始阶段：仿真验证
- 使用模拟数据进行算法验证
- 验证CAGS三层架构（战略层、战术层、修正层）的有效性
- 使用数学模型进行性能预测

### 2. 过渡阶段：真实环境原型
- 将仿真逻辑转换为真实HTTP请求
- 实现基于Range Requests的分片下载
- 集成训练好的AI模型进行决策

### 3. 当前阶段：端云协同部署
- 支持单机版和分布式版两种部署模式
- 服务端负责AI决策，客户端负责执行
- 实现完整的"感知→决策→执行→反馈"闭环

## 项目文件结构

```
cags_real_experiment/
├── cts_model.py              # AI模型定义（双塔EDL模型，4输出: Gamma,v,Alpha,Beta）
├── cags_scheduler.py         # CAGS调度算法
│   ├── CAGSStrategyLayer     # 战略层：基于势垒函数的非凸优化
│   ├── CAGSTacticalLayer     # 战术层：P-C流水线与乱序重排机制
│   └── CAGSCorrectionLayer   # 修正层：AIMD算法进行动态反馈控制
├── real_sensor.py            # 环境感知模块（参考probe设计）
├── real_downloader.py        # 基于Range请求的真实下载器
├── benchmark_executor.py     # 单机版主程序（支持cags/static/aimd三种模式）
├── cags_distributed/         # 分布式版组件
│   ├── server_brain.py       # 服务端AI决策服务（Flask API）
│   └── smart_client.py       # 客户端主程序
├── README.md                 # 项目说明文档
└── PROJECT_OVERVIEW.md       # 本详细说明文档
```

## 核心组件详解

### 1. cts_model.py - AI模型定义
- **CTSDualTowerModel**: 双塔证据深度学习模型
  - 输入: Client特征(4维) + Image特征(5维) + 算法ID
  - 输出: 4个参数(Gamma, v, Alpha, Beta)用于不确定性估计
  - 使用Softplus约束保证参数非负
  - 包含TransformerTower和FeatureTokenizer组件

### 2. cags_scheduler.py - 调度算法
- **CAGSStrategyLayer**: 战略层，实现基于势垒函数的非凸优化
  - 目标函数: Cost = α×T_trans + β×C_cpu + γ×R_risk
  - 考虑传输时间、CPU开销、传输风险
  - 支持不确定性感知的风险放大机制
  - 并发增益使用边际收益递减模型(n^0.9)

- **CAGSCorrectionLayer**: 修正层，实现AIMD算法
  - 成功路径：加性增加块大小(+256KB)
  - 失败路径：连续失败2次触发乘性减少(/2)
  - 支持rtt_ms参数用于高级优化

### 3. real_sensor.py - 环境感知
- **RealSensor**: 实现网络和系统特征探测
  - 测量RTT：HEAD请求测量往返时间
  - 估算带宽：下载样本数据估算
  - 系统探测：CPU性能、内存信息、系统负载
  - 提供完整客户端画像

### 4. real_downloader.py - 真实下载器
- **RealDownloader**: 基于Range Requests的多线程下载器
  - 支持并发下载多个分片
  - 实现文件预分配防止碎片
  - 支持进度跟踪和速度计算
  - 集成AIMD反馈机制

### 5. benchmark_executor.py - 单机版主程序
- 支持三种运行模式：
  - `cags`: AI自适应策略（加载模型→感知环境→AI决策→动态下载）
  - `static`: 静态策略（固定4MB分片+3并发，模拟Docker）
  - `aimd`: AIMD策略（固定初始配置，动态调整）
- 实现特征标准化以匹配训练分布
- 集成系统资源监控

### 6. cags_distributed/server_brain.py - 服务端
- Flask API服务，提供/negotiate端点
- 加载AI模型进行策略计算
- 实现特征标准化处理
- 返回最优分片大小和并发数

### 7. cags_distributed/smart_client.py - 客户端
- 感知本地环境
- 向服务端请求AI决策
- 使用返回策略执行下载
- 服务端不可达时使用fallback策略

## AI模型集成细节

### 模型文件
- **训练模型**: `ml_training/modeling/cts_best_model_full.pth`
- **模型类型**: 证据深度学习（Evidential Deep Learning）模型
- **输出参数**: Gamma(预测时间), v, Alpha, Beta(用于不确定性估计)

### 不确定性计算
- **公式**: U = β / (v × (α - 1) + 1e-6)
- **用途**: 风险放大机制，高不确定性触发保守策略
- **实现**: 在战略层集成不确定性权重

### 特征标准化
- **重要性**: 确保输入特征与训练分布一致
- **实现**: SimpleScaler类，使用Z-score标准化
- **参数**: 基于训练数据的均值和标准差

## 部署模式详解

### 单机版部署
```bash
# CAGS自适应策略
python benchmark_executor.py --mode cags --url http://47.120.14.12/real_test.bin

# 静态策略（基线）
python benchmark_executor.py --mode static --url http://47.120.14.12/real_test.bin

# AIMD策略（动态）
python benchmark_executor.py --mode aimd --url http://47.120.14.12/real_test.bin
```

### 分布式版部署
#### 服务端（4C8G边缘服务器）
```bash
cd cags_distributed
python server_brain.py
```
- 运行在`http://192.168.1.100:5000`
- 负责AI模型推理和策略计算
- 接收客户端环境信息，返回最优配置

#### 客户端（2C2G终端设备）
```bash
cd cags_distributed
python smart_client.py
```
- 感知本地环境并上报
- 接收服务端策略指导
- 执行分片下载任务

## 关键算法实现

### 1. 势垒函数优化
```python
# 传输时间计算（考虑并发收益递减）
effective_bw = bw_bps * (n ** 0.9)
t_trans = s / effective_bw

# CPU负载计算（上下文切换+系统调用开销）
syscall_overhead = 0.005 * (1024*1024 / s)
thread_overhead = 0.02 * n
task_load = thread_overhead + syscall_overhead
c_cpu = math.exp(4 * current_total_load)  # 指数势垒

# 风险成本（基于伯努利试验模型）
num_packets = s / MTU
prob_fail = 1 - (1 - predicted_loss_rate) ** num_packets
r_risk = (prob_fail * t_trans * 10) * risk_amplifier
```

### 2. AIMD动态调整
```python
def feedback(self, status, rtt_ms=None):
    if status == 'TIMEOUT':
        self.fail_streak += 1
        if self.fail_streak >= self.tolerance_threshold:  # 连续失败2次
            self.current_size = max(self.min_size, self.current_size // 2)
            self.fail_streak = 0
    elif status == 'SUCCESS':
        self.success_streak += 1
        self.fail_streak = 0  # 成功重置失败计数
        if self.success_streak > 5:
            if self.current_size < self.max_size:
                self.current_size = min(self.max_size, self.current_size + 256*1024)
                self.success_streak = 0
    return self.current_size
```

## 技术特点与创新

### 1. 不确定性感知机制
- 使用证据深度学习框架估计模型不确定性
- 将不确定性作为风险因子输入决策层
- 高不确定性触发保守传输策略

### 2. 势垒函数优化
- 非凸优化问题建模
- 指数势垒函数实现软约束
- 考虑边际收益递减效应

### 3. 自适应AIMD控制
- 动态调整分片大小
- 容忍短期网络波动
- 避免过度反应

### 4. 特征标准化
- 确保推理输入与训练分布一致
- 防止模型输出饱和
- 保证决策准确性

## 实验验证方案

### 1. 对比实验设计
- **CAGS模式**: AI+不确定性感知+AIMD
- **Static模式**: 固定配置（模拟Docker）
- **AIMD模式**: 动态调整但无AI决策

### 2. 评估指标
- 传输时间
- 传输成功率
- 瞬时速度稳定性
- RTO重传次数

### 3. 测试环境
- 弱网模拟（Clumsy/TC工具）
- 跨地域公网链路
- 异构硬件平台验证

## 项目状态总结

### 已完成
- ✓ AI模型训练与部署
- ✓ 真实HTTP分片下载实现
- ✓ 端云协同架构设计
- ✓ 不确定性感知机制
- ✓ 完整闭环反馈系统

### 可进一步优化
- 网络状态实时监控与反馈
- 更复杂的错误恢复机制
- 多种压缩算法支持
- 容器镜像仓库集成

### 项目价值
- 验证了AI驱动的自适应传输策略在真实环境中的有效性
- 为容器镜像传输优化提供了新的解决方案
- 体现了边缘计算场景下的智能调度思想