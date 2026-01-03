# 数据收集功能验证说明

## 概述
本项目包含两个用于验证数据收集功能的脚本：
1. `verify_data_collection.py` - 验证数据收集功能是否正常工作
2. `ml_training/quick_test.py` - 运行小型实验矩阵，快速验证功能

## 环境要求
- Python 3.7+
- Docker 引擎
- Docker Python SDK
- 系统需要有 `tc` 命令（如果不在云模式下运行）

## 验证数据收集功能

### 1. 运行验证脚本
```bash
python verify_data_collection.py
```

### 2. 预期输出
- 启动一个测试容器
- 执行一个简单的实验
- 验证数据收集是否正常
- 输出验证报告，包括：
  - JSON文件是否生成
  - 数据库记录是否创建
  - 关键字段是否完整
  - 解压性能数据是否收集

## 运行快速测试

### 1. 运行快速测试脚本
```bash
cd ml_training
python -m ml_training.quick_test
```

### 2. 预期输出
- 运行4次实验（1客户端 × 2镜像 × 2算法 × 1重复）
- 验证数据收集功能
- 输出实验结果统计
- 保存完整结果到JSON文件

## 验证要点

### 1. JSON文件验证
- 检查是否生成了独立的JSON文件
- 验证文件中包含以下字段：
  - `uuid`: 实验唯一ID
  - `profile_id`: 客户端配置ID
  - `image_name`: 镜像名称
  - `method`: 压缩方法
  - `cost_total`: 总耗时
  - `decompression_time`: 解压时间
  - `decompression_performance`: 解压性能数据
  - `host_cpu_load`: 宿主机CPU负载
  - `host_memory_usage`: 宿主机内存使用率
  - `status`: 实验状态

### 2. 解压性能数据验证
- 验证是否收集到解压性能数据
- 检查以下字段：
  - `decompression_time`: 解压时间
  - `cpu_usage`: CPU使用率
  - `memory_usage`: 内存使用情况
  - `disk_io`: 磁盘I/O
  - `method`: 压缩方法

### 3. 数据库记录验证
- 检查是否在SQLite数据库中创建了记录
- 验证记录数量是否与实验数量匹配
- 确认数据库中包含必要的列

## 云模式运行

如果在云服务器上运行，可以使用云模式跳过对`tc`命令的依赖：

```python
orchestrator = ExperimentOrchestrator(
    registry_url="localhost:5000",
    data_dir="/tmp/test_data",
    container_image="hello-world:latest",
    cloud_mode=True  # 启用云模式
)
```

## 故障排除

### 1. Docker权限问题
确保当前用户有运行Docker的权限：
```bash
sudo usermod -aG docker $USER
```

### 2. 端口冲突
如果出现端口冲突，请更改registry_url参数。

### 3. 磁盘空间不足
确保有足够的磁盘空间存储实验数据。

## 结果分析

验证脚本会输出：
- 成功/失败的实验数量
- 包含解压性能数据的实验数量
- 示例实验结果
- 完整结果保存位置

这些信息可以帮助您确认数据收集功能是否按预期工作。