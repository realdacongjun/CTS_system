# 端到端性能对比实验指南

## 实验目的

比较CTS（Context-aware Granularity Scaling）系统与原生Docker机制在不同网络环境下的性能差异。

## 实验设置

### 硬件配置

- **服务器**（4 vCPU, 8GB RAM）: IP 47.121.137.243
  - 运行Nginx服务，提供文件下载
- **客户端**（2 vCPU, 2GB RAM）
  - 运行实验脚本

### 实验文件

服务器上需要准备以下文件：

1. `generalized_text.tar` (100MB, 低熵，模拟文本文件)
2. `generalized_binary.tar` (100MB, 高熵，模拟二进制文件)

以及对应的压缩版本：

- `.tar.gz` - Gzip压缩（用于原生Docker模拟）
- `.tar.br` - Brotli压缩（用于CTS弱网策略）
- `.tar.lz4` - LZ4压缩（用于CTS强网策略）
- `.tar.zst` - Zstd压缩（用于CTS边缘网络策略）

## 实验流程

### 1. 服务器准备

在服务器上运行：
```bash
sudo python3 server_prep.py
```

此脚本会：
- 生成测试文件
- 为每个文件创建四种压缩格式（Gzip, Brotli, LZ4, Zstd）
- 将文件放置在Nginx根目录

### 2. 客户端实验

在客户端运行：
```bash
python3 e2e_experiment_runner.py --ip 47.121.137.243
```

此脚本会：

1. **场景A：IoT弱网环境**
   - 设置网络：2Mbps带宽，400ms延迟，5%丢包率
   - 运行Native客户端（下载.tar.gz文件）
   - 运行CTS客户端（下载.tar.br文件，Brotli策略）
   
2. **场景C：边缘网络环境**
   - 设置网络：20Mbps带宽，50ms延迟，1%丢包率
   - 运行Native客户端（下载.tar.gz文件）
   - 运行CTS客户端（下载.tar.zst文件，Zstd策略）
   
3. **场景B：云强网环境**
   - 设置网络：100Mbps带宽，20ms延迟，0%丢包率
   - 运行Native客户端（下载.tar.gz文件）
   - 运行CTS客户端（下载.tar.lz4文件，LZ4策略）

## 实验结果

- 所有结果将保存到`experiment_e2e_results_YYYYMMDD_HHMMSS.csv`
- 包含场景、客户端类型、文件类型、下载时间、吞吐量和速度提升比

## 注意事项

1. 需要安装必要的依赖库：
   ```bash
   pip install requests
   ```

2. 需要安装系统压缩工具：
   ```bash
   sudo apt-get install gzip brotli lz4 zstd
   ```

3. 网络控制需要sudo权限

4. 实验过程中会自动清理网络设置

## 结果分析

速度提升比（Speedup Ratio）= Native下载时间 / CTS下载时间

- 如果比值 > 1，表示CTS更快
- 如果比值 < 1，表示Native更快

## 四大天王算法定位

| 算法 | 文件后缀 | 典型场景 | 特征 | 训练标签映射 |
| --- | --- | --- | --- | --- |
| **Brotli** | `.tar.br` | **极弱网 (IoT/2G)** | 极致压缩，解压极慢 | `brotli-1` |
| **Zstd** | `.tar.zst` | **中等网络 (Edge/4G)** | **平衡之王** (压缩比接近Gzip，速度接近LZ4) | `zstd-1`, `zstd-3`, `zstd-6` |
| **LZ4** | `.tar.lz4` | **超强网 (Cloud/5G)** | 极速解压，压缩率低 | `lz4-fast`, `lz4-medium` |
| **Gzip** | `.tar.gz` | **基准 (Baseline)** | 兼容性好，但性能平庸 | `gzip-1` ~ `gzip-9` |