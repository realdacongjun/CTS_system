# CTS系统API集成更新说明

## 📋 更新概述

本次更新实现了方案二的完整API集成架构，使adaptive_downloader能够通过代理服务器API获取最优下载策略。

## 🔧 主要修改内容

### 1. adaptive_downloader.py 核心修改

#### 新增功能：
- **`get_strategy_from_proxy()`** 函数：调用代理服务器API获取下载策略
- **API参数传递**：支持传递环境状态和镜像特征给代理服务器
- **智能fallback机制**：当API不可用时自动降级到本地策略

#### 修改的函数：
- **`cts_download()`** 函数：
  - 新增 `env_state` 和 `image_features` 参数
  - `strategy` 和 `threads` 参数变为可选（默认None）
  - 集成API策略获取逻辑
  - 保持原有下载、解压、加载流程不变

### 2. experiment_runner.py 集成修改

#### 调用方式更新：
```python
cts_res = adaptive_downloader.cts_download(
    image, 
    strategy=None,      # 设为 None，完全由 API 决定
    threads=None,       # 设为 None，完全由 API 决定
    env_state=env_state, # 传入环境状态
    image_features=image_features, # 传入镜像特征
    clear_cache=True
)
```

## 🔄 完整工作流程

新的系统架构工作流程如下：

1. **`experiment_runner`** 收集环境状态和镜像特征
2. **`adaptive_downloader`** 调用代理服务器的 `/api/v1/strategy` API
3. **`proxy_server`** 内部加载 `CAGSDecisionEngine` 进行智能决策
4. **`adaptive_downloader`** 根据返回的下载URL执行并行下载
5. 完成解压和Docker加载流程

## 📊 API接口规范

### 请求接口
```
POST http://localhost:8000/api/v1/strategy
```

### 请求体格式
```json
{
    "image_name": "nginx:latest",
    "env_state": {
        "bandwidth_mbps": 100.0,
        "network_rtt": 20.0,
        "cpu_limit": 8.0,
        "mem_limit_mb": 16384.0,
        "theoretical_time": 10.0,
        "cpu_to_size_ratio": 0.08,
        "mem_to_size_ratio": 163.84,
        "network_score": 5.0
    },
    "image_features": {
        "total_size_mb": 100.0,
        "avg_layer_entropy": 0.8,
        "layer_count": 10,
        "text_ratio": 0.3,
        "zero_ratio": 0.1
    }
}
```

### 响应格式
```json
{
    "strategy": {
        "algo": "zstd_l3",
        "threads": 4,
        "download_url": "http://localhost:8000/data/zstd_l3/nginx_latest.tar.zst"
    },
    "metrics": {
        "pred_time_s": 15.2,
        "epistemic_unc": 0.3,
        "throughput_mbps": 52.6
    }
}
```

## ✅ 验证结果

运行 `test_api_integration.py` 的结果显示：
- ✅ API策略请求功能正常
- ✅ 函数签名兼容性良好
- ✅ 与experiment_runner集成成功

## 🚀 使用说明

### 前提条件
1. 确保 `proxy_server.py` 正在运行
2. 确保模型文件和配置正确

### 运行流程
```bash
# 1. 启动代理服务器
python proxy_server.py

# 2. 在新终端运行实验
python experiment_runner.py
```

### 测试验证
```bash
# 运行API集成测试
python test_api_integration.py

# 运行完整系统健康检查
python system_health_check.py
```

## ⚠️ 注意事项

1. **API连接失败处理**：当无法连接到代理服务器时，系统会自动降级到本地fallback策略
2. **参数兼容性**：保留了原有的参数以确保向后兼容
3. **错误处理**：完善的异常处理机制确保系统稳定性

## 📈 性能优势

相比原方案，新架构的优势：
- **实时决策**：每次下载都基于最新的环境状态做出最优决策
- **智能路由**：代理服务器可以根据实际条件选择最佳压缩策略
- **扩展性强**：易于添加新的决策算法和策略

---
*更新时间：2024年*
*版本：API集成版 v1.0*