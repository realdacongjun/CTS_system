# CTS实验框架快速入门 🚀

## 5分钟快速开始

### 第1步：环境检查
```bash
# 克隆项目并进入目录
cd tests

# 运行基础测试
python test_basic_functionality.py
```

### 第2步：配置修改
编辑 `configs/global_config.yaml`：
```yaml
registry:
  address: "你的Harbor地址/cts-images"
  username: "你的用户名"
  password: "你的密码"

network:
  interface: "你的网卡名"  # 如: eth0, ens33
```

### 第3步：执行实验
```bash
# 一键运行所有实验
python run_all_experiments.py
```

## 📁 输出结果

实验完成后，你会在以下目录找到结果：

- `data/processed/` - JSON格式的详细实验数据
- `figures/` - 论文级可视化图表（PNG格式）
- `logs/` - 详细的执行日志

## 🎯 关键指标快速查看

运行结束后，终端会显示核心结果摘要：
- 性能提升百分比
- 统计显著性检验
- 各组件贡献度分析
- 系统稳定性评估

## ❓ 遇到问题？

1. **权限问题**：使用 `sudo` 运行
2. **Docker问题**：检查 `systemctl status docker`
3. **网络问题**：确认网卡名称和仓库地址正确

## 📚 进阶使用

- 详细配置说明：查看 `USAGE_GUIDE.md`
- 完整技术文档：查看 `README.md`
- 自定义实验：修改 `experiments/` 目录下的脚本

---
*开始你的CTS系统评估之旅吧！* 🚀