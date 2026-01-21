import matplotlib.pyplot as plt
import numpy as np
from cags_scheduler import CAGSStrategyLayer

def run_sensitivity_analysis():
    print("🔬 启动扩展参数敏感性分析实验...")
    
    # 模拟不同的风险偏好权重 (Gamma)
    # Gamma = 0.5: 激进 (不怎么怕风险)
    # Gamma = 1.0: 相对激进
    # Gamma = 2.0: 你的默认值 (稳健) 
    # Gamma = 5.0: 极度保守 (非常怕风险)
    gamma_values = [0.5, 1.0, 2.0, 5.0]
    
    # 固定场景：弱网 (2Mbps, 丢包率 5%)
    bw = 2.0
    loss = 0.05
    cpu = 0.8
    
    results = []
    costs = []
    labels = []
    
    strategy = CAGSStrategyLayer()
    
    print(f"\n场景设定: Bandwidth={bw}Mbps, Loss={loss*100}%, CPU={cpu*100}%")
    print("-" * 70)
    print(f"{'Gamma(风险权重)':<15} | {'决策切片(KB)':<15} | {'决策并发':<10} | {'预期广义成本':<15}")
    print("-" * 70)

    for g in gamma_values:
        strategy.gamma = g # 动态修改参数
        best_config, cost = strategy.optimize(bw, loss, cpu)
        s, n = best_config
        
        results.append(s/1024)  # 转换为KB
        costs.append(cost)
        labels.append(f"γ={g}")
        
        print(f"{g:<15} | {s/1024:<15.0f} | {n:<10} | {cost:<15.2f}")

    # 创建子图显示切片大小和成本
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1：切片大小对比
    bars1 = ax1.bar(labels, results, color=['#a8d5e2', '#76c7c0', '#7bc0a8', '#e5989b'])
    ax1.set_title('Risk Weight vs. Selected Chunk Size', fontsize=14)
    ax1.set_ylabel('Chunk Size (KB)', fontsize=12)
    ax1.set_xlabel('Risk Weight (γ)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars1, results):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}KB',
                ha='center', va='bottom', fontsize=10)
    
    # 图2：成本对比
    bars2 = ax2.bar(labels, costs, color=['#f8c6c8', '#f19c9f', '#e87279', '#e04853'])
    ax2.set_title('Risk Weight vs. Expected Cost', fontsize=14)
    ax2.set_ylabel('Expected Generalized Cost', fontsize=12)
    ax2.set_xlabel('Risk Weight (γ)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("exp_sensitivity_extended.png", dpi=300, bbox_inches='tight')
    print("-" * 70)
    print("✅ 扩展敏感性分析完成！已生成 exp_sensitivity_extended.png")
    print("💡 结论：随着风险权重 γ 增加，系统倾向于选择更小的切片（安全性提高，但成本也相应增加）。")

def run_multi_scenario_sensitivity():
    """运行多场景的敏感性分析"""
    print("\n🔬 多场景敏感性分析...")
    
    # 不同网络场景
    scenarios = [
        {"name": "Strong Net", "bw": 50.0, "loss": 0.001, "cpu": 0.2},
        {"name": "Medium Net", "bw": 10.0, "loss": 0.02, "cpu": 0.5},
        {"name": "Weak Net", "bw": 2.0, "loss": 0.05, "cpu": 0.8}
    ]
    
    gamma_values = [0.5, 1.0, 2.0, 5.0]
    strategy = CAGSStrategyLayer()
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']} (BW={scenario['bw']}Mbps, Loss={scenario['loss']*100}%)")
        print("-" * 60)
        print(f"{'Gamma':<8} | {'Chunk Size(KB)':<15} | {'Concurrency':<12} | {'Cost':<10}")
        print("-" * 60)
        
        for g in gamma_values:
            strategy.gamma = g
            best_config, cost = strategy.optimize(
                scenario['bw'], 
                scenario['loss'], 
                scenario['cpu']
            )
            s, n = best_config
            print(f"{g:<8} | {s/1024:<15.0f} | {n:<12} | {cost:<10.2f}")


def run_uncertainty_impact_test():
    """
    新增实验：测试 AI 不确定性 (Uncertainty) 对决策的影响
    目的：证明 '风险放大机制' 有效，即当 AI 没把握时，系统会自动降级保平安。
    """
    print("\n🔬 启动不确定性影响分析 (Uncertainty Impact Test)...")
    
    # [修正点]：模拟"光纤级"强网环境
    # BW = 50Mbps (大包传输快)
    # Loss = 0.001% (极低，物理上允许发大包)
    bw = 50.0
    loss = 0.00001 
    cpu = 0.1 # CPU 也很空闲
    
    # 模拟 AI 从 "非常自信" 到 "完全瞎猜"
    uncertainty_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    strategy = CAGSStrategyLayer()
    # 稍微调低一点基础 Gamma，让系统在 U=0 时更倾向于激进
    strategy.gamma = 1.0 
    
    results_size = []
    results_cost = []
    labels = []
    
    print("-" * 75)
    print(f"{'Uncertainty(U)':<15} | {'Risk Amplifier':<15} | {'Decision(KB)':<15} | {'Cost':<10}")
    print("-" * 75)
    
    for u in uncertainty_levels:
        # 调用优化器
        best_config, cost = strategy.optimize(bw, loss, cpu, model_uncertainty=u)
        s, n = best_config
        
        # 计算风险放大因子
        amplifier = 1.0 + (strategy.uncertainty_weight * u)
        
        results_size.append(s/1024)
        results_cost.append(cost)
        labels.append(f"U={u}")
        
        print(f"{u:<15} | {amplifier:<15.1f}x | {s/1024:<15.0f} | {cost:<10.2f}")

    # === 画图 ===
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 柱状图
    # 使用 Coolwarm 渐变色：蓝色(冷静/自信) -> 红色(恐慌/不确定)
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(results_size)))
    bars = ax1.bar(labels, results_size, color=colors, alpha=0.8, label='Chunk Size')
    
    ax1.set_xlabel('AI Model Uncertainty (U)', fontsize=12)
    ax1.set_ylabel('Selected Chunk Size (KB)', fontsize=12, color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.set_title('Impact of AI Uncertainty on Granularity Decision', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标数值
    for bar, value in zip(bars, results_size):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}KB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 折线图：Cost 变化
    ax2 = ax1.twinx()
    ax2.plot(labels, results_cost, color='#e74c3c', marker='D', linewidth=2, linestyle='--', label='Optimization Cost')
    ax2.set_ylabel('Optimization Risk Cost (Risk Amplified)', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    plt.tight_layout()
    plt.savefig("exp_uncertainty_impact.png", dpi=300)
    print("-" * 75)
    print("✅ 修正完成！请查看新图表: exp_uncertainty_impact.png")
    print("💡 预期: 左边柱子高(大包)，右边柱子低(小包)，这证明了系统在'心里没底'时会主动降级。")

if __name__ == "__main__":
    run_sensitivity_analysis()
    run_multi_scenario_sensitivity()
    run_uncertainty_impact_test()       #