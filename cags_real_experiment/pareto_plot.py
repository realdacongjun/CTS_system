import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

# ================= 🔧 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "pareto_results_20260131_173001.csv") 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "paper_figures_v3_complete")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置 IEEE 论文通用绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ================= 🛠️ 数据加载与清洗 =================

def load_real_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return None
    try:
        with open(INPUT_FILE, 'r') as f:
            lines = f.readlines()
        header = lines[0]
        data_lines = [line for line in lines[1:] if not line.startswith("run_id")]
        from io import StringIO
        df = pd.read_csv(StringIO(header + "".join(data_lines)))
        
        # 清洗
        df = df.drop_duplicates(subset=['run_id'], keep='last')
        # 定义真实风险：ExitCode!=0 或 吞吐量极低 (<0.1Mbps)
        df['is_failure'] = (df['exit_code'] != 0) | (df['throughput_mbps'] < 0.1)
        
        print(f"✅ 数据加载成功，共 {len(df)} 条")
        return df
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None

def get_pareto_frontier(df):
    """计算前沿点"""
    valid_df = df[df['is_failure'] == False].copy()
    if valid_df.empty: return valid_df
    
    # 按成本排序
    sorted_df = valid_df.sort_values('cost_cpu_seconds')
    frontier = []
    curr_max_thr = -1.0
    
    for idx, row in sorted_df.iterrows():
        # 如果当前点的吞吐量比之前所有更低成本的点都高，则保留
        if row['throughput_mbps'] > curr_max_thr:
            frontier.append(row)
            curr_max_thr = row['throughput_mbps']
            
    return pd.DataFrame(frontier)

# ================= 🎨 绘图函数群 (5.1 - 5.5) =================

def plot_fig_5_1_sampling(df):
    """图 5.1: 真实分层采样分布"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 区分 Anchor (10/100MB) 和 Probe (300MB)
    df['type'] = df['file_size_mb'].apply(lambda x: 'Probe (300MB)' if x == 300 else 'Anchor (10/100MB)')
    colors = {'Anchor (10/100MB)': '#3498db', 'Probe (300MB)': '#e74c3c'}
    markers = {'Anchor (10/100MB)': 'o', 'Probe (300MB)': '^'}
    
    for t in df['type'].unique():
        sub = df[df['type'] == t]
        ax.scatter(sub['threads'], sub['cpu_quota'], sub['chunk_kb'], 
                   c=colors[t], marker=markers[t], s=50, label=t, alpha=0.8, edgecolors='w')
    
    ax.set_xlabel('Threads')
    ax.set_ylabel('CPU Quota')
    ax.set_zlabel('Chunk Size (KB)')
    ax.set_title('Figure 5.1: Stratified Sampling Design', pad=20)
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_5_1_Sampling.png"), dpi=300, bbox_inches='tight')
    print("✅ 图 5.1 完成")

def plot_fig_5_2_risk_barrier(df):
    """图 5.2: 风险势垒 (IoT)"""
    plt.figure(figsize=(12, 6))
    subset = df[df['scenario'].str.contains('IoT')].copy()
    
    # 状态分类
    subset['status'] = subset.apply(lambda x: 'Failed' if x['exit_code']!=0 
                                    else ('High Risk' if x['throughput_mbps'] < 0.5 else 'Feasible'), axis=1)
    palette = {'Failed': '#e74c3c', 'High Risk': '#f39c12', 'Feasible': '#2ecc71'}
    
    sns.scatterplot(data=subset, x='run_id', y='throughput_mbps', hue='status', palette=palette, s=80)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Min Barrier (0.5 Mbps)')
    
    plt.yscale('log')
    plt.title('Figure 5.2: Risk Barrier Mechanism', fontsize=16)
    plt.ylabel('Throughput (Mbps) [Log Scale]')
    plt.xlabel('Experiment Run ID')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_5_2_Risk_Barrier.png"), dpi=300, bbox_inches='tight')
    print("✅ 图 5.2 完成")

def plot_fig_5_3_morphology(df):
    """图 5.3: 帕累托形态对比 (分别取 10MB 和 100MB)"""
    plt.figure(figsize=(10, 7))
    
    configs = [
        {'sc': 'Cloud_Fast', 'size': 100, 'color': '#2ecc71', 'label': 'Cloud (Convex)'},
        {'sc': 'IoT_Weak',   'size': 10,  'color': '#e74c3c', 'label': 'IoT (Collapse)'}
    ]
    
    for cfg in configs:
        sub = df[(df['scenario'] == cfg['sc']) & (df['file_size_mb'] == cfg['size'])]
        frontier = get_pareto_frontier(sub)
        
        if not frontier.empty:
            # 归一化 (Min-Max)
            c = frontier['cost_cpu_seconds']
            t = frontier['throughput_mbps']
            norm_c = (c - c.min()) / (c.max() - c.min() + 1e-6)
            norm_t = (t - t.min()) / (t.max() - t.min() + 1e-6)
            
            plt.plot(norm_c, norm_t, marker='o', linewidth=3, label=cfg['label'], color=cfg['color'])

    plt.title('Figure 5.3: Pareto Frontier Morphology', fontsize=16)
    plt.xlabel('Normalized CPU Cost (Lower is Better)')
    plt.ylabel('Normalized Throughput (Higher is Better)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_5_3_Morphology.png"), dpi=300, bbox_inches='tight')
    print("✅ 图 5.3 完成")

def plot_fig_5_4_knee_point(df):
    """图 5.4: 动态膝点检测 (Weight Adaptation)"""
    # 这一张图最适合用 Cloud 场景 (100MB) 来展示，因为它的曲线是凸的，膝点移动明显
    sub = df[(df['scenario'] == 'Cloud_Fast') & (df['file_size_mb'] == 100)]
    frontier = get_pareto_frontier(sub)
    
    if frontier.empty:
        print("⚠️ 无法生成图 5.4 (缺少 Cloud 数据)")
        return

    plt.figure(figsize=(10, 6))
    
    # 1. 绘制前沿曲线
    plt.plot(frontier['cost_cpu_seconds'], frontier['throughput_mbps'], 'k--', label='Pareto Frontier', alpha=0.5)
    
    # 2. 模拟三种权重偏好，计算膝点
    # 归一化数据用于计算距离
    c = frontier['cost_cpu_seconds']
    t = frontier['throughput_mbps']
    norm_c = (c - c.min()) / (c.max() - c.min() + 1e-6)
    norm_t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    
    weights = [
        {'name': 'Energy First', 'wc': 0.8, 'wt': 0.2, 'color': '#27ae60', 'marker': 's'}, # 侧重省电
        {'name': 'Balanced',     'wc': 0.5, 'wt': 0.5, 'color': '#f39c12', 'marker': 'o'}, # 平衡
        {'name': 'Perf First',   'wc': 0.2, 'wt': 0.8, 'color': '#c0392b', 'marker': '^'}  # 侧重性能
    ]
    
    for w in weights:
        # 计算加权欧氏距离: dist = sqrt( wc*cost^2 + wt*(1-thr)^2 )
        # cost越小越好(0), thr越大越好(1)
        dist = np.sqrt(w['wc'] * norm_c**2 + w['wt'] * (1 - norm_t)**2)
        best_idx = dist.idxmin()
        best_point = frontier.loc[best_idx]
        
        # 绘制点
        plt.scatter(best_point['cost_cpu_seconds'], best_point['throughput_mbps'], 
                    s=150, c=w['color'], marker=w['marker'], label=f"{w['name']} ($w_c={w['wc']}$)", zorder=10, edgecolors='k')
        
        # 标注
        plt.annotate(f"{best_point['threads']}T", 
                     (best_point['cost_cpu_seconds'], best_point['throughput_mbps']),
                     xytext=(0, 15), textcoords='offset points', ha='center', fontsize=10, color=w['color'], fontweight='bold')

    plt.xscale('log') # 使用 Log 轴看得更清楚
    plt.title('Figure 5.4: Dynamic Knee Point Adaptation', fontsize=16)
    plt.xlabel('CPU Cost (s) [Log Scale]')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_5_4_Knee_Adaptation.png"), dpi=300, bbox_inches='tight')
    print("✅ 图 5.4 完成")

def plot_fig_5_5_gain_real(df):
    """图 5.5: 综合性能提升 (逻辑修正版)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # IoT Weak (10MB)
    iot_df = df[(df['scenario'] == 'IoT_Weak') & (df['file_size_mb'] == 10)].copy()
    if not iot_df.empty:
        iot_df = iot_df[iot_df['throughput_mbps'] < 20] # 清洗
        base_val = iot_df[iot_df['threads'] == 1]['throughput_mbps'].mean()
        opt_val = iot_df['throughput_mbps'].max()
        
        axes[0].bar(['Traditional\n(1 Thread)', 'Ours\n(Pareto)'], [base_val, opt_val], color=['gray', '#e74c3c'], width=0.5)
        gain = (opt_val - base_val)/base_val * 100 if base_val>0 else 0
        axes[0].text(1, opt_val, f"+{gain:.0f}%\n({opt_val:.1f} Mbps)", ha='center', va='bottom', fontsize=14, fontweight='bold', color='#c0392b')
        axes[0].set_title('IoT Weak: Throughput Gain', fontsize=14)
        axes[0].set_ylabel('Throughput (Mbps)')

    # Cloud Fast (100MB)
    cloud_df = df[(df['scenario'] == 'Cloud_Fast') & (df['file_size_mb'] == 100)].copy()
    if not cloud_df.empty:
        base_cost = cloud_df[cloud_df['threads'] == 16]['cost_cpu_seconds'].mean()
        valid = cloud_df[cloud_df['throughput_mbps'] > 800]
        if valid.empty: valid = cloud_df
        opt_cost = valid['cost_cpu_seconds'].min()
        
        axes[1].bar(['Traditional\n(16 Threads)', 'Ours\n(Pareto)'], [base_cost, opt_cost], color=['gray', '#2ecc71'], width=0.5)
        save = (base_cost - opt_cost)/base_cost * 100 if base_cost>0 else 0
        axes[1].text(1, opt_cost, f"-{save:.0f}%\n({opt_cost:.2f} s)", ha='center', va='bottom', fontsize=14, fontweight='bold', color='#27ae60')
        axes[1].set_title('Cloud Fast: Cost Reduction', fontsize=14)
        axes[1].set_ylabel('CPU Cost (s)')

    plt.suptitle('Figure 5.5: Comprehensive Performance Improvements', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_5_5_Gain.png"), dpi=300, bbox_inches='tight')
    print("✅ 图 5.5 完成")

# ================= 🚀 执行 =================

if __name__ == "__main__":
    print(f"🚀 开始生成全套论文图表 (V3)...")
    df = load_real_data()
    if df is not None:
        plot_fig_5_1_sampling(df)
        plot_fig_5_2_risk_barrier(df)
        plot_fig_5_3_morphology(df)
        plot_fig_5_4_knee_point(df)  # 👈 这里！它回来了！
        plot_fig_5_5_gain_real(df)
        print(f"\n🎉 5张图表全部生成完毕: {OUTPUT_DIR}")


