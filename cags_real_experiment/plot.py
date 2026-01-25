#!/usr/bin/env python3
"""
plot_results.py - 将实验生成的 CSV 数据转换为论文级别的图表
需要安装: pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob
import os

def plot_charts(csv_file):
    print(f"📊 Reading data from {csv_file}...")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 设置学术绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 论文常用字体
    plt.rcParams['font.size'] = 12

    # 定义颜色：Native用灰色(代表旧技术)，CTS用亮色(代表新技术)
    palette = {"Native": "#7f8c8d", "CTS": "#e74c3c"}

    # ==========================================
    # 图表 1: 端到端耗时对比 (Duration) - 越低越好
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x="Scenario", y="Time", hue="Type", data=df, palette=palette, edgecolor="black")
    
    # 在柱子上标注数值
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1fs', padding=3, fontsize=10)

    plt.title("End-to-End Download Duration (Lower is Better)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.xlabel("Network Scenario", fontsize=12)
    plt.legend(title="System Type")
    
    # 保存
    output_time = "fig_e2e_duration.png"
    plt.savefig(output_time, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_time}")
    plt.close()

    # ==========================================
    # 图表 2: 吞吐量对比 (Throughput) - 越高越好
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x="Scenario", y="Speed", hue="Type", data=df, palette=palette, edgecolor="black")
    
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f MB/s', padding=3, fontsize=10)

    plt.title("System Throughput Comparison (Higher is Better)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Throughput (MB/s)", fontsize=12)
    plt.xlabel("Network Scenario", fontsize=12)
    plt.legend(title="System Type")

    output_speed = "fig_e2e_throughput.png"
    plt.savefig(output_speed, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_speed}")
    plt.close()

    # ==========================================
    # 图表 3: 加速比 (Speedup Ratio) - 核心亮点
    # ==========================================
    # 只筛选 CTS 的行，因为 Native 的 Ratio 是空的或者 1.0
    df_cts = df[df['Type'] == 'CTS'].copy()
    
    plt.figure(figsize=(8, 5))
    # 使用渐变色表示加速程度
    ax3 = sns.barplot(x="Scenario", y="Ratio", data=df_cts, palette="viridis", edgecolor="black")

    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1fx', padding=3, fontsize=11, fontweight='bold')

    plt.title("CTS Speedup Ratio vs. Native Docker", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Speedup (x times)", fontsize=12)
    plt.xlabel("Network Scenario", fontsize=12)
    plt.axhline(y=1, color='r', linestyle='--', label="Baseline (1x)") # 画一条基准线
    plt.legend()

    output_ratio = "fig_e2e_speedup.png"
    plt.savefig(output_ratio, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_ratio}")
    plt.close()

if __name__ == "__main__":
    # 自动寻找最近生成的 csv 文件
    list_of_files = glob.glob('experiment_result_*.csv') 
    if not list_of_files:
        print("❌ No CSV files found! Run 'e2e_experiment_runner.py' first.")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        plot_charts(latest_file)