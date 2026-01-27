#!/usr/bin/env python3
"""
plot_matrix.py - 绘制全矩阵实验热力图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_heatmap(csv_file):
    df = pd.read_csv(csv_file)
    
    # 转换数据格式：行是镜像，列是场景，值是加速比
    pivot_table = df.pivot(index="Image", columns="Scenario", values="Speedup")
    
    # 调整列的顺序 (IoT -> Edge -> Cloud)
    pivot_table = pivot_table[['A-IoT', 'B-Edge', 'C-Cloud']]
    
    # 设置绘图
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.2)
    
    # 画热力图
    # annot=True: 在格子里显示数值
    # fmt=".1f": 保留1位小数
    # cmap="YlGnBu": 颜色从黄到蓝（或者用 'Reds' 表示热度）
    ax = sns.heatmap(pivot_table, annot=True, fmt=".1fx", cmap="YlOrRd", 
                     linewidths=.5, cbar_kws={'label': 'Speedup Ratio (Native / CTS)'})
    
    plt.title("CTS System Speedup Matrix\n(Across 4 Image Types & 3 Network Scenarios)", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("Image Content Type", fontsize=12)
    plt.xlabel("Network Scenario", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("fig_matrix_heatmap.png", dpi=300)
    print("✅ Heatmap generated: fig_matrix_heatmap.png")

if __name__ == "__main__":
    files = glob.glob('matrix_results_*.csv')
    if files:
        plot_heatmap(max(files, key=os.path.getctime))
    else:
        print("No CSV found!")