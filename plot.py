import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 设置论文级绘图风格 ---
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    print("正在加载数据...")
    df = pd.read_excel(file_path)
    # 只看成功的数据
    df = df[df['status'] == 'SUCCESS']
    # 过滤掉极其离谱的异常值 (比如 > 500秒的) 以免图形被拉伸太长
    df = df[df['total_time'] < 600]
    return df

def plot_1_overall_distribution(df):
    """图1: 整体耗时分布 (箱线图) - 看谁最稳"""
    plt.figure(figsize=(12, 6))
    
    # 按平均耗时排序
    order = df.groupby('method')['total_time'].median().sort_values().index
    
    sns.boxplot(x='method', y='total_time', data=df, order=order, palette="viridis", showfliers=False)
    
    plt.title('Figure 1: Overall Transfer Time Distribution by Algorithm', fontsize=16, fontweight='bold')
    plt.ylabel('Total Latency (Seconds)')
    plt.xlabel('Compression Method')
    plt.tight_layout()
    plt.savefig('chart_1_overall_ranking.png', dpi=300)
    print("✅ 图1已生成: 整体排名 (chart_1_overall_ranking.png)")

def plot_2_bandwidth_impact(df):
    """图2: 不同带宽下的性能趋势 (折线图) - 看交叉点"""
    plt.figure(figsize=(12, 6))
    
    # 为了图表清晰，我们将带宽分桶 (Binning)
    # 0-10, 10-50, 50-100, 100-500, >500
    bins = [0, 10, 50, 100, 500, 10000]
    labels = ['<10 Mbps', '10-50 Mbps', '50-100 Mbps', '100-500 Mbps', '>500 Mbps']
    df['bw_group'] = pd.cut(df['network_bw'], bins=bins, labels=labels)
    
    # 选取主要对手
    target_methods = ['gzip-6', 'zstd-3', 'zstd-19', 'lz4-fast']
    subset = df[df['method'].isin(target_methods)]
    
    sns.pointplot(x='bw_group', y='total_time', hue='method', data=subset, 
                  errorbar=None, markers=['o', 's', '^', 'D'], scale=1.2)
    
    plt.title('Figure 2: Impact of Network Bandwidth on Performance', fontsize=16, fontweight='bold')
    plt.ylabel('Average Transfer Time (Seconds)')
    plt.xlabel('Network Bandwidth Range')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('chart_2_bandwidth_impact.png', dpi=300)
    print("✅ 图2已生成: 带宽影响 (chart_2_bandwidth_impact.png)")

def plot_3_cpu_tradeoff(df):
    """图3: CPU 受限时的表现 (柱状图) - 看谁最吃资源"""
    plt.figure(figsize=(12, 6))
    
    # 筛选两种极端场景
    # 场景 A: 弱机 (CPU Limit < 1.0)
    weak_cpu = df[df['cpu_limit'] <= 1.0].copy()
    weak_cpu['Scenario'] = 'Weak CPU (< 1 Core)'
    
    # 场景 B: 强机 (CPU Limit >= 4.0)
    strong_cpu = df[df['cpu_limit'] >= 4.0].copy()
    strong_cpu['Scenario'] = 'Strong CPU (> 4 Cores)'
    
    combined = pd.concat([weak_cpu, strong_cpu])
    
    # 只看 Zstd-high 和 LZ4 的对比
    target_methods = ['zstd-19', 'zstd-3', 'lz4-fast', 'gzip-6']
    subset = combined[combined['method'].isin(target_methods)]
    
    sns.barplot(x='method', y='total_time', hue='Scenario', data=subset, 
                palette="RdBu", errorbar=None)
    
    plt.title('Figure 3: Performance under Different CPU Constraints', fontsize=16, fontweight='bold')
    plt.ylabel('Average Transfer Time (Seconds)')
    plt.xlabel('Compression Method')
    plt.tight_layout()
    plt.savefig('chart_3_cpu_impact.png', dpi=300)
    print("✅ 图3已生成: CPU影响 (chart_3_cpu_impact.png)")

if __name__ == "__main__":
    df = load_data('cts_data.xlsx')
    plot_1_overall_distribution(df)
    plot_2_bandwidth_impact(df)
    plot_3_cpu_tradeoff(df)