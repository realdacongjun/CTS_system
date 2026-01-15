import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 设置绘图风格 (论文专用风格)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif' # 防止中文乱码可能需要额外设置
plt.rcParams['axes.unicode_minus'] = False

def analyze_and_plot(file_path):
    print(f"正在读取数据: {file_path} ...")
    df = pd.read_excel(file_path)
    
    # 过滤掉失败的数据
    df = df[df['status'] == 'SUCCESS']
    
    # 将 total_time 转为数字
    df['total_time'] = pd.to_numeric(df['total_time'])
    
    print(f"有效数据条数: {len(df)}")

    # =================================================
    # 图 1: 带宽 vs 总耗时 (不同算法的对比)
    # =================================================
    plt.figure(figsize=(10, 6))
    
    # 选取几种典型算法来画图，避免太乱
    target_methods = ['gzip-6', 'zstd-3', 'lz4-fast']
    subset = df[df['method'].isin(target_methods)]
    
    # 绘制散点图 + 拟合线
    sns.lmplot(x='network_bw', y='total_time', hue='method', data=subset, 
               aspect=1.5, height=6, scatter_kws={'alpha':0.3})
    
    plt.title('Impact of Bandwidth on Transfer Time (Lower is Better)', fontsize=14)
    plt.xlabel('Network Bandwidth (Mbps)', fontsize=12)
    plt.ylabel('Total Latency (Seconds)', fontsize=12)
    plt.xlim(0, 200) # 重点看低带宽区域
    plt.ylim(0, 300)
    plt.tight_layout()
    plt.savefig('result_bandwidth_impact.png', dpi=300)
    print("✅ 已生成: result_bandwidth_impact.png")

    # =================================================
    # 图 2: 决策边界可视化 (最重要的图!)
    # 我们看看在什么情况下，哪个算法最快
    # =================================================
    print("正在计算最优策略分布...")
    
    # 对每组实验 (rep_id, image, network_bw, cpu_limit) 找到最快的 method
    # 这里我们简化一下，假设 id 临近的是同一组
    # 更好的做法是 group by 实验条件，这里演示一种简单可视化
    
    plt.figure(figsize=(10, 6))
    
    # 散点图：X轴=带宽，Y轴=CPU限制，颜色=最快算法
    # 这是一个近似处理，为了直观展示
    
    # 筛选出最快的数据
    best_methods = df.loc[df.groupby(['network_bw', 'cpu_limit', 'image'])['total_time'].idxmin()]
    
    sns.scatterplot(
        x='network_bw', 
        y='cpu_limit', 
        hue='method', 
        data=best_methods,
        palette='deep',
        s=50,
        alpha=0.8
    )
    
    plt.title('Optimal Compression Strategy Distribution', fontsize=14)
    plt.xlabel('Network Bandwidth (Mbps)', fontsize=12)
    plt.ylabel('Client CPU Limit (Cores)', fontsize=12)
    plt.xscale('log') # 因为带宽跨度大，用对数坐标更清晰
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('result_decision_boundary.png', dpi=300)
    print("✅ 已生成: result_decision_boundary.png")

if __name__ == "__main__":
    # 替换成你下载下来的 excel 文件名
    analyze_and_plot('cts_data.xlsx')