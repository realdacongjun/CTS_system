import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# ================= 配置区 =================
INPUT_FILE = "pareto_results_20260131_173001.csv"
OUTPUT_IMG = "pareto_curve.png"
OUTPUT_CSV = "pareto_cleaned_final.csv"
# =========================================

def analyze_and_plot():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"📖 正在读取: {INPUT_FILE}...")
    
    # 1. 智能读取与清洗
    # 有时候文件中间会因为追加写入包含多余的表头，需要过滤
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
    
    # 只保留第一行表头和所有数据行（排除中间重复出现的 header）
    header = lines[0]
    data_lines = [line for line in lines[1:] if not line.startswith("run_id")]
    
    from io import StringIO
    df = pd.read_csv(StringIO(header + "".join(data_lines)))
    
    print(f"   原始行数: {len(df)}")
    # 根据 run_id 去重
    df = df.drop_duplicates(subset=['run_id'], keep='last')
    print(f"   去重后行数: {len(df)}")
    
    # 保存清洗后的数据
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 清洗数据已保存至: {OUTPUT_CSV}")

    # 2. 帕累托前沿计算 (Pareto Frontier Calculation)
    # 目标：Cost 越低越好 (Min)，Throughput 越高越好 (Max)
    # 简单的筛选逻辑：如果在同样的 Cost 下，有更高的 Throughput，或者同样的 Throughput 下有更低的 Cost，则当前点被支配
    
    # 为了画图方便，我们按场景分组画
    scenarios = df['scenario'].unique()
    colors = {'IoT_Weak': 'red', 'Edge_Normal': 'green', 'Cloud_Fast': 'blue'}
    markers = {'IoT_Weak': 'o', 'Edge_Normal': '^', 'Cloud_Fast': 's'}

    plt.figure(figsize=(12, 8))
    
    for sc in scenarios:
        if "BASELINE" in sc: continue # 跳过 Baseline，以免干扰视线
        
        subset = df[df['scenario'] == sc].copy()
        subset = subset.sort_values('cost_cpu_seconds')
        
        # 绘制所有散点
        plt.scatter(subset['cost_cpu_seconds'], subset['throughput_mbps'], 
                    c=colors.get(sc, 'gray'), label=f"{sc} (All)", alpha=0.3, s=50)
        
        # 计算该场景下的帕累托前沿
        frontier_x = []
        frontier_y = []
        current_max_thr = -1.0
        
        for idx, row in subset.iterrows():
            # 如果当前点的吞吐量比之前所有低成本的点都高，那它就是一个“前沿点”
            if row['throughput_mbps'] > current_max_thr:
                frontier_x.append(row['cost_cpu_seconds'])
                frontier_y.append(row['throughput_mbps'])
                current_max_thr = row['throughput_mbps']
        
        # 绘制前沿连线
        plt.plot(frontier_x, frontier_y, c=colors.get(sc, 'black'), linestyle='-', linewidth=2, label=f"{sc} Frontier")
        
        # 标注最优配置（拐点/最大值）
        if frontier_x:
            plt.annotate(f"Max: {frontier_y[-1]:.1f} Mbps", 
                         (frontier_x[-1], frontier_y[-1]),
                         xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    # 3. 设置图表属性 (使用对数坐标，因为 Cloud 和 IoT 差异巨大)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title('Innovation II: Resource-Performance Pareto Frontier', fontsize=16)
    plt.xlabel('Computational Cost (CPU Seconds) [Log Scale] -> Lower is Better', fontsize=12)
    plt.ylabel('Network Throughput (Mbps) [Log Scale] -> Higher is Better', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"🖼️  图片已生成: {OUTPUT_IMG}")
    
    # 4. 打印核心结论数据
    print("\n" + "="*50)
    print("🚀 核心实验结论 (Key Findings)")
    print("="*50)
    
    for sc in scenarios:
        if "BASELINE" in sc: continue
        sub = df[df['scenario']==sc]
        max_thr = sub['throughput_mbps'].max()
        min_cost = sub['cost_cpu_seconds'].min()
        
        # 找到效率最高的点 (MB per CPU Second)
        best_eff_idx = sub['efficiency_mb_per_cpus'].idxmax()
        best_eff_row = sub.loc[best_eff_idx]
        
        print(f"Scenario: {sc}")
        print(f"  - Max Throughput: {max_thr:.2f} Mbps")
        print(f"  - Min CPU Cost:   {min_cost:.4f} s")
        print(f"  - Best Config:    Threads={best_eff_row['threads']}, Quota={best_eff_row['cpu_quota']}")
        print(f"  - Efficiency:     {best_eff_row['efficiency_mb_per_cpus']:.2f} MB/s/cpu")
        print("-" * 30)

if __name__ == "__main__":
    try:
        analyze_and_plot()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()