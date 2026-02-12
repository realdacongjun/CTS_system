import pandas as pd
import numpy as np
import json
import os

# ================= 🔧 路径自适应配置 (核心修改) =================
# 1. 获取当前脚本所在文件夹的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接文件路径 (不管你在哪里运行，都能找到)
INPUT_CSV = os.path.join(SCRIPT_DIR, "pareto_results_20260131_173001.csv") 
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "pareto_rules.json")


RELIABILITY_THRESHOLD = 0.1  # 允许最大 10% 的失败率 (弱网下可能稍高)

def filter_pareto_dominated(df):
    """
    (可选) 过滤掉被支配的点 (Dominated Points)
    逻辑：如果存在点 B，使得 B.Cost <= A.Cost 且 B.TP >= A.TP (且至少有一个不等)，则 A 被支配。
    """
    df = df.copy()
    is_dominated = []
    for index, row in df.iterrows():
        # 找到比当前行 Cost 更低且 TP 更高的行
        better_points = df[
            (df['Cost_Mean'] <= row['Cost_Mean']) & 
            (df['TP_Mean'] >= row['TP_Mean']) & 
            ((df['Cost_Mean'] < row['Cost_Mean']) | (df['TP_Mean'] > row['TP_Mean']))
        ]
        is_dominated.append(not better_points.empty)
    
    df['is_dominated'] = is_dominated
    # 只保留非支配点 (Pareto Frontier)
    return df[~df['is_dominated']]

def get_knee_point(df):
    """
    在帕累托前沿上寻找膝点 (Knee Point) - 增强版
    """
    df = df.copy()
    
    # --- 1. 获取绝对指标 ---
    t_max = df['TP_Mean'].max()
    t_min = df['TP_Mean'].min()
    c_max = df['Cost_Mean'].max()
    c_min = df['Cost_Mean'].min()
    
    # --- 2. 动态权重调整 (关键逻辑) ---
    # 默认权重：Cost 和 TP 同等重要
    w_cost = 1.0 
    w_tp = 1.0
    
    # [核心创新点]：如果最大吞吐量都很低（说明是弱网），则 Cost 权重极大
    # 这就是“帕累托坍缩”的数学体现：投入再多也没用，所以必须省 CPU
    if t_max < 20.0:  # 阈值可以设为 10-30 Mbps
        print(f"   [检测到弱网环境 (Max TP={t_max:.2f} < 20)] -> 启动节能优先模式 (Cost权重 x 5)")
        w_cost = 5.0  # 惩罚 CPU 开销
        w_tp = 0.5    # 降低吞吐量的诱惑
    
    # --- 3. 极差归一化 ---
    c_div = c_max - c_min if c_max != c_min else 1.0
    t_div = t_max - t_min if t_max != t_min else 1.0
    
    df['c_norm'] = (df['Cost_Mean'] - c_min) / c_div
    df['t_norm'] = (df['TP_Mean'] - t_min) / t_div
    
    # --- 4. 计算加权欧氏距离 ---
    # 理想点: c_norm=0 (Cost最小), t_norm=1 (TP最大)
    # Distance = sqrt( (w_c * cost)^2 + (w_t * (1-tp))^2 )
    df['dist_to_ideal'] = np.sqrt(
        (w_cost * df['c_norm'])**2 + 
        (w_tp * (1 - df['t_norm']))**2
    )
    
    # --- 5. 返回距离最小的点 ---
    best_idx = df['dist_to_ideal'].idxmin()
    return df.loc[best_idx]

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到文件 {INPUT_CSV}")
        return

    print(f"📖 读取数据: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 0. 预处理：计算失败率
    # 假设 exit_code != 0 为失败
    df['is_success'] = (df['exit_code'] == 0).astype(int)

    # 1. 聚合数据 (Group By 配置)
    # 注意：这里加入了 cpu_quota，因为它是你重要的调节参数
    # Group Key: 场景 + 决策变量(Threads, Quota, Chunk)
    group_cols = ['scenario', 'cpu_quota', 'threads', 'chunk_kb']
    
    summary = df.groupby(group_cols).agg({
        'throughput_mbps': 'mean',
        'cost_cpu_seconds': 'mean',
        'is_success': 'mean' # 成功率
    }).reset_index()
    
    # 重命名方便处理
    summary.rename(columns={
        'throughput_mbps': 'TP_Mean',
        'cost_cpu_seconds': 'Cost_Mean'
    }, inplace=True)
    
    summary['Fail_Rate'] = 1 - summary['is_success']

    # 2. 生成策略表
    policy_table = {}
    
    # 遍历每一个网络场景 (IoT, Edge, Cloud)
    for scenario in summary['scenario'].unique():
        if "BASELINE" in scenario: continue # 跳过基准测试
        
        print(f"\n🔍 分析场景: {scenario}")
        subset = summary[summary['scenario'] == scenario].copy()
        
        # [步骤 A] 可靠性筛选
        reliable_subset = subset[subset['Fail_Rate'] <= RELIABILITY_THRESHOLD]
        
        if reliable_subset.empty:
            print(f"   ⚠️ 警告: 该场景无可靠配置，回退到最低失败率配置")
            best_config = subset.loc[subset['Fail_Rate'].idxmin()]
        else:
            # [步骤 B] 帕累托非支配排序 (只看前沿上的点)
            pareto_frontier = filter_pareto_dominated(reliable_subset)
            print(f"   - 原始点数: {len(reliable_subset)} -> 前沿点数: {len(pareto_frontier)}")
            
            # [步骤 C] 膝点选择 (在性价比最高的地方切一刀)
            best_config = get_knee_point(pareto_frontier)

        # 3. 记录最优策略
        policy_table[scenario] = {
            "best_threads": int(best_config['threads']),
            "best_cpu_quota": float(best_config['cpu_quota']),
            "best_chunk_kb": int(best_config['chunk_kb']),
            "expected_throughput": round(float(best_config['TP_Mean']), 2),
            "expected_cost": round(float(best_config['Cost_Mean']), 4)
        }
        
        print(f"   ✅ 最终策略: CPU={best_config['cpu_quota']} | Threads={best_config['threads']} | Chunk={best_config['chunk_kb']}KB")
        print(f"      (预期性能: {best_config['TP_Mean']:.1f} Mbps, 代价: {best_config['Cost_Mean']:.3f} s)")

    # 4. 保存 JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(policy_table, f, indent=4)
    print(f"\n💾 策略文件已生成: {OUTPUT_JSON}")
    print("👉 你可以将此文件加载到 pareto_policy.py 中直接使用！")

if __name__ == "__main__":
    main()