import pandas as pd
import numpy as np
import json
import os

# ================= 🔧 配置 =================
INPUT_CSV = "thesis_final_dataset.csv"
OUTPUT_JSON = "pareto_rules.json"
RELIABILITY_THRESHOLD = 0.05  # 失败率超过 5% 的配置直接丢弃

def get_knee_point(df):
    """在帕累托前沿上寻找膝点 (Knee Point)"""
    # 归一化处理以便计算欧氏距离
    # 目标：Cost 越小越好 (0)，TP 越大越好 (1) -> 理想点为 [0, 1]
    df = df.copy()
    
    # 极简归一化
    c_min, c_max = df['Cost_Mean'].min(), df['Cost_Mean'].max()
    t_min, t_max = df['TP_Mean'].min(), df['TP_Mean'].max()
    
    # 防止除以0
    c_norm = (df['Cost_Mean'] - c_min) / (c_max - c_min + 1e-6)
    t_norm = (df['TP_Mean'] - t_min) / (t_max - t_min + 1e-6)
    
    # 计算到理想点 [0, 1] 的距离
    df['dist_to_ideal'] = np.sqrt(c_norm**2 + (1 - t_norm)**2)
    return df.loc[df['dist_to_ideal'].idxmin()]

def main():
    if not os.path.exists(INPUT_CSV):
        print("❌ 找不到 CSV 文件，请先运行实验套件脚本！")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 1. 聚合数据
    summary = df.groupby(['Network', 'FileID', 'Threads', 'Chunk_MB']).agg({
        'Throughput_MBps': 'mean',
        'Cost_Index': 'mean',
        'Error': lambda x: (x != 'Success').mean()
    }).reset_index()
    summary.columns = ['Network', 'FileID', 'Threads', 'Chunk_MB', 'TP_Mean', 'Cost_Mean', 'Fail_Rate']

    # 2. 筛选并提取策略
    policy_table = {}

    for net in summary['Network'].unique():
        policy_table[net] = {}
        for fid in summary['FileID'].unique():
            subset = summary[(summary['Network'] == net) & (summary['FileID'] == fid)]
            
            # [约束一] 可靠性筛选
            reliable_subset = subset[subset['Fail_Rate'] <= RELIABILITY_THRESHOLD]
            if reliable_subset.empty:
                # 如果都不可靠，被迫选失败率最低的那个
                best_config = subset.loc[subset['Fail_Rate'].idxmin()]
            else:
                # [约束二] 膝点检测 (自动隐含了帕累托逻辑)
                best_config = get_knee_point(reliable_subset)
            
            # 保存该场景下的最优决策
            policy_table[net][fid] = {
                "threads": int(best_config['Threads']),
                "chunk_mb": float(best_config['Chunk_MB']),
                "expected_tp": float(best_config['TP_Mean'])
            }
            print(f"✅ {net} | {fid} -> Opt: {int(best_config['Threads'])}T, {best_config['Chunk_MB']}MB")

    # 3. 固化为系统规则
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(policy_table, f, indent=4)
    print(f"\n💾 策略表已生成: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()