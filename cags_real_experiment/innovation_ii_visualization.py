# #!/usr/bin/env python3
# """
# 创新点II：多目标自适应传输优化 - 修正版可视化脚本
# Corrected Visualization Script for Innovation Point II: Multi-objective Adaptive Transmission Optimization

# 核心修正：
# ✅ 移除虚构丢包率指标（仅使用实测吞吐量/成本/时间）
# ✅ 修正术语："Parameter Space Coverage"替代"Stratified Sampling"
# ✅ 诚实展示离散帕累托点（不强行平滑连线）
# ✅ 所有图表基于真实测量指标，标注数据限制
# ✅ 学术级图表美化（Times New Roman, 高DPI, IEEE配色）
# """

# import pandas as pd
# import numpy as np
# import matplotlib
# import platform

# # ============== 【关键】中文字体设置（必须在 pyplot 之前，参考 chapter3_5.py） ==============
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei']
# elif system_name == 'Darwin':  # macOS
#     font_list = ['Heiti TC', 'PingFang HK']
# else:  # Linux
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # ======================================================================

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # ================= 🔧 全局配置 =================
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 自动查找最新实验数据文件
# data_files = glob.glob(os.path.join(SCRIPT_DIR, "pareto_results*.csv")) + \
#              glob.glob(os.path.join(SCRIPT_DIR, "*cleaned*.csv"))

# if data_files:
#     DATA_FILE = max(data_files, key=os.path.getctime)
#     print(f"📊 检测到数据文件: {os.path.basename(DATA_FILE)} ({len(data_files)} 个候选)")
# else:
#     print("❌ 未找到实验数据文件 (pareto_results_*.csv)")
#     exit(1)

# OUTPUT_DIR = os.path.join(SCRIPT_DIR, "innovation_ii_figures_corrected")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 设置学术论文绘图风格 (IEEE/ACM 标准)
# plt.rcParams.update({
#     'font.family': 'Times New Roman',
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 11,
#     'figure.titlesize': 18,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
#     'lines.linewidth': 2.0,
#     'lines.markersize': 8,
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linestyle': '--',
#     'grid.linewidth': 0.8
# })

# # IEEE 配色方案
# COLORS = {
#     'iot': '#e74c3c',      # 红色 - IoT弱网
#     'edge': '#f39c12',     # 橙色 - Edge边缘
#     'cloud': '#27ae60',    # 绿色 - Cloud云
#     'anchor': '#3498db',   # 蓝色 - Anchor系统扫描
#     'probe_small': '#9b59b6',  # 紫色 - Probe小文件
#     'probe_large': '#1abc9c',  # 青色 - Probe大文件
#     'baseline': '#95a5a6', # 灰色 - Baseline
#     'pareto': '#8e44ad'    # 深紫 - 帕累托点
# }

# # ================= 🛠️ 数据加载与验证 =================

# def load_and_validate_data():
#     """加载数据并验证指标真实性（仅保留实测指标）"""
#     try:
#         df = pd.read_csv(DATA_FILE)
#         print(f"✅ 成功加载数据: {len(df)} 条记录")
        
#         # 必需列验证
#         required_cols = ['run_id', 'exp_type', 'file_size_mb', 'scenario', 
#                         'cpu_quota', 'threads', 'chunk_kb', 'duration_s',
#                         'throughput_mbps', 'cost_cpu_seconds', 'efficiency_mb_per_cpus', 'exit_code']
        
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             print(f"❌ 缺少必要列: {missing_cols}")
#             return None
        
#         # 数据清洗
#         df = df.dropna(subset=['throughput_mbps', 'cost_cpu_seconds', 'duration_s'])
#         df = df[df['exit_code'] == 0]  # 仅保留成功实验
#         df = df[df['duration_s'] > 0]  # 移除零时长异常值
        
#         # 场景标准化（移除_BASELINE后缀用于分组）
#         df['network_type'] = df['scenario'].str.replace('_BASELINE', '', regex=False)
        
#         # 实验类型分类
#         df['is_baseline'] = df['exp_type'].str.contains('baseline', case=False)
#         df['is_anchor'] = df['exp_type'].str.contains('anchor', case=False) & ~df['is_baseline']
#         df['is_probe_small'] = df['exp_type'] == 'probe_small'
#         df['is_probe_large'] = df['exp_type'] == 'probe_large'
        
#         # ⚠️ 关键验证：确认无丢包率实测数据
#         has_loss_rate = any(col.lower().find('loss') >= 0 for col in df.columns)
#         if has_loss_rate:
#             print("⚠️ 警告: 检测到'loss'相关列，但TCP重传使应用层丢包率≈0%，不建议用于风险分析")
        
#         # 数据质量报告
#         print("\n📊 数据质量报告:")
#         print(f"   总记录数: {len(df):,}")
#         print(f"   Baseline记录: {df['is_baseline'].sum():,}")
#         print(f"   Anchor实验: {df['is_anchor'].sum():,}")
#         print(f"   Probe Small: {df['is_probe_small'].sum():,}")
#         print(f"   Probe Large: {df['is_probe_large'].sum():,}")
#         print(f"   吞吐量范围: {df['throughput_mbps'].min():.2f} - {df['throughput_mbps'].max():.2f} Mbps")
#         print(f"   CPU成本范围: {df['cost_cpu_seconds'].min():.4f} - {df['cost_cpu_seconds'].max():.4f} s")
#         print(f"   传输时间范围: {df['duration_s'].min():.2f} - {df['duration_s'].max():.2f} s")
        
#         # 场景分布
#         print("\n🌐 场景分布:")
#         for scenario, count in df['network_type'].value_counts().items():
#             sizes = sorted(df[df['network_type'] == scenario]['file_size_mb'].unique())
#             print(f"   {scenario:20s}: {count:4d} 条记录 | 文件大小: {sizes}")
        
#         return df
        
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # ================= 📐 帕累托前沿计算（高效+容差） =================

# def compute_pareto_frontier(df, maximize_col='throughput_mbps', minimize_col='cost_cpu_seconds'):
#     """
#     标准帕累托前沿计算 (完整支配关系检查)
#     返回非支配解集合（前沿点）
#     """
#     if len(df) == 0:
#         return pd.DataFrame()
    
#     # 完整的帕累托支配检查
#     pareto_points = []
#     for idx, candidate in df.iterrows():
#         is_dominated = False
#         for _, other in df.iterrows():
#             # 检查other是否支配candidate
#             if (other[maximize_col] >= candidate[maximize_col] and 
#                 other[minimize_col] <= candidate[minimize_col] and
#                 (other[maximize_col] > candidate[maximize_col] or 
#                  other[minimize_col] < candidate[minimize_col])):
#                 is_dominated = True
#                 break
#         if not is_dominated:
#             pareto_points.append(candidate)
    
#     pareto_df = pd.DataFrame(pareto_points)
#     if not pareto_df.empty:
#         # 按成本排序便于绘图
#         pareto_df = pareto_df.sort_values(minimize_col).reset_index(drop=True)
    
#     return pareto_df

# # ================= 🎨 图表生成函数 =================

# def plot_fig_5_1_stratified_sampling(df):
#     """图5.1: Anchor-Probe分层采样空间示意图"""
#     print("🎨 生成图5.1: 分层采样空间示意图...")
    
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 数据分类
#     anchor_data = df[df['exp_type'].isin(['anchor', 'anchor_baseline'])]
#     probe_data = df[df['exp_type'].str.contains('probe')]
    
#     # 绘制Anchor点（核心区域密集采样）- 减小符号大小
#     if not anchor_data.empty:
#         ax.scatter(anchor_data['threads'], anchor_data['cpu_quota'], 
#                   anchor_data['chunk_kb']/1024,  # 转换为MB
#                   c='#e74c3c', marker='o', s=30, alpha=0.7,  # 减小符号大小
#                   label='锚点 (核心区域)', edgecolors='black', linewidth=0.3)
    
#     # 绘制Probe点（边缘区域稀疏采样）- 减小符号大小
#     if not probe_data.empty:
#         ax.scatter(probe_data['threads'], probe_data['cpu_quota'], 
#                   probe_data['chunk_kb']/1024,
#                   c='#3498db', marker='^', s=40, alpha=0.8,  # 减小符号大小
#                   label='探测点 (边缘区域)', edgecolors='black', linewidth=0.3)
    
#     # 美化设置（中文化）
#     ax.set_xlabel('并发线程数', fontsize=14, labelpad=10)
#     ax.set_ylabel('CPU配额 (核)', fontsize=14, labelpad=10)
#     ax.set_zlabel('分片大小 (MB)', fontsize=14, labelpad=10)
    
#     ax.set_title('图5.1: 分层采样设计矩阵\n多尺度参数空间覆盖', 
#                 fontsize=16, pad=20)
    
#     # 添加图例和网格
#     ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
#     ax.grid(True, alpha=0.3)
    
#     # 设置视角
#     ax.view_init(elev=20, azim=45)
    
#     # 保存图片
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_1_Stratified_Sampling.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     print(f"✅ 图5.1完成: {output_path}")

# def plot_fig_5_2_stability_tradeoff(df):
#     """图5.2: 稳定性-性能权衡（基于实测传输时间变异系数）"""
#     print("\n🎨 生成图5.2: 稳定性-性能权衡...")
    
#     # 添加调试信息
#     print(f"   原始数据总数: {len(df)}")
#     print(f"   成功实验数 (exit_code=0): {len(df[df['exit_code'] == 0])}")
    
#     # 按场景+文件大小分组计算稳定性指标
#     stability_data = []
    
#     print("   数据分组详情:")
#     for (scenario, size), group in df.groupby(['network_type', 'file_size_mb']):
#         total_count = len(group)
#         non_baseline_count = len(group[~group['is_baseline']])
#         print(f"     {scenario}-{size}MB: 总计{total_count}, 非baseline{non_baseline_count}")
        
#         # 仅使用非baseline实验数据
#         group = group[~group['is_baseline']]
        
#         # 降低样本量要求：从5个改为3个，让更多有效的配置组合能够显示
#         if len(group) >= 3:  # 至少3次重复就有统计意义
#             stability_data.append({
#                 'scenario': scenario,
#                 'file_size_mb': size,
#                 'throughput_mean': group['throughput_mbps'].mean(),
#                 'throughput_std': group['throughput_mbps'].std(),
#                 'duration_mean': group['duration_s'].mean(),
#                 'duration_cv': (group['duration_s'].std() / 
#                                max(group['duration_s'].mean(), 1e-6)) * 100,  # 变异系数(%)
#                 'sample_size': len(group)
#             })
#             print(f"     ✓ 加入分析: {scenario}-{size}MB ({len(group)}个样本)")
#         else:
#             print(f"     ✗ 样本不足: {scenario}-{size}MB ({len(group)}个样本 < 3)")
    
#     if not stability_data:
#         print("   ⚠️  警告: 无足够数据计算稳定性指标（每组需≥3个样本）")
#         return {}
    
#     stability_df = pd.DataFrame(stability_data)
#     print(f"   最终用于绘图的数据点数: {len(stability_df)}")
    
#     # 创建图表 - 修改为按文件大小显示不同颜色的点
#     plt.figure(figsize=(12, 7.5))
    
#     # 场景和文件大小的颜色映射
#     color_map = {
#         ('IoT_Weak', 10): {'color': COLORS['iot'], 'marker': 'o', 'label': 'IoT弱网 (10MB)'},
#         ('Edge_Normal', 50): {'color': COLORS['edge'], 'marker': 's', 'label': '边缘网络 (50MB)'},
#         ('Edge_Normal', 300): {'color': '#d35400', 'marker': 'D', 'label': '边缘网络 (300MB)'},  # 深橙色
#         ('Cloud_Fast', 10): {'color': COLORS['cloud'], 'marker': '^', 'label': '云环境 (10MB)'},
#         ('Cloud_Fast', 100): {'color': '#27ae60', 'marker': 'v', 'label': '云环境 (100MB)'},   # 深绿色
#         ('Cloud_Fast', 300): {'color': '#16a085', 'marker': 'p', 'label': '云环境 (300MB)'}    # 青色
#     }
    
#     # 绘制每个配置组合（减小符号大小）
#     for idx, row in stability_df.iterrows():
#         key = (row['scenario'], row['file_size_mb'])
#         if key in color_map:
#             config = color_map[key]
#             plt.scatter(row['throughput_mean'], row['duration_cv'],
#                        c=config['color'], s=row['sample_size']*8,  # 减小气泡大小
#                        alpha=0.85, edgecolors='black', linewidth=1.0,  # 减小边框宽度
#                        marker=config['marker'], label=config['label'])
#             print(f"   绘制点: {config['label']} - 吞吐量:{row['throughput_mean']:.1f}Mbps, CV:{row['duration_cv']:.1f}%")
    
#     # 稳定性阈值线（基于实测分布：CV>30%视为不稳定）
#     plt.axhline(y=30, color='#c0392b', linestyle='--', linewidth=2.0, alpha=0.85,
#                 label='稳定性阈值 (CV=30%)')
    
#     # 坐标轴（中文化）
#     plt.xlabel('平均吞吐量 (Mbps)', fontsize=14)
#     plt.ylabel('传输时间变异系数 (CV, %)', fontsize=14)
    
#     # 标题（中文化）
#     plt.title('图5.2: 稳定性-性能权衡分析\n基于实测传输时间变异系数', 
#               fontsize=16, pad=20, fontweight='bold')
    
#     # 图例（调整位置避免重叠，中文化）
#     plt.legend(fontsize=10, loc='upper right', framealpha=0.95, ncol=2)
    
#     # 网格
#     plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
#     # Y轴范围
#     plt.ylim(0, min(plt.ylim()[1] * 1.15, 100))  # 限制在100%以内
    
#     # 添加注释框（中文化）
#     plt.text(0.98, 0.96, 'CV越低 → 稳定性越高\n(传输更可预测)', 
#             transform=plt.gca().transAxes, fontsize=11,
#             verticalalignment='top', horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.85, edgecolor='gray'))
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_2_Stability_Tradeoff.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成摘要
#     summary = {
#         'scenarios': stability_df['scenario'].nunique(),
#         'total_configs': len(stability_df),
#         'max_cv': stability_df['duration_cv'].max(),
#         'min_cv': stability_df['duration_cv'].min(),
#         'file_sizes': sorted(stability_df['file_size_mb'].unique())
#     }
    
#     print(f"   ✅ 完成: {summary['total_configs']} 个配置的稳定性分析")
#     print(f"      涉及文件大小: {summary['file_sizes']} MB")
#     print(f"      CV范围: {summary['min_cv']:.1f}% - {summary['max_cv']:.1f}%")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_3_pareto_discrete(df):
#     """图5.3: 离散帕累托点（诚实展示，不强行平滑）"""
#     print("\n🎨 生成图5.3: 离散帕累托前沿...")
    
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
#     scenarios = [
#         {'name': 'IoT_Weak', 'size': 10, 'title': 'IoT弱网 (10MB)', 'color': COLORS['iot']},
#         {'name': 'Edge_Normal', 'size': 50, 'title': '边缘网络 (50MB)', 'color': COLORS['edge']},
#         {'name': 'Cloud_Fast', 'size': 100, 'title': '云环境 (100MB)', 'color': COLORS['cloud']}
#     ]
    
#     all_summaries = []
    
#     for ax, scenario in zip(axes, scenarios):
#         # 筛选数据：指定场景+文件大小+非baseline
#         subset = df[(df['network_type'] == scenario['name']) & 
#                    (df['file_size_mb'] == scenario['size']) & 
#                    (~df['is_baseline'])]
        
#         # 数据量检查
#         if len(subset) < 8:
#             ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
#                    ha='center', va='center', fontsize=13, color='gray',
#                    fontweight='bold')
#             ax.set_title(scenario['title'], fontsize=14, color='gray', fontweight='bold')
#             ax.set_xlabel('CPU成本 (秒)', fontsize=12)
#             ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
#             all_summaries.append({'scenario': scenario['name'], 'points': len(subset), 'pareto': 0})
#             continue
        
#         # 使用统一的标准帕累托前沿计算
#         frontier = compute_pareto_frontier(subset)
        
#         # 绘制所有点（浅灰色背景）- 减小符号大小
#         ax.scatter(subset['cost_cpu_seconds'], subset['throughput_mbps'],
#                   c='#bdc3c7', s=25, alpha=0.65, edgecolors='none',  # 减小符号大小
#                   label=f'全部配置 ({len(subset)})')
        
#         # 绘制帕累托点（大圆点，不连线！）- 适当调整符号大小
#         if len(frontier) >= 3:
#             ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                       c=scenario['color'], s=80, alpha=0.92,  # 适当符号大小
#                       edgecolors='black', linewidth=1.2, marker='o',
#                       label=f'帕累托最优 ({len(frontier)})', zorder=5)
#         elif len(frontier) > 0:
#             # 少量点用星号强调 - 适当符号大小
#             ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                       c=scenario['color'], s=100, alpha=0.95,  # 适当符号大小
#                       edgecolors='black', linewidth=1.4, marker='*',
#                       label=f'帕累托 ({len(frontier)})', zorder=5)
        
#         # 坐标轴（中文化）
#         ax.set_xlabel('CPU成本 (秒)', fontsize=12)
#         ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
#         ax.set_title(scenario['title'], fontsize=14, fontweight='bold')
        
#         # 图例（右下角）
#         ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92)
        
#         # 标注数据量
#         ax.text(0.04, 0.96, f'n={len(subset)}', transform=ax.transAxes,
#                fontsize=11, verticalalignment='top', fontweight='bold',
#                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
#                         edgecolor=scenario['color'], alpha=0.85, linewidth=1.2))
        
#         # 保存场景摘要
#         all_summaries.append({
#             'scenario': scenario['name'],
#             'total_points': len(subset),
#             'pareto_points': len(frontier)
#         })
    
#     # 总标题（诚实标注离散采样，中文化）
#     fig.suptitle('图5.3: 不同网络环境下的帕累托最优配置\n(离散采样 - 无插值平滑)', 
#                 fontsize=17, y=1.04, fontweight='bold')
#     plt.tight_layout()
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_3_Pareto_Discrete.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 打印摘要
#     print(f"   ✅ 完成: 3个场景的帕累托分析")
#     for summary in all_summaries:
#         print(f"      {summary['scenario']:15s}: 总点数={summary['total_points']:3d} | 帕累托点={summary['pareto_points']:2d}")
#     print(f"   📁 保存至: {output_path}")
    
#     return all_summaries

# def plot_fig_5_4_knee_points(df):
#     """图5.4: 帕累托前沿上的膝点选择（修正版）"""
#     print("\n🎨 生成图5.4: 帕累托膝点选择...")
    
#     # 选择数据最丰富的场景：Cloud_Fast 100MB
#     cloud_data = df[(df['network_type'] == 'Cloud_Fast') & 
#                    (df['file_size_mb'] == 100) & 
#                    (~df['is_baseline'])]
    
#     if len(cloud_data) < 10:
#         print(f"   ⚠️  Cloud_Fast 100MB数据不足 (n={len(cloud_data)} < 10)，跳过图5.4")
#         return {}
    
#     # 计算帕累托前沿（使用标准算法）
#     frontier = compute_pareto_frontier(cloud_data)
    
#     # 降低要求：允许2个点也能生成图表（原来是3个点）
#     if len(frontier) < 2:
#         print(f"   ⚠️  帕累托前沿点数不足 (n={len(frontier)} < 2)，跳过图5.4")
#         return {}
    
#     print(f"   帕累托前沿点数: {len(frontier)}")
    
#     # 创建图表
#     plt.figure(figsize=(13, 8))
    
#     # 绘制所有实验点（浅灰色背景）- 减小符号大小
#     plt.scatter(cloud_data['cost_cpu_seconds'], cloud_data['throughput_mbps'],
#                c='#ecf0f1', s=25, alpha=0.5, edgecolors='none', label='全部配置')
    
#     # 绘制帕累托前沿（深色线条 + 点）- 适当符号大小
#     if len(frontier) > 1:
#         plt.plot(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                 'k-', linewidth=2.0, alpha=0.8, label='帕累托前沿')
#     plt.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                c='black', s=60, alpha=0.9, zorder=5, edgecolors='white', linewidth=1.2,
#                label=f'帕累托点 ({len(frontier)})')
    
#     # 定义膝点权重配置（当只有2个点时，简化配置）
#     if len(frontier) >= 3:
#         weights = [
#             {'name': '节能优先', 'wc': 0.8, 'wt': 0.2, 'color': COLORS['cloud'], 'marker': 'D'},
#             {'name': '平衡配置', 'wc': 0.5, 'wt': 0.5, 'color': COLORS['edge'], 'marker': 'o'},
#             {'name': '性能优先', 'wc': 0.2, 'wt': 0.8, 'color': COLORS['iot'], 'marker': '^'}
#         ]
#     else:
#         # 当只有2个帕累托点时，使用简单的两端配置
#         weights = [
#             {'name': '节能优先', 'wc': 0.9, 'wt': 0.1, 'color': COLORS['cloud'], 'marker': 'D'},
#             {'name': '性能优先', 'wc': 0.1, 'wt': 0.9, 'color': COLORS['iot'], 'marker': '^'}
#         ]
    
#     knee_points = []
    
#     # 关键修正：仅在帕累托前沿上计算膝点
#     # 归一化（仅在前沿点上）
#     c_min, c_max = frontier['cost_cpu_seconds'].min(), frontier['cost_cpu_seconds'].max()
#     t_min, t_max = frontier['throughput_mbps'].min(), frontier['throughput_mbps'].max()
    
#     frontier['c_norm'] = (frontier['cost_cpu_seconds'] - c_min) / max(c_max - c_min, 1e-8)
#     frontier['t_norm'] = (frontier['throughput_mbps'] - t_min) / max(t_max - t_min, 1e-8)
    
#     for weight in weights:
#         # L2距离计算（仅在帕累托前沿上）
#         distances = np.sqrt(
#             weight['wc'] * frontier['c_norm']**2 + 
#             weight['wt'] * (1 - frontier['t_norm'])**2
#         )
#         best_idx = distances.idxmin()
#         best_point = frontier.loc[best_idx]
#         knee_points.append((best_point, weight))
        
#         # 绘制膝点 - 适当符号大小
#         plt.scatter(best_point['cost_cpu_seconds'], best_point['throughput_mbps'],
#                    s=200, c=weight['color'], marker=weight['marker'],  # 适当符号大小
#                    edgecolors='black', linewidth=1.8, zorder=10,
#                    label=f"{weight['name']} (w_c={weight['wc']})")
        
#         # 标注配置信息（中文化）
#         config_text = f"{int(best_point['threads'])}线程\n{best_point['cpu_quota']:.1f}核"
#         plt.annotate(config_text, 
#                     (best_point['cost_cpu_seconds'], best_point['throughput_mbps']),
#                     xytext=(28, 30), textcoords='offset points',
#                     fontsize=11, fontweight='bold', color=weight['color'],
#                     bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
#                              alpha=0.92, edgecolor=weight['color'], linewidth=1.5),
#                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', 
#                                   color=weight['color'], lw=1.6, alpha=0.8))
    
#     # 连接膝点展示偏好轨迹（当有多个膝点时）
#     if len(knee_points) > 1:
#         trajectory_costs = [kp[0]['cost_cpu_seconds'] for kp in knee_points]
#         trajectory_throughputs = [kp[0]['throughput_mbps'] for kp in knee_points]
#         plt.plot(trajectory_costs, trajectory_throughputs, 
#                 color=COLORS['pareto'], linestyle='-.', linewidth=2.5, alpha=0.85,
#                 marker='x', markersize=8, markeredgecolor='black', markeredgewidth=1.2,
#                 label='偏好轨迹', zorder=6)
    
#     # 坐标轴（中文化）
#     plt.xlabel('CPU成本 (秒)', fontsize=14)
#     plt.ylabel('吞吐量 (Mbps)', fontsize=14)
    
#     # 标题（明确说明膝点计算位置，中文化）
#     plt.title('图5.4: 帕累托前沿上的膝点选择\n(仅在非支配解上计算)', 
#               fontsize=16, pad=22, fontweight='bold')
    
#     # 图例
#     plt.legend(loc='lower right', fontsize=11, framealpha=0.95, ncol=2)
    
#     # 网格
#     plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_4_Knee_Points.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成摘要
#     summary = {
#         'total_points': len(cloud_data),
#         'pareto_points': len(frontier),
#         'knee_points': len(knee_points),
#         'knee_configs': [(int(kp[0]['threads']), kp[0]['cpu_quota'], kp[1]['name']) 
#                         for kp in knee_points]
#     }
    
#     print(f"   ✅ 完成: Cloud_Fast 100MB场景")
#     print(f"      总配置数: {summary['total_points']}, 帕累托点: {summary['pareto_points']}, 膝点: {summary['knee_points']}")
#     for threads, cpu, name in summary['knee_configs']:
#         print(f"      {name:15s}: {threads}线程, {cpu:.1f}核")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_5_performance_gain(df):
#     """图5.5: 多场景性能提升对比"""
#     print("\n🎨 生成图5.5: 性能提升对比...")
    
#     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
#     # 场景配置
#     scenarios = [
#         {
#             'name': 'IoT_Weak',
#             'size': 10,
#             'title': 'IoT弱网场景',
#             'metric': '吞吐量提升',
#             'ylabel': '吞吐量 (Mbps)',
#             'baseline_label': '基线配置',
#             'optimized_label': '优化配置',
#             'color': COLORS['iot']
#         },
#         {
#             'name': 'Cloud_Fast', 
#             'size': 100,
#             'title': '云环境场景',
#             'metric': 'CPU成本降低',
#             'ylabel': 'CPU成本 (MB/CPU·秒)',
#             'baseline_label': '高成本配置',
#             'optimized_label': '低成本配置',
#             'color': COLORS['cloud']
#         }
#     ]
    
#     all_results = []
    
#     for ax, scenario in zip(axes, scenarios):
#         # 筛选数据
#         data = df[(df['network_type'] == scenario['name']) & 
#                  (df['file_size_mb'] == scenario['size']) & 
#                  (~df['is_baseline'])]
        
#         if len(data) < 10:
#             ax.text(0.5, 0.5, '数据不足', ha='center', va='center', 
#                    fontsize=14, color='gray')
#             ax.set_title(scenario['title'], fontsize=14, fontweight='bold')
#             continue
            
#         # 定义基准和优化配置
#         if scenario['name'] == 'IoT_Weak':
#             # IoT场景：低并发vs高并发
#             baseline_data = data[data['threads'] <= 2]
#             optimized_data = data[data['threads'] >= 8]
#             metric_col = 'throughput_mbps'
#         else:
#             # Cloud场景：高成本vs低成本  
#             cost_threshold = data['cost_cpu_seconds'].quantile(0.7)
#             baseline_data = data[data['cost_cpu_seconds'] >= cost_threshold]
#             optimized_data = data[data['cost_cpu_seconds'] <= data['cost_cpu_seconds'].quantile(0.3)]
#             metric_col = 'efficiency_mb_per_cpus'
        
#         if len(baseline_data) == 0 or len(optimized_data) == 0:
#             ax.text(0.5, 0.5, '配置分离失败', ha='center', va='center',
#                    fontsize=14, color='red')
#             ax.set_title(scenario['title'], fontsize=14, fontweight='bold')
#             continue
        
#         # 计算统计值
#         baseline_mean = baseline_data[metric_col].mean()
#         optimized_mean = optimized_data[metric_col].mean()
        
#         # 计算改善百分比
#         if scenario['name'] == 'IoT_Weak':
#             improvement_pct = ((optimized_mean - baseline_mean) / baseline_mean) * 100
#         else:
#             improvement_pct = ((baseline_mean - optimized_mean) / baseline_mean) * 100
        
#         # 绘制柱状图（适当符号大小）
#         bars = ax.bar([scenario['baseline_label'], scenario['optimized_label']], 
#                      [baseline_mean, optimized_mean],
#                      color=[COLORS['edge'], scenario['color']], 
#                      edgecolor='black', linewidth=1.2, width=0.6)
        
#         # 添加数值标签
#         for i, (bar, value) in enumerate(zip(bars, [baseline_mean, optimized_mean])):
#             height = bar.get_height()
#             if scenario['name'] == 'IoT_Weak':
#                 label = f'{value:.1f} Mbps'
#             else:
#                 label = f'{value:.0f} MB/CPU·s'
                
#             ax.text(bar.get_x() + bar.get_width()/2, height + max(height*0.05, 0.1),
#                    label, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
#         # 添加改善百分比标签
#         if improvement_pct > 0:
#             improvement_text = f'+{improvement_pct:.1f}%' if scenario['name'] == 'IoT_Weak' else f'-{improvement_pct:.1f}%'
#             ax.text(0.5, max(baseline_mean, optimized_mean) * 1.15, 
#                    improvement_text, ha='center', va='bottom', 
#                    fontsize=16, fontweight='bold', 
#                    color='darkgreen' if improvement_pct > 0 else 'darkred')
        
#         # 坐标轴设置（中文化）
#         ax.set_ylabel(scenario['ylabel'], fontsize=12)
#         ax.set_title(f'{scenario["title"]}\n{scenario["metric"]}: {improvement_pct:+.1f}%', 
#                     fontsize=14, fontweight='bold')
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
        
#         # 保存结果
#         all_results.append({
#             'scenario': scenario['name'],
#             'baseline': baseline_mean,
#             'optimized': optimized_mean,
#             'improvement_pct': improvement_pct
#         })
    
#     # 总标题（中文化）
#     fig.suptitle('图5.5: 多目标优化性能提升对比', fontsize=16, y=1.02, fontweight='bold')
#     plt.tight_layout()
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_5_Performance_Gain.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 打印摘要
#     print(f"   ✅ 完成: {len(all_results)}个场景的性能对比")
#     for result in all_results:
#         direction = "提升" if result['improvement_pct'] > 0 else "降低"
#         if result['scenario'] == 'IoT_Weak':
#             print(f"      {result['scenario']:15s}: +{result['improvement_pct']:.1f}% ({result['baseline']:.1f} → {result['optimized']:.1f} Mbps)")
#         else:
#             print(f"      {result['scenario']:15s}: {result['improvement_pct']:+.1f}% ({result['baseline']:.0f} → {result['optimized']:.0f} MB/CPU·s)")
#     print(f"   📁 保存至: {output_path}")
    
#     return all_results

# # ================= 🚀 主执行函数 =================

# def main():
#     print("=" * 75)
#     print("🚀 创新点II可视化图表生成器 (修正版)")
#     print("Corrected Visualization for Innovation Point II: Multi-objective Optimization")
#     print("=" * 75)
#     print(f"📁 数据文件: {os.path.basename(DATA_FILE)}")
#     print(f"💾 输出目录: {OUTPUT_DIR}")
#     print("-" * 75)
    
#     # 加载并验证数据
#     df = load_and_validate_data()
#     if df is None:
#         print("\n❌ 数据加载失败，程序终止")
#         return
    
#     # 生成所有图表
#     summaries = {}
    
#     summaries['fig5_1'] = plot_fig_5_1_stratified_sampling(df)
#     summaries['fig5_2'] = plot_fig_5_2_stability_tradeoff(df)
#     summaries['fig5_3'] = plot_fig_5_3_pareto_discrete(df)
#     summaries['fig5_4'] = plot_fig_5_4_knee_points(df)
#     summaries['fig5_5'] = plot_fig_5_5_performance_gain(df)
    
#     # 生成综合报告
#     print("\n" + "=" * 75)
#     print("✅ 所有图表生成完成!")
#     print("=" * 75)
#     print(f"📁 输出目录: {OUTPUT_DIR}")
#     print("\n📊 生成的图表文件:")
    
#     generated_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
#     for i, file in enumerate(generated_files, 1):
#         print(f"   {i}. {file}")
    
#     # 生成数据摘要报告
#     print("\n📋 实验数据摘要:")
#     print(f"   总记录数: {len(df):,}")
#     print(f"   有效场景: {df['network_type'].nunique()}")
#     print(f"   文件大小变体: {sorted(df['file_size_mb'].unique())} MB")
#     print(f"   CPU配额范围: {df['cpu_quota'].min():.1f} - {df['cpu_quota'].max():.1f} 核")
#     print(f"   线程数范围: {df['threads'].min()} - {df['threads'].max()}")
    
#     # 保存摘要到JSON
#     summary_report = {
#         'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'data_file': os.path.basename(DATA_FILE),
#         'total_records': len(df),
#         'figures_generated': len(generated_files),
#         'figure_summaries': summaries
#     }
    
#     import json
#     summary_path = os.path.join(OUTPUT_DIR, "visualization_summary.json")
#     with open(summary_path, 'w', encoding='utf-8') as f:
#         json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
#     print(f"\n💾 摘要报告已保存: {summary_path}")
#     print("=" * 75)
#     print("\n💡 使用建议:")
#     print("   1. 所有图表均基于实测指标（吞吐量/成本/时间），无虚构数据")
#     print("   2. 帕累托前沿以离散点展示，未进行插值平滑（符合数据真实性）")
#     print("   3. 图5.2使用传输时间变异系数(CV)定义稳定性，非丢包率")
#     print("   4. 论文中建议添加图注说明数据采样限制（见visualization_summary.json）")
#     print("=" * 75)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# """
# 创新点II：多目标自适应传输优化 - 中文版可视化脚本
# Corrected Visualization Script for Innovation Point II (Chinese Version)

# 核心特性：
# ✅ 中文显示完美支持（微软雅黑/黑体自动适配）
# ✅ 保留Times New Roman英文主体（符合学术出版规范）
# ✅ 所有图表基于实测指标（无虚构丢包率）
# ✅ 诚实展示离散帕累托点（不强行平滑）
# ✅ 学术级图表美化（300 DPI, IEEE配色）
# """

# import pandas as pd
# import numpy as np
# import matplotlib
# import platform
# import os
# import glob
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 0. 中文字体配置（必须在 import pyplot 之前！）
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
# elif system_name == 'Darwin':  # macOS
#     font_list = ['Heiti TC', 'PingFang HK', 'STHeiti']
# else:  # Linux
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC']

# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# # ==============================================================================
# # 1. 导入绘图库（必须在字体配置之后！）
# # ==============================================================================
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==============================================================================
# # 2. 全局配置
# # ==============================================================================
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 自动查找最新实验数据文件
# data_files = glob.glob(os.path.join(SCRIPT_DIR, "pareto_results*.csv")) + \
#              glob.glob(os.path.join(SCRIPT_DIR, "*cleaned*.csv"))

# if data_files:
#     DATA_FILE = max(data_files, key=os.path.getctime)
#     print(f"📊 检测到数据文件: {os.path.basename(DATA_FILE)} ({len(data_files)} 个候选)")
# else:
#     print("❌ 未找到实验数据文件 (pareto_results_*.csv)")
#     exit(1)

# OUTPUT_DIR = os.path.join(SCRIPT_DIR, "innovation_ii_figures_chinese")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 设置学术论文绘图风格 (IEEE/ACM 标准)
# # 注意：中文环境下保留Times New Roman用于英文/数字，中文自动用配置字体
# plt.rcParams.update({
#     'font.family': 'Times New Roman',  # 英文/数字用Times New Roman
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 11,
#     'figure.titlesize': 18,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
#     'lines.linewidth': 2.0,
#     'lines.markersize': 8,
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linestyle': '--',
#     'grid.linewidth': 0.8
# })

# # IEEE 配色方案
# COLORS = {
#     'iot': '#e74c3c',      # 红色 - IoT弱网
#     'edge': '#f39c12',     # 橙色 - Edge边缘
#     'cloud': '#27ae60',    # 绿色 - Cloud云
#     'anchor': '#3498db',   # 蓝色 - Anchor系统扫描
#     'probe_small': '#9b59b6',  # 紫色 - Probe小文件
#     'probe_large': '#1abc9c',  # 青色 - Probe大文件
#     'baseline': '#95a5a6', # 灰色 - Baseline
#     'pareto': '#8e44ad'    # 深紫 - 帕累托点
# }

# # ==============================================================================
# # 3. 数据加载与验证
# # ==============================================================================

# def load_and_validate_data():
#     """加载数据并验证指标真实性（仅保留实测指标）"""
#     try:
#         df = pd.read_csv(DATA_FILE)
#         print(f"✅ 成功加载数据: {len(df)} 条记录")
        
#         # 必需列验证
#         required_cols = ['run_id', 'exp_type', 'file_size_mb', 'scenario', 
#                         'cpu_quota', 'threads', 'chunk_kb', 'duration_s',
#                         'throughput_mbps', 'cost_cpu_seconds', 'efficiency_mb_per_cpus', 'exit_code']
        
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             print(f"❌ 缺少必要列: {missing_cols}")
#             return None
        
#         # 数据清洗
#         df = df.dropna(subset=['throughput_mbps', 'cost_cpu_seconds', 'duration_s'])
#         df = df[df['exit_code'] == 0]  # 仅保留成功实验
#         df = df[df['duration_s'] > 0]  # 移除零时长异常值
        
#         # 场景标准化（移除_BASELINE后缀用于分组）
#         df['network_type'] = df['scenario'].str.replace('_BASELINE', '', regex=False)
        
#         # 实验类型分类
#         df['is_baseline'] = df['exp_type'].str.contains('baseline', case=False)
#         df['is_anchor'] = df['exp_type'].str.contains('anchor', case=False) & ~df['is_baseline']
#         df['is_probe_small'] = df['exp_type'] == 'probe_small'
#         df['is_probe_large'] = df['exp_type'] == 'probe_large'
        
#         # ⚠️ 关键验证：确认无丢包率实测数据
#         has_loss_rate = any(col.lower().find('loss') >= 0 for col in df.columns)
#         if has_loss_rate:
#             print("⚠️ 警告: 检测到'loss'相关列，但TCP重传使应用层丢包率≈0%，不建议用于风险分析")
        
#         # 数据质量报告
#         print("\n📊 数据质量报告:")
#         print(f"   总记录数: {len(df):,}")
#         print(f"   Baseline记录: {df['is_baseline'].sum():,}")
#         print(f"   Anchor实验: {df['is_anchor'].sum():,}")
#         print(f"   Probe Small: {df['is_probe_small'].sum():,}")
#         print(f"   Probe Large: {df['is_probe_large'].sum():,}")
#         print(f"   吞吐量范围: {df['throughput_mbps'].min():.2f} - {df['throughput_mbps'].max():.2f} Mbps")
#         print(f"   CPU成本范围: {df['cost_cpu_seconds'].min():.4f} - {df['cost_cpu_seconds'].max():.4f} s")
#         print(f"   传输时间范围: {df['duration_s'].min():.2f} - {df['duration_s'].max():.2f} s")
        
#         # 场景分布
#         print("\n🌐 场景分布:")
#         for scenario, count in df['network_type'].value_counts().items():
#             sizes = sorted(df[df['network_type'] == scenario]['file_size_mb'].unique())
#             print(f"   {scenario:20s}: {count:4d} 条记录 | 文件大小: {sizes}")
        
#         return df
        
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # ==============================================================================
# # 4. 帕累托前沿计算（标准算法）
# # ==============================================================================

# def compute_pareto_frontier(df, maximize_col='throughput_mbps', minimize_col='cost_cpu_seconds'):
#     """
#     标准帕累托前沿计算 (完整支配关系检查)
#     返回非支配解集合（前沿点）
#     """
#     if len(df) == 0:
#         return pd.DataFrame()
    
#     # 完整的帕累托支配检查
#     pareto_points = []
#     for idx, candidate in df.iterrows():
#         is_dominated = False
#         for _, other in df.iterrows():
#             # 检查other是否支配candidate
#             if (other[maximize_col] >= candidate[maximize_col] and 
#                 other[minimize_col] <= candidate[minimize_col] and
#                 (other[maximize_col] > candidate[maximize_col] or 
#                  other[minimize_col] < candidate[minimize_col])):
#                 is_dominated = True
#                 break
#         if not is_dominated:
#             pareto_points.append(candidate)
    
#     pareto_df = pd.DataFrame(pareto_points)
#     if not pareto_df.empty:
#         # 按成本排序便于绘图
#         pareto_df = pareto_df.sort_values(minimize_col).reset_index(drop=True)
    
#     return pareto_df

# # ==============================================================================
# # 5. 图表生成函数（全部中文标签）
# # ==============================================================================

# def plot_fig_5_1_parameter_coverage(df):
#     """图5.1: 参数空间覆盖 - 系统扫描 + 极端点探测"""
#     print("\n🎨 生成图5.1: 参数空间覆盖...")
    
#     fig = plt.figure(figsize=(11, 7))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 按实验类型筛选（排除baseline）
#     anchor = df[df['is_anchor']]
#     probe_small = df[df['is_probe_small']]
#     probe_large = df[df['is_probe_large']]
    
#     # 用吞吐量着色（归一化）
#     all_throughput = pd.concat([anchor['throughput_mbps'], 
#                                 probe_small['throughput_mbps'],
#                                 probe_large['throughput_mbps']])
#     norm = plt.Normalize(vmin=all_throughput.min(), vmax=all_throughput.max())
#     cmap = plt.cm.viridis
    
#     # 绘制Anchor点（系统扫描）
#     if len(anchor) > 0:
#         colors = cmap(norm(anchor['throughput_mbps']))
#         ax.scatter(anchor['threads'], anchor['cpu_quota'], anchor['chunk_kb']/1024,
#                   c=colors, s=60, alpha=0.85, edgecolors='black', linewidth=0.6,
#                   label='锚点 (系统扫描)', depthshade=True)
    
#     # 绘制Probe Small点（小文件极端）
#     if len(probe_small) > 0:
#         colors = cmap(norm(probe_small['throughput_mbps']))
#         ax.scatter(probe_small['threads'], probe_small['cpu_quota'], 
#                   probe_small['chunk_kb']/1024,
#                   c=colors, s=100, alpha=0.9, marker='^', edgecolors='black', linewidth=0.8,
#                   label='探测点 (10MB)', depthshade=True)
    
#     # 绘制Probe Large点（大文件极端）
#     if len(probe_large) > 0:
#         colors = cmap(norm(probe_large['throughput_mbps']))
#         ax.scatter(probe_large['threads'], probe_large['cpu_quota'], 
#                   probe_large['chunk_kb']/1024,
#                   c=colors, s=100, alpha=0.9, marker='s', edgecolors='black', linewidth=0.8,
#                   label='探测点 (300MB)', depthshade=True)
    
#     # 坐标轴标签（中文）
#     ax.set_xlabel('并发线程数', fontsize=13, labelpad=12)
#     ax.set_ylabel('CPU配额 (核)', fontsize=13, labelpad=12)
#     ax.set_zlabel('分片大小 (KB)', fontsize=13, labelpad=12)
    
#     # 标题（中文）
#     ax.set_title('图5.1: 参数空间覆盖策略\n系统扫描与极端点探测', 
#                 fontsize=16, pad=18, fontweight='bold')
    
#     # 图例
#     ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
#     # 视角
#     ax.view_init(elev=28, azim=38)
    
#     # 颜色条
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
#     cbar.set_label('吞吐量 (Mbps)', rotation=270, labelpad=22, fontsize=12)
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_1_Parameter_Coverage.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成数据摘要
#     summary = {
#         'anchor_points': len(anchor),
#         'probe_small_points': len(probe_small),
#         'probe_large_points': len(probe_large),
#         'total_points': len(anchor) + len(probe_small) + len(probe_large)
#     }
    
#     print(f"   ✅ 完成: {summary['total_points']} 个配置点")
#     print(f"      锚点: {summary['anchor_points']}, 探测点(小): {summary['probe_small_points']}, 探测点(大): {summary['probe_large_points']}")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_2_stability_tradeoff(df):
#     """图5.2: 稳定性-性能权衡（基于实测传输时间变异系数）"""
#     print("\n🎨 生成图5.2: 稳定性-性能权衡...")
    
#     # 按场景+文件大小分组计算稳定性指标
#     stability_data = []
#     for (scenario, size), group in df.groupby(['network_type', 'file_size_mb']):
#         # 仅使用非baseline实验数据
#         group = group[~group['is_baseline']]
        
#         if len(group) >= 3:  # 至少3次重复才有统计意义（降低阈值）
#             stability_data.append({
#                 'scenario': scenario,
#                 'file_size_mb': size,
#                 'throughput_mean': group['throughput_mbps'].mean(),
#                 'throughput_std': group['throughput_mbps'].std(),
#                 'duration_mean': group['duration_s'].mean(),
#                 'duration_cv': (group['duration_s'].std() / 
#                                max(group['duration_s'].mean(), 1e-6)) * 100,  # 变异系数(%)
#                 'sample_size': len(group)
#             })
    
#     if not stability_data:
#         print("   ⚠️  警告: 无足够数据计算稳定性指标（每组需≥3个样本）")
#         return {}
    
#     stability_df = pd.DataFrame(stability_data)
    
#     # 创建图表
#     plt.figure(figsize=(12, 7.5))
    
#     # 场景映射（中文标签）
#     scenario_config = {
#         'IoT_Weak': {'label': 'IoT弱网', 'color': COLORS['iot'], 'marker': 'o'},
#         'Edge_Normal': {'label': '边缘网络', 'color': COLORS['edge'], 'marker': 's'},
#         'Cloud_Fast': {'label': '云环境', 'color': COLORS['cloud'], 'marker': '^'}
#     }
    
#     # 绘制每个场景
#     for scenario_key, config in scenario_config.items():
#         subset = stability_df[stability_df['scenario'] == scenario_key]
#         if not subset.empty:
#             plt.scatter(subset['throughput_mean'], subset['duration_cv'],
#                        c=config['color'], s=subset['sample_size']*15,  # 气泡大小反映样本量
#                        alpha=0.85, edgecolors='black', linewidth=1.3,
#                        marker=config['marker'], label=f"{config['label']} (n={len(subset)})")
    
#     # 稳定性阈值线（基于实测分布：CV>30%视为不稳定）
#     plt.axhline(y=30, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.85,
#                 label='稳定性阈值 (CV=30%)')
    
#     # 坐标轴（中文）
#     plt.xlabel('平均吞吐量 (Mbps)', fontsize=14)
#     plt.ylabel('传输时间变异系数 (CV, %)', fontsize=14)
    
#     # 标题（中文 + 诚实标注指标来源）
#     plt.title('图5.2: 稳定性-性能权衡分析\n基于实测传输时间变异系数（无丢包率测量）', 
#               fontsize=16, pad=20, fontweight='bold')
    
#     # 图例
#     plt.legend(fontsize=11, loc='upper right', framealpha=0.95)
    
#     # 网格
#     plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
#     # Y轴范围
#     plt.ylim(0, min(plt.ylim()[1] * 1.15, 100))  # 限制在100%以内
    
#     # 添加注释框（中文）
#     plt.text(0.98, 0.96, 'CV越低 → 稳定性越高\n(传输更可预测)', 
#             transform=plt.gca().transAxes, fontsize=11,
#             verticalalignment='top', horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.85, edgecolor='gray'))
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_2_Stability_Tradeoff.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成摘要
#     summary = {
#         'scenarios': stability_df['scenario'].nunique(),
#         'total_configs': len(stability_df),
#         'max_cv': stability_df['duration_cv'].max(),
#         'min_cv': stability_df['duration_cv'].min()
#     }
    
#     print(f"   ✅ 完成: {summary['total_configs']} 个配置的稳定性分析")
#     print(f"      CV范围: {summary['min_cv']:.1f}% - {summary['max_cv']:.1f}%")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_3_pareto_discrete(df):
#     """图5.3: 离散帕累托点（诚实展示，不强行平滑）"""
#     print("\n🎨 生成图5.3: 离散帕累托前沿...")
    
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
#     scenarios = [
#         {'name': 'IoT_Weak', 'size': 10, 'title': 'IoT弱网 (10MB)', 'color': COLORS['iot']},
#         {'name': 'Edge_Normal', 'size': 50, 'title': '边缘网络 (50MB)', 'color': COLORS['edge']},
#         {'name': 'Cloud_Fast', 'size': 100, 'title': '云环境 (100MB)', 'color': COLORS['cloud']}
#     ]
    
#     all_summaries = []
    
#     for ax, scenario in zip(axes, scenarios):
#         # 筛选数据：指定场景+文件大小+非baseline
#         subset = df[(df['network_type'] == scenario['name']) & 
#                    (df['file_size_mb'] == scenario['size']) & 
#                    (~df['is_baseline'])]
        
#         # 数据量检查
#         if len(subset) < 8:
#             ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
#                    ha='center', va='center', fontsize=13, color='gray',
#                    fontweight='bold')
#             ax.set_title(scenario['title'], fontsize=14, color='gray', fontweight='bold')
#             ax.set_xlabel('CPU成本 (秒)', fontsize=12)
#             ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
#             all_summaries.append({'scenario': scenario['name'], 'points': len(subset), 'pareto': 0})
#             continue
        
#         # 计算帕累托前沿
#         frontier = compute_pareto_frontier(subset)
        
#         # 绘制所有点（浅灰色背景）
#         ax.scatter(subset['cost_cpu_seconds'], subset['throughput_mbps'],
#                   c='#bdc3c7', s=50, alpha=0.65, edgecolors='none',
#                   label=f'全部配置 ({len(subset)})')
        
#         # 绘制帕累托点（大圆点，不连线！）
#         if len(frontier) >= 3:
#             ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                       c=scenario['color'], s=140, alpha=0.92, 
#                       edgecolors='black', linewidth=1.6, marker='o',
#                       label=f'帕累托最优 ({len(frontier)})', zorder=5)
#         elif len(frontier) > 0:
#             # 少量点用星号强调
#             ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                       c=scenario['color'], s=180, alpha=0.95, 
#                       edgecolors='black', linewidth=1.8, marker='*',
#                       label=f'帕累托点 ({len(frontier)})', zorder=5)
        
#         # 坐标轴（中文）
#         ax.set_xlabel('CPU成本 (秒)', fontsize=12)
#         ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
#         ax.set_title(scenario['title'], fontsize=14, fontweight='bold')
        
#         # 图例（右下角）
#         ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92)
        
#         # 标注数据量
#         ax.text(0.04, 0.96, f'n={len(subset)}', transform=ax.transAxes,
#                fontsize=11, verticalalignment='top', fontweight='bold',
#                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
#                         edgecolor=scenario['color'], alpha=0.85, linewidth=1.5))
        
#         # 保存场景摘要
#         all_summaries.append({
#             'scenario': scenario['name'],
#             'total_points': len(subset),
#             'pareto_points': len(frontier)
#         })
    
#     # 总标题（中文 + 诚实标注离散采样）
#     fig.suptitle('图5.3: 不同网络环境下的帕累托最优配置\n(离散采样 - 无插值平滑)', 
#                 fontsize=17, y=1.04, fontweight='bold')
#     plt.tight_layout()
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_3_Pareto_Discrete.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 打印摘要
#     print(f"   ✅ 完成: 3个场景的帕累托分析")
#     for summary in all_summaries:
#         print(f"      {summary['scenario']:15s}: 总点数={summary['total_points']:3d} | 帕累托点={summary['pareto_points']:2d}")
#     print(f"   📁 保存至: {output_path}")
    
#     return all_summaries

# def plot_fig_5_4_knee_points(df):
#     """图5.4: 帕累托前沿上的膝点选择（仅在前沿上计算）"""
#     print("\n🎨 生成图5.4: 帕累托膝点选择...")
    
#     # 选择数据最丰富的场景：Cloud_Fast 100MB
#     cloud_data = df[(df['network_type'] == 'Cloud_Fast') & 
#                    (df['file_size_mb'] == 100) & 
#                    (~df['is_baseline'])]
    
#     if len(cloud_data) < 10:
#         print(f"   ⚠️  Cloud_Fast 100MB数据不足 (n={len(cloud_data)} < 10)，跳过图5.4")
#         return {}
    
#     # 计算帕累托前沿
#     frontier = compute_pareto_frontier(cloud_data)
    
#     if len(frontier) < 2:  # 降低要求至2个点
#         print(f"   ⚠️  帕累托前沿点数不足 (n={len(frontier)} < 2)，跳过图5.4")
#         return {}
    
#     # 归一化（仅在前沿上）
#     c_min, c_max = frontier['cost_cpu_seconds'].min(), frontier['cost_cpu_seconds'].max()
#     t_min, t_max = frontier['throughput_mbps'].min(), frontier['throughput_mbps'].max()
    
#     frontier['c_norm'] = (frontier['cost_cpu_seconds'] - c_min) / max(c_max - c_min, 1e-8)
#     frontier['t_norm'] = (frontier['throughput_mbps'] - t_min) / max(t_max - t_min, 1e-8)
    
#     # 创建图表
#     plt.figure(figsize=(13, 8))
    
#     # 绘制所有实验点（浅灰色背景）
#     plt.scatter(cloud_data['cost_cpu_seconds'], cloud_data['throughput_mbps'],
#                c='#ecf0f1', s=45, alpha=0.5, edgecolors='none', label='全部配置')
    
#     # 绘制帕累托前沿（深灰色虚线 + 点）
#     if len(frontier) > 1:
#         plt.plot(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                 'k--', linewidth=2.2, alpha=0.7, label='帕累托前沿')
#     plt.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                c='black', s=90, alpha=0.85, zorder=5, edgecolors='white', linewidth=1.2)
    
#     # 定义膝点权重配置（仅3种清晰配置）
#     weights = [
#         {'name': '节能优先', 'wc': 0.8, 'wt': 0.2, 'color': COLORS['cloud'], 'marker': 'D'},
#         {'name': '平衡配置', 'wc': 0.5, 'wt': 0.5, 'color': COLORS['edge'], 'marker': 'o'},
#         {'name': '性能优先', 'wc': 0.2, 'wt': 0.8, 'color': COLORS['iot'], 'marker': '^'}
#     ]
    
#     knee_points = []
    
#     # 仅在帕累托前沿上计算膝点
#     for weight in weights:
#         # L2距离（归一化空间）
#         distances = np.sqrt(
#             weight['wc'] * frontier['c_norm']**2 + 
#             weight['wt'] * (1 - frontier['t_norm'])**2
#         )
#         best_idx = distances.idxmin()
#         best_point = frontier.loc[best_idx]
#         knee_points.append((best_point, weight))
        
#         # 绘制膝点
#         plt.scatter(best_point['cost_cpu_seconds'], best_point['throughput_mbps'],
#                    s=400, c=weight['color'], marker=weight['marker'], 
#                    edgecolors='black', linewidth=2.2, zorder=10,
#                    label=f"{weight['name']} (w_c={weight['wc']})")
        
#         # 标注配置信息（中文）
#         config_text = f"{int(best_point['threads'])}线程\n{best_point['cpu_quota']:.1f}核"
#         plt.annotate(config_text, 
#                     (best_point['cost_cpu_seconds'], best_point['throughput_mbps']),
#                     xytext=(28, 30), textcoords='offset points',
#                     fontsize=11, fontweight='bold', color=weight['color'],
#                     bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
#                              alpha=0.92, edgecolor=weight['color'], linewidth=1.8),
#                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', 
#                                   color=weight['color'], lw=2.0, alpha=0.8))
    
#     # 连接膝点展示偏好轨迹
#     if len(knee_points) > 1:
#         trajectory_costs = [kp[0]['cost_cpu_seconds'] for kp in knee_points]
#         trajectory_throughputs = [kp[0]['throughput_mbps'] for kp in knee_points]
#         plt.plot(trajectory_costs, trajectory_throughputs, 
#                 color=COLORS['pareto'], linestyle='-.', linewidth=3.0, alpha=0.85,
#                 marker='x', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
#                 label='偏好轨迹', zorder=6)
    
#     # 坐标轴（中文）
#     plt.xlabel('CPU成本 (秒)', fontsize=14)
#     plt.ylabel('吞吐量 (Mbps)', fontsize=14)
    
#     # 标题（中文 + 明确膝点计算位置）
#     plt.title('图5.4: 帕累托前沿上的膝点选择\n(仅在非支配解上计算)', 
#               fontsize=16, pad=22, fontweight='bold')
    
#     # 图例
#     plt.legend(loc='lower right', fontsize=11, framealpha=0.95, ncol=2)
    
#     # 网格
#     plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_4_Knee_Points.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成摘要
#     summary = {
#         'total_points': len(cloud_data),
#         'pareto_points': len(frontier),
#         'knee_points': len(knee_points),
#         'knee_configs': [(int(kp[0]['threads']), kp[0]['cpu_quota'], kp[1]['name']) 
#                         for kp in knee_points]
#     }
    
#     print(f"   ✅ 完成: Cloud_Fast 100MB场景")
#     print(f"      总配置数: {summary['total_points']}, 帕累托点: {summary['pareto_points']}, 膝点: {summary['knee_points']}")
#     for threads, cpu, name in summary['knee_configs']:
#         print(f"      {name:15s}: {threads}线程, {cpu:.1f}核")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_5_performance_gain(df):
#     """图5.5: 多场景性能提升对比（基于实测最优配置）"""
#     print("\n🎨 生成图5.5: 性能提升对比...")
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
#     # 定义对比场景
#     scenarios = [
#         {
#             'name': 'IoT_Weak', 
#             'size': 10,
#             'title': 'IoT弱网场景\n(10MB文件)',
#             'metric': 'throughput',
#             'ylabel': '吞吐量 (Mbps)',
#             'baseline_config': {'cpu_quota': 0.5, 'threads': 1},  # 保守配置
#             'optimal_config': {'cpu_quota': 2.0, 'threads': 16}   # 激进配置
#         },
#         {
#             'name': 'Cloud_Fast', 
#             'size': 100,
#             'title': '云环境场景\n(100MB文件)',
#             'metric': 'efficiency',
#             'ylabel': '资源效率 (MB/CPU·秒)',
#             'baseline_config': {'cpu_quota': 2.0, 'threads': 16},  # 高资源消耗
#             'optimal_config': {'cpu_quota': 0.5, 'threads': 4}     # 资源高效
#         }
#     ]
    
#     summaries = []
    
#     for idx, scenario in enumerate(scenarios):
#         ax = ax1 if idx == 0 else ax2
        
#         # 筛选场景数据
#         subset = df[(df['network_type'] == scenario['name']) & 
#                    (df['file_size_mb'] == scenario['size']) & 
#                    (~df['is_baseline'])]
        
#         if len(subset) < 10:
#             ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
#                    ha='center', va='center', fontsize=13, color='gray')
#             ax.set_title(scenario['title'], fontsize=14, color='gray')
#             summaries.append({'scenario': scenario['name'], 'valid': False})
#             continue
        
#         # 提取baseline配置性能
#         baseline_mask = (
#             (subset['cpu_quota'] == scenario['baseline_config']['cpu_quota']) & 
#             (subset['threads'] == scenario['baseline_config']['threads'])
#         )
#         baseline_data = subset[baseline_mask]
        
#         # 提取optimal配置性能
#         optimal_mask = (
#             (subset['cpu_quota'] == scenario['optimal_config']['cpu_quota']) & 
#             (subset['threads'] == scenario['optimal_config']['threads'])
#         )
#         optimal_data = subset[optimal_mask]
        
#         # 检查数据有效性
#         if len(baseline_data) == 0 or len(optimal_data) == 0:
#             ax.text(0.5, 0.5, '配置未找到', 
#                    ha='center', va='center', fontsize=13, color='gray')
#             ax.set_title(scenario['title'], fontsize=14, color='gray')
#             summaries.append({'scenario': scenario['name'], 'valid': False})
#             continue
        
#         # 计算指标
#         if scenario['metric'] == 'throughput':
#             baseline_val = baseline_data['throughput_mbps'].mean()
#             optimal_val = optimal_data['throughput_mbps'].mean()
#             improvement = ((optimal_val - baseline_val) / baseline_val) * 100
#             unit = 'Mbps'
#             higher_is_better = True
#         else:  # efficiency
#             baseline_val = baseline_data['efficiency_mb_per_cpus'].mean()
#             optimal_val = optimal_data['efficiency_mb_per_cpus'].mean()
#             improvement = ((optimal_val - baseline_val) / baseline_val) * 100
#             unit = 'MB/CPU·s'
#             higher_is_better = True
        
#         # 绘制柱状图
#         bars = ax.bar(
#             ['基线配置\n(保守)', '优化配置\n(自适应)'], 
#             [baseline_val, optimal_val],
#             color=[COLORS['baseline'], COLORS['cloud'] if idx==1 else COLORS['iot']], 
#             width=0.65, edgecolor='black', linewidth=1.5, alpha=0.9
#         )
        
#         # 标注数值
#         for i, (bar, value) in enumerate(zip(bars, [baseline_val, optimal_val])):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2, height + height*0.04,
#                    f'{value:.1f}\n{unit}', ha='center', va='bottom', 
#                    fontweight='bold', fontsize=12)
        
#         # 标注提升幅度
#         if abs(improvement) > 5:  # 显著提升才标注
#             sign = '+' if improvement > 0 else ''
#             color = '#27ae60' if improvement > 0 else '#e74c3c'
#             ax.text(1, optimal_val + optimal_val*0.12,
#                    f'{sign}{improvement:.0f}%\n提升', 
#                    ha='center', va='bottom', fontweight='bold', fontsize=13,
#                    color=color,
#                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
#                             alpha=0.9, edgecolor=color, linewidth=2))
        
#         # 坐标轴（中文）
#         ax.set_ylabel(scenario['ylabel'], fontsize=13)
#         ax.set_title(scenario['title'], fontsize=14, fontweight='bold', pad=15)
#         ax.set_ylim(0, max(baseline_val, optimal_val) * 1.35)
        
#         # 保存摘要
#         summaries.append({
#             'scenario': scenario['name'],
#             'baseline': baseline_val,
#             'optimal': optimal_val,
#             'improvement': improvement,
#             'unit': unit,
#             'valid': True
#         })
    
#     # 总标题（中文）
#     fig.suptitle('图5.5: 自适应配置的性能提升\n相比保守基线配置的实测改进', 
#                 fontsize=17, y=1.03, fontweight='bold')
#     plt.tight_layout()
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_5_Performance_Gain.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 打印摘要
#     print(f"   ✅ 完成: 2个场景的性能对比")
#     for summary in summaries:
#         if summary['valid']:
#             sign = '+' if summary['improvement'] > 0 else ''
#             print(f"      {summary['scenario']:15s}: {sign}{summary['improvement']:.1f}% "
#                   f"({summary['baseline']:.1f} → {summary['optimal']:.1f} {summary['unit']})")
#     print(f"   📁 保存至: {output_path}")
    
#     return summaries

# # ==============================================================================
# # 6. 主执行函数
# # ==============================================================================

# def main():
#     print("=" * 75)
#     print("🚀 创新点II可视化图表生成器 (中文版)")
#     print("Visualization for Innovation Point II: Multi-objective Optimization (Chinese)")
#     print("=" * 75)
#     print(f"📁 数据文件: {os.path.basename(DATA_FILE)}")
#     print(f"💾 输出目录: {OUTPUT_DIR}")
#     print("-" * 75)
    
#     # 加载并验证数据
#     df = load_and_validate_data()
#     if df is None:
#         print("\n❌ 数据加载失败，程序终止")
#         return
    
#     # 生成所有图表
#     summaries = {}
    
#     summaries['fig5_1'] = plot_fig_5_1_parameter_coverage(df)
#     summaries['fig5_2'] = plot_fig_5_2_stability_tradeoff(df)
#     summaries['fig5_3'] = plot_fig_5_3_pareto_discrete(df)
#     summaries['fig5_4'] = plot_fig_5_4_knee_points(df)
#     summaries['fig5_5'] = plot_fig_5_5_performance_gain(df)
    
#     # 生成综合报告
#     print("\n" + "=" * 75)
#     print("✅ 所有图表生成完成!")
#     print("=" * 75)
#     print(f"📁 输出目录: {OUTPUT_DIR}")
#     print("\n📊 生成的图表文件:")
    
#     generated_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
#     for i, file in enumerate(generated_files, 1):
#         print(f"   {i}. {file}")
    
#     # 生成数据摘要报告
#     print("\n📋 实验数据摘要:")
#     print(f"   总记录数: {len(df):,}")
#     print(f"   有效场景: {df['network_type'].nunique()}")
#     print(f"   文件大小变体: {sorted(df['file_size_mb'].unique())} MB")
#     print(f"   CPU配额范围: {df['cpu_quota'].min():.1f} - {df['cpu_quota'].max():.1f} 核")
#     print(f"   线程数范围: {df['threads'].min()} - {df['threads'].max()}")
    
#     # 保存摘要到JSON
#     summary_report = {
#         'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'data_file': os.path.basename(DATA_FILE),
#         'total_records': len(df),
#         'figures_generated': len(generated_files),
#         'figure_summaries': summaries
#     }
    
#     import json
#     summary_path = os.path.join(OUTPUT_DIR, "visualization_summary.json")
#     with open(summary_path, 'w', encoding='utf-8') as f:
#         json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
#     print(f"\n💾 摘要报告已保存: {summary_path}")
#     print("=" * 75)
#     print("\n💡 使用说明:")
#     print("   1. 所有图表基于实测指标（吞吐量/成本/时间），无虚构丢包率")
#     print("   2. 帕累托前沿以离散点展示，未进行插值平滑（符合数据真实性）")
#     print("   3. 图5.2使用传输时间变异系数(CV)定义稳定性")
#     print("   4. 中文显示已优化（Windows自动使用微软雅黑）")
#     print("   5. 图表符合IEEE/ACM出版标准（300 DPI, Times New Roman英文主体）")
#     print("=" * 75)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
创新点II：多目标自适应传输优化 - 中文版可视化脚本 (修复版)
Corrected Visualization Script for Innovation Point II (Chinese Version)

核心特性：
✅ 中文显示完美支持（基于参考代码的字体适配方案）
✅ 自动适配 Windows/macOS/Linux
✅ 所有图表基于实测指标
✅ 诚实展示离散帕累托点
✅ 学术级图表美化（300 DPI, IEEE配色）
"""

# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt # 先导入
# import seaborn as sns
# import platform
# import os
# import glob
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 0. 绘图配置 (自动适配中文 - 移植自参考代码)
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# # ==============================================================================
# # 1. 全局配置
# # ==============================================================================
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 自动查找最新实验数据文件
# data_files = glob.glob(os.path.join(SCRIPT_DIR, "pareto_results*.csv")) + \
#              glob.glob(os.path.join(SCRIPT_DIR, "*cleaned*.csv"))

# if data_files:
#     DATA_FILE = max(data_files, key=os.path.getctime)
#     print(f"📊 检测到数据文件: {os.path.basename(DATA_FILE)} ({len(data_files)} 个候选)")
# else:
#     # 如果没有找到文件，为了演示代码运行，这里可以生成一个伪数据生成器或者报错
#     # 为了保证代码可运行，这里提示错误
#     print("❌ 未找到实验数据文件 (pareto_results_*.csv)")
#     print("💡 请确保目录下存在数据文件，或修改脚本中的 DATA_FILE 路径")
#     # exit(1) # 注释掉退出，防止在没有数据时直接崩溃，仅作为演示

# OUTPUT_DIR = os.path.join(SCRIPT_DIR, "innovation_ii_figures_chinese")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 设置学术论文绘图风格 (IEEE/ACM 标准)
# # 注意：已移除 'font.family': 'Times New Roman' 以避免覆盖中文字体
# plt.rcParams.update({
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 11,
#     'figure.titlesize': 18,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
#     'lines.linewidth': 2.0,
#     'lines.markersize': 8,
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linestyle': '--',
#     'grid.linewidth': 0.8
# })

# # IEEE 配色方案
# COLORS = {
#     'iot': '#e74c3c',       # 红色 - IoT弱网
#     'edge': '#f39c12',      # 橙色 - Edge边缘
#     'cloud': '#27ae60',     # 绿色 - Cloud云
#     'anchor': '#3498db',    # 蓝色 - Anchor系统扫描
#     'probe_small': '#9b59b6',  # 紫色 - Probe小文件
#     'probe_large': '#1abc9c',  # 青色 - Probe大文件
#     'baseline': '#95a5a6', # 灰色 - Baseline
#     'pareto': '#8e44ad'    # 深紫 - 帕累托点
# }

# # ==============================================================================
# # 2. 数据加载与验证
# # ==============================================================================

# def load_and_validate_data():
#     """加载数据并验证指标真实性（仅保留实测指标）"""
#     try:
#         # 检查变量是否存在 (处理上面可能未找到文件的情况)
#         if 'DATA_FILE' not in globals():
#             return None

#         df = pd.read_csv(DATA_FILE)
#         print(f"✅ 成功加载数据: {len(df)} 条记录")
        
#         # 必需列验证
#         required_cols = ['run_id', 'exp_type', 'file_size_mb', 'scenario', 
#                          'cpu_quota', 'threads', 'chunk_kb', 'duration_s',
#                          'throughput_mbps', 'cost_cpu_seconds', 'efficiency_mb_per_cpus', 'exit_code']
        
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             print(f"❌ 缺少必要列: {missing_cols}")
#             return None
        
#         # 数据清洗
#         df = df.dropna(subset=['throughput_mbps', 'cost_cpu_seconds', 'duration_s'])
#         df = df[df['exit_code'] == 0]  # 仅保留成功实验
#         df = df[df['duration_s'] > 0]  # 移除零时长异常值
        
#         # 场景标准化（移除_BASELINE后缀用于分组）
#         df['network_type'] = df['scenario'].str.replace('_BASELINE', '', regex=False)
        
#         # 实验类型分类
#         df['is_baseline'] = df['exp_type'].str.contains('baseline', case=False)
#         df['is_anchor'] = df['exp_type'].str.contains('anchor', case=False) & ~df['is_baseline']
#         df['is_probe_small'] = df['exp_type'] == 'probe_small'
#         df['is_probe_large'] = df['exp_type'] == 'probe_large'
        
#         # ⚠️ 关键验证：确认无丢包率实测数据
#         has_loss_rate = any(col.lower().find('loss') >= 0 for col in df.columns)
#         if has_loss_rate:
#             print("⚠️ 警告: 检测到'loss'相关列，但TCP重传使应用层丢包率≈0%，不建议用于风险分析")
        
#         # 数据质量报告
#         print("\n📊 数据质量报告:")
#         print(f"   总记录数: {len(df):,}")
#         print(f"   Baseline记录: {df['is_baseline'].sum():,}")
#         print(f"   Anchor实验: {df['is_anchor'].sum():,}")
#         print(f"   Probe Small: {df['is_probe_small'].sum():,}")
#         print(f"   Probe Large: {df['is_probe_large'].sum():,}")
#         print(f"   吞吐量范围: {df['throughput_mbps'].min():.2f} - {df['throughput_mbps'].max():.2f} Mbps")
#         print(f"   CPU成本范围: {df['cost_cpu_seconds'].min():.4f} - {df['cost_cpu_seconds'].max():.4f} s")
#         print(f"   传输时间范围: {df['duration_s'].min():.2f} - {df['duration_s'].max():.2f} s")
        
#         # 场景分布
#         print("\n🌐 场景分布:")
#         for scenario, count in df['network_type'].value_counts().items():
#             sizes = sorted(df[df['network_type'] == scenario]['file_size_mb'].unique())
#             print(f"   {scenario:20s}: {count:4d} 条记录 | 文件大小: {sizes}")
        
#         return df
        
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # ==============================================================================
# # 3. 帕累托前沿计算（标准算法）
# # ==============================================================================

# def compute_pareto_frontier(df, maximize_col='throughput_mbps', minimize_col='cost_cpu_seconds'):
#     """
#     标准帕累托前沿计算 (完整支配关系检查)
#     返回非支配解集合（前沿点）
#     """
#     if len(df) == 0:
#         return pd.DataFrame()
    
#     # 完整的帕累托支配检查
#     pareto_points = []
#     for idx, candidate in df.iterrows():
#         is_dominated = False
#         for _, other in df.iterrows():
#             # 检查other是否支配candidate
#             if (other[maximize_col] >= candidate[maximize_col] and 
#                 other[minimize_col] <= candidate[minimize_col] and
#                 (other[maximize_col] > candidate[maximize_col] or 
#                  other[minimize_col] < candidate[minimize_col])):
#                 is_dominated = True
#                 break
#         if not is_dominated:
#             pareto_points.append(candidate)
    
#     pareto_df = pd.DataFrame(pareto_points)
#     if not pareto_df.empty:
#         # 按成本排序便于绘图
#         pareto_df = pareto_df.sort_values(minimize_col).reset_index(drop=True)
    
#     return pareto_df

# # ==============================================================================
# # 4. 图表生成函数（全部中文标签）
# # ==============================================================================

# def plot_fig_5_1_parameter_coverage(df):
#     """图5.1: 参数空间覆盖 - 系统扫描 + 极端点探测"""
#     print("\n🎨 生成图5.1: 参数空间覆盖...")
    
#     fig = plt.figure(figsize=(11, 7))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 按实验类型筛选（排除baseline）
#     anchor = df[df['is_anchor']]
#     probe_small = df[df['is_probe_small']]
#     probe_large = df[df['is_probe_large']]
    
#     # 用吞吐量着色（归一化）
#     all_throughput = pd.concat([anchor['throughput_mbps'], 
#                                 probe_small['throughput_mbps'],
#                                 probe_large['throughput_mbps']])
#     norm = plt.Normalize(vmin=all_throughput.min(), vmax=all_throughput.max())
#     cmap = plt.cm.viridis
    
#     # 绘制Anchor点（系统扫描）
#     if len(anchor) > 0:
#         colors = cmap(norm(anchor['throughput_mbps']))
#         ax.scatter(anchor['threads'], anchor['cpu_quota'], anchor['chunk_kb']/1024,
#                   c=colors, s=60, alpha=0.85, edgecolors='black', linewidth=0.6,
#                   label='锚点 (系统扫描)', depthshade=True)
    
#     # 绘制Probe Small点（小文件极端）
#     if len(probe_small) > 0:
#         colors = cmap(norm(probe_small['throughput_mbps']))
#         ax.scatter(probe_small['threads'], probe_small['cpu_quota'], 
#                   probe_small['chunk_kb']/1024,
#                   c=colors, s=100, alpha=0.9, marker='^', edgecolors='black', linewidth=0.8,
#                   label='探测点 (10MB)', depthshade=True)
    
#     # 绘制Probe Large点（大文件极端）
#     if len(probe_large) > 0:
#         colors = cmap(norm(probe_large['throughput_mbps']))
#         ax.scatter(probe_large['threads'], probe_large['cpu_quota'], 
#                   probe_large['chunk_kb']/1024,
#                   c=colors, s=100, alpha=0.9, marker='s', edgecolors='black', linewidth=0.8,
#                   label='探测点 (300MB)', depthshade=True)
    
#     # 坐标轴标签（中文）
#     ax.set_xlabel('并发线程数', fontsize=13, labelpad=12)
#     ax.set_ylabel('CPU配额 (核)', fontsize=13, labelpad=12)
#     ax.set_zlabel('分片大小 (KB)', fontsize=13, labelpad=12)
    
#     # 标题（中文）
#     ax.set_title('图5.1: 参数空间覆盖策略\n系统扫描与极端点探测', 
#                 fontsize=16, pad=18, fontweight='bold')
    
#     # 图例
#     ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
#     # 视角
#     ax.view_init(elev=28, azim=38)
    
#     # 颜色条
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
#     cbar.set_label('吞吐量 (Mbps)', rotation=270, labelpad=22, fontsize=12)
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_1_Parameter_Coverage.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成数据摘要
#     summary = {
#         'anchor_points': len(anchor),
#         'probe_small_points': len(probe_small),
#         'probe_large_points': len(probe_large),
#         'total_points': len(anchor) + len(probe_small) + len(probe_large)
#     }
    
#     print(f"   ✅ 完成: {summary['total_points']} 个配置点")
#     print(f"      锚点: {summary['anchor_points']}, 探测点(小): {summary['probe_small_points']}, 探测点(大): {summary['probe_large_points']}")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_2_stability_tradeoff(df):
#     """图5.2: 稳定性-性能权衡（基于实测传输时间变异系数）"""
#     print("\n🎨 生成图5.2: 稳定性-性能权衡...")
    
#     # 按场景+文件大小分组计算稳定性指标
#     stability_data = []
#     for (scenario, size), group in df.groupby(['network_type', 'file_size_mb']):
#         # 仅使用非baseline实验数据
#         group = group[~group['is_baseline']]
        
#         if len(group) >= 3:  # 至少3次重复才有统计意义（降低阈值）
#             stability_data.append({
#                 'scenario': scenario,
#                 'file_size_mb': size,
#                 'throughput_mean': group['throughput_mbps'].mean(),
#                 'throughput_std': group['throughput_mbps'].std(),
#                 'duration_mean': group['duration_s'].mean(),
#                 'duration_cv': (group['duration_s'].std() / 
#                                max(group['duration_s'].mean(), 1e-6)) * 100,  # 变异系数(%)
#                 'sample_size': len(group)
#             })
    
#     if not stability_data:
#         print("   ⚠️  警告: 无足够数据计算稳定性指标（每组需≥3个样本）")
#         return {}
    
#     stability_df = pd.DataFrame(stability_data)
    
#     # 创建图表
#     plt.figure(figsize=(12, 7.5))
    
#     # 场景映射（中文标签）
#     scenario_config = {
#         'IoT_Weak': {'label': 'IoT弱网', 'color': COLORS['iot'], 'marker': 'o'},
#         'Edge_Normal': {'label': '边缘网络', 'color': COLORS['edge'], 'marker': 's'},
#         'Cloud_Fast': {'label': '云环境', 'color': COLORS['cloud'], 'marker': '^'}
#     }
    
#     # 绘制每个场景
#     for scenario_key, config in scenario_config.items():
#         subset = stability_df[stability_df['scenario'] == scenario_key]
#         if not subset.empty:
#             plt.scatter(subset['throughput_mean'], subset['duration_cv'],
#                        c=config['color'], s=subset['sample_size']*15,  # 气泡大小反映样本量
#                        alpha=0.85, edgecolors='black', linewidth=1.3,
#                        marker=config['marker'], label=f"{config['label']} (n={len(subset)})")
    
#     # 稳定性阈值线（基于实测分布：CV>30%视为不稳定）
#     plt.axhline(y=30, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.85,
#                 label='稳定性阈值 (CV=30%)')
    
#     # 坐标轴（中文）
#     plt.xlabel('平均吞吐量 (Mbps)', fontsize=14)
#     plt.ylabel('传输时间变异系数 (CV, %)', fontsize=14)
    
#     # 标题（中文 + 诚实标注指标来源）
#     plt.title('图5.2: 稳定性-性能权衡分析\n基于实测传输时间变异系数（无丢包率测量）', 
#               fontsize=16, pad=20, fontweight='bold')
    
#     # 图例
#     plt.legend(fontsize=11, loc='upper right', framealpha=0.95)
    
#     # 网格
#     plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
#     # Y轴范围
#     plt.ylim(0, min(plt.ylim()[1] * 1.15, 100))  # 限制在100%以内
    
#     # 添加注释框（中文）
#     plt.text(0.98, 0.96, 'CV越低 → 稳定性越高\n(传输更可预测)', 
#             transform=plt.gca().transAxes, fontsize=11,
#             verticalalignment='top', horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.85, edgecolor='gray'))
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_2_Stability_Tradeoff.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成摘要
#     summary = {
#         'scenarios': stability_df['scenario'].nunique(),
#         'total_configs': len(stability_df),
#         'max_cv': stability_df['duration_cv'].max(),
#         'min_cv': stability_df['duration_cv'].min()
#     }
    
#     print(f"   ✅ 完成: {summary['total_configs']} 个配置的稳定性分析")
#     print(f"      CV范围: {summary['min_cv']:.1f}% - {summary['max_cv']:.1f}%")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_3_pareto_discrete(df):
#     """图5.3: 离散帕累托点（诚实展示，不强行平滑）"""
#     print("\n🎨 生成图5.3: 离散帕累托前沿...")
    
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
#     scenarios = [
#         {'name': 'IoT_Weak', 'size': 10, 'title': 'IoT弱网 (10MB)', 'color': COLORS['iot']},
#         {'name': 'Edge_Normal', 'size': 50, 'title': '边缘网络 (50MB)', 'color': COLORS['edge']},
#         {'name': 'Cloud_Fast', 'size': 100, 'title': '云环境 (100MB)', 'color': COLORS['cloud']}
#     ]
    
#     all_summaries = []
    
#     for ax, scenario in zip(axes, scenarios):
#         # 筛选数据：指定场景+文件大小+非baseline
#         subset = df[(df['network_type'] == scenario['name']) & 
#                    (df['file_size_mb'] == scenario['size']) & 
#                    (~df['is_baseline'])]
        
#         # 数据量检查
#         if len(subset) < 8:
#             ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
#                    ha='center', va='center', fontsize=13, color='gray',
#                    fontweight='bold')
#             ax.set_title(scenario['title'], fontsize=14, color='gray', fontweight='bold')
#             ax.set_xlabel('CPU成本 (秒)', fontsize=12)
#             ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
#             all_summaries.append({'scenario': scenario['name'], 'points': len(subset), 'pareto': 0})
#             continue
        
#         # 计算帕累托前沿
#         frontier = compute_pareto_frontier(subset)
        
#         # 绘制所有点（浅灰色背景）
#         ax.scatter(subset['cost_cpu_seconds'], subset['throughput_mbps'],
#                   c='#bdc3c7', s=50, alpha=0.65, edgecolors='none',
#                   label=f'全部配置 ({len(subset)})')
        
#         # 绘制帕累托点（大圆点，不连线！）
#         if len(frontier) >= 3:
#             ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                       c=scenario['color'], s=140, alpha=0.92, 
#                       edgecolors='black', linewidth=1.6, marker='o',
#                       label=f'帕累托最优 ({len(frontier)})', zorder=5)
#         elif len(frontier) > 0:
#             # 少量点用星号强调
#             ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                       c=scenario['color'], s=180, alpha=0.95, 
#                       edgecolors='black', linewidth=1.8, marker='*',
#                       label=f'帕累托点 ({len(frontier)})', zorder=5)
        
#         # 坐标轴（中文）
#         ax.set_xlabel('CPU成本 (秒)', fontsize=12)
#         ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
#         ax.set_title(scenario['title'], fontsize=14, fontweight='bold')
        
#         # 图例（右下角）
#         ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92)
        
#         # 标注数据量
#         ax.text(0.04, 0.96, f'n={len(subset)}', transform=ax.transAxes,
#                fontsize=11, verticalalignment='top', fontweight='bold',
#                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
#                          edgecolor=scenario['color'], alpha=0.85, linewidth=1.5))
        
#         # 保存场景摘要
#         all_summaries.append({
#             'scenario': scenario['name'],
#             'total_points': len(subset),
#             'pareto_points': len(frontier)
#         })
    
#     # 总标题（中文 + 诚实标注离散采样）
#     fig.suptitle('图5.3: 不同网络环境下的帕累托最优配置\n(离散采样 - 无插值平滑)', 
#                 fontsize=17, y=1.04, fontweight='bold')
#     plt.tight_layout()
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_3_Pareto_Discrete.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 打印摘要
#     print(f"   ✅ 完成: 3个场景的帕累托分析")
#     for summary in all_summaries:
#         print(f"      {summary['scenario']:15s}: 总点数={summary['total_points']:3d} | 帕累托点={summary['pareto_points']:2d}")
#     print(f"   📁 保存至: {output_path}")
    
#     return all_summaries

# def plot_fig_5_4_knee_points(df):
#     """图5.4: 帕累托前沿上的膝点选择（仅在前沿上计算）"""
#     print("\n🎨 生成图5.4: 帕累托膝点选择...")
    
#     # 选择数据最丰富的场景：Cloud_Fast 100MB
#     cloud_data = df[(df['network_type'] == 'Cloud_Fast') & 
#                    (df['file_size_mb'] == 100) & 
#                    (~df['is_baseline'])]
    
#     if len(cloud_data) < 10:
#         print(f"   ⚠️  Cloud_Fast 100MB数据不足 (n={len(cloud_data)} < 10)，跳过图5.4")
#         return {}
    
#     # 计算帕累托前沿
#     frontier = compute_pareto_frontier(cloud_data)
    
#     if len(frontier) < 2:  # 降低要求至2个点
#         print(f"   ⚠️  帕累托前沿点数不足 (n={len(frontier)} < 2)，跳过图5.4")
#         return {}
    
#     # 归一化（仅在前沿上）
#     c_min, c_max = frontier['cost_cpu_seconds'].min(), frontier['cost_cpu_seconds'].max()
#     t_min, t_max = frontier['throughput_mbps'].min(), frontier['throughput_mbps'].max()
    
#     frontier['c_norm'] = (frontier['cost_cpu_seconds'] - c_min) / max(c_max - c_min, 1e-8)
#     frontier['t_norm'] = (frontier['throughput_mbps'] - t_min) / max(t_max - t_min, 1e-8)
    
#     # 创建图表
#     plt.figure(figsize=(13, 8))
    
#     # 绘制所有实验点（浅灰色背景）
#     plt.scatter(cloud_data['cost_cpu_seconds'], cloud_data['throughput_mbps'],
#                c='#ecf0f1', s=45, alpha=0.5, edgecolors='none', label='全部配置')
    
#     # 绘制帕累托前沿（深灰色虚线 + 点）
#     if len(frontier) > 1:
#         plt.plot(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                 'k--', linewidth=2.2, alpha=0.7, label='帕累托前沿')
#     plt.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
#                c='black', s=90, alpha=0.85, zorder=5, edgecolors='white', linewidth=1.2)
    
#     # 定义膝点权重配置（仅3种清晰配置）
#     weights = [
#         {'name': '节能优先', 'wc': 0.8, 'wt': 0.2, 'color': COLORS['cloud'], 'marker': 'D'},
#         {'name': '平衡配置', 'wc': 0.5, 'wt': 0.5, 'color': COLORS['edge'], 'marker': 'o'},
#         {'name': '性能优先', 'wc': 0.2, 'wt': 0.8, 'color': COLORS['iot'], 'marker': '^'}
#     ]
    
#     knee_points = []
    
#     # 仅在帕累托前沿上计算膝点
#     for weight in weights:
#         # L2距离（归一化空间）
#         distances = np.sqrt(
#             weight['wc'] * frontier['c_norm']**2 + 
#             weight['wt'] * (1 - frontier['t_norm'])**2
#         )
#         best_idx = distances.idxmin()
#         best_point = frontier.loc[best_idx]
#         knee_points.append((best_point, weight))
        
#         # 绘制膝点
#         plt.scatter(best_point['cost_cpu_seconds'], best_point['throughput_mbps'],
#                    s=400, c=weight['color'], marker=weight['marker'], 
#                    edgecolors='black', linewidth=2.2, zorder=10,
#                    label=f"{weight['name']} (w_c={weight['wc']})")
        
#         # 标注配置信息（中文）
#         config_text = f"{int(best_point['threads'])}线程\n{best_point['cpu_quota']:.1f}核"
#         plt.annotate(config_text, 
#                     (best_point['cost_cpu_seconds'], best_point['throughput_mbps']),
#                     xytext=(28, 30), textcoords='offset points',
#                     fontsize=11, fontweight='bold', color=weight['color'],
#                     bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
#                              alpha=0.92, edgecolor=weight['color'], linewidth=1.8),
#                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', 
#                                   color=weight['color'], lw=2.0, alpha=0.8))
    
#     # 连接膝点展示偏好轨迹
#     if len(knee_points) > 1:
#         trajectory_costs = [kp[0]['cost_cpu_seconds'] for kp in knee_points]
#         trajectory_throughputs = [kp[0]['throughput_mbps'] for kp in knee_points]
#         plt.plot(trajectory_costs, trajectory_throughputs, 
#                 color=COLORS['pareto'], linestyle='-.', linewidth=3.0, alpha=0.85,
#                 marker='x', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
#                 label='偏好轨迹', zorder=6)
    
#     # 坐标轴（中文）
#     plt.xlabel('CPU成本 (秒)', fontsize=14)
#     plt.ylabel('吞吐量 (Mbps)', fontsize=14)
    
#     # 标题（中文 + 明确膝点计算位置）
#     plt.title('图5.4: 帕累托前沿上的膝点选择\n(仅在非支配解上计算)', 
#               fontsize=16, pad=22, fontweight='bold')
    
#     # 图例
#     plt.legend(loc='lower right', fontsize=11, framealpha=0.95, ncol=2)
    
#     # 网格
#     plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_4_Knee_Points.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 生成摘要
#     summary = {
#         'total_points': len(cloud_data),
#         'pareto_points': len(frontier),
#         'knee_points': len(knee_points),
#         'knee_configs': [(int(kp[0]['threads']), kp[0]['cpu_quota'], kp[1]['name']) 
#                         for kp in knee_points]
#     }
    
#     print(f"   ✅ 完成: Cloud_Fast 100MB场景")
#     print(f"      总配置数: {summary['total_points']}, 帕累托点: {summary['pareto_points']}, 膝点: {summary['knee_points']}")
#     for threads, cpu, name in summary['knee_configs']:
#         print(f"      {name:15s}: {threads}线程, {cpu:.1f}核")
#     print(f"   📁 保存至: {output_path}")
    
#     return summary

# def plot_fig_5_5_performance_gain(df):
#     """图5.5: 多场景性能提升对比（基于实测最优配置）"""
#     print("\n🎨 生成图5.5: 性能提升对比...")
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
#     # 定义对比场景
#     scenarios = [
#         {
#             'name': 'IoT_Weak', 
#             'size': 10,
#             'title': 'IoT弱网场景\n(10MB文件)',
#             'metric': 'throughput',
#             'ylabel': '吞吐量 (Mbps)',
#             'baseline_config': {'cpu_quota': 0.5, 'threads': 1},  # 保守配置
#             'optimal_config': {'cpu_quota': 2.0, 'threads': 16}   # 激进配置
#         },
#         {
#             'name': 'Cloud_Fast', 
#             'size': 100,
#             'title': '云环境场景\n(100MB文件)',
#             'metric': 'efficiency',
#             'ylabel': '资源效率 (MB/CPU·秒)',
#             'baseline_config': {'cpu_quota': 2.0, 'threads': 16},  # 高资源消耗
#             'optimal_config': {'cpu_quota': 0.5, 'threads': 4}      # 资源高效
#         }
#     ]
    
#     summaries = []
    
#     for idx, scenario in enumerate(scenarios):
#         ax = ax1 if idx == 0 else ax2
        
#         # 筛选场景数据
#         subset = df[(df['network_type'] == scenario['name']) & 
#                    (df['file_size_mb'] == scenario['size']) & 
#                    (~df['is_baseline'])]
        
#         if len(subset) < 10:
#             ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
#                    ha='center', va='center', fontsize=13, color='gray')
#             ax.set_title(scenario['title'], fontsize=14, color='gray')
#             summaries.append({'scenario': scenario['name'], 'valid': False})
#             continue
        
#         # 提取baseline配置性能
#         baseline_mask = (
#             (subset['cpu_quota'] == scenario['baseline_config']['cpu_quota']) & 
#             (subset['threads'] == scenario['baseline_config']['threads'])
#         )
#         baseline_data = subset[baseline_mask]
        
#         # 提取optimal配置性能
#         optimal_mask = (
#             (subset['cpu_quota'] == scenario['optimal_config']['cpu_quota']) & 
#             (subset['threads'] == scenario['optimal_config']['threads'])
#         )
#         optimal_data = subset[optimal_mask]
        
#         # 检查数据有效性
#         if len(baseline_data) == 0 or len(optimal_data) == 0:
#             ax.text(0.5, 0.5, '配置未找到', 
#                    ha='center', va='center', fontsize=13, color='gray')
#             ax.set_title(scenario['title'], fontsize=14, color='gray')
#             summaries.append({'scenario': scenario['name'], 'valid': False})
#             continue
        
#         # 计算指标
#         if scenario['metric'] == 'throughput':
#             baseline_val = baseline_data['throughput_mbps'].mean()
#             optimal_val = optimal_data['throughput_mbps'].mean()
#             improvement = ((optimal_val - baseline_val) / baseline_val) * 100
#             unit = 'Mbps'
#             higher_is_better = True
#         else:  # efficiency
#             baseline_val = baseline_data['efficiency_mb_per_cpus'].mean()
#             optimal_val = optimal_data['efficiency_mb_per_cpus'].mean()
#             improvement = ((optimal_val - baseline_val) / baseline_val) * 100
#             unit = 'MB/CPU·s'
#             higher_is_better = True
        
#         # 绘制柱状图
#         bars = ax.bar(
#             ['基线配置\n(保守)', '优化配置\n(自适应)'], 
#             [baseline_val, optimal_val],
#             color=[COLORS['baseline'], COLORS['cloud'] if idx==1 else COLORS['iot']], 
#             width=0.65, edgecolor='black', linewidth=1.5, alpha=0.9
#         )
        
#         # 标注数值
#         for i, (bar, value) in enumerate(zip(bars, [baseline_val, optimal_val])):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2, height + height*0.04,
#                    f'{value:.1f}\n{unit}', ha='center', va='bottom', 
#                    fontweight='bold', fontsize=12)
        
#         # 标注提升幅度
#         if abs(improvement) > 5:  # 显著提升才标注
#             sign = '+' if improvement > 0 else ''
#             color = '#27ae60' if improvement > 0 else '#e74c3c'
#             ax.text(1, optimal_val + optimal_val*0.12,
#                    f'{sign}{improvement:.0f}%\n提升', 
#                    ha='center', va='bottom', fontweight='bold', fontsize=13,
#                    color=color,
#                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
#                             alpha=0.9, edgecolor=color, linewidth=2))
        
#         # 坐标轴（中文）
#         ax.set_ylabel(scenario['ylabel'], fontsize=13)
#         ax.set_title(scenario['title'], fontsize=14, fontweight='bold', pad=15)
#         ax.set_ylim(0, max(baseline_val, optimal_val) * 1.35)
        
#         # 保存摘要
#         summaries.append({
#             'scenario': scenario['name'],
#             'baseline': baseline_val,
#             'optimal': optimal_val,
#             'improvement': improvement,
#             'unit': unit,
#             'valid': True
#         })
    
#     # 总标题（中文）
#     fig.suptitle('图5.5: 自适应配置的性能提升\n相比保守基线配置的实测改进', 
#                 fontsize=17, y=1.03, fontweight='bold')
#     plt.tight_layout()
    
#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "Fig_5_5_Performance_Gain.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # 打印摘要
#     print(f"   ✅ 完成: 2个场景的性能对比")
#     for summary in summaries:
#         if summary['valid']:
#             sign = '+' if summary['improvement'] > 0 else ''
#             print(f"      {summary['scenario']:15s}: {sign}{summary['improvement']:.1f}% "
#                   f"({summary['baseline']:.1f} → {summary['optimal']:.1f} {summary['unit']})")
#     print(f"   📁 保存至: {output_path}")
    
#     return summaries

# # ==============================================================================
# # 5. 主执行函数
# # ==============================================================================

# def main():
#     print("=" * 75)
#     print("🚀 创新点II可视化图表生成器 (中文修复版)")
#     print("Visualization for Innovation Point II: Multi-objective Optimization (Chinese)")
#     print("=" * 75)
    
#     # 确保有数据文件后再运行
#     if 'DATA_FILE' not in globals():
#         print("❌ 无法继续：未找到数据文件。")
#         return

#     print(f"📁 数据文件: {os.path.basename(DATA_FILE)}")
#     print(f"💾 输出目录: {OUTPUT_DIR}")
#     print("-" * 75)
    
#     # 加载并验证数据
#     df = load_and_validate_data()
#     if df is None:
#         print("\n❌ 数据加载失败，程序终止")
#         return
    
#     # 生成所有图表
#     summaries = {}
    
#     summaries['fig5_1'] = plot_fig_5_1_parameter_coverage(df)
#     summaries['fig5_2'] = plot_fig_5_2_stability_tradeoff(df)
#     summaries['fig5_3'] = plot_fig_5_3_pareto_discrete(df)
#     summaries['fig5_4'] = plot_fig_5_4_knee_points(df)
#     summaries['fig5_5'] = plot_fig_5_5_performance_gain(df)
    
#     # 生成综合报告
#     print("\n" + "=" * 75)
#     print("✅ 所有图表生成完成!")
#     print("=" * 75)
#     print(f"📁 输出目录: {OUTPUT_DIR}")
#     print("\n📊 生成的图表文件:")
    
#     generated_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
#     for i, file in enumerate(generated_files, 1):
#         print(f"   {i}. {file}")
    
#     # 生成数据摘要报告
#     print("\n📋 实验数据摘要:")
#     print(f"   总记录数: {len(df):,}")
#     print(f"   有效场景: {df['network_type'].nunique()}")
#     print(f"   文件大小变体: {sorted(df['file_size_mb'].unique())} MB")
#     print(f"   CPU配额范围: {df['cpu_quota'].min():.1f} - {df['cpu_quota'].max():.1f} 核")
#     print(f"   线程数范围: {df['threads'].min()} - {df['threads'].max()}")
    
#     # 保存摘要到JSON
#     summary_report = {
#         'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'data_file': os.path.basename(DATA_FILE),
#         'total_records': len(df),
#         'figures_generated': len(generated_files),
#         'figure_summaries': summaries
#     }
    
#     import json
#     summary_path = os.path.join(OUTPUT_DIR, "visualization_summary.json")
#     with open(summary_path, 'w', encoding='utf-8') as f:
#         json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
#     print(f"\n💾 摘要报告已保存: {summary_path}")
#     print("=" * 75)
#     print("\n💡 修复说明:")
#     print("   1. 已应用参考代码的字体适配方案（优先微软雅黑/PingFang/Droid Sans）")
#     print("   2. 已移除 'font.family': 'Times New Roman' 全局强制设置")
#     print("   3. 中文标签现在应该能正确显示，不再出现方块")
#     print("=" * 75)

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 绘图配置 (自动适配中文)
# ==============================================================================
system_name = platform.system()
if system_name == 'Windows':
    font_list = ['Microsoft YaHei', 'SimHei']
elif system_name == 'Darwin':
    font_list = ['Heiti TC', 'PingFang HK']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

matplotlib.rcParams['font.sans-serif'] = font_list + ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 全局配置
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 自动查找最新实验数据文件
# data_files = glob.glob(os.path.join(SCRIPT_DIR, "pareto_results*.csv")) + \
#              glob.glob(os.path.join(SCRIPT_DIR, "*cleaned*.csv"))

# if data_files:
#     DATA_FILE = max(data_files, key=os.path.getctime)
#     print(f"📊 检测到数据文件: {os.path.basename(DATA_FILE)} ({len(data_files)} 个候选)")
# else:
#     print("❌ 未找到实验数据文件 (pareto_results_*.csv)")
#     print("💡 请确保目录下存在数据文件，或修改脚本中的 DATA_FILE 路径")
#     # 创建一个示例数据用于测试
#     DATA_FILE = None

# 直接使用指定文件
DATA_FILE = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\cags_real_experiment\pareto_results_FINAL_CLEANED.csv"

# 验证文件存在
if not os.path.exists(DATA_FILE):
    print(f"❌ 数据文件不存在: {DATA_FILE}")
    exit(1)

print(f"📊 使用数据文件: {DATA_FILE}")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "innovation_ii_figures_chinese")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8
})

COLORS = {
    'iot': '#e74c3c',
    'edge': '#f39c12',
    'cloud': '#27ae60',
    'anchor': '#3498db',
    'probe_small': '#9b59b6',
    'probe_large': '#1abc9c',
    'baseline': '#95a5a6',
    'pareto': '#8e44ad'
}

# ==============================================================================
# 2. 数据加载与验证
# ==============================================================================

def load_and_validate_data():
    """加载数据并验证指标真实性"""
    try:
        if DATA_FILE is None or not os.path.exists(DATA_FILE):
            print("❌ 数据文件不存在")
            return None

        df = pd.read_csv(DATA_FILE)
        print(f"✅ 成功加载数据: {len(df)} 条记录")
        
        # 必需列验证
        required_cols = ['run_id', 'exp_type', 'file_size_mb', 'scenario', 
                         'cpu_quota', 'threads', 'chunk_kb', 'duration_s',
                         'throughput_mbps', 'cost_cpu_seconds', 'efficiency_mb_per_cpus', 'exit_code']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少必要列: {missing_cols}")
            return None
        
        # 数据清洗
        df = df.dropna(subset=['throughput_mbps', 'cost_cpu_seconds', 'duration_s'])
        df = df[df['exit_code'] == 0]
        df = df[df['duration_s'] > 0]
        
        # 场景标准化
        df['network_type'] = df['scenario'].str.replace('_BASELINE', '', regex=False)
        
        # 实验类型分类
        df['is_baseline'] = df['exp_type'].str.contains('baseline', case=False)
        df['is_anchor'] = df['exp_type'].str.contains('anchor', case=False) & ~df['is_baseline']
        df['is_probe_small'] = df['exp_type'] == 'probe_small'
        df['is_probe_large'] = df['exp_type'] == 'probe_large'
        
        # 数据质量报告
        print("\n📊 数据质量报告:")
        print(f"   总记录数: {len(df):,}")
        print(f"   Baseline记录: {df['is_baseline'].sum():,}")
        print(f"   Anchor实验: {df['is_anchor'].sum():,}")
        print(f"   Probe Small: {df['is_probe_small'].sum():,}")
        print(f"   Probe Large: {df['is_probe_large'].sum():,}")
        print(f"   吞吐量范围: {df['throughput_mbps'].min():.2f} - {df['throughput_mbps'].max():.2f} Mbps")
        print(f"   CPU成本范围: {df['cost_cpu_seconds'].min():.4f} - {df['cost_cpu_seconds'].max():.4f} s")
        
        # 场景分布
        print("\n🌐 场景分布:")
        for scenario, count in df['network_type'].value_counts().items():
            sizes = sorted(df[df['network_type'] == scenario]['file_size_mb'].unique())
            print(f"   {scenario:20s}: {count:4d} 条记录 | 文件大小: {sizes}")
        
        # 🔍 关键：验证各场景数据范围
        print("\n🔍 数据范围验证 (排除Baseline):")
        for (sc, sz), g in df.groupby(['network_type', 'file_size_mb']):
            g = g[~g['is_baseline']]
            if len(g) > 0:
                print(f"   {sc:15s} {sz:3d}MB: n={len(g):3d}, "
                      f"cost={g['cost_cpu_seconds'].min():.3f}-{g['cost_cpu_seconds'].max():.3f}, "
                      f"thr={g['throughput_mbps'].min():.1f}-{g['throughput_mbps'].max():.1f}")
        
        return df
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# 3. 帕累托前沿计算
# ==============================================================================

def compute_pareto_frontier(df, maximize_col='throughput_mbps', minimize_col='cost_cpu_seconds'):
    """标准帕累托前沿计算"""
    if len(df) == 0:
        return pd.DataFrame()
    
    pareto_points = []
    for idx, candidate in df.iterrows():
        is_dominated = False
        for _, other in df.iterrows():
            if (other[maximize_col] >= candidate[maximize_col] and 
                other[minimize_col] <= candidate[minimize_col] and
                (other[maximize_col] > candidate[maximize_col] or 
                 other[minimize_col] < candidate[minimize_col])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(candidate)
    
    pareto_df = pd.DataFrame(pareto_points)
    if not pareto_df.empty:
        pareto_df = pareto_df.sort_values(minimize_col).reset_index(drop=True)
    
    return pareto_df

# ==============================================================================
# 4. 图表生成函数
# ==============================================================================

def plot_fig_5_1_parameter_coverage(df):
    """图5.1: 参数空间覆盖"""
    print("\n🎨 生成图5.1: 参数空间覆盖...")
    
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    anchor = df[df['is_anchor']]
    probe_small = df[df['is_probe_small']]
    probe_large = df[df['is_probe_large']]
    
    all_throughput = pd.concat([anchor['throughput_mbps'], 
                                probe_small['throughput_mbps'],
                                probe_large['throughput_mbps']])
    norm = plt.Normalize(vmin=all_throughput.min(), vmax=all_throughput.max())
    cmap = plt.cm.viridis
    
    if len(anchor) > 0:
        colors = cmap(norm(anchor['throughput_mbps']))
        ax.scatter(anchor['threads'], anchor['cpu_quota'], anchor['chunk_kb']/1024,
                  c=colors, s=60, alpha=0.85, edgecolors='black', linewidth=0.6,
                  label='锚点 (系统扫描)', depthshade=True)
    
    if len(probe_small) > 0:
        colors = cmap(norm(probe_small['throughput_mbps']))
        ax.scatter(probe_small['threads'], probe_small['cpu_quota'], 
                  probe_small['chunk_kb']/1024,
                  c=colors, s=100, alpha=0.9, marker='^', edgecolors='black', linewidth=0.8,
                  label='探测点 (10MB)', depthshade=True)
    
    if len(probe_large) > 0:
        colors = cmap(norm(probe_large['throughput_mbps']))
        ax.scatter(probe_large['threads'], probe_large['cpu_quota'], 
                  probe_large['chunk_kb']/1024,
                  c=colors, s=100, alpha=0.9, marker='s', edgecolors='black', linewidth=0.8,
                  label='探测点 (300MB)', depthshade=True)
    
    ax.set_xlabel('并发线程数', fontsize=13, labelpad=12)
    ax.set_ylabel('CPU配额 (核)', fontsize=13, labelpad=12)
    ax.set_zlabel('分片大小 (KB)', fontsize=13, labelpad=12)
    ax.set_title('图5.1: 参数空间覆盖策略\n系统扫描与极端点探测', 
                fontsize=16, pad=18, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.view_init(elev=28, azim=38)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('吞吐量 (Mbps)', rotation=270, labelpad=22, fontsize=12)
    
    output_path = os.path.join(OUTPUT_DIR, "Fig_5_1_Parameter_Coverage.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    summary = {
        'anchor_points': len(anchor),
        'probe_small_points': len(probe_small),
        'probe_large_points': len(probe_large),
        'total_points': len(anchor) + len(probe_small) + len(probe_large)
    }
    
    print(f"   ✅ 完成: {summary['total_points']} 个配置点")
    print(f"   📁 保存至: {output_path}")
    
    return summary

def plot_fig_5_2_stability_tradeoff(df):
    """图5.2: 稳定性-性能权衡（修复：使用对数坐标避免视觉压缩）"""
    print("\n🎨 生成图5.2: 稳定性-性能权衡...")
    
    # 按场景+文件大小+exp_type分组计算稳定性（更细粒度）
    stability_data = []
    for (scenario, size, exp_type), group in df.groupby(['network_type', 'file_size_mb', 'exp_type']):
        group = group[~group['is_baseline']]
        
        if len(group) >= 2:  # 降低阈值到2个
            stability_data.append({
                'scenario': scenario,
                'file_size_mb': size,
                'exp_type': exp_type,
                'throughput_mean': group['throughput_mbps'].mean(),
                'throughput_std': group['throughput_mbps'].std(),
                'duration_mean': group['duration_s'].mean(),
                'duration_cv': (group['duration_s'].std() / 
                               max(group['duration_s'].mean(), 1e-6)) * 100,
                'sample_size': len(group)
            })
    
    if not stability_data:
        print("   ⚠️  警告: 无足够数据计算稳定性指标")
        return {}
    
    stability_df = pd.DataFrame(stability_data)
    
    # 🔧 修复：使用对数坐标避免视觉压缩
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    scenario_config = {
        'IoT_Weak': {'label': 'IoT弱网', 'color': COLORS['iot'], 'marker': 'o'},
        'Edge_Normal': {'label': '边缘网络', 'color': COLORS['edge'], 'marker': 's'},
        'Cloud_Fast': {'label': '云环境', 'color': COLORS['cloud'], 'marker': '^'}
    }
    
    # 左图：线性坐标（原始）
    for scenario_key, config in scenario_config.items():
        subset = stability_df[stability_df['scenario'] == scenario_key]
        if not subset.empty:
            ax1.scatter(subset['throughput_mean'], subset['duration_cv'],
                       c=config['color'], s=subset['sample_size']*20,
                       alpha=0.85, edgecolors='black', linewidth=1.3,
                       marker=config['marker'], label=f"{config['label']} (n={len(subset)})")
    
    ax1.axhline(y=30, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.85,
                label='稳定性阈值 (CV=30%)')
    ax1.set_xlabel('平均吞吐量 (Mbps) [线性]', fontsize=14)
    ax1.set_ylabel('传输时间变异系数 (CV, %)', fontsize=14)
    ax1.set_title('线性坐标 (注意：IoT/Edge被压缩)', fontsize=14)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.35)
    
    # 右图：对数坐标（修复）
    for scenario_key, config in scenario_config.items():
        subset = stability_df[stability_df['scenario'] == scenario_key]
        if not subset.empty:
            ax2.scatter(subset['throughput_mean'], subset['duration_cv'],
                       c=config['color'], s=subset['sample_size']*20,
                       alpha=0.85, edgecolors='black', linewidth=1.3,
                       marker=config['marker'], label=f"{config['label']} (n={len(subset)})")
    
    ax2.axhline(y=30, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.85,
                label='稳定性阈值 (CV=30%)')
    ax2.set_xscale('log')
    ax2.set_xlabel('平均吞吐量 (Mbps) [对数]', fontsize=14)
    ax2.set_ylabel('传输时间变异系数 (CV, %)', fontsize=14)
    ax2.set_title('对数坐标 (推荐：各场景清晰可见)', fontsize=14)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.35, which='both')
    
    fig.suptitle('图5.2: 稳定性-性能权衡分析\n基于实测传输时间变异系数', 
                fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "Fig_5_2_Stability_Tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    summary = {
        'scenarios': stability_df['scenario'].nunique(),
        'total_configs': len(stability_df),
        'max_cv': stability_df['duration_cv'].max(),
        'min_cv': stability_df['duration_cv'].min()
    }
    
    print(f"   ✅ 完成: {summary['total_configs']} 个配置的稳定性分析")
    print(f"      CV范围: {summary['min_cv']:.1f}% - {summary['max_cv']:.1f}%")
    print(f"   📁 保存至: {output_path}")
    
    return summary

def plot_fig_5_3_pareto_discrete(df):
    """图5.3: 离散帕累托点（修复：添加调试输出）"""
    print("\n🎨 生成图5.3: 离散帕累托前沿...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    scenarios = [
        {'name': 'IoT_Weak', 'size': 10, 'title': 'IoT弱网 (10MB)', 'color': COLORS['iot']},
        {'name': 'Edge_Normal', 'size': 50, 'title': '边缘网络 (50MB)', 'color': COLORS['edge']},
        {'name': 'Cloud_Fast', 'size': 100, 'title': '云环境 (100MB)', 'color': COLORS['cloud']}
    ]
    
    all_summaries = []
    
    for ax, scenario in zip(axes, scenarios):
        # 🔧 修复：明确调试输出
        print(f"\n   处理场景: {scenario['name']} ({scenario['size']}MB)")
        
        subset = df[(df['network_type'] == scenario['name']) & 
                   (df['file_size_mb'] == scenario['size']) & 
                   (~df['is_baseline'])]
        
        print(f"      筛选条件: network_type='{scenario['name']}', file_size_mb={scenario['size']}, is_baseline=False")
        print(f"      匹配记录: {len(subset)}")
        
        if len(subset) > 0:
            print(f"      成本范围: {subset['cost_cpu_seconds'].min():.4f} - {subset['cost_cpu_seconds'].max():.4f}")
            print(f"      吞吐范围: {subset['throughput_mbps'].min():.2f} - {subset['throughput_mbps'].max():.2f}")
        
        if len(subset) < 5:
            ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
                   ha='center', va='center', fontsize=13, color='gray',
                   fontweight='bold', transform=ax.transAxes)
            ax.set_title(scenario['title'], fontsize=14, color='gray', fontweight='bold')
            all_summaries.append({'scenario': scenario['name'], 'points': len(subset), 'pareto': 0})
            continue
        
        frontier = compute_pareto_frontier(subset)
        print(f"      帕累托前沿点: {len(frontier)}")
        
        # 绘制所有点
        ax.scatter(subset['cost_cpu_seconds'], subset['throughput_mbps'],
                  c='#bdc3c7', s=50, alpha=0.65, edgecolors='none',
                  label=f'全部配置 ({len(subset)})')
        
        # 绘制帕累托点
        if len(frontier) >= 2:
            ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
                      c=scenario['color'], s=140, alpha=0.92, 
                      edgecolors='black', linewidth=1.6, marker='o',
                      label=f'帕累托最优 ({len(frontier)})', zorder=5)
        elif len(frontier) > 0:
            ax.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
                      c=scenario['color'], s=180, alpha=0.95, 
                      edgecolors='black', linewidth=1.8, marker='*',
                      label=f'帕累托点 ({len(frontier)})', zorder=5)
        
        ax.set_xlabel('CPU成本 (秒)', fontsize=12)
        ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
        ax.set_title(scenario['title'], fontsize=14, fontweight='bold')
        ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92)
        
        # 标注数据量
        ax.text(0.04, 0.96, f'n={len(subset)}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=scenario['color'], alpha=0.85, linewidth=1.5))
        
        all_summaries.append({
            'scenario': scenario['name'],
            'total_points': len(subset),
            'pareto_points': len(frontier)
        })
    
    fig.suptitle('图5.3: 不同网络环境下的帕累托最优配置\n(离散采样 - 无插值平滑)', 
                fontsize=17, y=1.04, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "Fig_5_3_Pareto_Discrete.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n   ✅ 完成: 3个场景的帕累托分析")
    for summary in all_summaries:
        print(f"      {summary['scenario']:15s}: 总点数={summary['total_points']:3d} | 帕累托点={summary['pareto_points']:2d}")
    print(f"   📁 保存至: {output_path}")
    
    return all_summaries

def plot_fig_5_4_knee_points(df):
    """图5.4: 帕累托前沿上的膝点选择（修复：验证归一化和膝点计算）"""
    print("\n🎨 生成图5.4: 帕累托膝点选择...")
    
    cloud_data = df[(df['network_type'] == 'Cloud_Fast') & 
                   (df['file_size_mb'] == 100) & 
                   (~df['is_baseline'])]
    
    if len(cloud_data) < 10:
        print(f"   ⚠️  Cloud_Fast 100MB数据不足 (n={len(cloud_data)} < 10)")
        return {}
    
    frontier = compute_pareto_frontier(cloud_data)
    
    if len(frontier) < 2:
        print(f"   ⚠️  帕累托前沿点数不足 (n={len(frontier)} < 2)")
        return {}
    
    print(f"   帕累托前沿: {len(frontier)} 个点")
    print(f"   成本范围: {frontier['cost_cpu_seconds'].min():.4f} - {frontier['cost_cpu_seconds'].max():.4f}")
    print(f"   吞吐范围: {frontier['throughput_mbps'].min():.2f} - {frontier['throughput_mbps'].max():.2f}")
    
    # 🔧 修复：安全的归一化（防止除零）
    c_min, c_max = frontier['cost_cpu_seconds'].min(), frontier['cost_cpu_seconds'].max()
    t_min, t_max = frontier['throughput_mbps'].min(), frontier['throughput_mbps'].max()
    
    c_range = c_max - c_min
    t_range = t_max - t_min
    
    if c_range < 1e-9 or t_range < 1e-9:
        print("   ⚠️  数据范围太小，无法计算膝点")
        return {}
    
    frontier['c_norm'] = (frontier['cost_cpu_seconds'] - c_min) / c_range
    frontier['t_norm'] = (frontier['throughput_mbps'] - t_min) / t_range
    
    # 打印归一化后的前沿点
    print("\n   归一化帕累托前沿:")
    for idx, row in frontier.iterrows():
        print(f"      点{idx}: 成本={row['cost_cpu_seconds']:.4f}(norm={row['c_norm']:.3f}), "
              f"吞吐={row['throughput_mbps']:.2f}(norm={row['t_norm']:.3f}), "
              f"线程={int(row['threads'])}, 核={row['cpu_quota']:.1f}")
    
    plt.figure(figsize=(13, 8))
    
    # 绘制所有实验点
    plt.scatter(cloud_data['cost_cpu_seconds'], cloud_data['throughput_mbps'],
               c='#ecf0f1', s=45, alpha=0.5, edgecolors='none', label='全部配置')
    
    # 绘制帕累托前沿
    if len(frontier) > 1:
        plt.plot(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
                'k--', linewidth=2.2, alpha=0.7, label='帕累托前沿')
    plt.scatter(frontier['cost_cpu_seconds'], frontier['throughput_mbps'],
               c='black', s=90, alpha=0.85, zorder=5, edgecolors='white', linewidth=1.2)
    
    # 定义膝点权重
    weights = [
        {'name': '节能优先', 'wc': 0.8, 'wt': 0.2, 'color': COLORS['cloud'], 'marker': 'D'},
        {'name': '平衡配置', 'wc': 0.5, 'wt': 0.5, 'color': COLORS['edge'], 'marker': 'o'},
        {'name': '性能优先', 'wc': 0.2, 'wt': 0.8, 'color': COLORS['iot'], 'marker': '^'}
    ]
    
    knee_points = []
    
    print("\n   膝点计算:")
    for weight in weights:
        # L2距离（归一化空间）
        distances = np.sqrt(
            weight['wc'] * frontier['c_norm']**2 + 
            weight['wt'] * (1 - frontier['t_norm'])**2
        )
        best_idx = distances.idxmin()
        best_point = frontier.loc[best_idx]
        knee_points.append((best_point, weight))
        
        print(f"      {weight['name']}(w_c={weight['wc']}): "
              f"选中点成本={best_point['cost_cpu_seconds']:.4f}, "
              f"吞吐={best_point['throughput_mbps']:.2f}, "
              f"距离={distances.min():.4f}")
        
        # 绘制膝点
        plt.scatter(best_point['cost_cpu_seconds'], best_point['throughput_mbps'],
                   s=400, c=weight['color'], marker=weight['marker'], 
                   edgecolors='black', linewidth=2.2, zorder=10,
                   label=f"{weight['name']} (w_c={weight['wc']})")
        
        # 标注
        config_text = f"{int(best_point['threads'])}线程\n{best_point['cpu_quota']:.1f}核"
        plt.annotate(config_text, 
                    (best_point['cost_cpu_seconds'], best_point['throughput_mbps']),
                    xytext=(28, 30), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=weight['color'],
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                             alpha=0.92, edgecolor=weight['color'], linewidth=1.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', 
                                  color=weight['color'], lw=2.0, alpha=0.8))
    
    # 连接膝点
    if len(knee_points) > 1:
        trajectory_costs = [kp[0]['cost_cpu_seconds'] for kp in knee_points]
        trajectory_throughputs = [kp[0]['throughput_mbps'] for kp in knee_points]
        plt.plot(trajectory_costs, trajectory_throughputs, 
                color=COLORS['pareto'], linestyle='-.', linewidth=3.0, alpha=0.85,
                marker='x', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                label='偏好轨迹', zorder=6)
    
    plt.xlabel('CPU成本 (秒)', fontsize=14)
    plt.ylabel('吞吐量 (Mbps)', fontsize=14)
    plt.title('图5.4: 帕累托前沿上的膝点选择\n(仅在非支配解上计算)', 
              fontsize=16, pad=22, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95, ncol=2)
    plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
    output_path = os.path.join(OUTPUT_DIR, "Fig_5_4_Knee_Points.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    summary = {
        'total_points': len(cloud_data),
        'pareto_points': len(frontier),
        'knee_points': len(knee_points),
        'knee_configs': [(int(kp[0]['threads']), kp[0]['cpu_quota'], kp[1]['name']) 
                        for kp in knee_points]
    }
    
    print(f"\n   ✅ 完成: Cloud_Fast 100MB场景")
    for threads, cpu, name in summary['knee_configs']:
        print(f"      {name:15s}: {threads}线程, {cpu:.1f}核")
    print(f"   📁 保存至: {output_path}")
    
    return summary

def plot_fig_5_5_performance_gain(df):
    """图5.5: 多场景性能提升对比"""
    print("\n🎨 生成图5.5: 性能提升对比...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    scenarios = [
        {
            'name': 'IoT_Weak', 
            'size': 10,
            'title': 'IoT弱网场景\n(10MB文件)',
            'metric': 'throughput',
            'ylabel': '吞吐量 (Mbps)',
            'baseline_config': {'cpu_quota': 0.5, 'threads': 1},
            'optimal_config': {'cpu_quota': 2.0, 'threads': 4}  # 修正：IoT没有16线程数据
        },
        {
            'name': 'Cloud_Fast', 
            'size': 100,
            'title': '云环境场景\n(100MB文件)',
            'metric': 'efficiency',
            'ylabel': '资源效率 (MB/CPU·秒)',
            'baseline_config': {'cpu_quota': 2.0, 'threads': 16},
            'optimal_config': {'cpu_quota': 0.5, 'threads': 4}
        }
    ]
    
    summaries = []
    
    for idx, scenario in enumerate(scenarios):
        ax = ax1 if idx == 0 else ax2
        
        subset = df[(df['network_type'] == scenario['name']) & 
                   (df['file_size_mb'] == scenario['size']) & 
                   (~df['is_baseline'])]
        
        if len(subset) < 5:
            ax.text(0.5, 0.5, f'数据不足\n(n={len(subset)})', 
                   ha='center', va='center', fontsize=13, color='gray')
            ax.set_title(scenario['title'], fontsize=14, color='gray')
            summaries.append({'scenario': scenario['name'], 'valid': False})
            continue
        
        # 查找配置
        baseline_mask = (
            (subset['cpu_quota'] == scenario['baseline_config']['cpu_quota']) & 
            (subset['threads'] == scenario['baseline_config']['threads'])
        )
        baseline_data = subset[baseline_mask]
        
        optimal_mask = (
            (subset['cpu_quota'] == scenario['optimal_config']['cpu_quota']) & 
            (subset['threads'] == scenario['optimal_config']['threads'])
        )
        optimal_data = subset[optimal_mask]
        
        # 如果没找到精确匹配，找最接近的
        if len(baseline_data) == 0:
            # 找最低配额和线程的配置作为baseline
            baseline_data = subset.nsmallest(1, ['cpu_quota', 'threads'])
            print(f"   {scenario['name']}: 未找到精确baseline配置，使用最低配置替代")
        
        if len(optimal_data) == 0:
            # 找最高效率的配置
            optimal_data = subset.nlargest(1, 'efficiency_mb_per_cpus')
            print(f"   {scenario['name']}: 未找到精确optimal配置，使用最高效率替代")
        
        if len(baseline_data) == 0 or len(optimal_data) == 0:
            ax.text(0.5, 0.5, '配置未找到', 
                   ha='center', va='center', fontsize=13, color='gray')
            ax.set_title(scenario['title'], fontsize=14, color='gray')
            summaries.append({'scenario': scenario['name'], 'valid': False})
            continue
        
        # 计算指标
        if scenario['metric'] == 'throughput':
            baseline_val = baseline_data['throughput_mbps'].mean()
            optimal_val = optimal_data['throughput_mbps'].mean()
            unit = 'Mbps'
        else:
            baseline_val = baseline_data['efficiency_mb_per_cpus'].mean()
            optimal_val = optimal_data['efficiency_mb_per_cpus'].mean()
            unit = 'MB/CPU·s'
        
        improvement = ((optimal_val - baseline_val) / baseline_val) * 100
        
        # 绘制
        bars = ax.bar(
            ['基线配置\n(保守)', '优化配置\n(自适应)'], 
            [baseline_val, optimal_val],
            color=[COLORS['baseline'], COLORS['cloud'] if idx==1 else COLORS['iot']], 
            width=0.65, edgecolor='black', linewidth=1.5, alpha=0.9
        )
        
        for i, (bar, value) in enumerate(zip(bars, [baseline_val, optimal_val])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + height*0.04,
                   f'{value:.1f}\n{unit}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        if abs(improvement) > 1:
            sign = '+' if improvement > 0 else ''
            color = '#27ae60' if improvement > 0 else '#e74c3c'
            ax.text(1, optimal_val + optimal_val*0.15,
                   f'{sign}{improvement:.0f}%\n提升', 
                   ha='center', va='bottom', fontweight='bold', fontsize=12,
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                            alpha=0.9, edgecolor=color, linewidth=2))
        
        ax.set_ylabel(scenario['ylabel'], fontsize=13)
        ax.set_title(scenario['title'], fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, max(baseline_val, optimal_val) * 1.4)
        
        summaries.append({
            'scenario': scenario['name'],
            'baseline': baseline_val,
            'optimal': optimal_val,
            'improvement': improvement,
            'unit': unit,
            'valid': True
        })
    
    fig.suptitle('图5.5: 自适应配置的性能提升\n相比保守基线配置的实测改进', 
                fontsize=17, y=1.03, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "Fig_5_5_Performance_Gain.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ 完成: 性能对比")
    for summary in summaries:
        if summary['valid']:
            sign = '+' if summary['improvement'] > 0 else ''
            print(f"      {summary['scenario']:15s}: {sign}{summary['improvement']:.1f}% "
                  f"({summary['baseline']:.1f} → {summary['optimal']:.1f} {summary['unit']})")
    print(f"   📁 保存至: {output_path}")
    
    return summaries

# ==============================================================================
# 5. 主执行函数
# ==============================================================================

def main():
    print("=" * 75)
    print("🚀 创新点II可视化图表生成器 (修复版)")
    print("=" * 75)
    
    if DATA_FILE is None:
        print("❌ 无法继续：未找到数据文件。")
        return

    print(f"📁 数据文件: {os.path.basename(DATA_FILE)}")
    print(f"💾 输出目录: {OUTPUT_DIR}")
    print("-" * 75)
    
    df = load_and_validate_data()
    if df is None:
        print("\n❌ 数据加载失败，程序终止")
        return
    
    summaries = {}
    
    summaries['fig5_1'] = plot_fig_5_1_parameter_coverage(df)
    summaries['fig5_2'] = plot_fig_5_2_stability_tradeoff(df)
    summaries['fig5_3'] = plot_fig_5_3_pareto_discrete(df)
    summaries['fig5_4'] = plot_fig_5_4_knee_points(df)
    summaries['fig5_5'] = plot_fig_5_5_performance_gain(df)
    
    print("\n" + "=" * 75)
    print("✅ 所有图表生成完成!")
    print("=" * 75)
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("\n📊 生成的图表文件:")
    
    generated_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    for i, file in enumerate(generated_files, 1):
        print(f"   {i}. {file}")
    
    print("\n📋 实验数据摘要:")
    print(f"   总记录数: {len(df):,}")
    print(f"   有效场景: {df['network_type'].nunique()}")
    print(f"   文件大小变体: {sorted(df['file_size_mb'].unique())} MB")
    
    import json
    summary_report = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': os.path.basename(DATA_FILE) if DATA_FILE else None,
        'total_records': len(df),
        'figures_generated': len(generated_files),
        'figure_summaries': summaries
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "visualization_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 摘要报告已保存: {summary_path}")
    print("=" * 75)
    print("\n💡 修复内容:")
    print("   1. 图5.2: 添加对数坐标子图，避免IoT/Edge视觉压缩")
    print("   2. 图5.3: 添加详细调试输出，验证数据筛选")
    print("   3. 图5.4: 验证归一化计算，打印每个膝点的选择逻辑")
    print("   4. 图5.5: 添加配置回退机制，找不到精确匹配时选最接近的")
    print("=" * 75)

if __name__ == "__main__":
    main()