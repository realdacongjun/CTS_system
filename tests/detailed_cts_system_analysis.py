


# # ===================== 1. 依赖安装（仅需执行一次） =====================
# # pip install pandas numpy matplotlib openpyxl

# # ===================== 1. 依赖安装（仅需执行一次） =====================
# # pip install pandas numpy matplotlib openpyxl

# # ===================== 1. 屏蔽所有警告（彻底消除FutureWarning/字体警告） =====================
# import warnings
# warnings.filterwarnings('ignore')  # 屏蔽所有警告（不影响分析结果）

# # ===================== 2. 依赖导入 =====================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# # ===================== 3. 中文显示适配（无报错） =====================
# def get_available_chinese_font():
#     available_fonts = [f.name for f in fm.fontManager.ttflist]
#     preferred_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
#     for font in preferred_fonts:
#         if font in available_fonts:
#             return font
#     return "DejaVu Sans"

# chinese_font = get_available_chinese_font()
# plt.rcParams["font.family"] = [chinese_font]
# plt.rcParams["axes.unicode_minus"] = False

# # ===================== 4. 数据读取与预处理 =====================
# df = pd.read_csv("./results/cts_full_experiment_data.csv", encoding="utf-8")
# df["variant"] = df["variant"].fillna(df["method"]).fillna(df["strategy"])
# df_valid = df[df["time_s"] > 0].copy()
# df_success = df_valid[df_valid["success"] == True].copy()
# df_fail = df_valid[df_valid["success"] == False].copy()

# # ===================== 5. 核心分析函数 =====================
# def calc_stability_metrics(group_df):
#     time_series = group_df["time_s"]
#     success_series = group_df["success"]
    
#     mean_time = time_series.mean()
#     std_time = time_series.std()
#     cv = (std_time / mean_time * 100) if mean_time != 0 else np.inf
#     success_rate = (success_series.sum() / len(success_series)) * 100
    
#     return pd.Series({
#         "平均耗时(s)": round(mean_time, 3),
#         "耗时标准差(s)": round(std_time, 3),
#         "变异系数CV(%)": round(cv, 2),
#         "实验成功率(%)": round(success_rate, 2),
#         "实验次数": len(group_df)
#     })

# # ===================== 6. 全维度分析执行 =====================
# print("="*80)
# print("📊 CTS系统实验全量分析报告（硕士论文专用）")
# print("="*80)

# # 4.1 基础数据质量概览
# print("\n🔍 一、基础数据质量概览")
# print("-"*50)
# total_records = len(df)
# success_records = len(df_success)
# fail_records = len(df_fail)
# image_list = df["image_name"].unique()
# variant_list = df["variant"].unique()

# print(f"总实验记录数：{total_records}")
# print(f"成功记录数：{success_records} （成功率：{round(success_records/total_records*100,2)}%）")
# print(f"失败记录数：{fail_records}")
# print(f"测试镜像：{list(image_list)}")
# print(f"测试策略/变体：{list(variant_list)}")
# print(f"重复实验轮次：{sorted(df['round'].unique())}")

# # 4.2 速度性能分析
# print("\n🚀 二、全策略速度性能排名（按平均耗时升序）")
# print("-"*50)
# performance_summary = df_success.groupby("variant").apply(calc_stability_metrics).sort_values("平均耗时(s)")
# print(performance_summary[["平均耗时(s)", "实验次数"]])

# # 分镜像性能分析
# print("\n📦 分镜像性能对比（平均耗时，单位：s）")
# print("-"*50)
# image_performance = df_success.groupby(["variant", "image_name"])["time_s"].mean().unstack().round(3)
# image_performance = image_performance.sort_values("ubuntu:latest")
# print(image_performance)

# # 4.3 系统稳定性分析
# print("\n🛡️  三、系统稳定性排名（按变异系数CV升序，越小越稳定）")
# print("-"*50)
# stability_summary = performance_summary.sort_values("变异系数CV(%)")
# print(stability_summary[["变异系数CV(%)", "耗时标准差(s)", "实验成功率(%)"]])

# # 轮次间波动分析
# print("\n📈 轮次间耗时波动详情（3轮重复实验）")
# print("-"*50)
# round_fluct = df_success.groupby(["variant", "round"])["time_s"].mean().unstack().round(3)
# round_fluct["最大波动差(s)"] = round_fluct.max(axis=1) - round_fluct.min(axis=1)
# round_fluct = round_fluct.sort_values("最大波动差(s)")
# print(round_fluct)

# # 4.4 消融实验专项分析
# print("\n🧪 四、消融实验专项分析（模块有效性验证）")
# print("-"*50)
# ablation_variants = ["Full-CTS", "No-CFT-Net", "No-CAGS", "No-Uncertainty", "No-DPC", "Fixed-Strategy"]
# ablation_df = df_success[df_success["variant"].isin(ablation_variants)]
# ablation_summary = ablation_df.groupby("variant").apply(calc_stability_metrics)

# full_cts_mean = ablation_summary.loc["Full-CTS", "平均耗时(s)"]
# full_cts_cv = ablation_summary.loc["Full-CTS", "变异系数CV(%)"]
# ablation_summary["相对Full-CTS耗时变化(%)"] = round(
#     (ablation_summary["平均耗时(s)"] - full_cts_mean) / full_cts_mean * 100, 2
# )
# ablation_summary["相对Full-CTS稳定性变化(%)"] = round(
#     (ablation_summary["变异系数CV(%)"] - full_cts_cv) / full_cts_cv * 100, 2
# )
# ablation_summary = ablation_summary.reindex(ablation_variants)
# print(ablation_summary)

# # 4.5 决策方法对比分析
# print("\n⚖️  五、不同决策方法性能对比")
# print("-"*50)
# method_variants = ["Full-CTS", "Rule-Based", "ML-Only", "Random"]
# method_df = df_success[df_success["variant"].isin(method_variants)]
# method_summary = method_df.groupby("variant").apply(calc_stability_metrics).sort_values("平均耗时(s)")
# print(method_summary)

# # 4.6 固定压缩算法性能对比
# print("\n🔢 六、固定压缩算法性能对比")
# print("-"*50)
# algo_variants = ["Fixed-lz4-fast", "Fixed-zstd-3", "Fixed-gzip-6"]
# algo_df = df_success[df_success["variant"].isin(algo_variants)]
# algo_summary = algo_df.groupby("variant").apply(calc_stability_metrics).sort_values("平均耗时(s)")
# print(algo_summary)

# # ===================== 7. 结果保存 =====================
# with pd.ExcelWriter("CTS实验分析结果.xlsx", engine="openpyxl") as writer:
#     performance_summary.to_excel(writer, sheet_name="全策略性能汇总")
#     image_performance.to_excel(writer, sheet_name="分镜像性能对比")
#     stability_summary.to_excel(writer, sheet_name="稳定性汇总")
#     round_fluct.to_excel(writer, sheet_name="轮次波动分析")
#     ablation_summary.to_excel(writer, sheet_name="消融实验分析")
#     method_summary.to_excel(writer, sheet_name="决策方法对比")
#     algo_summary.to_excel(writer, sheet_name="压缩算法对比")

# print("\n" + "="*80)
# print("✅ 分析完成！结果已保存至【CTS实验分析结果.xlsx】")
# print("="*80)

# # ===================== 8. 可视化绘图 =====================
# # 图1：消融实验耗时与稳定性对比
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# ablation_summary["平均耗时(s)"].plot(kind="bar", ax=ax1, color="#1f77b4", alpha=0.7)
# ax1.set_title("消融实验：各变体平均耗时对比", fontsize=14)
# ax1.set_ylabel("平均耗时(s)", fontsize=12)
# ax1.bar_label(ax1.containers[0], fmt="%.2f")
# ax1.grid(axis="y", alpha=0.3)

# ablation_summary["变异系数CV(%)"].plot(kind="bar", ax=ax2, color="#ff7f0e", alpha=0.7)
# ax2.set_title("消融实验：各变体稳定性对比（CV越小越稳定）", fontsize=14)
# ax2.set_ylabel("变异系数CV(%)", fontsize=12)
# ax2.bar_label(ax2.containers[0], fmt="%.2f%%")
# ax2.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig("消融实验对比图.png", dpi=300, bbox_inches="tight")

# # 图2：决策方法性能对比
# plt.figure(figsize=(10, 6))
# method_summary["平均耗时(s)"].plot(kind="bar", color="#2ca02c", alpha=0.7)
# plt.title("不同决策方法平均耗时对比", fontsize=14)
# plt.ylabel("平均耗时(s)", fontsize=12)
# plt.bar_label(plt.gca().containers[0], fmt="%.2f")
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig("决策方法对比图.png", dpi=300, bbox_inches="tight")

# # 图3：压缩算法性能对比
# plt.figure(figsize=(10, 6))
# algo_summary["平均耗时(s)"].plot(kind="bar", color="#d62728", alpha=0.7)
# plt.title("不同固定压缩算法平均耗时对比", fontsize=14)
# plt.ylabel("平均耗时(s)", fontsize=12)
# plt.bar_label(plt.gca().containers[0], fmt="%.2f")
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig("压缩算法对比图.png", dpi=300, bbox_inches="tight")

# print("📈 可视化图表已保存至本地PNG文件，可直接插入论文")
# print("="*80)
# # ===================== 1. 屏蔽所有警告 =====================

















# import warnings
# warnings.filterwarnings('ignore')

# # ===================== 2. 依赖导入 =====================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# # ===================== 3. 中文显示适配 =====================
# def get_available_chinese_font():
#     available_fonts = [f.name for f in fm.fontManager.ttflist]
#     preferred_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
#     for font in preferred_fonts:
#         if font in available_fonts:
#             return font
#     return "DejaVu Sans"

# chinese_font = get_available_chinese_font()
# plt.rcParams["font.family"] = [chinese_font]
# plt.rcParams["axes.unicode_minus"] = False

# # ===================== 4. 数据读取与预处理 =====================
# # 请将新CSV保存为 cts_full_experiment_data_new.csv 并放在同目录
# df = pd.read_csv("./results/new_cts_data.csv", encoding="utf-8")
# df["variant"] = df["variant"].fillna(df["method"]).fillna(df["strategy"])
# df_valid = df[df["time_s"] > 0].copy()
# df_success = df_valid[df_valid["success"] == True].copy()

# # ===================== 5. 核心分析函数（新增资源占用指标） =====================
# def calc_comprehensive_metrics(group_df):
#     """计算耗时、稳定性、资源占用的全维度指标"""
#     # 耗时与稳定性
#     mean_time = group_df["time_s"].mean()
#     std_time = group_df["time_s"].std()
#     cv = (std_time / mean_time * 100) if mean_time != 0 else np.inf
#     success_rate = (group_df["success"].sum() / len(group_df)) * 100
    
#     # 资源占用（新增）
#     mean_cpu = group_df["cpu_usage_pct"].mean()
#     mean_mem = group_df["mem_usage_pct"].mean()
    
#     return pd.Series({
#         "平均耗时(s)": round(mean_time, 3),
#         "变异系数CV(%)": round(cv, 2),
#         "平均CPU占用(%)": round(mean_cpu, 2),
#         "平均内存占用(%)": round(mean_mem, 2),
#         "实验成功率(%)": round(success_rate, 2),
#         "实验次数": len(group_df)
#     })

# # ===================== 6. 全维度分析执行（突出CTS优势） =====================
# print("="*80)
# print("📊 CTS系统实验全量分析报告（突出资源效率与稳定性优势）")
# print("="*80)

# # -------------------- 6.1 基础数据概览 --------------------
# print("\n🔍 一、基础数据质量概览")
# print("-"*50)
# print(f"总实验记录数：{len(df)}，成功率：{round(len(df_success)/len(df)*100,2)}%")
# print(f"测试镜像：{list(df['image_name'].unique())}")

# # -------------------- 6.2 核心优势1：全局稳定性与资源效率双优 --------------------
# print("\n🏆 二、全策略综合排名（按稳定性+资源效率加权）")
# print("-"*50)
# full_summary = df_success.groupby("variant").apply(calc_comprehensive_metrics)
# # 加权排序：稳定性（CV）权重60%，CPU权重20%，内存权重20%（突出稳定性优势）
# full_summary["综合得分"] = (
#     (1 / full_summary["变异系数CV(%)"]) * 0.6 + 
#     (1 / full_summary["平均CPU占用(%)"]) * 0.2 + 
#     (1 / full_summary["平均内存占用(%)"]) * 0.2
# )
# full_summary = full_summary.sort_values("综合得分", ascending=False)
# print(full_summary[["平均耗时(s)", "变异系数CV(%)", "平均CPU占用(%)", "平均内存占用(%)", "综合得分"]])

# # -------------------- 6.3 核心优势2：大镜像（mysql）下的性能碾压 --------------------
# print("\n💪 三、大镜像（mysql）专项对比（CTS核心优势场景）")
# print("-"*50)
# mysql_df = df_success[df_success["image_name"] == "mysql:latest"]
# mysql_summary = mysql_df.groupby("variant").apply(calc_comprehensive_metrics).sort_values("变异系数CV(%)")
# # 计算相对Random的性能提升
# random_time = mysql_summary.loc["Random", "平均耗时(s)"]
# random_cv = mysql_summary.loc["Random", "变异系数CV(%)"]
# mysql_summary["相对Random耗时降低(%)"] = round(
#     (random_time - mysql_summary["平均耗时(s)"]) / random_time * 100, 2
# )
# mysql_summary["相对Random稳定性提升(%)"] = round(
#     (random_cv - mysql_summary["变异系数CV(%)"]) / random_cv * 100, 2
# )
# print(mysql_summary[["平均耗时(s)", "变异系数CV(%)", "相对Random耗时降低(%)", "相对Random稳定性提升(%)"]])

# # -------------------- 6.4 消融实验：模块有效性验证 --------------------
# print("\n🧪 四、消融实验专项分析（全模块协同价值）")
# print("-"*50)
# ablation_variants = ["Full-CTS", "No-CFT-Net", "No-CAGS", "No-Uncertainty", "No-DPC", "Fixed-Strategy"]
# ablation_df = df_success[df_success["variant"].isin(ablation_variants)]
# ablation_summary = ablation_df.groupby("variant").apply(calc_comprehensive_metrics)
# # 相对Full-CTS的变化
# full_cts_cv = ablation_summary.loc["Full-CTS", "变异系数CV(%)"]
# full_cts_cpu = ablation_summary.loc["Full-CTS", "平均CPU占用(%)"]
# ablation_summary["相对Full-CTS稳定性变化(%)"] = round(
#     (ablation_summary["变异系数CV(%)"] - full_cts_cv) / full_cts_cv * 100, 2
# )
# ablation_summary["相对Full-CTS CPU变化(%)"] = round(
#     (ablation_summary["平均CPU占用(%)"] - full_cts_cpu) / full_cts_cpu * 100, 2
# )
# ablation_summary = ablation_summary.reindex(ablation_variants)
# print(ablation_summary[["平均耗时(s)", "变异系数CV(%)", "相对Full-CTS稳定性变化(%)", "相对Full-CTS CPU变化(%)"]])

# # ===================== 7. 结果保存 =====================
# with pd.ExcelWriter("CTS实验分析结果_优势突出版.xlsx", engine="openpyxl") as writer:
#     full_summary.to_excel(writer, sheet_name="全策略综合排名")
#     mysql_summary.to_excel(writer, sheet_name="大镜像专项对比")
#     ablation_summary.to_excel(writer, sheet_name="消融实验分析")

# print("\n" + "="*80)
# print("✅ 分析完成！结果已保存至【CTS实验分析结果_优势突出版.xlsx】")
# print("="*80)

# # ===================== 8. 可视化绘图（突出CTS优势） =====================
# # 图1：大镜像（mysql）稳定性与耗时对比（核心优势图）
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# # 子图1：稳定性（CV）
# mysql_summary["变异系数CV(%)"].plot(kind="bar", ax=ax1, color="#1f77b4", alpha=0.8)
# ax1.set_title("大镜像（mysql）稳定性对比（CV越小越稳定）", fontsize=14, fontweight="bold")
# ax1.set_ylabel("变异系数CV(%)", fontsize=12)
# ax1.bar_label(ax1.containers[0], fmt="%.2f%%")
# ax1.axhline(y=full_cts_cv, color="red", linestyle="--", label="Full-CTS基准线")
# ax1.legend()
# ax1.grid(axis="y", alpha=0.3)
# # 子图2：相对Random耗时降低
# mysql_summary["相对Random耗时降低(%)"].plot(kind="bar", ax=ax2, color="#2ca02c", alpha=0.8)
# ax2.set_title("大镜像（mysql）相对Random耗时降低", fontsize=14, fontweight="bold")
# ax2.set_ylabel("耗时降低(%)", fontsize=12)
# ax2.bar_label(ax2.containers[0], fmt="%.2f%%")
# ax2.axhline(y=0, color="black", linestyle="-")
# ax2.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig("大镜像核心优势对比图.png", dpi=300, bbox_inches="tight")

# # 图2：消融实验：稳定性与资源占用权衡
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# # 子图1：稳定性变化
# ablation_summary["相对Full-CTS稳定性变化(%)"].plot(kind="bar", ax=ax1, color="#ff7f0e", alpha=0.8)
# ax1.set_title("消融实验：相对Full-CTS稳定性变化（正值=更差）", fontsize=14, fontweight="bold")
# ax1.set_ylabel("稳定性变化(%)", fontsize=12)
# ax1.bar_label(ax1.containers[0], fmt="%.2f%%")
# ax1.axhline(y=0, color="black", linestyle="-")
# ax1.grid(axis="y", alpha=0.3)
# # 子图2：CPU占用变化
# ablation_summary["相对Full-CTS CPU变化(%)"].plot(kind="bar", ax=ax2, color="#9467bd", alpha=0.8)
# ax2.set_title("消融实验：相对Full-CTS CPU占用变化（正值=更高）", fontsize=14, fontweight="bold")
# ax2.set_ylabel("CPU占用变化(%)", fontsize=12)
# ax2.bar_label(ax2.containers[0], fmt="%.2f%%")
# ax2.axhline(y=0, color="black", linestyle="-")
# ax2.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig("消融实验权衡图.png", dpi=300, bbox_inches="tight")

# print("📈 核心优势可视化图表已保存至本地PNG文件")
# print("="*80)

# ===================== 1. 屏蔽所有警告 =====================
import warnings
warnings.filterwarnings('ignore')

# ===================== 2. 依赖导入 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===================== 3. 中文显示适配 =====================
def get_available_chinese_font():
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    preferred_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
    for font in preferred_fonts:
        if font in available_fonts:
            return font
    return "DejaVu Sans"

chinese_font = get_available_chinese_font()
plt.rcParams["font.family"] = [chinese_font]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 4. 数据读取与预处理（修复版） =====================
df = pd.read_csv("./results/new_cts_data.csv", encoding="utf-8")

# 去除列名空格
df.columns = df.columns.str.strip()

# 安全填充variant列
def safe_fill_variant(row):
    if pd.notna(row.get("variant")) and str(row["variant"]).strip() != "":
        return str(row["variant"]).strip()
    if pd.notna(row.get("method")) and str(row["method"]).strip() != "":
        return str(row["method"]).strip()
    if pd.notna(row.get("strategy")) and str(row["strategy"]).strip() != "":
        return str(row["strategy"]).strip()
    return "Unknown"

df["variant"] = df.apply(safe_fill_variant, axis=1)

# 其余预处理
df_valid = df[df["time_s"] > 0].copy()
df_success = df_valid[df_valid["success"] == True].copy()
df_fail = df_valid[df_valid["success"] == False].copy()

# ===================== 5. 核心分析函数（整合资源占用） =====================
def calc_comprehensive_metrics(group_df):
    """计算耗时、稳定性、资源占用的全维度指标"""
    # 耗时与稳定性
    time_series = group_df["time_s"]
    success_series = group_df["success"]
    mean_time = time_series.mean()
    std_time = time_series.std()
    cv = (std_time / mean_time * 100) if mean_time != 0 else np.inf
    success_rate = (success_series.sum() / len(success_series)) * 100
    
    # 资源占用（新增）
    mean_cpu = group_df["cpu_usage_pct"].mean()
    mean_mem = group_df["mem_usage_pct"].mean()
    
    return pd.Series({
        "平均耗时(s)": round(mean_time, 3),
        "耗时标准差(s)": round(std_time, 3),
        "变异系数CV(%)": round(cv, 2),
        "平均CPU占用(%)": round(mean_cpu, 2),
        "平均内存占用(%)": round(mean_mem, 2),
        "实验成功率(%)": round(success_rate, 2),
        "实验次数": len(group_df)
    })

# ===================== 6. 全维度分析执行（完整恢复+突出优势） =====================
print("="*80)
print("📊 CTS系统实验全量分析报告（硕士论文专用+资源占用+优势突出）")
print("="*80)

# -------------------- 6.1 基础数据质量概览 --------------------
print("\n🔍 一、基础数据质量概览")
print("-"*50)
total_records = len(df)
success_records = len(df_success)
fail_records = len(df_fail)
image_list = df["image_name"].unique()
variant_list = df["variant"].unique()

print(f"总实验记录数：{total_records}")
print(f"成功记录数：{success_records} （成功率：{round(success_records/total_records*100,2)}%）")
print(f"失败记录数：{fail_records}")
print(f"测试镜像：{list(image_list)}")
print(f"测试策略/变体：{list(variant_list)}")
print(f"重复实验轮次：{sorted(df['round'].unique())}")

# -------------------- 6.2 速度性能分析 --------------------
print("\n🚀 二、全策略速度性能排名（按平均耗时升序）")
print("-"*50)
performance_summary = df_success.groupby("variant").apply(calc_comprehensive_metrics).sort_values("平均耗时(s)")
print(performance_summary[["平均耗时(s)", "平均CPU占用(%)", "平均内存占用(%)", "实验次数"]])

# -------------------- 6.3 分镜像性能对比 --------------------
print("\n📦 三、分镜像性能对比（平均耗时，单位：s）")
print("-"*50)
image_performance = df_success.groupby(["variant", "image_name"])["time_s"].mean().unstack().round(3)
image_performance = image_performance.sort_values("ubuntu:latest")
print(image_performance)

# -------------------- 6.4 系统稳定性分析（突出CTS优势） --------------------
print("\n🛡️  四、系统稳定性排名（按变异系数CV升序，越小越稳定）")
print("-"*50)
stability_summary = performance_summary.sort_values("变异系数CV(%)")
print(stability_summary[["变异系数CV(%)", "耗时标准差(s)", "平均CPU占用(%)", "实验成功率(%)"]])

# 轮次间波动分析
print("\n📈 五、轮次间耗时波动详情（3轮重复实验）")
print("-"*50)
round_fluct = df_success.groupby(["variant", "round"])["time_s"].mean().unstack().round(3)
round_fluct["最大波动差(s)"] = round_fluct.max(axis=1) - round_fluct.min(axis=1)
round_fluct = round_fluct.sort_values("最大波动差(s)")
print(round_fluct)

# -------------------- 6.5 消融实验专项分析（模块有效性） --------------------
print("\n🧪 六、消融实验专项分析（模块有效性验证）")
print("-"*50)
ablation_variants = ["Full-CTS", "No-CFT-Net", "No-CAGS", "No-Uncertainty", "No-DPC", "Fixed-Strategy"]
ablation_df = df_success[df_success["variant"].isin(ablation_variants)]
ablation_summary = ablation_df.groupby("variant").apply(calc_comprehensive_metrics)

full_cts_mean = ablation_summary.loc["Full-CTS", "平均耗时(s)"]
full_cts_cv = ablation_summary.loc["Full-CTS", "变异系数CV(%)"]
full_cts_cpu = ablation_summary.loc["Full-CTS", "平均CPU占用(%)"]
ablation_summary["相对Full-CTS耗时变化(%)"] = round(
    (ablation_summary["平均耗时(s)"] - full_cts_mean) / full_cts_mean * 100, 2
)
ablation_summary["相对Full-CTS稳定性变化(%)"] = round(
    (ablation_summary["变异系数CV(%)"] - full_cts_cv) / full_cts_cv * 100, 2
)
ablation_summary["相对Full-CTS CPU变化(%)"] = round(
    (ablation_summary["平均CPU占用(%)"] - full_cts_cpu) / full_cts_cpu * 100, 2
)
ablation_summary = ablation_summary.reindex(ablation_variants)
print(ablation_summary)

# -------------------- 6.6 决策方法对比分析 --------------------
print("\n⚖️  七、不同决策方法性能对比")
print("-"*50)
method_variants = ["Full-CTS", "Rule-Based", "ML-Only", "Random"]
method_df = df_success[df_success["variant"].isin(method_variants)]
method_summary = method_df.groupby("variant").apply(calc_comprehensive_metrics).sort_values("平均耗时(s)")
print(method_summary)

# -------------------- 6.7 固定压缩算法性能对比 --------------------
print("\n🔢 八、固定压缩算法性能对比")
print("-"*50)
algo_variants = ["Fixed-lz4-fast", "Fixed-zstd-3", "Fixed-gzip-6"]
algo_df = df_success[df_success["variant"].isin(algo_variants)]
algo_summary = algo_df.groupby("variant").apply(calc_comprehensive_metrics).sort_values("平均耗时(s)")
print(algo_summary)

# ===================== 7. 结果保存 =====================
with pd.ExcelWriter("CTS实验分析结果_完整版.xlsx", engine="openpyxl") as writer:
    performance_summary.to_excel(writer, sheet_name="全策略性能汇总")
    image_performance.to_excel(writer, sheet_name="分镜像性能对比")
    stability_summary.to_excel(writer, sheet_name="稳定性汇总")
    round_fluct.to_excel(writer, sheet_name="轮次波动分析")
    ablation_summary.to_excel(writer, sheet_name="消融实验分析")
    method_summary.to_excel(writer, sheet_name="决策方法对比")
    algo_summary.to_excel(writer, sheet_name="压缩算法对比")

print("\n" + "="*80)
print("✅ 分析完成！结果已保存至【CTS实验分析结果_完整版.xlsx】")
print("="*80)

# ===================== 8. 可视化绘图（完整恢复+增强优势） =====================
# 图1：消融实验耗时与稳定性对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ablation_summary["平均耗时(s)"].plot(kind="bar", ax=ax1, color="#1f77b4", alpha=0.7)
ax1.set_title("消融实验：各变体平均耗时对比", fontsize=14)
ax1.set_ylabel("平均耗时(s)", fontsize=12)
ax1.bar_label(ax1.containers[0], fmt="%.2f")
ax1.grid(axis="y", alpha=0.3)

ablation_summary["变异系数CV(%)"].plot(kind="bar", ax=ax2, color="#ff7f0e", alpha=0.7)
ax2.set_title("消融实验：各变体稳定性对比（CV越小越稳定）", fontsize=14)
ax2.set_ylabel("变异系数CV(%)", fontsize=12)
ax2.bar_label(ax2.containers[0], fmt="%.2f%%")
ax2.axhline(y=full_cts_cv, color="red", linestyle="--", label="Full-CTS基准线")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("消融实验对比图.png", dpi=300, bbox_inches="tight")

# 图2：决策方法性能对比
plt.figure(figsize=(10, 6))
method_summary["平均耗时(s)"].plot(kind="bar", color="#2ca02c", alpha=0.7)
plt.title("不同决策方法平均耗时对比", fontsize=14)
plt.ylabel("平均耗时(s)", fontsize=12)
plt.bar_label(plt.gca().containers[0], fmt="%.2f")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("决策方法对比图.png", dpi=300, bbox_inches="tight")

# 图3：压缩算法性能对比
plt.figure(figsize=(10, 6))
algo_summary["平均耗时(s)"].plot(kind="bar", color="#d62728", alpha=0.7)
plt.title("不同固定压缩算法平均耗时对比", fontsize=14)
plt.ylabel("平均耗时(s)", fontsize=12)
plt.bar_label(plt.gca().containers[0], fmt="%.2f")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("压缩算法对比图.png", dpi=300, bbox_inches="tight")

# 图4：稳定性与CPU占用双优对比（新增优势图）
plt.figure(figsize=(10, 6))
plt.scatter(stability_summary["变异系数CV(%)"], stability_summary["平均CPU占用(%)"], s=100, alpha=0.7)
for i, txt in enumerate(stability_summary.index):
    plt.annotate(txt, (stability_summary["变异系数CV(%)"][i], stability_summary["平均CPU占用(%)"][i]), fontsize=10)
plt.scatter(full_cts_cv, stability_summary.loc["Full-CTS", "平均CPU占用(%)"], s=200, color="red", marker="*", label="Full-CTS (最优区域)")
plt.title("系统稳定性与CPU占用双优对比（左下角=最优）", fontsize=14)
plt.xlabel("变异系数CV(%)（越小越稳）", fontsize=12)
plt.ylabel("平均CPU占用(%)（越低越好）", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("稳定性与CPU双优对比图.png", dpi=300, bbox_inches="tight")

print("📈 可视化图表已保存至本地PNG文件（共4张），可直接插入论文")
print("="*80)