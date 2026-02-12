import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import warnings
import sys
import os
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
import platform

# --- 🚀 核心修复代码开始 ---
system_name = platform.system()
if system_name == 'Windows':
    # Windows 优先用微软雅黑，保底用黑体
    font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
elif system_name == 'Darwin':
    # Mac OS 优先用黑体-简
    font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
else:
    # Linux (Docker/Ubuntu) 通常没有微软字体，优先用 WenQuanYi 或 Droid Sans Fallback
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']

# 这一行是魔法：自动寻找系统里存在的第一个中文字体
matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题
# --- 🚀 核心修复代码结束 ---

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml_training.modeling.train import CTSDualTowerModel, TransformerTower, FeatureTokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FairComparisonEvaluator:
    """公平对比评估器 - 使用完整算法集进行对比"""
    
    def __init__(self):
        self.model = None
        self.scaler_c = StandardScaler()
        self.scaler_i = StandardScaler()
        self.enc_algo = LabelEncoder()
        self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        self.col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
        
    def load_existing_model(self):
        """加载已训练的CFT-Net模型（使用完整10种算法）"""
        print("加载现有的CFT-Net模型（10种算法版本）...")
        
        # 模型路径
        model_path = os.path.join('..', 'modeling', 'cts_best_model_full_modified.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到预训练模型: {model_path}")
        
        # 初始化模型（使用训练时的完整参数）
        self.model = CTSDualTowerModel(
            client_feats=len(self.col_client),
            image_feats=len(self.col_image),
            num_algos=10,  # 使用完整的10种算法
            embed_dim=32
        )
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ 成功加载CFT-Net模型（10种算法）")
    
    def load_real_training_data(self):
        """加载真实的训练数据"""
        print("加载真实的cts_data.xlsx训练数据...")
        
        # 数据路径
        data_path = os.path.join('..', 'modeling', 'cts_data.xlsx')
        feature_path = os.path.join('..', 'modeling', 'image_features_database.csv')
        
        # 读取数据
        df_exp = pd.read_excel(data_path)
        df_feat = pd.read_csv(feature_path)
        
        # 数据预处理（与训练时保持一致）
        rename_map = {
            "image": "image_name", "method": "algo_name",
            "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
            "mem_limit": "mem_limit_mb"
        }
        df_exp = df_exp.rename(columns=rename_map)
        
        if 'total_time' not in df_exp.columns:
            possible_cols = [c for c in df_exp.columns if 'total_tim' in c]
            if possible_cols: 
                df_exp = df_exp.rename(columns={possible_cols[0]: 'total_time'})
        
        # 过滤有效数据
        df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
        
        if 'mem_limit_mb' not in df_exp.columns: 
            df_exp['mem_limit_mb'] = 1024.0
        
        # 合并特征数据
        df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        print(f"✅ 加载数据完成，样本数: {len(df)}")
        
        # 显示算法分布
        print("\n算法分布分析:")
        algo_counts = df['algo_name'].value_counts()
        total_samples = len(df)
        for algo, count in algo_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {algo:15s}: {count:4d} 样本 ({percentage:5.1f}%)")
        
        # 显示性能统计
        print("\n各算法性能统计:")
        algo_stats = df.groupby('algo_name')['total_time'].agg(['mean', 'std', 'count'])
        algo_stats = algo_stats.sort_values('mean')
        for algo in algo_stats.index:
            mean_time = algo_stats.loc[algo, 'mean']
            std_time = algo_stats.loc[algo, 'std']
            count = algo_stats.loc[algo, 'count']
            print(f"  {algo:15s}: 平均 {mean_time:6.2f}s ± {std_time:5.2f}s (n={count})")
        
        return df
    
    def prepare_features(self, df):
        """准备特征数据"""
        print("准备特征数据...")
        
        # 特征列
        col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
        
        # 标准化处理（与训练时保持一致）
        X_client = self.scaler_c.fit_transform(df[col_client].values)
        X_client = self.scaler_c.fit_transform(df[col_client].values)
        X_image = self.scaler_i.fit_transform(df[col_image].values)
        X_algo = self.enc_algo.fit_transform(df['algo_name'].values)
        
        # 目标值处理 - 应用log变换处理长尾分布
        y_original = df['total_time'].values
        y_log_transformed = np.log1p(y_original)  # log(1+x)变换
        
        print(f"目标值分布统计:")
        print(f"  原始值范围: {y_original.min():.2f} - {y_original.max():.2f} 秒")
        print(f"  原始值均值: {y_original.mean():.2f} ± {y_original.std():.2f} 秒")
        print(f"  变异系数: {y_original.std()/y_original.mean():.3f}")
        print(f"  Log变换后范围: {y_log_transformed.min():.2f} - {y_log_transformed.max():.2f}")
        print(f"  Log变换后均值: {y_log_transformed.mean():.2f} ± {y_log_transformed.std():.2f}")
        
        return X_client, X_image, X_algo, y_log_transformed, y_original
    
    def train_all_models_on_same_data(self, df):
        """在相同数据上训练所有模型进行公平对比"""
        print("=== 在相同真实数据上训练所有模型 ===")
        
        # 准备特征
        X_client, X_image, X_algo, y_log, y_orig = self.prepare_features(df)
        
        # 分割训练测试集
        split_idx = int(len(df) * 0.8)
        X_train = (X_client[:split_idx], X_image[:split_idx], X_algo[:split_idx])
        X_test = (X_client[split_idx:], X_image[split_idx:], X_algo[split_idx:])
        y_train_orig = y_orig[:split_idx]
        y_test_orig = y_orig[split_idx:]
        y_train_log = y_log[:split_idx]
        y_test_log = y_log[split_idx:]
        
        # 准备合并特征用于传统机器学习模型
        X_train_combined = np.hstack([
            X_train[0],  # 客户端特征
            X_train[1],  # 镜像特征
            X_train[2].reshape(-1, 1)  # 算法特征
        ])
        X_test_combined = np.hstack([
            X_test[0],
            X_test[1], 
            X_test[2].reshape(-1, 1)
        ])
        
        # 处理数据中的无效值
        X_train_combined = np.nan_to_num(X_train_combined, nan=0.0)
        X_test_combined = np.nan_to_num(X_test_combined, nan=0.0)
        y_train_log = np.nan_to_num(y_train_log, nan=np.median(y_train_log))
        y_test_log = np.nan_to_num(y_test_log, nan=np.median(y_test_log))
        
        results = {}
        
        # 1. 训练线性回归（在log空间训练）
        print("训练 Linear Regression (log-space)...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_combined, y_train_log)
        lr_pred_log = lr_model.predict(X_test_combined)
        lr_pred_log = np.clip(lr_pred_log, 0.1, np.log1p(1200.0))  # 限制在合理范围内
        lr_pred_orig = np.expm1(lr_pred_log)  # 转换回原始尺度
        results['Linear Regression'] = {
            'model': lr_model,
            'predictions': lr_pred_orig,
            'rmse': np.sqrt(mean_squared_error(y_test_orig, lr_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, lr_pred_orig),
            'r2': r2_score(y_test_orig, lr_pred_orig)
        }
        
        # 2. 训练随机森林（在log空间训练）
        print("训练 Random Forest (log-space)...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_combined, y_train_log)
        rf_pred_log = rf_model.predict(X_test_combined)
        rf_pred_log = np.clip(rf_pred_log, 0.1, np.log1p(1200.0))
        rf_pred_orig = np.expm1(rf_pred_log)
        results['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred_orig,
            'rmse': np.sqrt(mean_squared_error(y_test_orig, rf_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, rf_pred_orig),
            'r2': r2_score(y_test_orig, rf_pred_orig)
        }
        
        # 3. 训练梯度提升（在log空间训练）
        print("训练 Gradient Boosting (log-space)...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_combined, y_train_log)
        gb_pred_log = gb_model.predict(X_test_combined)
        gb_pred_log = np.clip(gb_pred_log, 0.1, np.log1p(1200.0))
        gb_pred_orig = np.expm1(gb_pred_log)
        results['Gradient Boosting'] = {
            'model': gb_model,
            'predictions': gb_pred_orig,
            'rmse': np.sqrt(mean_squared_error(y_test_orig, gb_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, gb_pred_orig),
            'r2': r2_score(y_test_orig, gb_pred_orig)
        }
        
        # 4. 使用预训练的CFT-Net（完整10种算法）- 注意：CFT-Net已经在log空间训练
        print("使用预训练的 CFT-Net（10种算法）...")
        cftnet_pred = self.predict_with_cftnet_full_algorithms(X_test[0], X_test[1], X_test[2])
        results['CFT-Net (10 algorithms)'] = {
            'predictions': cftnet_pred,
            'rmse': np.sqrt(mean_squared_error(y_test_orig, cftnet_pred)),
            'mae': mean_absolute_error(y_test_orig, cftnet_pred),
            'r2': r2_score(y_test_orig, cftnet_pred)
        }
        
        return results, y_test_orig
    
    def predict_with_cftnet_full_algorithms(self, X_client, X_image, X_algo):
        """使用CFT-Net进行预测（使用完整10种算法）"""
        # 转换为torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        cx = torch.FloatTensor(X_client).to(device)
        ix = torch.FloatTensor(X_image).to(device)
        ax = torch.LongTensor(X_algo).to(device)
        
        # 预测
        with torch.no_grad():
            preds = self.model(cx, ix, ax)
            gamma = preds[:, 0]  # 只需要gamma作为预测值
            
        # 转换回原始尺度
        predictions = np.expm1(gamma.cpu().numpy())
        predictions = np.nan_to_num(predictions, nan=np.median(predictions))
        predictions = np.clip(predictions, 0.1, 1200.0)
        
        return predictions
    
    def generate_comparison_table(self, results):
        """生成模型性能对比表格"""
        print("\n表3.1 模型预测性能对比（基于真实数据，完整算法集）")
        print("=" * 75)
        print(f"{'模型':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("=" * 75)
        
        # 找到最好的基线模型用于比较
        baseline_models = {k: v for k, v in results.items() if 'CFT-Net' not in k}
        best_baseline = min(baseline_models.items(), key=lambda x: x[1]['rmse'])
        best_baseline_rmse = best_baseline[1]['rmse']
        best_baseline_name = best_baseline[0]
        
        for name, result in results.items():
            improvement = ""
            if 'CFT-Net' in name:
                improvement_direction = "提升" if result['rmse'] < best_baseline_rmse else "下降"
                improvement_percent = abs((best_baseline_rmse - result['rmse'])/best_baseline_rmse*100)
                improvement = f" (相比{best_baseline_name}{'提升' if result['rmse'] < best_baseline_rmse else '下降'} {improvement_percent:.1f}%)"
            
            print(f"{name:<25} {result['rmse']:<12.4f} {result['mae']:<12.4f} {result['r2']:<12.4f}{improvement}")
        
        print("=" * 75)
        
        # 保存为CSV
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'R2': result['r2']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv('table_3_1_model_comparison.csv', index=False)
    
    def generate_prediction_scatter_plots(self, results, y_true):
        """生成预测值vs真实值散点图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('图3.4 模型预测准确性对比（完整算法集）', fontsize=16, fontweight='bold')
        
        # 使用实际存在的模型
        available_models = [name for name in ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'CFT-Net (10 algorithms)'] if name in results]
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for i, model in enumerate(available_models[:4]):  # 最多显示4个模型
            if i < len(positions):
                row, col = positions[i]
                ax = axes[row, col]
                y_pred = results[model]['predictions']
                
                # 绘制散点图
                ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                
                # 绘制完美预测线
                min_val = min(min(y_true), min(y_pred))
                max_val = max(max(y_true), max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                # 计算指标
                rmse = results[model]['rmse']
                r2 = results[model]['r2']
                
                ax.set_xlabel('真实传输时间 (秒)')
                ax.set_ylabel('预测传输时间 (秒)')
                ax.set_title(f'{model}\nRMSE={rmse:.3f}, R²={r2:.3f}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_3_4_prediction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_stats(self, results):
        """生成性能统计摘要"""
        cftnet_result = results['CFT-Net (10 algorithms)']
        
        # 找到最好的基线模型
        baseline_models = {k: v for k, v in results.items() if 'CFT-Net' not in k}
        best_baseline = min(baseline_models.items(), key=lambda x: x[1]['rmse'])
        best_baseline_result = best_baseline[1]
        best_baseline_name = best_baseline[0]
        
        rmse_improvement = (best_baseline_result['rmse'] - cftnet_result['rmse']) / best_baseline_result['rmse'] * 100
        r2_value = cftnet_result['r2'] * 100
        
        print(f"\n=== 第三章关键统计 ===")
        improvement_word = "提升" if rmse_improvement > 0 else "下降"
        print(f"CFT-Net(10算法)的RMSE为 {cftnet_result['rmse']:.3f}，相比{best_baseline_name}({best_baseline_result['rmse']:.3f}){improvement_word}了 {abs(rmse_improvement):.1f}%。")
        print(f"R²达到 {r2_value:.1f}%，说明模型解释了{r2_value:.1f}%的性能波动。")
        
        # 保存统计结果
        stats = {
            'cftnet_rmse': cftnet_result['rmse'],
            'best_baseline_rmse': best_baseline_result['rmse'],
            'best_baseline_name': best_baseline_name,
            'rmse_improvement_percent': rmse_improvement,
            'cftnet_r2_percent': r2_value
        }
        
        with open('chapter3_2_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats

def main():
    """主函数"""
    print("=== 第三章实验：基于真实数据的公平模型对比（完整算法集）===")
    
    # 初始化评估器
    evaluator = FairComparisonEvaluator()
    
    # 加载现有模型
    evaluator.load_existing_model()
    
    # 加载真实训练数据
    df = evaluator.load_real_training_data()
    
    # 在相同数据上训练所有模型进行公平对比
    results, y_test = evaluator.train_all_models_on_same_data(df)
    
    # 生成对比表格
    evaluator.generate_comparison_table(results)
    
    # 生成散点图
    evaluator.generate_prediction_scatter_plots(results, y_test)
    
    # 生成统计摘要
    stats = evaluator.generate_performance_stats(results)
    
    print(f"\n=== 实验完成 ===")
    print("生成的文件:")
    print("- table_3_1_model_comparison.csv: 模型性能对比表")
    print("- figure_3_4_prediction_accuracy.png: 预测准确性对比图")
    print("- chapter3_2_statistics.json: 性能提升统计")

if __name__ == "__main__":
    main()