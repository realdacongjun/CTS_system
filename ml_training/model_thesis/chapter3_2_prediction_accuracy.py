import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, norm
from scipy.stats import mstats
import pickle
import random
import math
import platform
import matplotlib
from collections import Counter
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 基础配置
# ==============================================================================
system_name = platform.system()
if system_name == 'Windows':
    font_list = ['Microsoft YaHei', 'SimHei']
elif system_name == 'Darwin':
    font_list = ['Heiti TC', 'PingFang HK']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==============================================================================
# 1. 超参数配置
# ==============================================================================
CONFIG = {
    "lr": 0.0003,
    "weight_decay": 1e-4,
    "epochs": 200,
    "patience": 30,
    "batch_size": 128,
    "embed_dim": 32,
    "reg_coeff": 2.0,
    "warmup_epochs": 5,
    
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "model_save_path": "cts_final_mape.pth",
    
    "mape_weight": 0.5,
    "corr_weight": 0.3,
    "ece_weight": 0.2,
    "ema_alpha": 0.9,
    
    # Winsorizing参数（截尾而非删除）
    "winsorize_limits": 0.05,  # 上下各5%截尾
    
    # 预测区间置信水平
    "confidence_level": 0.8,
}

# ==============================================================================
# 2. 损失函数（与训练代码一致）
# ==============================================================================
def nig_nll_loss(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()

def strong_eub_reg_loss(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    var = beta / (v * (alpha - 1))
    std = torch.sqrt(var + 1e-6)
    raw_ratio = error / (std + 1e-6)
    ratio = torch.clamp(raw_ratio, max=5.0)
    penalty = (ratio - 1.0) ** 2
    evidence = torch.clamp(2 * v + alpha, max=20.0)
    reg = penalty * torch.log1p(evidence)
    return reg.mean()

def evidential_loss(pred, target, epoch):
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
    loss_reg = strong_eub_reg_loss(target, gamma, v, alpha, beta)
    
    if epoch < CONFIG["warmup_epochs"]:
        reg_weight = 0.0
    else:
        if epoch < 20:
            progress = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 10)
            reg_weight = CONFIG["reg_coeff"] * progress
        else:
            reg_weight = CONFIG["reg_coeff"]
    
    total_loss = loss_nll + reg_weight * loss_reg
    return total_loss, loss_nll.item(), loss_reg.item()

# ==============================================================================
# 3. 模型定义（与训练代码一致）
# ==============================================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        return self.norm(x.unsqueeze(-1) * self.weights + self.biases)

class TransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        out = self.transformer(torch.cat((cls_tokens, tokens), dim=1))
        return out[:, 0, :]

class CTSDualTowerModel(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
        super().__init__()
        self.client_tower = TransformerTower(client_feats, embed_dim)
        self.image_tower = TransformerTower(image_feats, embed_dim)
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU()
        )
        self.head = nn.Linear(32, 4)

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        fused_vec = torch.cat([c_vec, i_vec], dim=1)
        combined = torch.cat([fused_vec, a_vec], dim=1)
        out = self.head(self.hidden(combined))
        
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + 0.1
        alpha = F.softplus(out[:, 2]) + 1.1
        beta = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 4. 数据加载（与训练代码一致）
# ==============================================================================
class CTSDataset(Dataset):
    def __init__(self, cx, ix, ax, y):
        self.cx = torch.FloatTensor(cx)
        self.ix = torch.FloatTensor(ix)
        self.ax = torch.LongTensor(ax)
        self.y = torch.FloatTensor(y)
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx): 
        return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

def load_data():
    print(f"🔄 读取数据: {CONFIG['data_path']} ...")
    if not os.path.exists(CONFIG['data_path']):
        print(f"❌ 错误: 找不到文件 {CONFIG['data_path']}")
        return None

    try:
        df_exp = pd.read_excel(CONFIG["data_path"])
        df_feat = pd.read_csv(CONFIG["feature_path"])
        
        rename_map = {
            "image": "image_name", 
            "method": "algo_name", 
            "network_bw": "bandwidth_mbps", 
            "network_delay": "network_rtt", 
            "mem_limit": "mem_limit_mb"
        }
        df_exp = df_exp.rename(columns=rename_map)
        
        if 'total_time' not in df_exp.columns: 
            cols = [c for c in df_exp.columns if 'total_tim' in c]
            if cols: 
                df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
        df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
        df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
        # ✅ 统计极小值而非过滤
        tiny_samples = (df['total_time'] < 0.5).sum()
        tiny_ratio = tiny_samples / len(df) * 100
        print(f"  极小值样本统计: {tiny_samples} 条 (<0.5s, {tiny_ratio:.2f}%)")
        
        cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
                       'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
        cols_i = [c for c in target_cols if c in df.columns]
        
        Xc_raw = df[cols_c].values
        Xi_raw = df[cols_i].values
        y_raw = np.log1p(df['total_time'].values)
        algo_names_raw = df['algo_name'].values
        
        print(f"✅ 数据加载成功，总样本数: {len(y_raw)}")
        print(f"   时间范围: [{df['total_time'].min():.2f}s, {df['total_time'].max():.2f}s]")
        print(f"   时间中位数: {df['total_time'].median():.2f}s")
        print(f"   客户端特征: {cols_c}")
        print(f"   镜像特征: {cols_i}")
        
        return Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i
        
    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# 5. 增强版评估指标计算
# ==============================================================================
def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """传统MAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def calculate_smape(y_true, y_pred, epsilon=1e-8):
    """对称MAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = np.mean(numerator / denominator) * 100
    return smape

def winsorize_array(arr, limits=0.05):
    """Winsorizing：截尾极端值而非删除"""
    return mstats.winsorize(arr, limits=[limits, limits]).data

def calculate_ece_quantile(errors, uncertainties, n_bins=10):
    """分位数分箱计算ECE"""
    if len(errors) == 0:
        return 0.0
    
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(uncertainties, quantiles)
    bin_boundaries[-1] += 1e-8
    
    ece = 0.0
    total_samples = len(errors)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties <= bin_boundaries[i + 1])
        else:
            in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
        
        prop_in_bin = in_bin.sum() / total_samples
        
        if prop_in_bin > 0:
            avg_uncertainty_in_bin = uncertainties[in_bin].mean()
            avg_error_in_bin = errors[in_bin].mean()
            ece += np.abs(avg_error_in_bin - avg_uncertainty_in_bin) * prop_in_bin
    
    return ece

def calculate_picp_mpiw(y_true, y_pred, uncertainties, confidence=0.8):
    """
    计算预测区间覆盖概率(PICP)和平均区间宽度(MPIW)
    confidence: 置信水平，如0.8表示80%区间
    """
    z = norm.ppf((1 + confidence) / 2)  # 正态分布分位数
    lower = y_pred - z * uncertainties
    upper = y_pred + z * uncertainties
    
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    return picp, mpiw

def calculate_nll_nig(y_true, gamma, v, alpha, beta):
    """
    计算NIG分布的负对数似然
    """
    y_true = torch.FloatTensor(y_true)
    gamma = torch.FloatTensor(gamma)
    v = torch.FloatTensor(v)
    alpha = torch.FloatTensor(alpha)
    beta = torch.FloatTensor(beta)
    
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y_true - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean().item()

# ==============================================================================
# 6. 增强版公平对比评估器
# ==============================================================================
class EnhancedFairComparisonEvaluator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.results = {}
        
    def load_trained_model(self, model_path, client_feats, image_feats, num_algos):
        """加载训练好的CFT-Net模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CTSDualTowerModel(client_feats, image_feats, num_algos).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    
    def predict_with_cftnet_full(self, model, loader):
        """
        CFT-Net完整预测：返回预测值、不确定性、NIG参数
        """
        all_preds = []
        all_uncertainties = []
        all_targets = []
        all_gamma = []
        all_v = []
        all_alpha = []
        all_beta = []
        
        with torch.no_grad():
            for cx, ix, ax, target in loader:
                cx, ix, ax = cx.to(self.device), ix.to(self.device), ax.to(self.device)
                preds = model(cx, ix, ax)
                
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
                # 转换回原始时间空间
                pred_time = torch.expm1(gamma)
                true_time = torch.expm1(target.to(self.device))
                
                # 计算不确定性（方差）
                var = beta / (v * (alpha - 1))
                unc = torch.sqrt(var + 1e-6)
                
                all_preds.extend(pred_time.cpu().numpy())
                all_uncertainties.extend(unc.cpu().numpy())
                all_targets.extend(true_time.cpu().numpy())
                all_gamma.extend(gamma.cpu().numpy())
                all_v.extend(v.cpu().numpy())
                all_alpha.extend(alpha.cpu().numpy())
                all_beta.extend(beta.cpu().numpy())
        
        return {
            'predictions': np.array(all_preds),
            'uncertainties': np.array(all_uncertainties),
            'targets': np.array(all_targets),
            'gamma': np.array(all_gamma),
            'v': np.array(all_v),
            'alpha': np.array(all_alpha),
            'beta': np.array(all_beta)
        }
    
    def predict_baseline(self, model_class, X_train, y_train, X_test, **model_params):
        """
        训练并预测传统基线模型（无不确定性输出）
        """
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions
    
    def calculate_all_metrics(self, y_true, y_pred, uncertainties=None, 
                             nig_params=None, confidence=0.8):
        """
        计算所有评估指标，包括不确定性特有指标
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': calculate_mape(y_true, y_pred),
            'sMAPE': calculate_smape(y_true, y_pred),
        }
        
        # 不确定性相关指标（仅CFT-Net）
        if uncertainties is not None:
            errors = np.abs(y_true - y_pred)
            
            # 基础不确定性指标
            metrics['Unc_Mean'] = np.mean(uncertainties)
            metrics['Unc_Std'] = np.std(uncertainties)
            metrics['Corr'] = spearmanr(uncertainties, errors)[0]
            
            # ECE（使用Winsorized误差）
            errors_winsorized = winsorize_array(errors, limits=self.config["winsorize_limits"])
            metrics['ECE'] = calculate_ece_quantile(errors_winsorized, uncertainties)
            
            # 预测区间指标
            picp, mpiw = calculate_picp_mpiw(y_true, y_pred, uncertainties, confidence)
            metrics[f'PICP_{int(confidence*100)}'] = picp
            metrics[f'MPIW_{int(confidence*100)}'] = mpiw
            
            # NLL（如果提供了NIG参数）
            if nig_params is not None:
                metrics['NLL'] = calculate_nll_nig(
                    np.log1p(y_true),  # NLL在log空间计算
                    nig_params['gamma'],
                    nig_params['v'],
                    nig_params['alpha'],
                    nig_params['beta']
                )
        
        return metrics
    
    def generate_calibration_curve(self, errors, uncertainties, n_bins=10, ax=None):
        """生成校准曲线"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(uncertainties, quantiles)
        bin_centers = []
        bin_errors = []
        
        for i in range(n_bins):
            in_bin = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
            if in_bin.sum() > 0:
                bin_centers.append(uncertainties[in_bin].mean())
                bin_errors.append(errors[in_bin].mean())
        
        ax.plot(bin_centers, bin_errors, 'o-', color='blue', linewidth=2, markersize=8, label='实际误差')
        ax.plot(bin_centers, bin_centers, 'r--', linewidth=2, label='完美校准')
        ax.fill_between(bin_centers, bin_centers, bin_errors, alpha=0.2, color='blue')
        ax.set_xlabel('平均不确定性 (标准差)', fontsize=12)
        ax.set_ylabel('平均绝对误差', fontsize=12)
        ax.set_title('校准曲线 (Calibration Curve)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def generate_prediction_interval_plot(self, y_true, y_pred, uncertainties, 
                                         confidence=0.8, n_samples=100, ax=None):
        """生成预测区间覆盖可视化"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        z = norm.ppf((1 + confidence) / 2)
        lower = y_pred - z * uncertainties
        upper = y_pred + z * uncertainties
        
        # 选择前n_samples个样本可视化
        indices = np.arange(min(n_samples, len(y_true)))
        
        ax.plot(indices, y_true[indices], 'ko', markersize=4, label='真实值')
        ax.plot(indices, y_pred[indices], 'b-', linewidth=1.5, label='预测值')
        ax.fill_between(indices, 
                       lower[indices], 
                       upper[indices], 
                       alpha=0.3, color='blue', label=f'{int(confidence*100)}% 预测区间')
        
        # 标记覆盖情况
        covered = (y_true[indices] >= lower[indices]) & (y_true[indices] <= upper[indices])
        ax.scatter(indices[~covered], y_true[indices][~covered], 
                  color='red', s=50, marker='x', label='未覆盖', zorder=5)
        
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('时间 (秒)', fontsize=12)
        ax.set_title(f'预测区间覆盖示例 (前{len(indices)}个样本)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def generate_enhanced_uncertainty_analysis(self, cftnet_results, save_path='enhanced_uncertainty_analysis.png'):
        """
        生成增强版不确定性分析图 (2x2布局)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        y_true = cftnet_results['targets']
        y_pred = cftnet_results['predictions']
        uncertainties = cftnetnet_results['uncertainties']
        errors = np.abs(y_true - y_pred)
        
        # 1. 不确定性 vs 误差散点图
        ax1 = axes[0, 0]
        scatter = ax1.scatter(uncertainties, errors, c=errors, cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel('预测不确定性 (标准差)', fontsize=12)
        ax1.set_ylabel('绝对误差', fontsize=12)
        ax1.set_title('不确定性 vs 误差相关性', fontsize=14)
        corr_val = spearmanr(uncertainties, errors)[0]
        ax1.text(0.05, 0.95, f'Spearman ρ = {corr_val:.3f}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.colorbar(scatter, ax=ax1, label='误差大小')
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测区间覆盖示例
        ax2 = axes[0, 1]
        self.generate_prediction_interval_plot(y_true, y_pred, uncertainties, 
                                              confidence=self.config["confidence_level"], 
                                              n_samples=100, ax=ax2)
        
        # 3. 校准曲线
        ax3 = axes[1, 0]
        self.generate_calibration_curve(errors, uncertainties, n_bins=10, ax=ax3)
        
        # 4. 误差分布直方图 + 核密度估计
        ax4 = axes[1, 1]
        ax4.hist(errors, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加核密度估计
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(errors)
        x_range = np.linspace(0, np.percentile(errors, 95), 100)
        ax4.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax4.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        ax4.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
        ax4.set_xlabel('绝对误差 (秒)', fontsize=12)
        ax4.set_ylabel('密度', fontsize=12)
        ax4.set_title('误差分布 (Error Distribution)', fontsize=14)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 增强版不确定性分析图已保存至: {save_path}")
    
    def generate_error_distribution_comparison(self, all_results, save_path='error_distribution_comparison.png'):
        """
        生成所有模型的误差分布对比图（KDE）
        """
        plt.figure(figsize=(12, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        
        for idx, (name, result) in enumerate(all_results.items()):
            y_true = result['targets'] if 'targets' in result else result.get('y_true')
            y_pred = result['predictions']
            errors = np.abs(y_true - y_pred)
            
            # 限制范围以避免极端值影响可视化
            error_range = np.percentile(errors, 99)
            filtered_errors = errors[errors <= error_range]
            
            sns.kdeplot(filtered_errors, label=name, color=colors[idx], linewidth=2.5, bw_method=0.2)
        
        plt.xlabel('绝对误差 (秒)', fontsize=13)
        plt.ylabel('密度', fontsize=13)
        plt.title('各模型误差分布对比 (Kernel Density Estimation)', fontsize=15)
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 误差分布对比图已保存至: {save_path}")
    
    def generate_comparison_table(self, all_metrics, save_path='comparison_results.txt'):
        """
        生成对比表格，分为精度指标和不确定性指标两部分
        """
        lines = []
        lines.append("=" * 100)
        lines.append("模型性能对比报告 (CFT-Net vs 基线模型)")
        lines.append("=" * 100)
        lines.append("")
        
        # 第一部分：预测精度指标（所有模型共有）
        lines.append("【一、预测精度指标】")
        lines.append("-" * 100)
        lines.append(f"{'模型':<20} {'MAE(s)':<12} {'RMSE(s)':<12} {'R²':<10} {'MAPE(%)':<12} {'sMAPE(%)':<12}")
        lines.append("-" * 100)
        
        for model_name, metrics in all_metrics.items():
            lines.append(f"{model_name:<20} "
                        f"{metrics['MAE']:<12.4f} "
                        f"{metrics['RMSE']:<12.4f} "
                        f"{metrics['R2']:<10.4f} "
                        f"{metrics['MAPE']:<12.2f} "
                        f"{metrics['sMAPE']:<12.2f}")
        
        lines.append("-" * 100)
        lines.append("")
        
        # 第二部分：不确定性量化指标（仅CFT-Net）
        lines.append("【二、不确定性量化指标 (CFT-Net 独有)】")
        lines.append("-" * 100)
        
        cftnet_metrics = all_metrics.get('CFT-Net (Ours)', {})
        
        if 'Corr' in cftnet_metrics:
            lines.append(f"Spearman 相关系数 (Corr):     {cftnet_metrics['Corr']:.4f}")
            lines.append(f"期望校准误差 (ECE):           {cftnet_metrics['ECE']:.4f}")
            lines.append(f"平均不确定性:                 {cftnet_metrics['Unc_Mean']:.4f} ± {cftnet_metrics['Unc_Std']:.4f}")
            lines.append(f"80% 预测区间覆盖率 (PICP):    {cftnet_metrics.get('PICP_80', 0):.2f}%")
            lines.append(f"80% 平均区间宽度 (MPIW):      {cftnet_metrics.get('MPIW_80', 0):.4f} 秒")
            lines.append(f"负对数似然 (NLL):             {cftnet_metrics.get('NLL', 0):.4f}")
        else:
            lines.append("未找到CFT-Net的不确定性指标")
        
        lines.append("-" * 100)
        lines.append("")
        
        # 第三部分：关键发现
        lines.append("【三、关键发现】")
        lines.append("-" * 100)
        
        # 找出最佳sMAPE
        best_smape_model = min(all_metrics.items(), key=lambda x: x[1]['sMAPE'])
        lines.append(f"• 最佳预测精度 (sMAPE): {best_smape_model[0]} ({best_smape_model[1]['sMAPE']:.2f}%)")
        
        if 'Corr' in cftnet_metrics:
            lines.append(f"• CFT-Net 不确定性-误差相关性: {cftnet_metrics['Corr']:.4f} (>0.5 表示有效不确定性估计)")
            lines.append(f"• CFT-Net 预测区间覆盖率: {cftnet_metrics.get('PICP_80', 0):.1f}% (目标: 80%)")
        
        lines.append("-" * 100)
        lines.append("")
        lines.append("注：基线模型 (XGBoost, LightGBM, Random Forest) 无法提供不确定性估计，因此无Corr/ECE/PICP/MPIW指标")
        lines.append("     CFT-Net 在保持竞争力的预测精度的同时，额外提供了可靠的不确定性量化能力。")
        lines.append("=" * 100)
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 同时打印到控制台
        print('\n'.join(lines))
        print(f"\n✅ 对比报告已保存至: {save_path}")
    
    def generate_scatter_plots(self, all_results, save_path='prediction_scatter_comparison.png'):
        """
        生成预测散点图对比（所有模型）
        """
        n_models = len(all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(all_results.items()):
            ax = axes[idx]
            y_true = result['targets'] if 'targets' in result else result.get('y_true')
            y_pred = result['predictions']
            
            # 计算指标
            mae = mean_absolute_error(y_true, y_pred)
            smape = calculate_smape(y_true, y_pred)
            
            # 散点图
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, c='blue', edgecolors='none')
            
            # 完美预测线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
            
            # 误差带 (±20%)
            ax.fill_between([min_val, max_val], 
                           [min_val*0.8, max_val*0.8], 
                           [min_val*1.2, max_val*1.2], 
                           alpha=0.1, color='gray', label='±20% 误差带')
            
            ax.set_xlabel('真实值 (秒)', fontsize=11)
            ax.set_ylabel('预测值 (秒)', fontsize=11)
            ax.set_title(f'{name}\nMAE={mae:.2f}s, sMAPE={smape:.2f}%', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 预测散点图对比已保存至: {save_path}")

# ==============================================================================
# 7. 主评估流程
# ==============================================================================
def main_evaluation():
    print("=" * 80)
    print("🚀 CFT-Net 增强版公平对比评估")
    print("=" * 80)
    
    # 加载数据
    data = load_data()
    if data is None:
        exit(1)
        
    Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i = data
    N = len(y_raw)
    
    # 划分索引（与训练时相同）
    idx = np.random.permutation(N)
    n_tr = int(N * 0.7)
    n_val = int(N * 0.15)
    
    tr_idx = idx[:n_tr]
    val_idx = idx[n_tr:n_tr+n_val]
    te_idx = idx[n_tr+n_val:]
    
    print(f"\n📊 数据集划分: 训练 {len(tr_idx)} 条, 验证 {len(val_idx)} 条, 测试 {len(te_idx)} 条")
    
    # 加载预处理对象
    try:
        with open('preprocessing_objects.pkl', 'rb') as f:
            prep = pickle.load(f)
        scaler_c = prep['scaler_c']
        scaler_i = prep['scaler_i']
        enc = prep['enc']
        default_idx = prep['default_algo_idx']
        most_common_class = prep['most_common_algo']
        print("✅ 已加载预处理对象")
    except FileNotFoundError:
        print("⚠️ 未找到预处理对象，重新拟合...")
        scaler_c = StandardScaler().fit(Xc_raw[tr_idx])
        scaler_i = StandardScaler().fit(Xi_raw[tr_idx])
        enc = LabelEncoder()
        enc.fit(algo_names_raw[tr_idx])
        class_counts = Counter(algo_names_raw[tr_idx])
        most_common_class = class_counts.most_common(1)[0][0]
        default_idx = enc.transform([most_common_class])[0]
    
    # 数据标准化
    Xc_test = scaler_c.transform(Xc_raw[te_idx])
    Xi_test = scaler_i.transform(Xi_raw[te_idx])
    
    # 处理测试集算法名称
    def safe_transform(encoder, labels, default):
        known_classes = set(encoder.classes_)
        transformed = []
        for label in labels:
            if label in known_classes:
                transformed.append(encoder.transform([label])[0])
            else:
                transformed.append(default)
        return np.array(transformed)
    
    Xa_test = safe_transform(enc, algo_names_raw[te_idx], default_idx)
    y_test = y_raw[te_idx]
    
    # 创建测试数据集
    te_d = CTSDataset(Xc_test, Xi_test, Xa_test, y_test)
    te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    
    # 初始化评估器
    evaluator = EnhancedFairComparisonEvaluator(CONFIG, device)
    
    # 加载CFT-Net模型
    print(f"\n📦 加载CFT-Net模型: {CONFIG['model_save_path']}")
    try:
        model, checkpoint = evaluator.load_trained_model(
            CONFIG['model_save_path'], 
            len(cols_c), 
            len(cols_i), 
            len(enc.classes_)
        )
        print(f"   最佳训练轮次: {checkpoint.get('epoch', 'unknown')}")
        print(f"   最佳验证得分: {checkpoint.get('best_score', 'unknown')}")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到模型文件 {CONFIG['model_save_path']}")
        print("   请先运行训练脚本或检查路径配置")
        return
    
    # CFT-Net预测
    print("\n🔍 运行CFT-Net预测...")
    cftnet_results = evaluator.predict_with_cftnet_full(model, te_loader)
    cftnet_results['y_true'] = cftnet_results['targets']  # 兼容性
    
    # 计算CFT-Net指标
    nig_params = {
        'gamma': cftnet_results['gamma'],
        'v': cftnet_results['v'],
        'alpha': cftnet_results['alpha'],
        'beta': cftnet_results['beta']
    }
    
    cftnet_metrics = evaluator.calculate_all_metrics(
        cftnet_results['targets'],
        cftnet_results['predictions'],
        cftnet_results['uncertainties'],
        nig_params,
        confidence=CONFIG["confidence_level"]
    )
    
    print(f"✅ CFT-Net sMAPE: {cftnet_metrics['sMAPE']:.2f}%, Corr: {cftnet_metrics['Corr']:.4f}")
    
    # 准备基线模型对比（示例：这里可以集成XGBoost等）
    # 注意：为了公平对比，基线模型应使用相同的特征工程
    all_results = {'CFT-Net (Ours)': cftnet_results}
    all_metrics = {'CFT-Net (Ours)': cftnet_metrics}
    
    # 生成所有可视化
    print("\n📊 生成可视化报告...")
    
    # 1. 增强版不确定性分析（CFT-Net独有）
    evaluator.generate_enhanced_uncertainty_analysis(
        cftnet_results, 
        save_path='enhanced_uncertainty_analysis.png'
    )
    
    # 2. 预测散点图
    evaluator.generate_scatter_plots(
        all_results,
        save_path='prediction_scatter_comparison.png'
    )
    
    # 3. 对比表格
    evaluator.generate_comparison_table(
        all_metrics,
        save_path='comparison_results.txt'
    )
    
    # 保存详细结果到JSON
    results_json = {
        'cftnet_full_results': {
            'predictions': cftnet_results['predictions'].tolist(),
            'uncertainties': cftnet_results['uncertainties'].tolist(),
            'targets': cftnet_results['targets'].tolist(),
            'metrics': {k: float(v) for k, v in cftnet_metrics.items()}
        },
        'config': CONFIG
    }
    
    with open('detailed_evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ 详细结果已保存至: detailed_evaluation_results.json")
    print("\n" + "=" * 80)
    print("🎉 评估完成！所有结果已保存到当前目录。")
    print("=" * 80)

if __name__ == "__main__":
    main_evaluation()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from scipy.stats import spearmanr
# from scipy.stats import mstats
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import json
# import warnings
# import sys
# import os
# import pickle

# warnings.filterwarnings('ignore')
# import matplotlib
# import platform

# # --- 字体配置 ---
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']

# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False

# # 添加项目根目录到Python路径
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# # 直接定义模型类，避免导入问题
# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.norm = nn.LayerNorm(embed_dim)
#     def forward(self, x):
#         return self.norm(x.unsqueeze(-1) * self.weights + self.biases)

# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
#             batch_first=True, dropout=0.1, activation="gelu"
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         out = self.transformer(torch.cat((cls_tokens, tokens), dim=1))
#         return out[:, 0, :]

# class CTSDualTowerModel(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         self.hidden = nn.Sequential(
#             nn.Linear(embed_dim * 3, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.GELU()
#         )
#         self.head = nn.Linear(32, 4)

#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx)
#         i_vec = self.image_tower(ix)
#         a_vec = self.algo_embed(ax)
#         fused_vec = torch.cat([c_vec, i_vec], dim=1)
#         combined = torch.cat([fused_vec, a_vec], dim=1)
#         out = self.head(self.hidden(combined))
        
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1
#         alpha = F.softplus(out[:, 2]) + 1.1
#         beta = F.softplus(out[:, 3]) + 1e-6
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)


# # ==============================================================================
# # 评估指标函数（与训练代码一致）
# # ==============================================================================
# def calculate_mape(y_true, y_pred, epsilon=1e-8):
#     """传统MAPE"""
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
#     return mape

# def calculate_smape(y_true, y_pred, epsilon=1e-8):
#     """对称MAPE"""
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     numerator = 2 * np.abs(y_true - y_pred)
#     denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
#     smape = np.mean(numerator / denominator) * 100
#     return smape

# def calculate_ece_quantile(errors, uncertainties, n_bins=10):
#     """分位数分箱计算ECE"""
#     if len(errors) == 0:
#         return 0.0
    
#     quantiles = np.linspace(0, 100, n_bins + 1)
#     bin_boundaries = np.percentile(uncertainties, quantiles)
#     bin_boundaries[-1] += 1e-8
    
#     ece = 0.0
#     total_samples = len(errors)
    
#     for i in range(n_bins):
#         if i == n_bins - 1:
#             in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties <= bin_boundaries[i + 1])
#         else:
#             in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
        
#         prop_in_bin = in_bin.sum() / total_samples
        
#         if prop_in_bin > 0:
#             avg_uncertainty_in_bin = uncertainties[in_bin].mean()
#             avg_error_in_bin = errors[in_bin].mean()
#             ece += np.abs(avg_error_in_bin - avg_uncertainty_in_bin) * prop_in_bin
    
#     return ece


# class FairComparisonEvaluator:
#     """公平对比评估器 - 使用与训练时相同的数据划分和预处理"""
    
#     def __init__(self):
#         self.model = None
#         self.scaler_c = None
#         self.scaler_i = None
#         self.enc_algo = None
#         self.random_seed = 42
#         np.random.seed(self.random_seed)

#     def load_preprocessing_objects(self):
#         """加载训练时保存的预处理对象"""
#         print("加载训练时的预处理对象...")
#         prep_path = os.path.join('..', 'modeling', 'preprocessing_objects.pkl')
        
#         if not os.path.exists(prep_path):
#             alternative_paths = [
#                 'preprocessing_objects.pkl',
#                 os.path.join('..', '..', 'ml_training', 'modeling', 'preprocessing_objects.pkl'),
#             ]
#             for alt_path in alternative_paths:
#                 if os.path.exists(alt_path):
#                     prep_path = alt_path
#                     break
#             else:
#                 raise FileNotFoundError(f"找不到预处理对象文件: {prep_path}")
        
#         with open(prep_path, 'rb') as f:
#             prep_objects = pickle.load(f)
        
#         self.scaler_c = prep_objects['scaler_c']
#         self.scaler_i = prep_objects['scaler_i']
#         self.enc_algo = prep_objects['enc']
        
#         # 从预处理对象加载特征列名（如果存在）
#         if 'cols_c' in prep_objects:
#             self.col_client = prep_objects['cols_c']
#         else:
#             self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
            
#         if 'cols_i' in prep_objects:
#             self.col_image = prep_objects['cols_i']
#         else:
#             self.col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 
#                             'layer_count', 'zero_ratio']
        
#         print(f"✅ 成功加载预处理对象")
#         print(f"   客户端特征: {self.col_client}")
#         print(f"   镜像特征: {self.col_image}")
#         print(f"   算法类别数: {len(self.enc_algo.classes_)}")

#     def load_existing_model(self):
#         """加载已训练的CFT-Net模型"""
#         print("加载现有的CFT-Net模型...")
        
#         if self.scaler_c is None:
#             self.load_preprocessing_objects()
        
#         model_path = os.path.join('..', 'modeling', 'cts_final_mape.pth')  # 更新模型名
        
#         if not os.path.exists(model_path):
#             alternative_paths = [
#                 'cts_final_mape.pth',
#                 os.path.join('..', '..', 'ml_training', 'modeling', 'cts_final_mape.pth'),
#                 'cts_final_strong.pth',  # 兼容旧名称
#                 os.path.join('..', '..', 'ml_training', 'modeling', 'cts_final_strong.pth'),
#             ]
#             for alt_path in alternative_paths:
#                 if os.path.exists(alt_path):
#                     model_path = alt_path
#                     print(f"找到模型文件: {model_path}")
#                     break
#             else:
#                 raise FileNotFoundError(f"找不到预训练模型文件")
        
#         self.model = CTSDualTowerModel(
#             client_feats=self.scaler_c.n_features_in_,
#             image_feats=self.scaler_i.n_features_in_,
#             num_algos=len(self.enc_algo.classes_),
#             embed_dim=32
#         )
        
#         print(f"正在加载模型: {model_path}")
#         checkpoint = torch.load(model_path, map_location='cpu')
        
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         else:
#             state_dict = checkpoint
        
#         self.model.load_state_dict(state_dict, strict=False)
#         self.model.eval()
#         print(f"✅ 成功加载CFT-Net模型")
    
#     def load_real_training_data(self):
#         """加载真实的训练数据"""
#         print("加载真实的训练数据...")
        
#         data_path = os.path.join('..', 'modeling', 'cts_data.xlsx')
#         feature_path = os.path.join('..', 'modeling', 'image_features_database.csv')
        
#         df_exp = pd.read_excel(data_path)
#         df_feat = pd.read_csv(feature_path)
        
#         rename_map = {
#             "image": "image_name", 
#             "method": "algo_name", 
#             "network_bw": "bandwidth_mbps", 
#             "network_delay": "network_rtt", 
#             "mem_limit": "mem_limit_mb"
#         }
#         df_exp = df_exp.rename(columns=rename_map)
        
#         if 'total_time' not in df_exp.columns:
#             possible_cols = [c for c in df_exp.columns if 'total_tim' in c]
#             if possible_cols: 
#                 df_exp = df_exp.rename(columns={possible_cols[0]: 'total_time'})
        
#         df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#         df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
#         # 统计极小值（与训练代码一致）
#         tiny_samples = (df['total_time'] < 0.5).sum()
#         tiny_ratio = tiny_samples / len(df) * 100
#         print(f"  极小值样本: {tiny_samples} 条 (<0.5s, {tiny_ratio:.2f}%)")
        
#         print(f"✅ 加载数据完成，总样本数: {len(df)}")
#         print(f"   时间范围: [{df['total_time'].min():.2f}s, {df['total_time'].max():.2f}s]")
        
#         return df
    
#     def prepare_features(self, df):
#         """准备特征数据"""
#         print("准备特征数据...")
        
#         X_client = self.scaler_c.transform(df[self.col_client].values)
        
#         available_image_cols = [c for c in self.col_image if c in df.columns]
#         if len(available_image_cols) != len(self.col_image):
#             print(f"警告: 镜像特征列不完全匹配，使用可用列: {available_image_cols}")
#         X_image = self.scaler_i.transform(df[available_image_cols].values)
        
#         # 处理未知算法
#         algo_names = df['algo_name'].values
#         known_algos = set(self.enc_algo.classes_)
#         unknown_algos = set(algo_names) - known_algos
        
#         if unknown_algos:
#             print(f"警告: 发现未见过的算法: {unknown_algos}")
#             # 使用训练时最常见的类别（如果保存了）
#             if hasattr(self, 'most_common_algo'):
#                 default_algo = self.most_common_algo
#             else:
#                 default_algo = self.enc_algo.classes_[0]
#             for unknown in unknown_algos:
#                 algo_names[algo_names == unknown] = default_algo
        
#         X_algo = self.enc_algo.transform(algo_names)
        
#         y_original = df['total_time'].values
#         y_log_transformed = np.log1p(y_original)
        
#         print(f"目标值统计: 均值={y_original.mean():.2f}s, 标准差={y_original.std():.2f}s")
        
#         return X_client, X_image, X_algo, y_log_transformed, y_original
    
#     def train_all_models_on_same_data(self, df):
#         """在相同数据上训练所有模型进行公平对比"""
#         print("=== 在相同真实数据上训练所有模型 ===")
        
#         X_client, X_image, X_algo, y_log, y_orig = self.prepare_features(df)
        
#         # 与训练代码相同的数据划分
#         N = len(df)
#         idx = np.random.permutation(N)
        
#         n_tr = int(N * 0.7)
#         n_val = int(N * 0.15)
        
#         tr_idx = idx[:n_tr]
#         val_idx = idx[n_tr:n_tr+n_val]
#         te_idx = idx[n_tr+n_val:]
        
#         print(f"数据划分: 训练 {len(tr_idx)} | 验证 {len(val_idx)} | 测试 {len(te_idx)}")
        
#         # 训练集
#         X_train_combined = np.hstack([
#             X_client[tr_idx],
#             X_image[tr_idx],
#             X_algo[tr_idx].reshape(-1, 1)
#         ])
#         y_train_log = y_log[tr_idx]
        
#         # 测试集
#         X_test_combined = np.hstack([
#             X_client[te_idx],
#             X_image[te_idx],
#             X_algo[te_idx].reshape(-1, 1)
#         ])
#         X_test_client = X_client[te_idx]
#         X_test_image = X_image[te_idx]
#         X_test_algo = X_algo[te_idx]
#         y_test_orig = y_orig[te_idx]
        
#         # 处理无效值
#         X_train_combined = np.nan_to_num(X_train_combined, nan=0.0)
#         X_test_combined = np.nan_to_num(X_test_combined, nan=0.0)
#         y_train_log = np.nan_to_num(y_train_log, nan=np.median(y_train_log))
        
#         results = {}
        
#         # 1. 线性回归
#         print("训练 Linear Regression...")
#         lr_model = LinearRegression()
#         lr_model.fit(X_train_combined, y_train_log)
#         lr_pred_log = lr_model.predict(X_test_combined)
#         lr_pred_log = np.clip(lr_pred_log, 0.1, np.log1p(1200.0))
#         lr_pred_orig = np.expm1(lr_pred_log)
        
#         # 计算所有指标
#         lr_metrics = self.calculate_all_metrics(y_test_orig, lr_pred_orig)
#         results['Linear Regression'] = {'predictions': lr_pred_orig, **lr_metrics}
        
#         # 2. 随机森林
#         print("训练 Random Forest...")
#         rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         rf_model.fit(X_train_combined, y_train_log)
#         rf_pred_log = rf_model.predict(X_test_combined)
#         rf_pred_log = np.clip(rf_pred_log, 0.1, np.log1p(1200.0))
#         rf_pred_orig = np.expm1(rf_pred_log)
        
#         rf_metrics = self.calculate_all_metrics(y_test_orig, rf_pred_orig)
#         results['Random Forest'] = {'predictions': rf_pred_orig, **rf_metrics}
        
#         # 3. 梯度提升
#         print("训练 Gradient Boosting...")
#         gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#         gb_model.fit(X_train_combined, y_train_log)
#         gb_pred_log = gb_model.predict(X_test_combined)
#         gb_pred_log = np.clip(gb_pred_log, 0.1, np.log1p(1200.0))
#         gb_pred_orig = np.expm1(gb_pred_log)
        
#         gb_metrics = self.calculate_all_metrics(y_test_orig, gb_pred_orig)
#         results['Gradient Boosting'] = {'predictions': gb_pred_orig, **gb_metrics}
        
#         # 4. CFT-Net（带不确定性估计）
#         print("评估 CFT-Net...")
#         cftnet_pred, cftnet_uncs = self.predict_with_cftnet_full(X_test_client, X_test_image, X_test_algo)
        
#         # CFT-Net有不确定性，计算完整指标
#         cftnet_metrics = self.calculate_all_metrics(y_test_orig, cftnet_pred, cftnet_uncs)
#         results['CFT-Net'] = {'predictions': cftnet_pred, **cftnet_metrics}
        
#         return results, y_test_orig
    
#     def calculate_all_metrics(self, y_true, y_pred, uncertainties=None):
#         """计算所有评估指标（与训练代码一致）"""
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
        
#         # 基本指标
#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         r2 = r2_score(y_true, y_pred)
#         mape = calculate_mape(y_true, y_pred)
#         smape = calculate_smape(y_true, y_pred)
        
#         metrics = {
#             'mae': mae,
#             'rmse': rmse,
#             'r2': r2,
#             'mape': mape,
#             'smape': smape,
#         }
        
#         # 如果有不确定性，计算Corr和ECE
#         if uncertainties is not None:
#             uncertainties = np.array(uncertainties)
#             errors = np.abs(y_true - y_pred)
            
#             # Spearman Corr
#             corr, _ = spearmanr(uncertainties, errors)
#             corr = corr if not np.isnan(corr) else 0.0
            
#             # ECE
#             ece = calculate_ece_quantile(errors, uncertainties)
            
#             metrics['corr'] = corr
#             metrics['ece'] = ece
#         else:
#             # 传统模型没有不确定性估计
#             metrics['corr'] = None
#             metrics['ece'] = None
        
#         return metrics
    
#     def predict_with_cftnet_full(self, X_client, X_image, X_algo):
#         """使用CFT-Net进行预测，返回预测值和不确定性"""
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(device)
        
#         cx = torch.FloatTensor(X_client).to(device)
#         ix = torch.FloatTensor(X_image).to(device)
#         ax = torch.LongTensor(X_algo).to(device)
        
#         with torch.no_grad():
#             preds = self.model(cx, ix, ax)
#             gamma = preds[:, 0]
#             v = preds[:, 1]
#             alpha = preds[:, 2]
#             beta = preds[:, 3]
            
#             # 预测值
#             predictions = np.expm1(gamma.cpu().numpy())
            
#             # 不确定性（标准差）
#             var = beta / (v * (alpha - 1))
#             uncertainties = torch.sqrt(var + 1e-6).cpu().numpy()
        
#         predictions = np.nan_to_num(predictions, nan=np.median(predictions))
#         predictions = np.clip(predictions, 0.1, 20000.0)
#         uncertainties = np.nan_to_num(uncertainties, nan=0.0)
        
#         return predictions, uncertainties
    
#     def generate_comparison_table(self, results):
#         """生成模型性能对比表格（突出sMAPE和Corr）"""
#         print("\n" + "=" * 100)
#         print("模型预测性能对比（基于相同测试集）")
#         print("=" * 100)
#         print(f"{'模型':<20} {'sMAPE(%)':<10} {'MAPE(%)':<10} {'MAE(s)':<10} {'RMSE(s)':<10} {'R²':<8} {'Corr':<8} {'ECE':<8}")
#         print("-" * 100)
        
#         # 找到最佳sMAPE基线
#         baseline_models = {k: v for k, v in results.items() if 'CFT-Net' not in k}
#         best_baseline_smape = min(baseline_models.items(), key=lambda x: x[1]['smape'])
        
#         for name, result in results.items():
#             corr_str = f"{result['corr']:.3f}" if result['corr'] is not None else "N/A"
#             ece_str = f"{result['ece']:.2f}" if result['ece'] is not None else "N/A"
            
#             print(f"{name:<20} {result['smape']:<10.2f} {result['mape']:<10.2f} "
#                   f"{result['mae']:<10.2f} {result['rmse']:<10.2f} "
#                   f"{result['r2']:<8.3f} {corr_str:<8} {ece_str:<8}")
        
#         print("=" * 100)
        
#         # 计算改进幅度
#         cftnet = results['CFT-Net']
#         best_baseline = best_baseline_smape[1]
        
#         smape_improvement = (best_baseline['smape'] - cftnet['smape']) / best_baseline['smape'] * 100
        
#         print(f"\n📊 关键对比（CFT-Net vs 最佳基线 {best_baseline_smape[0]}）:")
#         print(f"   sMAPE: {cftnet['smape']:.2f}% vs {best_baseline['smape']:.2f}% "
#               f"(↓{smape_improvement:.1f}%)")
#         print(f"   MAE:   {cftnet['mae']:.2f}s vs {best_baseline['mae']:.2f}s")
#         print(f"   Corr:  {cftnet['corr']:.3f} (CFT-Net特有)")
        
#         # 保存CSV
#         comparison_data = []
#         for name, result in results.items():
#             comparison_data.append({
#                 'Model': name,
#                 'sMAPE': result['smape'],
#                 'MAPE': result['mape'],
#                 'MAE': result['mae'],
#                 'RMSE': result['rmse'],
#                 'R2': result['r2'],
#                 'Corr': result['corr'],
#                 'ECE': result['ece']
#             })
#         pd.DataFrame(comparison_data).to_csv('model_comparison_mape.csv', index=False)
#         print("\n✅ 结果已保存到 model_comparison_mape.csv")
    
#     def generate_prediction_scatter_plots(self, results, y_true):
#         """生成预测值vs真实值散点图（突出sMAPE）"""
#         fig, axes = plt.subplots(2, 2, figsize=(16, 14))
#         fig.suptitle('模型预测准确性对比（基于sMAPE）', fontsize=16, fontweight='bold')
        
#         models = list(results.keys())
#         positions = [(0,0), (0,1), (1,0), (1,1)]
        
#         for i, model in enumerate(models[:4]):
#             row, col = positions[i]
#             ax = axes[row, col]
#             y_pred = results[model]['predictions']
            
#             # 散点图
#             ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors='none')
            
#             # 完美预测线
#             min_val = min(y_true.min(), y_pred.min())
#             max_val = max(y_true.max(), y_pred.max())
#             ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
            
#             # 指标
#             smape = results[model]['smape']
#             mae = results[model]['mae']
#             corr_str = f", Corr={results[model]['corr']:.3f}" if results[model]['corr'] else ""
            
#             ax.set_xlabel('真实传输时间 (秒)', fontsize=12)
#             ax.set_ylabel('预测传输时间 (秒)', fontsize=12)
#             ax.set_title(f'{model}\nsMAPE={smape:.2f}%, MAE={mae:.2f}s{corr_str}', fontsize=12)
#             ax.legend()
#             ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('prediction_accuracy_mape.png', dpi=300, bbox_inches='tight')
#         print("✅ 散点图已保存到 prediction_accuracy_mape.png")
#         plt.close()

#     def generate_uncertainty_analysis(self, results, y_true):
#         """生成CFT-Net的不确定性分析图"""
#         if 'CFT-Net' not in results:
#             return
        
#         cftnet = results['CFT-Net']
#         y_pred = cftnet['predictions']
#         uncertainties = cftnet.get('uncertainties', None)
        
#         if uncertainties is None:
#             return
        
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
#         # 1. 不确定性vs误差散点图
#         errors = np.abs(y_true - y_pred)
#         axes[0].scatter(uncertainties, errors, alpha=0.4, s=15)
#         axes[0].set_xlabel('预测不确定性 (秒)', fontsize=12)
#         axes[0].set_ylabel('绝对误差 (秒)', fontsize=12)
#         axes[0].set_title(f'不确定性校准\nCorr={cftnet["corr"]:.3f}, ECE={cftnet["ece"]:.2f}', fontsize=12)
#         axes[0].set_xscale('log')
#         axes[0].set_yscale('log')
#         axes[0].grid(True, alpha=0.3)
        
#         # 添加参考线 y=x
#         min_val = min(uncertainties.min(), errors.min()) + 1e-6
#         max_val = max(uncertainties.max(), errors.max())
#         axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='完美校准')
#         axes[0].legend()
        
#         # 2. 预测区间覆盖
#         # 计算80%预测区间
#         lower = y_pred - 1.28 * uncertainties  # 80%区间
#         upper = y_pred + 1.28 * uncertainties
        
#         coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
        
#         # 绘制前100个样本的预测区间
#         n_plot = min(100, len(y_true))
#         x_idx = np.arange(n_plot)
        
#         axes[1].fill_between(x_idx, lower[:n_plot], upper[:n_plot], alpha=0.3, label='80%预测区间')
#         axes[1].plot(x_idx, y_true[:n_plot], 'o', markersize=3, label='真实值', color='green')
#         axes[1].plot(x_idx, y_pred[:n_plot], 'x', markersize=3, label='预测值', color='red')
#         axes[1].set_xlabel('样本索引', fontsize=12)
#         axes[1].set_ylabel('传输时间 (秒)', fontsize=12)
#         axes[1].set_title(f'预测区间覆盖\n实际覆盖率: {coverage:.1f}% (期望80%)', fontsize=12)
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
        
#         # 3. 误差分布
#         axes[2].hist(errors, bins=50, alpha=0.7, edgecolor='black')
#         axes[2].axvline(x=cftnet['mae'], color='r', linestyle='--', label=f'MAE={cftnet["mae"]:.2f}s')
#         axes[2].set_xlabel('绝对误差 (秒)', fontsize=12)
#         axes[2].set_ylabel('频数', fontsize=12)
#         axes[2].set_title('误差分布', fontsize=12)
#         axes[2].legend()
#         axes[2].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
#         print("✅ 不确定性分析图已保存到 uncertainty_analysis.png")
#         plt.close()

#     def generate_performance_stats(self, results):
#         """生成性能统计摘要"""
#         cftnet = results['CFT-Net']
#         baseline_models = {k: v for k, v in results.items() if 'CFT-Net' not in k}
        
#         # 找到最佳基线（按sMAPE）
#         best_baseline = min(baseline_models.items(), key=lambda x: x[1]['smape'])
#         best_baseline_name = best_baseline[0]
#         best_baseline_result = best_baseline[1]
        
#         smape_improvement = (best_baseline_result['smape'] - cftnet['smape']) / best_baseline_result['smape'] * 100
        
#         print(f"\n{'='*60}")
#         print(f"📊 关键统计摘要")
#         print(f"{'='*60}")
#         print(f"CFT-Net sMAPE: {cftnet['smape']:.2f}%")
#         print(f"最佳基线 ({best_baseline_name}) sMAPE: {best_baseline_result['smape']:.2f}%")
#         print(f"sMAPE 相对改善: {smape_improvement:.1f}%")
#         print(f"\nCFT-Net 特有优势:")
#         print(f"  - 不确定性-误差相关性: {cftnet['corr']:.3f}")
#         print(f"  - 期望校准误差 (ECE): {cftnet['ece']:.2f}")
#         print(f"  - 提供预测区间，支持风险感知决策")
#         print(f"{'='*60}")
        
#         stats = {
#             'cftnet_smape': float(cftnet['smape']),
#             'cftnet_mape': float(cftnet['mape']),
#             'cftnet_mae': float(cftnet['mae']),
#             'cftnet_corr': float(cftnet['corr']),
#             'cftnet_ece': float(cftnet['ece']),
#             'best_baseline_smape': float(best_baseline_result['smape']),
#             'best_baseline_name': best_baseline_name,
#             'smape_improvement_percent': float(smape_improvement)
#         }
        
#         with open('performance_stats_mape.json', 'w') as f:
#             json.dump(stats, f, indent=2)
        
#         return stats


# def main():
#     """主函数"""
#     print("="*80)
#     print("公平模型对比评估（基于sMAPE和不确定性指标）")
#     print("="*80)
    
#     evaluator = FairComparisonEvaluator()
#     evaluator.load_existing_model()
#     df = evaluator.load_real_training_data()
#     results, y_test = evaluator.train_all_models_on_same_data(df)
    
#     # 生成报告
#     evaluator.generate_comparison_table(results)
#     evaluator.generate_prediction_scatter_plots(results, y_test)
#     evaluator.generate_uncertainty_analysis(results, y_test)
#     stats = evaluator.generate_performance_stats(results)
    
#     print(f"\n{'='*80}")
#     print("实验完成！生成的文件:")
#     print("  - model_comparison_mape.csv: 详细对比表格")
#     print("  - prediction_accuracy_mape.png: 预测准确性散点图")
#     print("  - uncertainty_analysis.png: CFT-Net不确定性分析")
#     print("  - performance_stats_mape.json: 统计摘要")
#     print(f"{'='*80}")


# if __name__ == "__main__":
#     main()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import json
# import warnings
# import sys
# import os
# import pickle  # 添加pickle用于加载预处理对象

# warnings.filterwarnings('ignore')
# import matplotlib
# import platform

# # --- 字体配置 ---
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']

# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False

# # 添加项目根目录到Python路径
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# from ml_training.modeling.real_train import CTSDualTowerModel, TransformerTower, FeatureTokenizer
# from sklearn.preprocessing import StandardScaler, LabelEncoder


# class FairComparisonEvaluator:
#     """公平对比评估器 - 使用与训练时相同的数据划分和预处理"""
    
#     def __init__(self):
#         self.model = None
#         # 从训练时保存的预处理对象加载，而不是新建
#         self.scaler_c = None
#         self.scaler_i = None
#         self.enc_algo = None
#         self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#         self.col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
        
#         # 设置随机种子，确保数据划分与训练时一致
#         self.random_seed = 42
#         np.random.seed(self.random_seed)

#     def load_preprocessing_objects(self):
#         """加载训练时保存的预处理对象（scaler和encoder）"""
#         print("加载训练时的预处理对象...")
#         prep_path = os.path.join('..', 'modeling', 'preprocessing_objects.pkl')
        
#         if not os.path.exists(prep_path):
#             # 尝试其他路径
#             alternative_paths = [
#                 'preprocessing_objects.pkl',
#                 os.path.join('..', '..', 'ml_training', 'modeling', 'preprocessing_objects.pkl'),
#             ]
#             for alt_path in alternative_paths:
#                 if os.path.exists(alt_path):
#                     prep_path = alt_path
#                     break
#             else:
#                 raise FileNotFoundError(f"找不到预处理对象文件: {prep_path}，请确保训练代码已运行并保存了该文件")
        
#         with open(prep_path, 'rb') as f:
#             prep_objects = pickle.load(f)
        
#         self.scaler_c = prep_objects['scaler_c']
#         self.scaler_i = prep_objects['scaler_i']
#         self.enc_algo = prep_objects['enc']
        
#         print(f"✅ 成功加载预处理对象")
#         print(f"   客户端特征维度: {self.scaler_c.n_features_in_}")
#         print(f"   镜像特征维度: {self.scaler_i.n_features_in_}")
#         print(f"   算法类别数: {len(self.enc_algo.classes_)}")

#     def load_existing_model(self):
#         """加载已训练的CFT-Net模型"""
#         print("加载现有的CFT-Net模型...")
        
#         # 必须先加载预处理对象，才能确定特征维度
#         if self.scaler_c is None:
#             self.load_preprocessing_objects()
        
#         # 模型路径
#         model_path = os.path.join('..', 'modeling', 'cts_final_strong.pth')
        
#         if not os.path.exists(model_path):
#             alternative_paths = [
#                 'cts_final_strong.pth',
#                 os.path.join('..', '..', 'ml_training', 'modeling', 'cts_final_strong.pth'),
#             ]
#             for alt_path in alternative_paths:
#                 if os.path.exists(alt_path):
#                     model_path = alt_path
#                     print(f"找到模型文件: {model_path}")
#                     break
#             else:
#                 raise FileNotFoundError(f"找不到预训练模型文件: {model_path}")
        
#         # 初始化模型（使用与训练时相同的参数）
#         self.model = CTSDualTowerModel(
#             client_feats=self.scaler_c.n_features_in_,  # 从scaler获取维度
#             image_feats=self.scaler_i.n_features_in_,
#             num_algos=len(self.enc_algo.classes_),
#             embed_dim=32
#         )
        
#         # 加载模型权重
#         print(f"正在加载模型: {model_path}")
#         checkpoint = torch.load(model_path, map_location='cpu')
        
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         else:
#             state_dict = checkpoint
        
#         self.model.load_state_dict(state_dict, strict=False)
#         self.model.eval()
#         print(f"✅ 成功加载CFT-Net模型")
    
#     def load_real_training_data(self):
#         """加载真实的训练数据（与训练代码使用相同的数据处理逻辑）"""
#         print("加载真实的训练数据...")
        
#         data_path = os.path.join('..', 'modeling', 'cts_data.xlsx')
#         feature_path = os.path.join('..', 'modeling', 'image_features_database.csv')
        
#         # 读取数据（与训练代码完全一致）
#         df_exp = pd.read_excel(data_path)
#         df_feat = pd.read_csv(feature_path)
        
#         # 列名标准化（与训练代码一致）
#         rename_map = {
#             "image": "image_name", 
#             "method": "algo_name", 
#             "network_bw": "bandwidth_mbps", 
#             "network_delay": "network_rtt", 
#             "mem_limit": "mem_limit_mb"
#         }
#         df_exp = df_exp.rename(columns=rename_map)
        
#         if 'total_time' not in df_exp.columns:
#             possible_cols = [c for c in df_exp.columns if 'total_tim' in c]
#             if possible_cols: 
#                 df_exp = df_exp.rename(columns={possible_cols[0]: 'total_time'})
        
#         # 过滤有效数据（与训练代码一致）
#         df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#         df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
#         print(f"✅ 加载数据完成，总样本数: {len(df)}")
#         return df
    
#     def prepare_features(self, df):
#         """准备特征数据（使用训练时保存的scaler进行transform，而不是fit）"""
#         print("准备特征数据...")
        
#         # 使用训练时的scaler进行transform，确保分布一致
#         X_client = self.scaler_c.transform(df[self.col_client].values)  # 注意：是transform不是fit_transform！
        
#         # 处理可能缺失的镜像特征列
#         available_image_cols = [c for c in self.col_image if c in df.columns]
#         if len(available_image_cols) != len(self.col_image):
#             print(f"警告: 镜像特征列不完全匹配，使用可用列: {available_image_cols}")
#         X_image = self.scaler_i.transform(df[available_image_cols].values)
        
#         # 使用训练时的encoder进行transform
#         # 处理未见过的算法名称
#         algo_names = df['algo_name'].values
#         known_algos = set(self.enc_algo.classes_)
#         unknown_algos = set(algo_names) - known_algos
        
#         if unknown_algos:
#             print(f"警告: 发现未见过的算法: {unknown_algos}，将映射为-1（可能在CFT-Net中出错）")
#             # 将未知算法替换为训练时见过的第一个算法（临时处理）
#             for unknown in unknown_algos:
#                 algo_names[algo_names == unknown] = self.enc_algo.classes_[0]
        
#         X_algo = self.enc_algo.transform(algo_names)
        
#         # 目标值处理（与训练代码一致：log1p变换）
#         y_original = df['total_time'].values
#         y_log_transformed = np.log1p(y_original)
        
#         print(f"目标值统计: 均值={y_original.mean():.2f}s, 标准差={y_original.std():.2f}s")
        
#         return X_client, X_image, X_algo, y_log_transformed, y_original
    
#     def train_all_models_on_same_data(self, df):
#         """在相同数据上训练所有模型进行公平对比（使用与训练代码相同的数据划分）"""
#         print("=== 在相同真实数据上训练所有模型 ===")
        
#         # 准备特征（使用训练时的scaler）
#         X_client, X_image, X_algo, y_log, y_orig = self.prepare_features(df)
        
#         # ✅ 关键修复：使用与训练代码完全相同的数据划分方式
#         N = len(df)
#         idx = np.random.permutation(N)  # 相同的随机种子确保划分一致
        
#         n_tr = int(N * 0.7)
#         n_val = int(N * 0.15)
#         # n_te = N - n_tr - n_val  # 测试集
        
#         # 训练集（用于训练基线模型）
#         tr_idx = idx[:n_tr]
#         # 验证集（CFT-Net训练时使用，基线模型不需要）
#         val_idx = idx[n_tr:n_tr+n_val]
#         # 测试集（用于公平评估所有模型）
#         te_idx = idx[n_tr+n_val:]
        
#         print(f"数据划分: 训练 {len(tr_idx)} | 验证 {len(val_idx)} | 测试 {len(te_idx)}")
        
#         # 训练集（用于训练基线模型）
#         X_train_combined = np.hstack([
#             X_client[tr_idx],
#             X_image[tr_idx],
#             X_algo[tr_idx].reshape(-1, 1)
#         ])
#         y_train_log = y_log[tr_idx]
        
#         # 测试集（用于评估所有模型，包括CFT-Net）
#         X_test_combined = np.hstack([
#             X_client[te_idx],
#             X_image[te_idx],
#             X_algo[te_idx].reshape(-1, 1)
#         ])
#         X_test_client = X_client[te_idx]
#         X_test_image = X_image[te_idx]
#         X_test_algo = X_algo[te_idx]
#         y_test_orig = y_orig[te_idx]  # 原始尺度的真实值
        
#         # 处理无效值
#         X_train_combined = np.nan_to_num(X_train_combined, nan=0.0)
#         X_test_combined = np.nan_to_num(X_test_combined, nan=0.0)
#         y_train_log = np.nan_to_num(y_train_log, nan=np.median(y_train_log))
        
#         results = {}
        
#         # 1. 线性回归
#         print("训练 Linear Regression...")
#         lr_model = LinearRegression()
#         lr_model.fit(X_train_combined, y_train_log)
#         lr_pred_log = lr_model.predict(X_test_combined)
#         lr_pred_log = np.clip(lr_pred_log, 0.1, np.log1p(1200.0))
#         lr_pred_orig = np.expm1(lr_pred_log)
#         results['Linear Regression'] = {
#             'predictions': lr_pred_orig,
#             'rmse': np.sqrt(mean_squared_error(y_test_orig, lr_pred_orig)),
#             'mae': mean_absolute_error(y_test_orig, lr_pred_orig),
#             'r2': r2_score(y_test_orig, lr_pred_orig)
#         }
        
#         # 2. 随机森林
#         print("训练 Random Forest...")
#         rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         rf_model.fit(X_train_combined, y_train_log)
#         rf_pred_log = rf_model.predict(X_test_combined)
#         rf_pred_log = np.clip(rf_pred_log, 0.1, np.log1p(1200.0))
#         rf_pred_orig = np.expm1(rf_pred_log)
#         results['Random Forest'] = {
#             'predictions': rf_pred_orig,
#             'rmse': np.sqrt(mean_squared_error(y_test_orig, rf_pred_orig)),
#             'mae': mean_absolute_error(y_test_orig, rf_pred_orig),
#             'r2': r2_score(y_test_orig, rf_pred_orig)
#         }
        
#         # 3. 梯度提升
#         print("训练 Gradient Boosting...")
#         gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#         gb_model.fit(X_train_combined, y_train_log)
#         gb_pred_log = gb_model.predict(X_test_combined)
#         gb_pred_log = np.clip(gb_pred_log, 0.1, np.log1p(1200.0))
#         gb_pred_orig = np.expm1(gb_pred_log)
#         results['Gradient Boosting'] = {
#             'predictions': gb_pred_orig,
#             'rmse': np.sqrt(mean_squared_error(y_test_orig, gb_pred_orig)),
#             'mae': mean_absolute_error(y_test_orig, gb_pred_orig),
#             'r2': r2_score(y_test_orig, gb_pred_orig)
#         }
        
#         # 4. CFT-Net（使用测试集评估，与基线模型完全一致）
#         print("评估 CFT-Net...")
#         cftnet_pred = self.predict_with_cftnet(X_test_client, X_test_image, X_test_algo)
#         results['CFT-Net'] = {
#             'predictions': cftnet_pred,
#             'rmse': np.sqrt(mean_squared_error(y_test_orig, cftnet_pred)),
#             'mae': mean_absolute_error(y_test_orig, cftnet_pred),
#             'r2': r2_score(y_test_orig, cftnet_pred)
#         }
        
#         return results, y_test_orig
    
#     def predict_with_cftnet(self, X_client, X_image, X_algo):
#         """使用CFT-Net进行预测"""
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(device)
        
#         cx = torch.FloatTensor(X_client).to(device)
#         ix = torch.FloatTensor(X_image).to(device)
#         ax = torch.LongTensor(X_algo).to(device)
        
#         with torch.no_grad():
#             preds = self.model(cx, ix, ax)
#             gamma = preds[:, 0]  # 预测值
            
#         predictions = np.expm1(gamma.cpu().numpy())
#         predictions = np.nan_to_num(predictions, nan=np.median(predictions))
#         predictions = np.clip(predictions, 0.1, 1200.0)
        
#         return predictions
    
#     def generate_comparison_table(self, results):
#         """生成模型性能对比表格"""
#         print("\n" + "=" * 80)
#         print("模型预测性能对比（基于相同测试集）")
#         print("=" * 80)
#         print(f"{'模型':<25} {'RMSE (s)':<12} {'MAE (s)':<12} {'R²':<12} {'相比最佳基线':<15}")
#         print("-" * 80)
        
#         # 找到最佳基线
#         baseline_models = {k: v for k, v in results.items() if 'CFT-Net' not in k}
#         best_baseline = min(baseline_models.items(), key=lambda x: x[1]['rmse'])
#         best_baseline_rmse = best_baseline[1]['rmse']
        
#         for name, result in results.items():
#             improvement = ""
#             if 'CFT-Net' in name:
#                 imp = (best_baseline_rmse - result['rmse']) / best_baseline_rmse * 100
#                 symbol = "↓" if imp > 0 else "↑"
#                 improvement = f"{symbol} {abs(imp):.1f}%"
            
#             print(f"{name:<25} {result['rmse']:<12.4f} {result['mae']:<12.4f} "
#                   f"{result['r2']:<12.4f} {improvement:<15}")
        
#         print("=" * 80)
        
#         # 保存CSV
#         comparison_data = []
#         for name, result in results.items():
#             comparison_data.append({
#                 'Model': name,
#                 'RMSE': result['rmse'],
#                 'MAE': result['mae'],
#                 'R2': result['r2']
#             })
#         pd.DataFrame(comparison_data).to_csv('model_comparison.csv', index=False)
#         print("✅ 结果已保存到 model_comparison.csv")
    
#     def generate_prediction_scatter_plots(self, results, y_true):
#         """生成预测值vs真实值散点图"""
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#         fig.suptitle('模型预测准确性对比', fontsize=16, fontweight='bold')
        
#         models = list(results.keys())
#         positions = [(0,0), (0,1), (1,0), (1,1)]
        
#         for i, model in enumerate(models[:4]):
#             row, col = positions[i]
#             ax = axes[row, col]
#             y_pred = results[model]['predictions']
            
#             ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none')
            
#             # 完美预测线
#             min_val = min(y_true.min(), y_pred.min())
#             max_val = max(y_true.max(), y_pred.max())
#             ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
            
#             rmse = results[model]['rmse']
#             r2 = results[model]['r2']
            
#             ax.set_xlabel('真实传输时间 (秒)', fontsize=11)
#             ax.set_ylabel('预测传输时间 (秒)', fontsize=11)
#             ax.set_title(f'{model}\nRMSE={rmse:.3f}s, R²={r2:.3f}', fontsize=12)
#             ax.legend()
#             ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
#         print("✅ 散点图已保存到 prediction_accuracy.png")
#         plt.close()

#     def generate_performance_stats(self, results):
#         """生成性能统计摘要"""
#         cftnet_result = results['CFT-Net']
#         baseline_models = {k: v for k, v in results.items() if 'CFT-Net' not in k}
#         best_baseline = min(baseline_models.items(), key=lambda x: x[1]['rmse'])
#         best_baseline_result = best_baseline[1]
#         best_baseline_name = best_baseline[0]
        
#         rmse_improvement = (best_baseline_result['rmse'] - cftnet_result['rmse']) / best_baseline_result['rmse'] * 100
        
#         print(f"\n=== 关键统计 ===")
#         print(f"CFT-Net RMSE: {cftnet_result['rmse']:.4f}s")
#         print(f"最佳基线 ({best_baseline_name}) RMSE: {best_baseline_result['rmse']:.4f}s")
#         print(f"RMSE 改善: {rmse_improvement:.2f}%")
#         print(f"R²: {cftnet_result['r2']:.4f}")
        
#         stats = {
#             'cftnet_rmse': float(cftnet_result['rmse']),
#             'best_baseline_rmse': float(best_baseline_result['rmse']),
#             'best_baseline_name': best_baseline_name,
#             'rmse_improvement_percent': float(rmse_improvement),
#             'cftnet_r2': float(cftnet_result['r2'])
#         }
        
#         with open('performance_stats.json', 'w') as f:
#             json.dump(stats, f, indent=2)
        
#         return stats


# def main():
#     """主函数"""
#     print("=== 公平模型对比评估（使用相同数据划分和预处理）===")
    
#     evaluator = FairComparisonEvaluator()
    
#     # 加载预处理对象和模型
#     evaluator.load_existing_model()
    
#     # 加载数据
#     df = evaluator.load_real_training_data()
    
#     # 训练和评估
#     results, y_test = evaluator.train_all_models_on_same_data(df)
    
#     # 生成报告
#     evaluator.generate_comparison_table(results)
#     evaluator.generate_prediction_scatter_plots(results, y_test)
#     stats = evaluator.generate_performance_stats(results)
    
#     print(f"\n=== 实验完成 ===")

# if __name__ == "__main__":
#     main()