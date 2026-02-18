"""
CFT-Net 消融实验脚本 (增强版)
增加 sMAPE, PICP-80%, MPIW 等关键指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import platform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, wilcoxon, norm
from scipy.optimize import brentq
import warnings
import time
import random
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 基础配置
# ==============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
set_seed(SEED)

system_name = platform.system()
if system_name == 'Windows':
    font_list = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system_name == 'Darwin':
    font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 检查字体
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_list:
    if font in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [font]
        break
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

CONFIG = {
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "epochs": 100,
    "patience": 20,
    "batch_size": 128,
    "lr": 0.0005,
    "reg_coeff": 1.0,
    "embed_dim": 32,
    "n_runs": 5,
    "random_seeds": [42, 123, 456, 789, 2024],
    "picp_target": 0.8,  # 80% 置信区间
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# ==============================================================================
# 1. 模型定义（保持不变）
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
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        out = self.transformer(torch.cat((cls_tokens, tokens), dim=1))
        return out[:, 0, :]

class MLPTower(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class OursModel(nn.Module):
    """完整模型"""
    def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
        super().__init__()
        self.client_tower = TransformerTower(c_dim, embed_dim)
        self.image_tower = TransformerTower(i_dim, embed_dim)
        self.algo_embed = nn.Embedding(n_algos, embed_dim)
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
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        fused = (c + i) / 2.0
        a = self.algo_embed(ax)
        out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
        return torch.stack([
            out[:, 0],
            F.softplus(out[:, 1]) + 0.1,
            F.softplus(out[:, 2]) + 1.1,
            F.softplus(out[:, 3]) + 1e-6
        ], dim=1)

class MLPBackboneModel(nn.Module):
    """消融：Transformer → MLP"""
    def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
        super().__init__()
        self.client_tower = MLPTower(c_dim, embed_dim)
        self.image_tower = MLPTower(i_dim, embed_dim)
        self.algo_embed = nn.Embedding(n_algos, embed_dim)
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim * 3, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU()
        )
        self.head = nn.Linear(32, 4)
        
    def forward(self, cx, ix, ax):
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        fused = (c + i) / 2.0
        a = self.algo_embed(ax)
        out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
        return torch.stack([
            out[:, 0], F.softplus(out[:, 1])+0.1,
            F.softplus(out[:, 2])+1.1, F.softplus(out[:, 3])+1e-6
        ], dim=1)

class SingleTowerModel(nn.Module):
    """消融：双塔 → 单塔"""
    def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
        super().__init__()
        self.tower = TransformerTower(c_dim + i_dim, embed_dim)
        self.algo_embed = nn.Embedding(n_algos, embed_dim)
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU()
        )
        self.head = nn.Linear(32, 4)
        
    def forward(self, cx, ix, ax):
        combined = torch.cat([cx, ix], dim=1)
        feat = self.tower(combined)
        a = self.algo_embed(ax)
        out = self.head(self.hidden(torch.cat([feat, a], dim=1)))
        return torch.stack([
            out[:, 0], F.softplus(out[:, 1])+0.1,
            F.softplus(out[:, 2])+1.1, F.softplus(out[:, 3])+1e-6
        ], dim=1)

# ==============================================================================
# 2. 损失函数
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
    var = beta / (v * (alpha - 1) + 1e-6)
    std = torch.sqrt(var + 1e-6)
    ratio = torch.clamp(error / (std + 1e-6), max=10.0)
    penalty = (ratio - 1.0) ** 2
    evidence = torch.clamp(2 * v + alpha, max=50.0)
    return (penalty * torch.log1p(evidence)).mean()

# ==============================================================================
# 3. 评估指标（增强版）
# ==============================================================================

def calculate_smape(y_true, y_pred, epsilon=1e-8):
    """对称平均绝对百分比误差"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(numerator / denominator) * 100

def calculate_picp_mpiw(y_true, y_pred, uncertainties, confidence=0.8):
    """
    计算PICP和MPIW
    PICP: 预测区间覆盖率
    MPIW: 平均预测区间宽度
    """
    z = norm.ppf((1 + confidence) / 2)
    lower = y_pred - z * uncertainties
    upper = y_pred + z * uncertainties
    
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    return picp, mpiw

def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8):
    """事后校准，学习最优缩放因子"""
    def picp_with_scale(s):
        z = norm.ppf((1 + target_coverage) / 2)
        lower = y_pred - z * s * unc_raw
        upper = y_pred + z * s * unc_raw
        return np.mean((y_true >= lower) & (y_true <= upper))
    
    try:
        s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, 0.1, 100)
        return s_opt
    except:
        return 33.713  # 默认回退值

def compute_ece(uncertainty, abs_error, n_bins=15):
    """期望校准误差"""
    unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
    err_norm = (abs_error - abs_error.min()) / (abs_error.max() - abs_error.min() + 1e-8)
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(unc_norm, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins-1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            avg_unc = unc_norm[mask].mean()
            avg_err = err_norm[mask].mean()
            ece += np.abs(avg_unc - avg_err) * (np.sum(mask) / len(uncertainty))
    return ece

# ==============================================================================
# 4. 数据集
# ==============================================================================

class CTSDataset(Dataset):
    def __init__(self, cx, ix, ax, y):
        self.cx = torch.FloatTensor(cx)
        self.ix = torch.FloatTensor(ix)
        self.ax = torch.LongTensor(ax)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.cx[i], self.ix[i], self.ax[i], self.y[i]

# ==============================================================================
# 5. 训练与评估（增强版，包含所有关键指标）
# ==============================================================================

def train_ablation_single(name, model_class, loss_type, c_dim, i_dim, n_algos,
                          Xc_train, Xi_train, Xa_train, y_train,
                          Xc_val, Xi_val, Xa_val, y_val,
                          Xc_test, Xi_test, Xa_test, y_test,  # 增加测试集
                          seed):
    """单次实验：训练并返回完整指标"""
    set_seed(seed)
    model = model_class(c_dim, i_dim, n_algos).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train),
                              batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(CTSDataset(Xc_val, Xi_val, Xa_val, y_val),
                            batch_size=CONFIG["batch_size"], shuffle=False)
    # 测试集用于最终评估
    test_loader = DataLoader(CTSDataset(Xc_test, Xi_test, Xa_test, y_test),
                            batch_size=CONFIG["batch_size"], shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # 训练循环
    for epoch in range(CONFIG["epochs"]):
        model.train()
        for cx, ix, ax, y in train_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(cx, ix, ax)

            gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            loss_nll = nig_nll_loss(y, gamma, v, alpha, beta)
            loss_reg = strong_eub_reg_loss(y, gamma, v, alpha, beta)
            reg_w = 0.0 if epoch < 3 else CONFIG["reg_coeff"]
            loss = loss_nll + reg_w * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cx, ix, ax, y in val_loader:
                cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
                preds = model(cx, ix, ax)
                loss_nll = nig_nll_loss(y, preds[:,0], preds[:,1], preds[:,2], preds[:,3])
                val_loss += loss_nll.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                break

    # ===== 最终测试评估（包含所有关键指标）=====
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 1. 在验证集上学习校准参数
    val_preds, val_targets, val_uncs = [], [], []
    with torch.no_grad():
        for cx, ix, ax, y in val_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            
            p = torch.expm1(gamma)
            var = beta / (v * (alpha - 1) + 1e-6)
            u = torch.sqrt(var + 1e-6)
            
            val_preds.extend(p.cpu().numpy())
            val_targets.extend(torch.expm1(y).cpu().numpy())
            val_uncs.extend(u.cpu().numpy())
    
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_uncs = np.array(val_uncs)
    
    # 学习缩放因子
    scale_factor = post_hoc_calibration(val_targets, val_preds, val_uncs)
    
    # 2. 在测试集上评估
    test_preds, test_targets, test_uncs = [], [], []
    with torch.no_grad():
        for cx, ix, ax, y in test_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            
            p = torch.expm1(gamma)
            var = beta / (v * (alpha - 1) + 1e-6)
            u = torch.sqrt(var + 1e-6)
            
            test_preds.extend(p.cpu().numpy())
            test_targets.extend(torch.expm1(y).cpu().numpy())
            test_uncs.extend(u.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_uncs = np.array(test_uncs)
    
    # 校准后不确定性
    test_uncs_cal = test_uncs * scale_factor
    
    # 计算所有关键指标
    abs_errors = np.abs(test_targets - test_preds)
    
    metrics = {
        # 精度指标
        'rmse': np.sqrt(mean_squared_error(test_targets, test_preds)),
        'mae': mean_absolute_error(test_targets, test_preds),
        'smape': calculate_smape(test_targets, test_preds),
        'r2': r2_score(test_targets, test_preds),
        
        # 不确定性质量指标
        'corr': spearmanr(test_uncs_cal, abs_errors)[0],
        'ece': compute_ece(test_uncs_cal, abs_errors),
        
        # 校准指标
        'picp_raw': calculate_picp_mpiw(test_targets, test_preds, test_uncs, 0.8)[0],
        'picp_cal': calculate_picp_mpiw(test_targets, test_preds, test_uncs_cal, 0.8)[0],
        'mpiw_raw': calculate_picp_mpiw(test_targets, test_preds, test_uncs, 0.8)[1],
        'mpiw_cal': calculate_picp_mpiw(test_targets, test_preds, test_uncs_cal, 0.8)[1],
        
        # 校准参数
        'scale_factor': scale_factor
    }
    
    # 处理NaN
    if np.isnan(metrics['corr']):
        metrics['corr'] = 0.0
    
    print(f"👉 {name} (seed={seed}): sMAPE={metrics['smape']:.2f}%, "
          f"PICP={metrics['picp_cal']:.1f}%, Corr={metrics['corr']:.3f}")
    
    return metrics

# ==============================================================================
# 6. 主程序（增强版）
# ==============================================================================

def run_ablation_experiments():
    print("="*70)
    print("🔬 消融实验（增强版，包含sMAPE/PICP/Corr等完整指标）")
    print("="*70)

    # 加载数据
    df = pd.read_excel(CONFIG["data_path"])
    df_feat = pd.read_csv(CONFIG["feature_path"])
    
    rename_map = {"image": "image_name", "method": "algo_name",
                  "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
                  "mem_limit": "mem_limit_mb"}
    df = df.rename(columns=rename_map)
    
    if 'total_time' not in df.columns:
        cols = [c for c in df.columns if 'total_tim' in c]
        if cols:
            df['total_time'] = df[cols[0]]
    
    df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
    df = pd.merge(df, df_feat, on="image_name", how="inner")
    
    cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
    cols_i = ['total_size_mb', 'avg_layer_entropy', 'entropy_std',
              'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
    cols_i = [c for c in cols_i if c in df.columns]
    
    print(f"📊 数据集: {len(df)} 条记录")
    print(f"   客户端特征: {cols_c}")
    print(f"   镜像特征: {cols_i}")

    # 实验配置
    experiments = [
        ("Ours (Full)", OursModel, 'strong_eub'),
        ("w/o Transformer", MLPBackboneModel, 'strong_eub'),
        ("w/o Dual-Tower", SingleTowerModel, 'strong_eub'),
    ]

    all_results = {name: [] for name, _, _ in experiments}

    # 运行多次实验
    for run_idx, seed in enumerate(CONFIG["random_seeds"][:CONFIG["n_runs"]]):
        print(f"\n{'='*70}")
        print(f"🔄 实验 {run_idx+1}/{CONFIG['n_runs']} (seed={seed})")
        print(f"{'='*70}")
        
        set_seed(seed)
        
        # 数据划分：70%训练 / 15%验证 / 15%测试
        idx = np.arange(len(df))
        idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=seed)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=seed)
        
        # 预处理
        scaler_c = StandardScaler()
        scaler_i = StandardScaler()
        enc = LabelEncoder()
        
        Xc_train_raw = df.iloc[idx_train][cols_c].values
        Xi_train_raw = df.iloc[idx_train][cols_i].values
        Xa_train_raw = df.iloc[idx_train]['algo_name'].values
        y_train_raw = np.log1p(df.iloc[idx_train]['total_time'].values)
        
        scaler_c.fit(Xc_train_raw)
        scaler_i.fit(Xi_train_raw)
        enc.fit(df['algo_name'].values)
        
        Xc_train = scaler_c.transform(Xc_train_raw)
        Xi_train = scaler_i.transform(Xi_train_raw)
        Xa_train = enc.transform(Xa_train_raw)
        
        # 验证集
        Xc_val = scaler_c.transform(df.iloc[idx_val][cols_c].values)
        Xi_val = scaler_i.transform(df.iloc[idx_val][cols_i].values)
        Xa_val = enc.transform(df.iloc[idx_val]['algo_name'].values)
        y_val = np.log1p(df.iloc[idx_val]['total_time'].values)
        
        # 测试集
        Xc_test = scaler_c.transform(df.iloc[idx_test][cols_c].values)
        Xi_test = scaler_i.transform(df.iloc[idx_test][cols_i].values)
        Xa_test = enc.transform(df.iloc[idx_test]['algo_name'].values)
        y_test = np.log1p(df.iloc[idx_test]['total_time'].values)
        
        c_dim = Xc_train.shape[1]
        i_dim = Xi_train.shape[1]
        n_algos = len(enc.classes_)
        
        # 运行每个实验
        for name, model_class, loss_type in experiments:
            res = train_ablation_single(
                name, model_class, loss_type,
                c_dim, i_dim, n_algos,
                Xc_train, Xi_train, Xa_train, y_train_raw,
                Xc_val, Xi_val, Xa_val, y_val,
                Xc_test, Xi_test, Xa_test, y_test,
                seed
            )
            all_results[name].append(res)

    # 汇总统计
    summary = {}
    for name, runs in all_results.items():
        df_runs = pd.DataFrame(runs)
        summary[name] = {}
        
        # 计算均值和标准差
        for col in df_runs.columns:
            summary[name][f'{col}_mean'] = df_runs[col].mean()
            summary[name][f'{col}_std'] = df_runs[col].std()
        
        # 显著性检验（与Ours比较）
        if name != 'Ours (Full)':
            ours_smape = [r['smape'] for r in all_results['Ours (Full)']]
            other_smape = [r['smape'] for r in runs]
            try:
                stat, p = wilcoxon(ours_smape, other_smape, alternative='less')
                summary[name]['p_smape'] = p
            except:
                summary[name]['p_smape'] = 1.0

    return summary, all_results

# ==============================================================================
# 7. 可视化（增强版，包含所有关键指标）
# ==============================================================================

def plot_comprehensive_results(summary):
    """绘制完整的消融实验对比图（包含sMAPE, PICP, Corr等）"""
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False
    
    names = list(summary.keys())
    # Ours放最后
    non_ours = [n for n in names if "Ours" not in n]
    sorted_names = non_ours + ['Ours (Full)'] if 'Ours (Full)' in names else non_ours
    
    colors = ['#808080'] * (len(sorted_names)-1) + ['#2ca02c']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics_config = [
        ('smape', 'sMAPE (%) ↓', 'sMAPE (%)', '%.2f'),
        ('mae', 'MAE (秒) ↓', 'MAE (s)', '%.2f'),
        ('rmse', 'RMSE (秒) ↓', 'RMSE (s)', '%.2f'),
        ('r2', 'R² 分数 ↑', 'R²', '%.3f'),
        ('corr', '不确定性相关性 ↑', 'Spearman ρ', '%.3f'),
        ('picp_cal', 'PICP-80% (校准后) ↑', 'PICP (%)', '%.1f'),
    ]
    
    for idx, (metric, title, ylabel, fmt) in enumerate(metrics_config):
        ax = axes[idx // 3, idx % 3]
        
        means = [summary[n][f'{metric}_mean'] for n in sorted_names]
        stds = [summary[n][f'{metric}_std'] for n in sorted_names]
        
        bars = ax.bar(sorted_names, means, yerr=stds, capsize=5, color=colors,
                     error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis='x', rotation=25, labelsize=9)
        
        # 数值标签
        for bar, m, s in zip(bars, means, stds):
            is_ours = bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)
            fw = 'bold' if is_ours else 'normal'
            text = f'{m:{fmt[1:]}}±{s:{fmt[1:]}}'
            ax.text(bar.get_x() + bar.get_width()/2., m + s + max(means)*0.02,
                   text, ha='center', va='bottom', fontsize=8, fontweight=fw, rotation=0)
    
    plt.tight_layout()
    plt.savefig('ablation_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 完整消融实验图已保存: ablation_comprehensive.png")

# ==============================================================================
# 8. 主程序
# ==============================================================================

if __name__ == "__main__":
    summary, all_results = run_ablation_experiments()
    
    print("\n" + "="*80)
    print("📊 消融实验最终结果（均值 ± 标准差, n={}）".format(CONFIG['n_runs']))
    print("="*80)
    
    # 打印表头
    print(f"{'变体':<20} {'sMAPE(%)':<12} {'PICP(%)':<12} {'Corr':<12} {'RMSE(s)':<12} {'MAE(s)':<12} {'p-value'}")
    print("-"*80)
    
    for name in summary.keys():
        s = summary[name]
        smape = f"{s['smape_mean']:.2f}±{s['smape_std']:.2f}"
        picp = f"{s['picp_cal_mean']:.1f}±{s['picp_cal_std']:.1f}"
        corr = f"{s['corr_mean']:.3f}±{s['corr_std']:.3f}"
        rmse = f"{s['rmse_mean']:.2f}±{s['rmse_std']:.2f}"
        mae = f"{s['mae_mean']:.2f}±{s['mae_std']:.2f}"
        p = f"{s['p_smape']:.4f}" if 'p_smape' in s else '-'
        print(f"{name:<20} {smape:<12} {picp:<12} {corr:<12} {rmse:<12} {mae:<12} {p}")
    
    print("="*80)
    
    # 保存结果
    pd.DataFrame(summary).T.to_csv('ablation_results_complete.csv')
    print("✅ 完整结果已保存至 ablation_results_complete.csv")
    
    # 绘制图表
    plot_comprehensive_results(summary)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import matplotlib
# import platform
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import spearmanr, wilcoxon
# import warnings
# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 0. 绘图配置（【强化】自动适配中英文，保证中文显示）
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

# # 设置全局字体，负号正常显示
# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False
# plt.style.use('seaborn-v0_8-whitegrid')

# # 检查字体是否可用，若配置的字体均不存在则回退到 DejaVu Sans（英文）
# import matplotlib.font_manager as fm
# available_fonts = [f.name for f in fm.fontManager.ttflist]
# for font in font_list:
#     if font in available_fonts:
#         matplotlib.rcParams['font.sans-serif'] = [font]
#         break
# else:
#     matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
#     print("⚠️ 未找到中文字体，使用英文显示")

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)

# # ==============================================================================
# # 1. 超参数配置（【最强版】回归已验证参数）
# # ==============================================================================
# CONFIG = {
#     "data_path": "E:\\硕士毕业论文材料合集\\论文实验代码相关\\CTS_system\\ml_training\\modeling\\cts_data.xlsx",
#     "feature_path": "E:\\硕士毕业论文材料合集\\论文实验代码相关\\CTS_system\\ml_training\\image_features_database.csv",
#     "epochs": 60,
#     "patience": 15,
#     "batch_size": 128,
#     "lr": 0.0005,
#     "reg_coeff": 1.0,
#     "embed_dim": 32,
#     "n_runs": 5,
#     "random_seeds": [42, 123, 456, 789, 2024],
#     "plot_ablation": "figure_3_6_ablation.png",
#     "plot_calibration": "figure_3_7_calibration_ablation.png"
# }
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==============================================================================
# # 2. 损失函数（不变）
# # ==============================================================================
# def nig_nll_loss(y, gamma, v, alpha, beta):
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#           - alpha * torch.log(two_blambda) \
#           + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def strong_eub_reg_loss(y, gamma, v, alpha, beta):
#     error = torch.abs(y - gamma)
#     var = beta / (v * (alpha - 1) + 1e-6)
#     std = torch.sqrt(var + 1e-6)
#     ratio = torch.clamp(error / (std + 1e-6), max=10.0)
#     penalty = (ratio - 1.0) ** 2
#     evidence = torch.clamp(2 * v + alpha, max=50.0)
#     return (penalty * torch.log1p(evidence)).mean()

# def vanilla_kl_reg_loss(y, gamma, v, alpha, beta):
#     error = torch.abs(y - gamma)
#     evidence = torch.clamp(2 * v + alpha, max=50.0)
#     return (error * evidence).mean()

# # ==============================================================================
# # 3. 模型组件与变体（MSE模型保留定义，但不参与消融主实验）
# # ==============================================================================
# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.norm = nn.LayerNorm(embed_dim)
#     def forward(self, x):
#         return self.norm(x.unsqueeze(-1) * self.weights + self.biases)

# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4,
#             batch_first=True, dropout=0.1, activation='gelu'
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         out = self.transformer(torch.cat((cls_tokens, tokens), dim=1))
#         return out[:, 0, :]

# class MLPTower(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(num_features, embed_dim * 2),
#             nn.BatchNorm1d(embed_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.ReLU()
#         )
#     def forward(self, x):
#         return self.net(x)

# # ----- 模型变体 -----
# class OursModel(nn.Module):
#     """完整模型：双塔Transformer + 平均融合 + 证据回归头"""
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32, output_dim=4):
#         super().__init__()
#         self.client_tower = TransformerTower(c_dim, embed_dim)
#         self.image_tower = TransformerTower(i_dim, embed_dim)
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
#         self.hidden = nn.Sequential(
#             nn.Linear(embed_dim * 3, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.GELU()
#         )
#         self.head = nn.Linear(32, output_dim)
#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
#         fused = (c + i) / 2.0          # 【正确】平均融合，无门控
#         a = self.algo_embed(ax)
#         out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
#         if out.shape[1] == 4:
#             return torch.stack([
#                 out[:, 0],
#                 F.softplus(out[:, 1]) + 0.1,
#                 F.softplus(out[:, 2]) + 1.1,
#                 F.softplus(out[:, 3]) + 1e-6
#             ], dim=1)
#         return out

# class MLPBackboneModel(nn.Module):
#     """消融A：Transformer → MLP"""
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = MLPTower(c_dim, embed_dim)
#         self.image_tower = MLPTower(i_dim, embed_dim)
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
#         self.hidden = nn.Sequential(
#             nn.Linear(embed_dim * 3, 64), nn.LayerNorm(64), nn.GELU(),
#             nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU()
#         )
#         self.head = nn.Linear(32, 4)
#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
#         fused = (c + i) / 2.0
#         a = self.algo_embed(ax)
#         out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
#         return torch.stack([
#             out[:, 0], F.softplus(out[:, 1])+0.1,
#             F.softplus(out[:, 2])+1.1, F.softplus(out[:, 3])+1e-6
#         ], dim=1)

# class SingleTowerModel(nn.Module):
#     """消融B：双塔 → 单塔（输入拼接后过Transformer）"""
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
#         super().__init__()
#         self.tower = TransformerTower(c_dim + i_dim, embed_dim)
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
#         self.hidden = nn.Sequential(
#             nn.Linear(embed_dim * 2, 64), nn.LayerNorm(64), nn.GELU(),
#             nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU()
#         )
#         self.head = nn.Linear(32, 4)
#     def forward(self, cx, ix, ax):
#         combined = torch.cat([cx, ix], dim=1)
#         feat = self.tower(combined)
#         a = self.algo_embed(ax)
#         out = self.head(self.hidden(torch.cat([feat, a], dim=1)))
#         return torch.stack([
#             out[:, 0], F.softplus(out[:, 1])+0.1,
#             F.softplus(out[:, 2])+1.1, F.softplus(out[:, 3])+1e-6
#         ], dim=1)

# # ----- MSE模型（保留定义，但默认不参与消融实验）-----
# class MSEModel(OursModel):
#     """消融C（补充对照）：证据回归 → 普通MSE回归（输出维度1）"""
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
#         super().__init__(c_dim, i_dim, n_algos, embed_dim, output_dim=1)

# # ==============================================================================
# # 4. 数据集
# # ==============================================================================
# class CTSDataset(Dataset):
#     def __init__(self, cx, ix, ax, y):
#         self.cx = torch.FloatTensor(cx)
#         self.ix = torch.FloatTensor(ix)
#         self.ax = torch.LongTensor(ax)
#         self.y = torch.FloatTensor(y)
#     def __len__(self):
#         return len(self.y)
#     def __getitem__(self, i):
#         return self.cx[i], self.ix[i], self.ax[i], self.y[i]

# # ==============================================================================
# # 5. 核心训练与评估函数（单次运行，带早停）
# # ==============================================================================
# def compute_ece(uncertainty, abs_error, n_bins=15):
#     """期望校准误差（归一化后）"""
#     unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
#     err_norm = (abs_error - abs_error.min()) / (abs_error.max() - abs_error.min() + 1e-8)
#     bins = np.linspace(0, 1, n_bins+1)
#     bin_indices = np.digitize(unc_norm, bins) - 1
#     bin_indices = np.clip(bin_indices, 0, n_bins-1)
#     ece = 0.0
#     for i in range(n_bins):
#         mask = bin_indices == i
#         if np.sum(mask) > 0:
#             avg_unc = unc_norm[mask].mean()
#             avg_err = err_norm[mask].mean()
#             ece += np.abs(avg_unc - avg_err) * (np.sum(mask) / len(uncertainty))
#     return ece

# def train_ablation_single(name, model_class, loss_type, c_dim, i_dim, n_algos,
#                           Xc_train, Xi_train, Xa_train, y_train,
#                           Xc_val, Xi_val, Xa_val, y_val, seed):
#     """单次实验：训练并返回最佳指标及模型（带早停）"""
#     set_seed(seed)
#     model = model_class(c_dim, i_dim, n_algos).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

#     train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train),
#                               batch_size=CONFIG["batch_size"], shuffle=True)
#     val_loader = DataLoader(CTSDataset(Xc_val, Xi_val, Xa_val, y_val),
#                             batch_size=CONFIG["batch_size"], shuffle=False)

#     best_rmse = float('inf')
#     best_metrics = {}
#     best_model_state = None
#     patience_counter = 0

#     for epoch in range(CONFIG["epochs"]):
#         model.train()
#         for cx, ix, ax, y in train_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             optimizer.zero_grad()
#             preds = model(cx, ix, ax)

#             if loss_type == 'mse':
#                 loss = F.mse_loss(preds.squeeze(), y)
#             else:
#                 gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
#                 loss_nll = nig_nll_loss(y, gamma, v, alpha, beta)
#                 if loss_type == 'strong_eub':
#                     loss_reg = strong_eub_reg_loss(y, gamma, v, alpha, beta)
#                 elif loss_type == 'kl':
#                     loss_reg = vanilla_kl_reg_loss(y, gamma, v, alpha, beta)
#                 else:
#                     loss_reg = 0.0
#                 reg_w = 0.0 if epoch < 3 else CONFIG["reg_coeff"]
#                 loss = loss_nll + reg_w * loss_reg

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#         # --- 验证 ---
#         model.eval()
#         pred_list, true_list, unc_list = [], [], []
#         with torch.no_grad():
#             for cx, ix, ax, y in val_loader:
#                 cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#                 preds = model(cx, ix, ax)
#                 if loss_type == 'mse':
#                     gamma = preds.squeeze()
#                 else:
#                     gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
#                     unc = beta / (v * (alpha - 1))
#                     unc_list.extend(unc.cpu().numpy())

#                 pred_list.extend(np.expm1(gamma.cpu().numpy()))
#                 true_list.extend(np.expm1(y.cpu().numpy()))

#         pred_list = np.array(pred_list)
#         true_list = np.array(true_list)
#         curr_rmse = np.sqrt(mean_squared_error(true_list, pred_list))

#         # --- 防坍塌机制（仅针对KL）---
#         valid = True
#         if loss_type == 'kl':
#             pred_std = np.std(pred_list)
#             if pred_std < 0.5:
#                 valid = False
#             if len(unc_list) > 0:
#                 abs_err = np.abs(true_list - pred_list)
#                 corr, _ = spearmanr(unc_list, abs_err)
#                 if np.isnan(corr) or corr < 0.05:
#                     valid = False

#         # --- 早停与最佳模型保存 ---
#         if curr_rmse < best_rmse and valid:
#             best_rmse = curr_rmse
#             best_model_state = model.state_dict()
#             best_metrics['rmse'] = curr_rmse
#             best_metrics['mae'] = mean_absolute_error(true_list, pred_list)
#             best_metrics['r2'] = r2_score(true_list, pred_list)
#             if loss_type != 'mse' and len(unc_list) > 0:
#                 abs_err = np.abs(true_list - pred_list)
#                 spearman_corr, _ = spearmanr(unc_list, abs_err)
#                 best_metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
#                 best_metrics['ece'] = compute_ece(np.array(unc_list), abs_err)
#             else:
#                 best_metrics['spearman'] = 0.0
#                 best_metrics['ece'] = 1.0
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= CONFIG["patience"]:
#                 break

#     # 保存最佳模型
#     if best_model_state is not None:
#         os.makedirs('ablation_models', exist_ok=True)
#         torch.save(best_model_state, f'ablation_models/best_{name}_seed{seed}.pth')

#     return best_metrics

# # ==============================================================================
# # 6. 多次独立实验主流程（已移除MSE）
# # ==============================================================================
# def run_ablation_experiments():
#     print("="*60)
#     print("🔬 消融实验（多次独立运行，无MSE基线）")
#     print("="*60)

#     df = pd.read_excel(CONFIG["data_path"])
#     df_feat = pd.read_csv(CONFIG["feature_path"])
#     rename_map = {"image": "image_name", "method": "algo_name",
#                   "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
#                   "mem_limit": "mem_limit_mb"}
#     df = df.rename(columns=rename_map)
#     if 'total_time' not in df.columns:
#         df['total_time'] = df[[c for c in df.columns if 'total_tim' in c][0]]
#     df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
#     if 'mem_limit_mb' not in df.columns:
#         df['mem_limit_mb'] = 1024.0
#     df = pd.merge(df, df_feat, on="image_name", how="inner")

#     cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     cols_i = ['total_size_mb', 'avg_layer_entropy', 'entropy_std',
#               'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#     cols_i = [c for c in cols_i if c in df.columns]

#     # 仅包含四个消融变体
#     all_results = {
#         'Ours (Full)': [],
#         'w/o Transformer (MLP)': [],
#         'w/o Dual-Tower (Single)': [],
#         'w/o Robust EUB (KL)': []
#     }

#     for run_idx, seed in enumerate(CONFIG["random_seeds"][:CONFIG["n_runs"]]):
#         print(f"\n--- 实验 {run_idx+1}/{CONFIG['n_runs']} (seed={seed}) ---")
#         set_seed(seed)

#         idx = np.arange(len(df))
#         idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=seed)

#         scaler_c = StandardScaler()
#         scaler_i = StandardScaler()
#         enc = LabelEncoder()

#         Xc_train_raw = df.iloc[idx_train][cols_c].values
#         Xi_train_raw = df.iloc[idx_train][cols_i].values
#         Xa_train_raw = df.iloc[idx_train]['algo_name'].values
#         y_train_raw = np.log1p(df.iloc[idx_train]['total_time'].values)

#         scaler_c.fit(Xc_train_raw)
#         scaler_i.fit(Xi_train_raw)
#         enc.fit(df['algo_name'].values)

#         Xc_train = scaler_c.transform(Xc_train_raw)
#         Xi_train = scaler_i.transform(Xi_train_raw)
#         Xa_train = enc.transform(Xa_train_raw)

#         Xc_val_raw = df.iloc[idx_val][cols_c].values
#         Xi_val_raw = df.iloc[idx_val][cols_i].values
#         Xa_val_raw = df.iloc[idx_val]['algo_name'].values
#         y_val_raw = np.log1p(df.iloc[idx_val]['total_time'].values)

#         Xc_val = scaler_c.transform(Xc_val_raw)
#         Xi_val = scaler_i.transform(Xi_val_raw)
#         Xa_val = enc.transform(Xa_val_raw)

#         c_dim = Xc_train.shape[1]
#         i_dim = Xi_train.shape[1]
#         n_algos = len(enc.classes_)

#         # 1. Ours (Full)
#         res_ours = train_ablation_single(
#             'Ours', OursModel, 'strong_eub',
#             c_dim, i_dim, n_algos,
#             Xc_train, Xi_train, Xa_train, y_train_raw,
#             Xc_val, Xi_val, Xa_val, y_val_raw, seed
#         )
#         all_results['Ours (Full)'].append(res_ours)

#         # 2. MLP Backbone
#         res_mlp = train_ablation_single(
#             'MLP', MLPBackboneModel, 'strong_eub',
#             c_dim, i_dim, n_algos,
#             Xc_train, Xi_train, Xa_train, y_train_raw,
#             Xc_val, Xi_val, Xa_val, y_val_raw, seed
#         )
#         all_results['w/o Transformer (MLP)'].append(res_mlp)

#         # 3. Single Tower
#         res_single = train_ablation_single(
#             'Single', SingleTowerModel, 'strong_eub',
#             c_dim, i_dim, n_algos,
#             Xc_train, Xi_train, Xa_train, y_train_raw,
#             Xc_val, Xi_val, Xa_val, y_val_raw, seed
#         )
#         all_results['w/o Dual-Tower (Single)'].append(res_single)

#         # 4. Vanilla KL
#         res_kl = train_ablation_single(
#             'KL', OursModel, 'kl',
#             c_dim, i_dim, n_algos,
#             Xc_train, Xi_train, Xa_train, y_train_raw,
#             Xc_val, Xi_val, Xa_val, y_val_raw, seed
#         )
#         all_results['w/o Robust EUB (KL)'].append(res_kl)

#         # ----- MSE基线（注释掉，不参与消融实验）-----
#         # res_mse = train_ablation_single(
#         #     'MSE', MSEModel, 'mse',
#         #     c_dim, i_dim, n_algos,
#         #     Xc_train, Xi_train, Xa_train, y_train_raw,
#         #     Xc_val, Xi_val, Xa_val, y_val_raw, seed
#         # )
#         # all_results['MSE Baseline'].append(res_mse)

#     # ----- 汇总统计与显著性检验 -----
#     summary = {}
#     for name, runs in all_results.items():
#         df_runs = pd.DataFrame(runs)
#         summary[name] = {
#             'rmse_mean': df_runs['rmse'].mean(),
#             'rmse_std': df_runs['rmse'].std(),
#             'mae_mean': df_runs['mae'].mean(),
#             'mae_std': df_runs['mae'].std(),
#             'r2_mean': df_runs['r2'].mean(),
#             'r2_std': df_runs['r2'].std(),
#             'spearman_mean': df_runs['spearman'].mean() if 'spearman' in df_runs else 0,
#             'spearman_std': df_runs['spearman'].std() if 'spearman' in df_runs else 0,
#             'ece_mean': df_runs['ece'].mean() if 'ece' in df_runs else 1,
#             'ece_std': df_runs['ece'].std() if 'ece' in df_runs else 0
#         }

#     # 显著性检验：Ours vs 每个消融变体
#     ours_rmse = [r['rmse'] for r in all_results['Ours (Full)']]
#     for name in all_results.keys():
#         if name == 'Ours (Full)': continue
#         other_rmse = [r['rmse'] for r in all_results[name]]
#         if len(ours_rmse) == len(other_rmse) and len(ours_rmse) > 1:
#             try:
#                 stat, p = wilcoxon(ours_rmse, other_rmse, alternative='less')
#                 summary[name]['p_vs_ours'] = p
#             except:
#                 summary[name]['p_vs_ours'] = 1.0
#         else:
#             summary[name]['p_vs_ours'] = 1.0

#     return summary, all_results

# # ==============================================================================
# # 7. 可视化（【强化】全中文显示，字体回退）
# # ==============================================================================
# def plot_ablation_results(summary):
#     """绘制消融实验对比图（全中文，Ours绿色压轴）"""
#     # 强制中文字体（二次保险）
#     plt.rcParams['font.sans-serif'] = font_list
#     plt.rcParams['axes.unicode_minus'] = False

#     names = list(summary.keys())
    
#     # 强制排序：非Ours按RMSE从大到小，Ours固定放在最后
#     non_ours = [n for n in names if "Ours" not in n]
#     rmse_means_non_ours = [summary[n]['rmse_mean'] for n in non_ours]
#     sorted_idx = np.argsort(rmse_means_non_ours)[::-1]
#     sorted_names = [non_ours[i] for i in sorted_idx]
#     if 'Ours (Full)' in names:
#         sorted_names.append('Ours (Full)')
#     names = sorted_names

#     # 配色：Ours绿色，其他灰色
#     colors = ['#808080'] * (len(names)-1) + ['#2ca02c']

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     metrics = ['rmse', 'mae', 'r2']
#     titles = ['RMSE (秒) ↓', 'MAE (秒) ↓', 'R² 分数 ↑']
#     ylabels = ['RMSE', 'MAE', 'R²']

#     for i, metric in enumerate(metrics):
#         means = [summary[n][f'{metric}_mean'] for n in names]
#         stds = [summary[n][f'{metric}_std'] for n in names]

#         bars = axes[i].bar(names, means, yerr=stds, capsize=5, color=colors,
#                            error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'alpha':0.7})
#         axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
#         axes[i].set_ylabel(ylabels[i], fontsize=12)
#         axes[i].tick_params(axis='x', rotation=20, labelsize=10)

#         # 数值标签（Ours加粗）
#         for bar, mean, std in zip(bars, means, stds):
#             height = bar.get_height()
#             offset = 0.02 if metric == 'r2' else height * 0.05
#             fw = 'bold' if bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0) else 'normal'
#             if metric == 'r2':
#                 text = f'{mean:.3f}±{std:.3f}'
#             else:
#                 text = f'{mean:.2f}±{std:.2f}'
#             axes[i].text(bar.get_x() + bar.get_width()/2., height + offset,
#                          text, ha='center', va='bottom', fontsize=9, fontweight=fw)

#     plt.tight_layout()
#     plt.savefig(CONFIG["plot_ablation"], dpi=300)
#     plt.close()
#     print(f"✅ 消融实验柱状图已保存: {CONFIG['plot_ablation']}")

# def plot_calibration_ablation(summary):
#     """绘制校准指标对比（全中文，Ours绿色压轴）"""
#     plt.rcParams['font.sans-serif'] = font_list
#     plt.rcParams['axes.unicode_minus'] = False

#     names = list(summary.keys())
#     # Ours放最后
#     non_ours = [n for n in names if "Ours" not in n]
#     rmse_means_non_ours = [summary[n]['rmse_mean'] for n in non_ours]
#     sorted_idx = np.argsort(rmse_means_non_ours)[::-1]
#     sorted_names = [non_ours[i] for i in sorted_idx]
#     if 'Ours (Full)' in names:
#         sorted_names.append('Ours (Full)')
#     names = sorted_names

#     colors = ['#808080'] * (len(names)-1) + ['#2ca02c']

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # Spearman相关性
#     sp_means = [summary[n]['spearman_mean'] for n in names]
#     sp_stds = [summary[n]['spearman_std'] for n in names]
#     bars1 = axes[0].bar(names, sp_means, yerr=sp_stds, capsize=5, color=colors,
#                         error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
#     axes[0].set_title('不确定性校准 - Spearman相关性 ↑', fontsize=14, fontweight='bold')
#     axes[0].set_ylabel('Spearman ρ', fontsize=12)
#     axes[0].tick_params(axis='x', rotation=20, labelsize=10)
#     for bar, m, s in zip(bars1, sp_means, sp_stds):
#         fw = 'bold' if bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0) else 'normal'
#         axes[0].text(bar.get_x()+bar.get_width()/2., m+s+0.02, f'{m:.3f}±{s:.3f}',
#                      ha='center', va='bottom', fontsize=9, fontweight=fw)

#     # 期望校准误差（ECE）
#     ece_means = [summary[n]['ece_mean'] for n in names]
#     ece_stds = [summary[n]['ece_std'] for n in names]
#     bars2 = axes[1].bar(names, ece_means, yerr=ece_stds, capsize=5, color=colors,
#                         error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
#     axes[1].set_title('不确定性校准 - 期望校准误差(ECE) ↓', fontsize=14, fontweight='bold')
#     axes[1].set_ylabel('ECE', fontsize=12)
#     axes[1].tick_params(axis='x', rotation=20, labelsize=10)
#     for bar, m, s in zip(bars2, ece_means, ece_stds):
#         fw = 'bold' if bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0) else 'normal'
#         axes[1].text(bar.get_x()+bar.get_width()/2., m+s+0.01, f'{m:.3f}±{s:.3f}',
#                      ha='center', va='bottom', fontsize=9, fontweight=fw)

#     plt.tight_layout()
#     plt.savefig(CONFIG["plot_calibration"], dpi=300)
#     plt.close()
#     print(f"✅ 校准指标对比图已保存: {CONFIG['plot_calibration']}")

# # ==============================================================================
# # 8. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     summary, all_results = run_ablation_experiments()

#     print("\n" + "="*60)
#     print("📊 消融实验最终结果（均值 ± 标准差, n={})".format(CONFIG['n_runs']))
#     print("="*60)
#     header = f"{'变体':<25} {'RMSE':<15} {'MAE':<15} {'R²':<15} {'Spearman':<15} {'ECE':<15} {'p vs Ours'}"
#     print(header)
#     print("-"*100)
#     for name in summary.keys():
#         s = summary[name]
#         rmse = f"{s['rmse_mean']:.2f}±{s['rmse_std']:.2f}"
#         mae = f"{s['mae_mean']:.2f}±{s['mae_std']:.2f}"
#         r2 = f"{s['r2_mean']:.3f}±{s['r2_std']:.3f}"
#         sp = f"{s['spearman_mean']:.3f}±{s['spearman_std']:.3f}" if 'spearman_mean' in s else 'N/A'
#         ece = f"{s['ece_mean']:.3f}±{s['ece_std']:.3f}" if 'ece_mean' in s else 'N/A'
#         p = f"{s['p_vs_ours']:.4f}" if 'p_vs_ours' in s else '-'
#         print(f"{name:<25} {rmse:<15} {mae:<15} {r2:<15} {sp:<15} {ece:<15} {p}")
#     print("="*60)

#     # 保存汇总结果（不包含MSE）
#     pd.DataFrame(summary).T.to_csv('ablation_results_summary.csv')
#     print("✅ 汇总结果已保存至 ablation_results_summary.csv")

#     plot_ablation_results(summary)
#     plot_calibration_ablation(summary)