
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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
from datetime import datetime
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
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# ==============================================================================
# 1. 超参数配置（修复版：调整初始化参数）
# ==============================================================================
CONFIG = {
    "lr": 0.0005,
    "weight_decay": 5e-5,
    "epochs": 300,
    # "epochs": 150,
    "patience": 40,
    # "patience": 30,
    "batch_size": 64,
    
    "embed_dim": 32,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 32,
    
    # EDL参数
    "reg_coeff": 0.0050,
    "warmup_epochs": 30,
    "alpha_init": 1.5,
    "beta_init": 1.0,
    "v_init": 0.5,
    
    # PICP目标（仅用于监控）
    "picp_target": 0.8,
    
    # 数据路径
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "model_save_path": f"cts_fixed_{datetime.now().strftime('%m%d_%H%M')}_seed{SEED}.pth",
    
    # 评估权重
    "mape_weight": 0.4,
    "corr_weight": 0.3,
    "ece_weight": 0.3,
    
    "ema_alpha": 0.9,
    "winsorize_limits": 0.05,
    "use_winsorized_for_selection": False,
    
    "use_mixup": True,
    "mixup_alpha": 0.2,
}

# ==============================================================================
# 2. 模型架构
# ==============================================================================
class LightweightFeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        out = x * self.embeddings + self.bias
        return self.norm(out)

class LightweightTransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim=16, nhead=2):
        super().__init__()
        self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=CONFIG["dim_feedforward"],
            batch_first=True, 
            dropout=0.1,
            activation="gelu"
        )
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        out = self.encoder(x)
        return out[:, 0, :]

class CompactCFTNet(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=16):
        super().__init__()
        self.client_tower = LightweightTransformerTower(client_feats, embed_dim, nhead=2)
        self.image_tower = LightweightTransformerTower(image_feats, embed_dim, nhead=2)
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
        
        self._init_weights()
        self.alpha_init = CONFIG["alpha_init"]
        self.beta_init = CONFIG["beta_init"]
        self.v_init = CONFIG["v_init"]
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, cx, ix, ax):
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        a = self.algo_embed(ax)
        
        fused = torch.cat([c, i, a], dim=-1)
        out = self.fusion(fused)
        
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + self.v_init
        alpha = F.softplus(out[:, 2]) + self.alpha_init
        beta = F.softplus(out[:, 3]) + self.beta_init   # 使用配置值，保持一致性
        
        return torch.stack([gamma, v, alpha, beta], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ==============================================================================
# 3. 损失函数（PICP仅作为监控，不参与梯度）
# ==============================================================================
def nig_nll_loss(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()

def improved_eub_loss(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    var = beta / (v * (alpha - 1) + 1e-6)
    std = torch.sqrt(var)
    
    confidence = 1.0 / (std + 1e-6)
    raw_ratio = error * confidence
    ratio = torch.clamp(raw_ratio, max=10.0)
    
    penalty = (ratio - 1.0) ** 2
    evidence = torch.clamp(2 * v + alpha, max=20.0)
    reg = penalty * torch.log1p(evidence)
    
    return reg.mean()

def calculate_picp_mpiw_torch(y_true, y_pred, uncertainties, confidence=0.8):
    """
    PyTorch版本的PICP/MPIW计算（用于训练时监控）
    """
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    lower = y_pred - z * uncertainties
    upper = y_pred + z * uncertainties
    
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    picp = covered.mean().item()
    mpiw = (upper - lower).mean().item()
    return picp, mpiw

def calculate_score(metrics):
    """
    多目标综合得分计算（越小越好）
    所有子指标归一化到[0,1]区间，确保可比性
    """
    # 1. sMAPE: 假设优秀<10%，可接受<30%，差>50%
    smape_raw = metrics['sMAPE']
    smape_norm = min(smape_raw / 30.0, 2.0)  # 30%->1.0, 60%->2.0(封顶)
    
    # 2. Corr: 高相关性好(0)，低相关性差(1)
    # 使用(1-corr)/2将[-1,1]映射到[0,1]，再取绝对值确保单调
    corr_raw = metrics['Corr']
    corr_norm = (1.0 - corr_raw) / 2.0  # Corr=1->0, Corr=0->0.5, Corr=-1->1
    
    # 3. ECE: 校准误差，假设优秀<5，可接受<20，差>50
    ece_raw = metrics['ECE']
    ece_norm = min(ece_raw / 20.0, 2.0)  # 20->1.0, 40->2.0(封顶)
    
    # 4. PICP: 目标是80%，偏差越大越差
    picp_raw = metrics['PICP_80']
    picp_gap = abs(picp_raw - 80.0) / 80.0  # 偏差百分比，0~1.25(当PICP=0或160)
    picp_norm = min(picp_gap, 2.0)  # 封顶2.0
    
    # 5. 综合得分（加权平均）
    # 权重和=1.0，确保可比性
    score = (0.5 * smape_norm + 
             0.2 * corr_norm + 
             0.15 * ece_norm + 
             0.15 * picp_norm)
    
    # 调试信息（训练时打印）
    if False:  # 设为True查看细节
        print(f"sMAPE:{smape_raw:.2f}->{smape_norm:.3f} | "
              f"Corr:{corr_raw:.3f}->{corr_norm:.3f} | "
              f"ECE:{ece_raw:.2f}->{ece_norm:.3f} | "
              f"PICP:{picp_raw:.1f}->{picp_norm:.3f} | "
              f"Score:{score:.4f}")
    
    return score

def evidential_loss(pred, target, epoch, config):
    """
    组合损失：NLL + EUB（PICP仅作为监控指标，不参与梯度）
    """
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    
    loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
    loss_reg = improved_eub_loss(target, gamma, v, alpha, beta)
    
    # 计算不确定性（始终需要，用于返回均值）
    var = beta / (v * (alpha - 1) + 1e-6)
    uncertainties = torch.sqrt(var + 1e-6)
    
    # PICP监控值（仅用于训练日志）
    picp_val = 0.0
    if epoch >= config["warmup_epochs"]:
        with torch.no_grad():
            y_true_time = torch.expm1(target)
            y_pred_time = torch.expm1(gamma)
            picp_val, _ = calculate_picp_mpiw_torch(
                y_true_time, y_pred_time, uncertainties, config["picp_target"]
            )
    
    # 动态权重（仅对EUB正则化）
    if epoch < config["warmup_epochs"]:
        reg_weight = 0.0
    else:
        progress = min(1.0, (epoch - config["warmup_epochs"]) / 10)
        reg_weight = config["reg_coeff"] * progress
    
    total_loss = loss_nll + reg_weight * loss_reg
    return total_loss, loss_nll.item(), loss_reg.item(), picp_val, uncertainties.mean().item()
# ==============================================================================
# 4. 数据加载
# ==============================================================================
class MixupCTSDataset(Dataset):
    def __init__(self, cx, ix, ax, y, use_mixup=True, alpha=0.2):
        self.cx = torch.FloatTensor(cx)
        self.ix = torch.FloatTensor(ix)
        self.ax = torch.LongTensor(ax)
        self.y = torch.FloatTensor(y)
        self.use_mixup = use_mixup
        self.alpha = alpha
        self.num_samples = len(y)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if not self.use_mixup or random.random() > 0.5:
            return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]
        else:
            idx2 = random.randint(0, self.num_samples - 1)
            lam = np.random.beta(self.alpha, self.alpha)
            
            cx_mix = lam * self.cx[idx] + (1 - lam) * self.cx[idx2]
            ix_mix = lam * self.ix[idx] + (1 - lam) * self.ix[idx2]
            y_mix = lam * self.y[idx] + (1 - lam) * self.y[idx2]
            ax_mix = self.ax[idx] if lam > 0.5 else self.ax[idx2]
            
            return cx_mix, ix_mix, ax_mix, y_mix

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
        
        tiny_samples = (df['total_time'] < 0.5).sum()
        print(f"  极小值样本: {tiny_samples} 条 (<0.5s, {tiny_samples/len(df)*100:.2f}%)")
        
        cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
                       'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
        cols_i = [c for c in target_cols if c in df.columns]
        
        Xc_raw = df[cols_c].values
        Xi_raw = df[cols_i].values
        y_raw = np.log1p(df['total_time'].values)
        algo_names_raw = df['algo_name'].values
        
        print(f"✅ 数据加载成功: {len(y_raw)} 条")
        print(f"   时间范围: [{df['total_time'].min():.2f}s, {df['total_time'].max():.2f}s]")
        print(f"   客户端特征{len(cols_c)}个: {cols_c}")
        print(f"   镜像特征{len(cols_i)}个: {cols_i}")
        
        return Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i
        
    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# 5. 评估指标
# ==============================================================================
def calculate_mape(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_smape(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(numerator / denominator) * 100

def calculate_ece_quantile(errors, uncertainties, n_bins=10):
    if len(errors) == 0:
        return 0.0
    
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(uncertainties, quantiles)
    bin_boundaries[-1] += 1e-8
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i+1])
        if i == n_bins - 1:
            in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties <= bin_boundaries[i+1])
        
        prop_in_bin = in_bin.sum() / len(errors)
        if prop_in_bin > 0:
            avg_unc = uncertainties[in_bin].mean()
            avg_err = errors[in_bin].mean()
            ece += np.abs(avg_err - avg_unc) * prop_in_bin
    
    return ece

def calculate_picp_mpiw(y_true, y_pred, uncertainties, confidence=0.8):
    z = norm.ppf((1 + confidence) / 2)
    lower = y_pred - z * uncertainties
    upper = y_pred + z * uncertainties
    
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    return picp, mpiw

def convert_to_native(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(v) for v in obj]
    return obj

from scipy.optimize import brentq

def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, search_range=(0.1, 50)):
    """
    在验证集上学习最优缩放因子，使PICP达到目标覆盖率
    """
    def picp_with_scale(s):
        lower = y_pred - s * unc_raw
        upper = y_pred + s * unc_raw
        return np.mean((y_true >= lower) & (y_true <= upper))
    
    s_min, s_max = search_range
    picp_min = picp_with_scale(s_min)
    picp_max = picp_with_scale(s_max)
    
    if picp_min >= target_coverage:
        return s_min
    elif picp_max <= target_coverage:
        print(f"警告：最大缩放 {s_max} 仅能达到 PICP={picp_max:.1f}%，小于目标 {target_coverage*100}%")
        return s_max
    else:
        try:
            s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, s_min, s_max)
            return s_opt
        except:
            scales = np.linspace(s_min, s_max, 200)
            picps = [picp_with_scale(s) for s in scales]
            best_idx = np.argmin(np.abs(np.array(picps) - target_coverage))
            return scales[best_idx]

# class MetricsNormalizer:
#     def __init__(self, window_size=5, ema_alpha=0.9):
#         self.smape_history = []
#         self.ece_history = []
#         self.window_size = window_size
#         self.ema_alpha = ema_alpha
#         self.smape_scale = 100.0
#         self.ece_scale = 50.0
        
#     def update(self, smape, ece):
#         self.smape_history.append(smape)
#         self.ece_history.append(ece)
        
#         if len(self.smape_history) >= self.window_size:
#             recent_smape = self.smape_history[-self.window_size:]
#             recent_ece = self.ece_history[-self.window_size:]
            
#             new_smape_scale = np.percentile(recent_smape, 90)
#             new_ece_scale = np.percentile(recent_ece, 90)
            
#             self.smape_scale = self.ema_alpha * self.smape_scale + (1 - self.ema_alpha) * new_smape_scale
#             self.ece_scale = self.ema_alpha * self.ece_scale + (1 - self.ema_alpha) * new_ece_scale
            
#     def normalize(self, smape, corr, ece):
#         smape_norm = smape / (self.smape_scale + 1e-8)
#         ece_norm = ece / (self.ece_scale + 1e-8)
#         corr_norm = (1.0 - corr) / 2.0
#         return smape_norm, corr_norm, ece_norm

# ==============================================================================
# 6. 训练主循环（修复版）
# ==============================================================================
def train_model():
    data = load_data()
    if data is None:
        return None
        
    Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i = data
    N = len(y_raw)
    
    idx = np.random.permutation(N)
    n_tr, n_val = int(N * 0.7), int(N * 0.15)
    tr_idx, val_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]
    
    print(f"\n📊 数据集划分: 训练 {len(tr_idx)} | 验证 {len(val_idx)} | 测试 {len(te_idx)}")
    
    scaler_c = StandardScaler().fit(Xc_raw[tr_idx])
    scaler_i = StandardScaler().fit(Xi_raw[tr_idx])
    
    Xc_train = scaler_c.transform(Xc_raw[tr_idx])
    Xc_val = scaler_c.transform(Xc_raw[val_idx])
    Xc_test = scaler_c.transform(Xc_raw[te_idx])
    
    Xi_train = scaler_i.transform(Xi_raw[tr_idx])
    Xi_val = scaler_i.transform(Xi_raw[val_idx])
    Xi_test = scaler_i.transform(Xi_raw[te_idx])
    
    enc = LabelEncoder()
    enc.fit(algo_names_raw[tr_idx])
    num_algos = len(enc.classes_)
    
    class_counts = Counter(algo_names_raw[tr_idx])
    most_common_class = class_counts.most_common(1)[0][0]
    default_idx = enc.transform([most_common_class])[0]
    print(f"   算法类别数: {num_algos}, 默认算法: {most_common_class}")
    
    def safe_transform(labels, default):
        known = set(enc.classes_)
        return np.array([enc.transform([l])[0] if l in known else default for l in labels])
    
    Xa_train = enc.transform(algo_names_raw[tr_idx])
    Xa_val = safe_transform(algo_names_raw[val_idx], default_idx)
    Xa_test = safe_transform(algo_names_raw[te_idx], default_idx)
    
    y_train, y_val, y_test = y_raw[tr_idx], y_raw[val_idx], y_raw[te_idx]
    
    with open('preprocessing_objects.pkl', 'wb') as f:
        pickle.dump({
            'scaler_c': scaler_c, 'scaler_i': scaler_i, 'enc': enc,
            'cols_c': cols_c, 'cols_i': cols_i,
            'default_algo_idx': default_idx, 'most_common_algo': most_common_class
        }, f)
    
    tr_d = MixupCTSDataset(Xc_train, Xi_train, Xa_train, y_train, 
                          use_mixup=CONFIG["use_mixup"], alpha=CONFIG["mixup_alpha"])
    val_d = MixupCTSDataset(Xc_val, Xi_val, Xa_val, y_val, use_mixup=False)
    te_d = MixupCTSDataset(Xc_test, Xi_test, Xa_test, y_test, use_mixup=False)
    
    tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
    te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 使用设备: {device}")
    
    model = CompactCFTNet(len(cols_c), len(cols_i), num_algos, CONFIG["embed_dim"]).to(device)
    num_params = model.count_parameters()
    print(f"📦 模型参数量: {num_params:,} (~{num_params/1000:.1f}K)")
    print(f"   架构: 双塔 + EDL (alpha≥{CONFIG['alpha_init']}, beta≥{CONFIG['beta_init']}, v≥{CONFIG['v_init']})")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], 
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    # normalizer = MetricsNormalizer(window_size=5, ema_alpha=CONFIG["ema_alpha"])
    best_score = float('inf')
    best_metrics = {}
    patience_counter = 0
    
    history = {k: [] for k in ['loss', 'mae', 'smape', 'mape', 'rmse', 'corr', 'ece', 
                               'picp_80', 'mpiw_80', 'mean_unc', 'score', 'lr']}
    
    print(f"\n🏃 开始训练 (目标PICP: {CONFIG['picp_target']*100}%)...")
    
    for epoch in range(CONFIG["epochs"]):
        # ---- 训练阶段 ----
        model.train()
        epoch_loss = 0
        epoch_picp = 0
        batch_count = 0
        
        for cx, ix, ax, target in tr_loader:
            cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(cx, ix, ax)
            loss, nll, reg, picp, mean_unc = evidential_loss(pred, target, epoch, CONFIG)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_picp += picp
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        avg_picp_train = epoch_picp / batch_count * 100  # 转换为百分比
        
        # ---- 验证阶段 ----
        model.eval()
        val_preds, val_targets, val_uncs = [], [], []
        
        with torch.no_grad():
            for cx, ix, ax, target in val_loader:
                cx, ix, ax = cx.to(device), ix.to(device), ax.to(device)
                preds = model(cx, ix, ax)
                
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                pred_time = torch.expm1(gamma)
                true_time = torch.expm1(target.to(device))
                
                var = beta / (v * (alpha - 1))
                unc = torch.sqrt(var + 1e-6)
                
                val_preds.extend(pred_time.cpu().numpy())
                val_targets.extend(true_time.cpu().numpy())
                val_uncs.extend(unc.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_uncs = np.array(val_uncs)
        val_errs = np.abs(val_targets - val_preds)
        
        # 计算验证集指标
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_smape = calculate_smape(val_targets, val_preds)
        val_mape = calculate_mape(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_corr = spearmanr(val_uncs, val_errs)[0]
        val_corr = 0.0 if np.isnan(val_corr) else val_corr
        
        val_ece = calculate_ece_quantile(val_errs, val_uncs)
        val_picp, val_mpiw = calculate_picp_mpiw(val_targets, val_preds, val_uncs, 0.8)
        
        current_metrics = {
        'sMAPE': val_smape,
        'Corr': val_corr,
        'ECE': val_ece,
        'PICP_80': val_picp}
    
    
        # 调用新的静态评分函数
        val_score = calculate_score(current_metrics)

        # 动态归一化与得分
        # normalizer.update(val_smape, val_ece)
        # smape_n, corr_n, ece_n = normalizer.normalize(val_smape, val_corr, val_ece)
        # val_score = (CONFIG["mape_weight"] * smape_n + 
        #             CONFIG["corr_weight"] * corr_n + 
        #             CONFIG["ece_weight"] * ece_n)
        
        # 记录历史
        for k, v in zip(history.keys(), 
                       [avg_loss, val_mae, val_smape, val_mape, val_rmse, 
                        val_corr, val_ece, val_picp, val_mpiw, val_uncs.mean(),
                        val_score, optimizer.param_groups[0]['lr']]):
            history[k].append(v)
        
        scheduler.step(val_score)
        # ---- 打印与保存 ----
        print(f"Ep{epoch+1:03d} | Loss:{avg_loss:.3f} | "
              f"sMAPE:{val_smape:.2f}% | "
              f"Corr:{val_corr:.3f} | "
              f"PICP:{val_picp:.1f}% (train:{avg_picp_train:.1f}%) MPIW:{val_mpiw:.1f}s | "
              f"MeanUnc:{val_uncs.mean():.2f}s", end="")
        
        if val_score < best_score:
            best_score = val_score
            best_metrics = {
                'smape': val_smape, 'mape': val_mape, 'mae': val_mae,
                'rmse': val_rmse, 'corr': val_corr, 'ece': val_ece,
                'picp_80': val_picp, 'mpiw_80': val_mpiw, 'epoch': epoch
            }
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'best_metrics': best_metrics,
                'config': CONFIG,
                # 'normalizer_stats': {'smape_scale': normalizer.smape_scale, 
                #                     'ece_scale': normalizer.ece_scale}
            }, CONFIG["model_save_path"])
            print(f" ⭐ BEST")
        else:
            patience_counter += 1
            print(f" (pat:{patience_counter}/{CONFIG['patience']})")
        
        if patience_counter >= CONFIG["patience"]:
            print(f"\n⏹️ 早停触发 (Best Epoch {best_metrics['epoch']+1})")
            break
    
    # ---- 事后校准与测试 ----
    print("\n🔍 加载最佳模型进行测试与校准...")
    checkpoint = torch.load(CONFIG["model_save_path"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 验证集上收集用于校准
    val_preds_cal, val_targets_cal, val_uncs_cal = [], [], []
    with torch.no_grad():
        for cx, ix, ax, target in val_loader:
            cx, ix, ax = cx.to(device), ix.to(device), ax.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            
            pred_time = torch.expm1(gamma)
            true_time = torch.expm1(target.to(device))
            var = beta / (v * (alpha - 1))
            unc = torch.sqrt(var + 1e-6)
            
            val_preds_cal.extend(pred_time.cpu().numpy())
            val_targets_cal.extend(true_time.cpu().numpy())
            val_uncs_cal.extend(unc.cpu().numpy())
    
    val_preds_cal = np.array(val_preds_cal)
    val_targets_cal = np.array(val_targets_cal)
    val_uncs_cal = np.array(val_uncs_cal)
    
    print(f"   原始验证集PICP: {calculate_picp_mpiw(val_targets_cal, val_preds_cal, val_uncs_cal, 0.8)[0]:.1f}%")
    scale_factor = post_hoc_calibration(val_targets_cal, val_preds_cal, val_uncs_cal, target_coverage=0.8, search_range=(0.1, 50))
    print(f"   学习到的缩放因子: {scale_factor:.3f}")
    
    # 测试集
    test_preds, test_targets, test_uncs_raw = [], [], []
    with torch.no_grad():
        for cx, ix, ax, target in te_loader:
            cx, ix, ax = cx.to(device), ix.to(device), ax.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            
            pred_time = torch.expm1(gamma)
            true_time = torch.expm1(target.to(device))
            var = beta / (v * (alpha - 1))
            unc = torch.sqrt(var + 1e-6)
            
            test_preds.extend(pred_time.cpu().numpy())
            test_targets.extend(true_time.cpu().numpy())
            test_uncs_raw.extend(unc.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_uncs_raw = np.array(test_uncs_raw)
    
    test_uncs_calibrated = test_uncs_raw * scale_factor
    
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_smape = calculate_smape(test_targets, test_preds)
    test_mape = calculate_mape(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_corr = spearmanr(test_uncs_raw, np.abs(test_targets - test_preds))[0]
    test_corr = 0.0 if np.isnan(test_corr) else test_corr
    
    picp_raw, mpiw_raw = calculate_picp_mpiw(test_targets, test_preds, test_uncs_raw, 0.8)
    picp_cal, mpiw_cal = calculate_picp_mpiw(test_targets, test_preds, test_uncs_calibrated, 0.8)
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果 (带事后校准)")
    print(f"{'='*60}")
    print(f"预测精度:")
    print(f"  MAE:    {test_mae:.3f}s  |  RMSE: {test_rmse:.3f}s")
    print(f"  sMAPE:  {test_smape:.2f}%  ⭐")
    print(f"  MAPE:   {test_mape:.2f}%")
    print(f"\n不确定性（原始 vs 校准后）:")
    print(f"  Corr:   {test_corr:.4f}")
    print(f"  PICP80: {picp_raw:.1f}% → {picp_cal:.1f}% (目标: 80%)")
    print(f"  MPIW:   {mpiw_raw:.2f}s → {mpiw_cal:.2f}s")
    print(f"  缩放因子: {scale_factor:.3f}x")
    print(f"{'='*60}")
    
    # 保存结果
    results = {
        'model': 'CompactCFT-Net-Calibrated',
        'num_params': num_params,
        'seed': SEED,
        'calibration': {
            'scale_factor': float(scale_factor),
            'val_picp_before': float(calculate_picp_mpiw(val_targets_cal, val_preds_cal, val_uncs_cal, 0.8)[0]),
        },
        'test_metrics': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'smape': float(test_smape),
            'mape': float(test_mape),
            'corr': float(test_corr),
            'picp_80_raw': float(picp_raw),
            'picp_80_calibrated': float(picp_cal),
            'mpiw_raw': float(mpiw_raw),
            'mpiw_calibrated': float(mpiw_cal),
        },
        'best_val_metrics': convert_to_native(best_metrics),
        'config': convert_to_native(CONFIG)
    }
    
    results = convert_to_native(results)
    result_path = f'results_calibrated_{datetime.now().strftime("%m%d_%H%M")}_seed{SEED}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存至: {result_path}")
    return results

def plot_training_history(history, best_epoch):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    metrics = [
        ('loss', 'Training Loss'), ('smape', 'Validation sMAPE (%)'),
        ('mae', 'Validation MAE (s)'), ('rmse', 'Validation RMSE (s)'),
        ('corr', 'Uncertainty Correlation'), ('ece', 'Expected Calibration Error'),
        ('picp_80', 'PICP 80% (%)'), ('mpiw_80', 'MPIW 80% (s)')
    ]
    
    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 4, idx % 4]
        ax.plot(history[key], label=key.upper(), linewidth=1.5)
        ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, 
                  label=f'Best Epoch {best_epoch+1}')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.upper())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'training_history_fixed_{datetime.now().strftime("%m%d_%H%M")}.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 训练历史图已保存至: {save_path}")

if __name__ == "__main__":
    results = train_model()