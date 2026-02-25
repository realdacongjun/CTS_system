
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
# 1. 超参数配置（优化版：精度优先）
# ==============================================================================
CONFIG = {
    "lr": 0.001,             
    "weight_decay": 1e-5,     # 降低正则化，缓解欠拟合
    "epochs": 800,            # 增加训练轮数
    "patience": 100,          # 增加早停耐心
    "batch_size": 256,         # 增大Batch，稳定梯度
    
    "embed_dim": 64,         
    "nhead": 4,              
    "num_layers": 2,
    "dim_feedforward": 128,
    
    # EDL参数
    "reg_coeff": 0.1,        
    "warmup_epochs": 0,      
    "alpha_init": 2.0,
    "beta_init": 1.0,
    "v_init": 1.0,
    
    # PICP目标 (大幅降低权重，优先保精度，靠事后校准补PICP)
    "picp_target": 0.9,      
    "lambda_picp": 1.0,       # 从10降到1
    "lambda_mpiw": 0.001,     # 从0.01降到0.001
    
    # 长尾样本加权
    "tail_weight": 3.0,        # 长尾样本权重倍数
    "tail_quantile": 0.90,     # 定义长尾的分位数
    
    # 数据路径
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "model_save_path": f"cts_optimized_{datetime.now().strftime('%m%d_%H%M')}_seed{SEED}.pth",
    
    # 评估权重
    "mape_weight": 0.4,
    "corr_weight": 0.3,
    "ece_weight": 0.3,
    
    "ema_alpha": 0.9,
    "winsorize_limits": 0.05,
    "use_winsorized_for_selection": False,
    
    "use_mixup": True,
    "mixup_alpha": 0.4,
}

# ==============================================================================
# 2. 模型架构（保持原样，未改动）
# ==============================================================================
class LightweightFeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_features, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_normal_(self.embeddings)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        out = x * self.embeddings + self.bias
        return self.norm(out)

class LightweightTransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=CONFIG["dim_feedforward"],
            batch_first=True, 
            dropout=0.2,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        out = self.encoder(x)
        return out[:, 0, :]

class CompactCFTNetV2(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=64):
        super().__init__()
        self.client_tower = LightweightTransformerTower(client_feats, embed_dim, nhead=CONFIG['nhead'], num_layers=CONFIG['num_layers'])
        self.image_tower = LightweightTransformerTower(image_feats, embed_dim, nhead=CONFIG['nhead'], num_layers=CONFIG['num_layers'])
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        self.shared_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        self.head_mean = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self.head_uncertainty = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 3)
        )
        
        self._init_weights()
        self.alpha_init = CONFIG["alpha_init"]
        self.beta_init = CONFIG["beta_init"]
        self.v_init = CONFIG["v_init"]
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, cx, ix, ax):
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        a = self.algo_embed(ax)
        
        fused = torch.cat([c, i, a], dim=-1)
        shared = self.shared_fusion(fused)
        
        gamma = self.head_mean(shared).squeeze(-1)
        unc_out = self.head_uncertainty(shared)
        
        v = F.softplus(unc_out[:, 0]) + self.v_init
        alpha = F.softplus(unc_out[:, 1]) + self.alpha_init
        beta = F.softplus(unc_out[:, 2]) + self.beta_init
        
        return torch.stack([gamma, v, alpha, beta], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ==============================================================================
# 3. 损失函数（优化版：支持长尾加权 + 降低PICP权重）
# ==============================================================================
def nig_nll_loss(y, gamma, v, alpha, beta, reduction='mean'):
    """标准的NIG负对数似然，支持reduction参数"""
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    
    if reduction == 'mean':
        return nll.mean()
    return nll

def get_uncertainty_metrics(gamma, v, alpha, beta):
    mean = gamma
    var = beta / (v * (alpha - 1) + 1e-6)
    std = torch.sqrt(var + 1e-6)
    return mean, std

def picp_loss(y_true, y_pred, std, target=0.9):
    z = norm.ppf((1 + target) / 2)
    
    lower = y_pred - z * std
    upper = y_pred + z * std
    
    d_low = F.relu(lower - y_true)
    d_high = F.relu(y_true - upper)
    distance = d_low + d_high
    
    mpiw = upper - lower
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    
    loss_coverage = torch.mean(distance**2) 
    loss_sharpness = torch.mean(covered * mpiw**2)
    
    return loss_coverage, loss_sharpness

def evidential_loss_v2(pred, target, epoch, config):
    """
    优化版损失：
    1. 支持长尾样本加权
    2. 降低PICP权重，优先保证拟合精度
    """
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    
    # --- 新增：长尾样本加权 ---
    target_ori = torch.expm1(target)
    q_tail = torch.quantile(target_ori, config['tail_quantile'])
    sample_weights = torch.where(target_ori > q_tail, config['tail_weight'], 1.0)
    # -------------------------
    
    # 1. 基础NLL损失 (改为加权平均)
    loss_nll_per_sample = nig_nll_loss(target, gamma, v, alpha, beta, reduction='none')
    loss_nll = (loss_nll_per_sample * sample_weights).mean()
    
    # 2. 计算不确定性指标
    y_pred_log, std_log = get_uncertainty_metrics(gamma, v, alpha, beta)
    
    # 3. PICP & MPIW 损失
    loss_cov, loss_sharp = picp_loss(target, y_pred_log, std_log, target=config['picp_target'])
    
    # 4. 动态权重 (降低整体权重)
    progress = min(1.0, epoch / 100.0) 
    w_picp = config['lambda_picp'] * progress
    w_mpiw = config['lambda_mpiw'] * progress
    
    total_loss = loss_nll + w_picp * loss_cov + w_mpiw * loss_sharp
    
    # 监控指标
    with torch.no_grad():
        y_true_ori = torch.expm1(target)
        y_pred_ori = torch.expm1(gamma)
        std_ori = std_log * torch.expm1(gamma)
        
        z = norm.ppf((1 + 0.8) / 2)
        lower = y_pred_ori - z * std_ori
        upper = y_pred_ori + z * std_ori
        covered = ((y_true_ori >= lower) & (y_true_ori <= upper)).float()
        picp_val = covered.mean().item()
        mpiw_val = (upper - lower).mean().item()

    return total_loss, loss_nll.item(), loss_cov.item(), picp_val, std_ori.mean().item()

# ==============================================================================
# 4. 数据加载 (优化版：新增物理交叉特征)
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
        
        # ==========================================
        # 【核心优化】新增物理交叉特征
        # ==========================================
        print("🔧 新增强物理交叉特征...")
        
        # 1. 最核心特征：理论传输时间 (文件大小 / 有效带宽)
        # 注意：bandwidth是Mbps，转成MB/s需要除以8
        df['theoretical_time'] = df['total_size_mb'] / (df['bandwidth_mbps'] / 8 + 1e-8)
        
        # 2. 资源压力特征
        df['cpu_to_size_ratio'] = df['cpu_limit'] / (df['total_size_mb'] + 1e-8)
        df['mem_to_size_ratio'] = df['mem_limit_mb'] / (df['total_size_mb'] + 1e-8)
        
        # 3. 网络综合指标
        df['network_score'] = df['bandwidth_mbps'] / (df['network_rtt'] + 1e-8)
        
        # 更新特征列表：把新特征加入客户端特征
        cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb', 
                  'theoretical_time', 'cpu_to_size_ratio', 'mem_to_size_ratio', 'network_score']
        
        # 镜像特征保持不变
        target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
                       'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
        cols_i = [c for c in target_cols if c in df.columns]
        
        # 打印新增的特征
        print(f"   客户端特征数: {len(cols_c)} (新增4个物理特征)")
        print(f"   镜像特征数: {len(cols_i)}")
        # ==========================================
        
        df['total_time_log'] = np.log1p(df['total_time'])
        
        Xc_raw = df[cols_c].values
        Xi_raw = df[cols_i].values
        y_raw = df['total_time_log'].values
        algo_names_raw = df['algo_name'].values
        
        print(f"✅ 数据加载成功: {len(y_raw)} 条")
        print(f"   时间范围: [{df['total_time'].min():.2f}s, {df['total_time'].max():.2f}s]")
        
        return Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i, df['total_time'].values
        
    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# 5. 评估工具 (保持原样)
# ==============================================================================
def calculate_smape(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(numerator / denominator) * 100

def calculate_ece_quantile(errors, uncertainties, n_bins=10):
    if len(errors) == 0: return 0.0
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(uncertainties, quantiles)
    bin_boundaries[-1] += 1e-8
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i+1])
        if i == n_bins - 1: in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties <= bin_boundaries[i+1])
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
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [convert_to_native(v) for v in obj]
    return obj

# ==============================================================================
# 6. 训练主循环
# ==============================================================================
def train_model():
    data = load_data()
    if data is None: return None
        
    Xc_raw, Xi_raw, algo_names_raw, y_raw_log, cols_c, cols_i, y_raw_ori = data
    N = len(y_raw_log)
    
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
    
    def safe_transform(labels, default):
        known = set(enc.classes_)
        return np.array([enc.transform([l])[0] if l in known else default for l in labels])
    
    Xa_train = enc.transform(algo_names_raw[tr_idx])
    Xa_val = safe_transform(algo_names_raw[val_idx], default_idx)
    Xa_test = safe_transform(algo_names_raw[te_idx], default_idx)
    
    y_train_log, y_val_log, y_test_log = y_raw_log[tr_idx], y_raw_log[val_idx], y_raw_log[te_idx]
    y_val_ori, y_test_ori = y_raw_ori[val_idx], y_raw_ori[te_idx]
    
    # 保存预处理对象，注意保存新的cols_c和cols_i
    with open('preprocessing_objects_optimized.pkl', 'wb') as f:
        pickle.dump({
            'scaler_c': scaler_c, 'scaler_i': scaler_i, 'enc': enc,
            'cols_c': cols_c, 'cols_i': cols_i
        }, f)
    
    tr_d = MixupCTSDataset(Xc_train, Xi_train, Xa_train, y_train_log, 
                          use_mixup=CONFIG["use_mixup"], alpha=CONFIG["mixup_alpha"])
    val_d = MixupCTSDataset(Xc_val, Xi_val, Xa_val, y_val_log, use_mixup=False)
    te_d = MixupCTSDataset(Xc_test, Xi_test, Xa_test, y_test_log, use_mixup=False)
    
    tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
    te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 使用设备: {device}")
    
    # 注意：这里传入新的 client_feats 长度
    model = CompactCFTNetV2(len(cols_c), len(cols_i), num_algos, CONFIG["embed_dim"]).to(device)
    num_params = model.count_parameters()
    print(f"📦 模型参数量: {num_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], 
                           weight_decay=CONFIG["weight_decay"])
    
    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=1e-6
    )
    
    best_score = float('inf')
    patience_counter = 0
    
    print(f"\n🏃 开始优化版训练 (精度优先)...")
    
    for epoch in range(CONFIG["epochs"]):
        # ---- 训练 ----
        model.train()
        epoch_loss = 0
        for cx, ix, ax, target in tr_loader:
            cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(cx, ix, ax)
            loss, nll, cov, picp_train, _ = evidential_loss_v2(pred, target, epoch, CONFIG)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step(epoch)
        
        # ---- 验证 ----
        model.eval()
        val_preds_ori = []
        val_uncs_ori = []
        
        with torch.no_grad():
            for cx, ix, ax, target_log in val_loader:
                cx, ix, ax = cx.to(device), ix.to(device), ax.to(device)
                preds = model(cx, ix, ax)
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
                pred_log = gamma
                pred_ori = torch.expm1(pred_log)
                
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_log = torch.sqrt(var_log + 1e-6)
                std_ori = torch.exp(pred_log) * std_log 
                
                val_preds_ori.extend(pred_ori.cpu().numpy())
                val_uncs_ori.extend(std_ori.cpu().numpy())
        
        val_preds_ori = np.array(val_preds_ori)
        val_uncs_ori = np.array(val_uncs_ori)
        val_errs_ori = np.abs(y_val_ori - val_preds_ori)
        
        val_mae = mean_absolute_error(y_val_ori, val_preds_ori)
        val_smape = calculate_smape(y_val_ori, val_preds_ori)
        val_rmse = np.sqrt(mean_squared_error(y_val_ori, val_preds_ori))
        val_corr = spearmanr(val_uncs_ori, val_errs_ori)[0] if len(val_uncs_ori) > 10 else 0.0
        val_corr = 0.0 if np.isnan(val_corr) else val_corr
        
        val_picp, val_mpiw = calculate_picp_mpiw(y_val_ori, val_preds_ori, val_uncs_ori, 0.8)
        
        # 综合得分：更关注RMSE和sMAPE
        picp_gap = abs(val_picp - 80.0) / 80.0
        val_score = (val_rmse / 30.0) + (val_smape / 20.0) + (1.0 - val_corr) + 1.0 * picp_gap
        
        print(f"Ep{epoch+1:03d} | Loss:{epoch_loss/len(tr_loader):.2f} | "
              f"RMSE:{val_rmse:.2f}s | sMAPE:{val_smape:.2f}% | "
              f"Corr:{val_corr:.3f} | "
              f"PICP:{val_picp:.1f}% | MPIW:{val_mpiw:.2f}s", end="")
        
        if val_score < best_score:
            best_score = val_score
            best_metrics = {
                'rmse': val_rmse, 'smape': val_smape, 'corr': val_corr,
                'picp': val_picp, 'mpiw': val_mpiw, 'epoch': epoch
            }
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f" ⭐ BEST")
        else:
            patience_counter += 1
            print(f" (pat:{patience_counter}/{CONFIG['patience']})")
        
        if patience_counter >= CONFIG["patience"]:
            print(f"\n⏹️ 早停触发 (Best Epoch {best_metrics['epoch']+1})")
            break
    
    # ---- 测试 ----
    print("\n🔍 最终测试...")
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    model.eval()
    
    test_preds, test_uncs = [], []
    with torch.no_grad():
        for cx, ix, ax, _ in te_loader:
            cx, ix, ax = cx.to(device), ix.to(device), ax.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            
            pred_ori = torch.expm1(gamma)
            var_log = beta / (v * (alpha - 1) + 1e-6)
            std_ori = torch.exp(gamma) * torch.sqrt(var_log)
            
            test_preds.extend(pred_ori.cpu().numpy())
            test_uncs.extend(std_ori.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_uncs = np.array(test_uncs)
    test_errs = np.abs(y_test_ori - test_preds)
    
    test_mae = mean_absolute_error(y_test_ori, test_preds)
    test_smape = calculate_smape(y_test_ori, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test_ori, test_preds))
    test_corr = spearmanr(test_uncs, test_errs)[0] if len(test_uncs) > 10 else 0.0
    test_picp, test_mpiw = calculate_picp_mpiw(y_test_ori, test_preds, test_uncs, 0.8)
    
    print(f"\n{'='*60}")
    print(f"📊 优化版最终测试结果")
    print(f"{'='*60}")
    print(f"精度 (目标：降低):")
    print(f"  RMSE:  {test_rmse:.2f}s")
    print(f"  sMAPE: {test_smape:.2f}%")
    print(f"\n不确定性质量 (目标：升高):")
    print(f"  Corr:  {test_corr:.4f}")
    print(f"  PICP:  {test_picp:.1f}% (目标: 80%)")
    print(f"  MPIW:  {test_mpiw:.2f}s")
    print(f"{'='*60}")
    print(f"\n💡 提示：如果PICP未达标，请使用evaluation.py中的分层校准进行事后补全。")
    
    return test_preds, test_uncs, y_test_ori

if __name__ == "__main__":
    preds, uncs, trues = train_model()