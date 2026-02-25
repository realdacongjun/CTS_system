import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import time
import random
import platform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, norm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全局配置
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# ==============================================================================
# 1. 数据定义
# ==============================================================================
class CTSDataset(Dataset):
    def __init__(self, cx, ix, ax, y):
        self.cx = torch.FloatTensor(cx)
        self.ix = torch.FloatTensor(ix)
        self.ax = torch.LongTensor(ax)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

def load_global_data():
    print("📦 正在全局加载数据...")
    df_exp = pd.read_excel("cts_data.xlsx")
    df_feat = pd.read_csv("image_features_database.csv")
    
    rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", 
                  "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
    df_exp = df_exp.rename(columns=rename_map)
    
    if 'total_time' not in df_exp.columns:
        cols = [c for c in df_exp.columns if 'total_tim' in c]
        if cols: df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
    df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
    df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    
    # 客户端特征
    base_client_cols = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
    cols_c = [c for c in base_client_cols if c in df.columns]
    
    if 'total_size_mb' in df.columns and 'bandwidth_mbps' in df.columns:
        df['theoretical_time'] = df['total_size_mb'] / (df['bandwidth_mbps'] / 8 + 1e-8)
        cols_c.append('theoretical_time')
    if 'cpu_limit' in df.columns and 'total_size_mb' in df.columns:
        df['cpu_to_size_ratio'] = df['cpu_limit'] / (df['total_size_mb'] + 1e-8)
        cols_c.append('cpu_to_size_ratio')
    if 'mem_limit_mb' in df.columns and 'total_size_mb' in df.columns:
        df['mem_to_size_ratio'] = df['mem_limit_mb'] / (df['total_size_mb'] + 1e-8)
        cols_c.append('mem_to_size_ratio')
    if 'bandwidth_mbps' in df.columns and 'network_rtt' in df.columns:
        df['network_score'] = df['bandwidth_mbps'] / (df['network_rtt'] + 1e-8)
        cols_c.append('network_score')

    # 镜像特征 (自动检测)
    possible_image_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
                           'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
    cols_i = [c for c in possible_image_cols if c in df.columns]
    
    print(f"   客户端特征: {len(cols_c)} 维 | 镜像特征: {len(cols_i)} 维")
    df['total_time_log'] = np.log1p(df['total_time'])
    
    # 划分
    N = len(df)
    idx = np.random.permutation(N)
    n_tr, n_val = int(N * 0.7), int(N * 0.15)
    tr_idx, val_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]
    
    # 标准化
    scaler_c = StandardScaler().fit(df[cols_c].values[tr_idx])
    scaler_i = StandardScaler().fit(df[cols_i].values[tr_idx])
    enc = LabelEncoder().fit(df['algo_name'].values[tr_idx])
    
    data_dict = {
        'cols_c': cols_c, 'cols_i': cols_i,
        'scaler_c': scaler_c, 'scaler_i': scaler_i, 'enc': enc,
    }
    
    for name, idx_list in zip(['train', 'val', 'test'], [tr_idx, val_idx, te_idx]):
        data_dict[f'Xc_{name}'] = scaler_c.transform(df[cols_c].values[idx_list])
        data_dict[f'Xi_{name}'] = scaler_i.transform(df[cols_i].values[idx_list])
        
        def safe_trans(labels):
            known = set(enc.classes_)
            return np.array([enc.transform([l])[0] if l in known else 0 for l in labels])
            
        data_dict[f'Xa_{name}'] = safe_trans(df['algo_name'].values[idx_list])
        data_dict[f'y_{name}_log'] = df['total_time_log'].values[idx_list]
        data_dict[f'y_{name}_ori'] = df['total_time'].values[idx_list]

    print("✅ 全局数据加载完成！")
    return data_dict

# ==============================================================================
# 2. 模型定义
# ==============================================================================
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[128]):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class LightweightFeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_features, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_normal_(self.embeddings)
    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.norm(x * self.embeddings + self.bias)

class TransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim=64):
        super().__init__()
        self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128,
            batch_first=True, dropout=0.2, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return self.encoder(torch.cat([cls, tokens], dim=1))[:, 0, :]

class UnifiedModel(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, arch_cfg):
        super().__init__()
        self.arch_cfg = arch_cfg
        self.embed_dim = 64
        
        self.algo_embed = nn.Embedding(num_algos, self.embed_dim)
        
        if arch_cfg['tower']:
            if arch_cfg['backbone'] == 'Trans':
                self.client_net = TransformerTower(client_feats, self.embed_dim)
                self.image_net = TransformerTower(image_feats, self.embed_dim)
            else:
                self.client_net = MLPBlock(client_feats, self.embed_dim, [128])
                self.image_net = MLPBlock(image_feats, self.embed_dim, [128])
            fusion_dim = self.embed_dim * 3
        else:
            self.backbone = MLPBlock(client_feats + image_feats, self.embed_dim, [128])
            fusion_dim = self.embed_dim + self.embed_dim

        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.LayerNorm(128), nn.GELU(), 
            nn.Dropout(0.2), nn.Linear(128, self.embed_dim)
        )
        
        if arch_cfg['edl']:
            if arch_cfg['decoupled']:
                self.head_mean = nn.Sequential(nn.Linear(self.embed_dim, 32), nn.GELU(), nn.Linear(32, 1))
                self.head_uncertainty = nn.Sequential(
                    nn.Linear(self.embed_dim, 32), nn.LayerNorm(32), nn.GELU(), 
                    nn.Dropout(0.1), nn.Linear(32, 3)
                )
            else:
                self.head_single = nn.Linear(self.embed_dim, 4)
            self.alpha_init, self.beta_init, self.v_init = 2.0, 1.0, 1.0
        else:
            self.head_baseline = nn.Linear(self.embed_dim, 1)

    def forward(self, cx, ix, ax):
        a = self.algo_embed(ax)
        if self.arch_cfg['tower']:
            c = self.client_net(cx)
            i = self.image_net(ix)
            fused_feat = torch.cat([c, i, a], dim=-1)
        else:
            feat_emb = self.backbone(torch.cat([cx, ix], dim=-1))
            fused_feat = torch.cat([feat_emb, a], dim=-1)
            
        shared = self.shared_fusion(fused_feat)
        
        if self.arch_cfg['edl']:
            if self.arch_cfg['decoupled']:
                gamma = self.head_mean(shared).squeeze(-1)
                unc_out = self.head_uncertainty(shared)
            else:
                out = self.head_single(shared)
                gamma, unc_out = out[:, 0], out[:, 1:]
            v = F.softplus(unc_out[:, 0]) + self.v_init
            alpha = F.softplus(unc_out[:, 1]) + self.alpha_init
            beta = F.softplus(unc_out[:, 2]) + self.beta_init
            return torch.stack([gamma, v, alpha, beta], dim=1)
        else:
            return self.head_baseline(shared).squeeze(-1)

# ==============================================================================
# 3. 训练与全指标评估
# ==============================================================================
def nig_nll_loss(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()

def train_single_variant(arch_name, arch_cfg, data_dict):
    print(f"\n{'='*60}")
    print(f"🧪 开始训练变体: {arch_name}")
    print(f"{'='*60}")
    
    tr_loader = DataLoader(CTSDataset(data_dict['Xc_train'], data_dict['Xi_train'], data_dict['Xa_train'], data_dict['y_train_log']), 
                           batch_size=256, shuffle=True)
    val_loader = DataLoader(CTSDataset(data_dict['Xc_val'], data_dict['Xi_val'], data_dict['Xa_val'], data_dict['y_val_log']), 
                           batch_size=256)
    
    model = UnifiedModel(len(data_dict['cols_c']), len(data_dict['cols_i']), len(data_dict['enc'].classes_), arch_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    
    best_rmse = float('inf')
    best_path = f"temp_{arch_name.replace(':', '').replace(' ', '_')}.pth"
    
    for epoch in range(400):
        model.train()
        for cx, ix, ax, y in tr_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(cx, ix, ax)
            if arch_cfg['edl']: loss = nig_nll_loss(y, pred[:,0], pred[:,1], pred[:,2], pred[:,3])
            else: loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            model.eval()
            val_preds = []
            with torch.no_grad():
                for cx, ix, ax, _ in val_loader:
                    p = model(cx.to(device), ix.to(device), ax.to(device))
                    if arch_cfg['edl']: p = p[:, 0]
                    val_preds.extend(torch.expm1(p).cpu().numpy())
            val_rmse = np.sqrt(mean_squared_error(data_dict['y_val_ori'], val_preds))
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), best_path)
                print(f"  Ep {epoch:03d} | Val RMSE: {val_rmse:.2f} ⭐")
    
    print(f"✅ {arch_name} 训练完成")
    return best_path

def evaluate_variant_full(arch_name, arch_cfg, model_path, data_dict):
    """完整评估：精度、不确定性、参数量、推理速度"""
    print(f"📊 正在全指标评估 {arch_name}...")
    
    model = UnifiedModel(len(data_dict['cols_c']), len(data_dict['cols_i']), len(data_dict['enc'].classes_), arch_cfg).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    te_loader = DataLoader(CTSDataset(data_dict['Xc_test'], data_dict['Xi_test'], data_dict['Xa_test'], data_dict['y_test_log']), 
                           batch_size=256)
    
    # 1. 预测收集
    test_preds, test_uncs = [], []
    with torch.no_grad():
        for cx, ix, ax, _ in te_loader:
            p = model(cx.to(device), ix.to(device), ax.to(device))
            if arch_cfg['edl']:
                gamma, v, alpha, beta = p[:,0], p[:,1], p[:,2], p[:,3]
                pred_ori = torch.expm1(gamma)
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_ori = torch.exp(gamma) * torch.sqrt(var_log)
                test_preds.extend(pred_ori.cpu().numpy())
                test_uncs.extend(std_ori.cpu().numpy())
            else:
                test_preds.extend(torch.expm1(p).cpu().numpy())
                test_uncs.extend(np.zeros_like(p.cpu().numpy()))
    
    test_preds = np.array(test_preds)
    test_uncs = np.array(test_uncs)
    y_true = data_dict['y_test_ori']
    
    # 2. 精度指标
    rmse = np.sqrt(mean_squared_error(y_true, test_preds))
    mae = mean_absolute_error(y_true, test_preds)
    r2 = r2_score(y_true, test_preds)
    smape = 200 * np.mean(np.abs(y_true - test_preds) / (np.abs(y_true) + np.abs(test_preds) + 1e-8))
    
    # 3. 不确定性指标
    corr = 0.0
    picp = 0.0
    if arch_cfg['edl']:
        corr = spearmanr(test_uncs, np.abs(y_true - test_preds))[0]
        corr = 0.0 if np.isnan(corr) else corr
        z = norm.ppf((1 + 0.8) / 2)
        lower = test_preds - z * test_uncs
        upper = test_preds + z * test_uncs
        picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    
    # 4. 参数量 (K)
    params_k = sum(p.numel() for p in model.parameters()) / 1000.0
    
    # 5. 推理时间 (CPU上测量，公平对比)
    model_cpu = model.cpu()
    model_cpu.eval()
    times = []
    # Warmup
    with torch.no_grad():
        _ = model_cpu(torch.FloatTensor(data_dict['Xc_test'][:10]), 
                      torch.FloatTensor(data_dict['Xi_test'][:10]), 
                      torch.LongTensor(data_dict['Xa_test'][:10]))
    # 正式测量
    with torch.no_grad():
        start = time.perf_counter()
        for i in range(0, len(data_dict['Xc_test']), 256):
            _ = model_cpu(torch.FloatTensor(data_dict['Xc_test'][i:i+256]), 
                          torch.FloatTensor(data_dict['Xi_test'][i:i+256]), 
                          torch.LongTensor(data_dict['Xa_test'][i:i+256]))
        total_time = time.perf_counter() - start
    time_per_sample_ms = (total_time / len(data_dict['Xc_test'])) * 1000
    
    return {
        "Model": arch_name,
        "R²": f"{r2:.4f}",
        "sMAPE (%)": f"{smape:.2f}",
        "MAE (s)": f"{mae:.2f}",
        "Corr": f"{corr:.3f}" if arch_cfg['edl'] else "-",
        "PICP-80 (%)": f"{picp:.1f}" if arch_cfg['edl'] else "-",
        "Params (K)": f"{params_k:.1f}",
        "Time (ms)": f"{time_per_sample_ms:.3f}",
    }

# ==============================================================================
# 4. 主程序
# ==============================================================================
if __name__ == "__main__":
    variants = [
        ("V1: MLP-Single",  {"backbone": "MLP", "tower": False, "edl": False, "decoupled": False}),
        ("V2: MLP-Tower",   {"backbone": "MLP", "tower": True,  "edl": False, "decoupled": False}),
        ("V3: Trans-Tower", {"backbone": "Trans", "tower": True, "edl": False, "decoupled": False}),
        ("V4: Trans-EDL",   {"backbone": "Trans", "tower": True, "edl": True,  "decoupled": False}),
        ("V5: Full (Ours)", {"backbone": "Trans", "tower": True, "edl": True,  "decoupled": True}),
    ]
    
    try:
        data = load_global_data()
        results_list = []
        
        for name, cfg in variants:
            safe_name = name.replace(" ", "_").replace(":", "")
            path = train_single_variant(safe_name, cfg, data)
            metrics = evaluate_variant_full(name, cfg, path, data)
            results_list.append(metrics)
            if os.path.exists(path): os.remove(path)
        
        # 打印结果
        print(f"\n\n{'='*100}")
        print(f"🎉 消融实验完成！最终结果汇总 (可直接复制)")
        print(f"{'='*100}\n")
        
        df_res = pd.DataFrame(results_list)
        print(df_res.to_markdown(index=False))
        
        # 保存CSV
        df_res.to_csv("ablation_results_full.csv", index=False, encoding='utf-8-sig')
        print(f"\n💡 结果已保存至 ablation_results_full.csv")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()