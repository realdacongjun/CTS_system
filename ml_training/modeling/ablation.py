"""
CFT-Net 消融实验脚本 (Ablation Study)
策略：减法原则 (Subtraction Strategy)
1. Ours (Full): Transformer + EDL + Mixup
2. w/o EDL: 替换为 MSE Loss，输出层改为1维
3. w/o Transformer: 替换为 MLP 塔
4. w/o Mixup: 关闭数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random
import pickle
import platform
import matplotlib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, norm
from collections import Counter
import warnings
import time
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# ==============================================================================
# 1. 灵活的模型定义 (支持消融切换)
# ==============================================================================

# --- 组件 A: MLP 塔 (用于 w/o Transformer) ---
class MLPTower(nn.Module):
    def __init__(self, num_features, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

# --- 组件 B: Transformer 塔 (Ours) ---
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
    def __init__(self, num_features, embed_dim=32, nhead=2):
        super().__init__()
        self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=32,
            batch_first=True, dropout=0.1, activation="gelu"
        )
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        out = self.encoder(x)
        return out[:, 0, :]

# --- 组件 C: FTTransformer 塔 (新增消融变体) ---
class FTTransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim=32, nhead=4):
        super().__init__()
        # 特征嵌入层
        self.feature_embedding = nn.Embedding(num_features, embed_dim)
        # 类别特征嵌入（这里假设所有特征都是数值型，实际使用时可能需要调整）
        self.numerical_embedding = nn.Linear(num_features, embed_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim*2,
            batch_first=True, 
            dropout=0.1, 
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输出投影
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: [batch_size, num_features]
        batch_size, num_features = x.shape
        
        # 数值特征嵌入
        numerical_emb = self.numerical_embedding(x)  # [batch_size, embed_dim]
        
        # 添加残差连接和层归一化
        out = self.transformer(numerical_emb.unsqueeze(1))  # [batch_size, 1, embed_dim]
        out = out.squeeze(1)  # [batch_size, embed_dim]
        
        return self.output_projection(out)

# --- 主模型 (FlexibleCFTNet) - 更新 ---
class FlexibleCFTNet(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=32, 
                 use_transformer=True, use_edl=True, transformer_type='lightweight'):
        super().__init__()
        self.use_edl = use_edl
        self.transformer_type = transformer_type
        
        # 1. 选择骨干网络
        if use_transformer:
            if transformer_type == 'fttransformer':
                self.client_tower = FTTransformerTower(client_feats, embed_dim)
                self.image_tower = FTTransformerTower(image_feats, embed_dim)
            else:  # lightweight transformer
                self.client_tower = LightweightTransformerTower(client_feats, embed_dim)
                self.image_tower = LightweightTransformerTower(image_feats, embed_dim)
        else:
            self.client_tower = MLPTower(client_feats, embed_dim)
            self.image_tower = MLPTower(image_feats, embed_dim)
            
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        # 2. 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            # 关键修改：EDL输出4维，MSE输出1维
            nn.Linear(32, 4 if use_edl else 1) 
        )
        
        # EDL 初始化参数
        self.alpha_init = 1.5
        self.beta_init = 1.0
        self.v_init = 0.5

    def forward(self, cx, ix, ax):
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        a = self.algo_embed(ax)
        
        fused = torch.cat([c, i, a], dim=-1)
        out = self.fusion(fused)
        
        if self.use_edl:
            gamma = out[:, 0]
            v = F.softplus(out[:, 1]) + self.v_init
            alpha = F.softplus(out[:, 2]) + self.alpha_init
            beta = F.softplus(out[:, 3]) + self.beta_init
            return torch.stack([gamma, v, alpha, beta], dim=1)
        else:
            # MSE 模式：直接输出预测值 (gamma)
            return out[:, 0]

# ==============================================================================
# 2. 辅助组件 (Loss, Data, Metrics)
# ==============================================================================
# EDL Loss
def nig_nll_loss(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) - alpha * torch.log(two_blambda) + \
          (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) + \
          torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()

def improved_eub_loss(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    var = beta / (v * (alpha - 1) + 1e-6)
    std = torch.sqrt(var)
    confidence = 1.0 / (std + 1e-6)
    ratio = torch.clamp(error * confidence, max=10.0)
    reg = ((ratio - 1.0) ** 2) * torch.log1p(torch.clamp(2 * v + alpha, max=20.0))
    return reg.mean()

def calculate_smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def calculate_picp(y_true, y_pred, unc):
    z = 1.28 # 80% CI
    lower = y_pred - z * unc
    upper = y_pred + z * unc
    return np.mean((y_true >= lower) & (y_true <= upper)) * 100

# 数据集
class MixupCTSDataset(Dataset):
    def __init__(self, cx, ix, ax, y, use_mixup=True, alpha=0.2):
        self.cx, self.ix, self.ax, self.y = torch.FloatTensor(cx), torch.FloatTensor(ix), torch.LongTensor(ax), torch.FloatTensor(y)
        self.use_mixup = use_mixup; self.alpha = alpha; self.num_samples = len(y)
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        if not self.use_mixup or random.random() > 0.5: 
            return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]
        idx2 = random.randint(0, self.num_samples - 1); lam = np.random.beta(self.alpha, self.alpha)
        return lam * self.cx[idx] + (1 - lam) * self.cx[idx2], \
               lam * self.ix[idx] + (1 - lam) * self.ix[idx2], \
               self.ax[idx] if lam > 0.5 else self.ax[idx2], \
               lam * self.y[idx] + (1 - lam) * self.y[idx2]

# 加载数据
def load_data_simple():
    print("🔄 加载数据...")
    if not os.path.exists("cts_data.xlsx"): return None
    df_exp = pd.read_excel("cts_data.xlsx")
    df_feat = pd.read_csv("image_features_database.csv")
    rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
    df_exp = df_exp.rename(columns=rename_map)
    if 'total_time' not in df_exp.columns:
        cols = [c for c in df_exp.columns if 'total_tim' in c]
        if cols: df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
    df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
    df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    
    cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
    cols_i = [c for c in ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio'] if c in df.columns]
    
    Xc = df[cols_c].values
    Xi = df[cols_i].values
    y = np.log1p(df['total_time'].values)
    algo_names = df['algo_name'].values
    
    return Xc, Xi, algo_names, y, cols_c, cols_i

# ==============================================================================
# 3. 消融实验运行器 - 修改加载预训练模型部分
# ==============================================================================
def load_pretrained_model(model, model_path):
    """加载预训练模型权重"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ 成功加载预训练模型: {model_path}")
        return True
    except Exception as e:
        print(f"❌ 加载预训练模型失败: {e}")
        return False

def run_ablation_experiment(variant_name, config, data_bundle):
    print(f"\n{'='*60}")
    print(f"🧪 运行实验: {variant_name}")
    print(f"{'='*60}")
    
    # 1. 数据准备
    Xc_tr, Xi_tr, Xa_tr, y_tr = data_bundle['train']
    Xc_val, Xi_val, Xa_val, y_val = data_bundle['val']
    Xc_te, Xi_te, Xa_te, y_te = data_bundle['test']
    
    use_mixup = config.get('use_mixup', True)
    tr_loader = DataLoader(MixupCTSDataset(Xc_tr, Xi_tr, Xa_tr, y_tr, use_mixup=use_mixup), batch_size=64, shuffle=True)
    val_loader = DataLoader(MixupCTSDataset(Xc_val, Xi_val, Xa_val, y_val, use_mixup=False), batch_size=64)
    te_loader = DataLoader(MixupCTSDataset(Xc_te, Xi_te, Xa_te, y_te, use_mixup=False), batch_size=64)
    
    # 2. 模型初始化
    model = FlexibleCFTNet(
        client_feats=Xc_tr.shape[1],
        image_feats=Xi_tr.shape[1],
        num_algos=10, # 假设
        embed_dim=32,
        use_transformer=config.get('use_transformer', True),
        use_edl=config.get('use_edl', True),
        transformer_type=config.get('transformer_type', 'lightweight')
    ).to(device)
    
    # 3. 特殊处理：如果是Ours变体且指定了预训练模型路径，则加载预训练权重
    pretrained_model_path = config.get('pretrained_model_path')
    if variant_name == "Ours (Full)" and pretrained_model_path and os.path.exists(pretrained_model_path):
        if load_pretrained_model(model, pretrained_model_path):
            print("🌟 使用预训练模型权重进行测试")
            # 如果使用预训练模型，跳过训练阶段
            best_state = model.state_dict()
        else:
            print("⚠️ 预训练模型加载失败，使用随机初始化训练")
            best_state = train_model_normal(model, tr_loader, val_loader, config)
    else:
        # 正常训练流程
        best_state = train_model_normal(model, tr_loader, val_loader, config)
    
    # 4. 最终测试
    model.load_state_dict(best_state)
    model.eval()
    
    preds, targets, uncs = [], [], []
    with torch.no_grad():
        for cx, ix, ax, target in te_loader:
            cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
            out = model(cx, ix, ax)
            
            if config['use_edl']:
                gamma, v, alpha, beta = out[:,0], out[:,1], out[:,2], out[:,3]
                p = torch.expm1(gamma)
                u = torch.sqrt(beta / (v * (alpha - 1) + 1e-6))
            else:
                p = torch.expm1(out) # 此时out是1维
                u = torch.zeros_like(p) # 无不确定性
            
            preds.extend(p.cpu().numpy())
            targets.extend(torch.expm1(target).cpu().numpy())
            uncs.extend(u.cpu().numpy())
            
    preds, targets, uncs = np.array(preds), np.array(targets), np.array(uncs)
    
    # 计算指标
    smape = calculate_smape(targets, preds)
    
    if config['use_edl']:
        # 使用你之前训练好的最佳缩放因子进行校准，保证公平
        scale_factor = 33.713 
        uncs_cal = uncs * scale_factor
        corr = spearmanr(uncs_cal, np.abs(targets - preds))[0]
        picp = calculate_picp(targets, preds, uncs_cal)
    else:
        corr = np.nan
        picp = np.nan
        
    print(f"👉 结果: sMAPE={smape:.2f}%, Corr={corr:.3f}, PICP={picp:.1f}%")
    
    return {
        'Variant': variant_name,
        'sMAPE (%)': f"{smape:.2f}",
        'Corr': f"{corr:.3f}" if not np.isnan(corr) else "N/A",
        'PICP-80 (%)': f"{picp:.1f}" if not np.isnan(picp) else "N/A"
    }

def train_model_normal(model, tr_loader, val_loader, config):
    """正常的模型训练函数"""
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    best_state = None
    patience = 20
    counter = 0
    
    # 训练循环 (加速版: 100 Epochs)
    epochs = 200
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for cx, ix, ax, target in tr_loader:
            cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(cx, ix, ax)
            
            if config['use_edl']:
                # EDL Loss
                gamma, v, alpha, beta = out[:,0], out[:,1], out[:,2], out[:,3]
                nll = nig_nll_loss(target, gamma, v, alpha, beta)
                reg = improved_eub_loss(target, gamma, v, alpha, beta)
                reg_w = 0.005 * min(1.0, epoch/30) # Warmup
                loss = nll + reg_w * reg
            else:
                # MSE Loss
                loss = F.mse_loss(out, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cx, ix, ax, target in val_loader:
                cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
                out = model(cx, ix, ax)
                if config['use_edl']:
                    loss = nig_nll_loss(target, out[:,0], out[:,1], out[:,2], out[:,3]) # 只看NLL
                else:
                    loss = F.mse_loss(out, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"⏹️ 早停于 Epoch {epoch}")
            break
            
        if (epoch+1) % 20 == 0:
            print(f"Ep{epoch+1:03d} | Val Loss: {val_loss:.4f}")
    
    return best_state

# ==============================================================================
# 4. 主程序 - 修改实验配置
# ==============================================================================
if __name__ == "__main__":
    # 1. 准备数据
    data_raw = load_data_simple()
    if data_raw is None: exit()
    
    Xc, Xi, algo_names, y, _, _ = data_raw
    N = len(y)
    idx = np.random.RandomState(42).permutation(N)
    n_tr, n_val = int(N*0.7), int(N*0.15)
    
    # 预处理
    scaler_c = StandardScaler().fit(Xc[idx[:n_tr]])
    scaler_i = StandardScaler().fit(Xi[idx[:n_tr]])
    enc = LabelEncoder().fit(algo_names[idx[:n_tr]])
    
    def process_split(indices):
        xc = scaler_c.transform(Xc[indices])
        xi = scaler_i.transform(Xi[indices])
        xa = np.array([enc.transform([l])[0] if l in enc.classes_ else 0 for l in algo_names[indices]])
        return xc, xi, xa, y[indices]
        
    data_bundle = {
        'train': process_split(idx[:n_tr]),
        'val':   process_split(idx[n_tr:n_tr+n_val]),
        'test':  process_split(idx[n_tr+n_val:])
    }
    
    # 2. 定义消融配置 - 添加新的变体
    experiments = [
        ("Ours (Full)", {
            'use_transformer': True,  
            'use_edl': True,  
            'use_mixup': True,
            'transformer_type': 'lightweight',
            'pretrained_model_path': "E:\\硕士毕业论文材料合集\\论文实验代码相关\\CTS_system\\ml_training\\modeling\\cts_fixed_0217_1727_seed42.pth"
        }),
        ("w/o EDL", {
            'use_transformer': True,  
            'use_edl': False, 
            'use_mixup': True,
            'transformer_type': 'lightweight'
        }),
        ("w/o Transformer", {
            'use_transformer': False, 
            'use_edl': True,  
            'use_mixup': True
        }),
        ("w/o Mixup", {
            'use_transformer': True,  
            'use_edl': True,  
            'use_mixup': False,
            'transformer_type': 'lightweight'
        }),
        ("Dual-Tower + FTTransformer", {
            'use_transformer': True,  
            'use_edl': True,  
            'use_mixup': True,
            'transformer_type': 'fttransformer'
        }),
    ]
    
    results = []
    
    # 3. 运行实验
    print(f"\n🔥 开始消融实验 (共 {len(experiments)} 组)...")
    start_time = time.time()
    
    for name, cfg in experiments:
        res = run_ablation_experiment(name, cfg, data_bundle)
        results.append(res)
        
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ 所有实验完成，耗时 {total_time:.1f} 分钟")
    
    # 4. 输出结果
    df = pd.DataFrame(results)
    print("\n📊 最终消融实验结果 (Ablation Study Results):")
    print(df.to_string(index=False))
    
    df.to_csv("final_ablation_results.csv", index=False)
    print("\n结果已保存至 final_ablation_results.csv")