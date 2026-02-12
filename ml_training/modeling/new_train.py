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
from scipy.stats import spearmanr
import pickle
import random
import math
import platform
import matplotlib

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
# 1. 超参数配置 (终极强约束版)
# ==============================================================================
CONFIG = {
    "lr": 0.0005,              
    "weight_decay": 1e-4,      
    "epochs": 200,             
    "patience": 15,            # 激进早停
    "batch_size": 128,         
    "embed_dim": 32,           
    
    # 强约束参数
    "reg_coeff": 1.0,          # 【强】拉满惩罚
    "warmup_epochs": 3,        # 【快】几乎立即介入
    
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "model_save_path": "cts_final_strong.pth",
}

# ==============================================================================
# 2. 损失函数：Symmetric Strong EUB
# ==============================================================================
def nig_nll_loss(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()

def strong_eub_reg_loss(y, gamma, v, alpha, beta):
    """
    终极保底版：强约束 + 对称惩罚
    目标：强制 Ratio = Error/Std 接近 1
    """
    error = torch.abs(y - gamma)
    
    # 计算标准差
    var = beta / (v * (alpha - 1) + 1e-6)
    std = torch.sqrt(var + 1e-6)
    
    # 比率计算 (带截断)
    raw_ratio = error / (std + 1e-6)
    ratio = torch.clamp(raw_ratio, max=5.0) # 防止梯度爆炸
    
    # 对称惩罚 (Symmetric Penalty)
    # Ratio > 1 (盲目自信) -> (Ratio-1)^2 -> 惩罚
    # Ratio < 1 (过度保守) -> (Ratio-1)^2 -> 惩罚
    # 逼迫模型学会 calibration
    penalty = (ratio - 1.0)**2 
    
    # Evidence 截断
    evidence = torch.clamp(2 * v + alpha, max=20.0)
    
    # 最终正则
    reg = penalty * torch.log1p(evidence)
    
    return reg.mean()

def evidential_loss(pred, target, epoch):
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    
    loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
    loss_reg = strong_eub_reg_loss(target, gamma, v, alpha, beta)
    
    # 快速 Warmup
    if epoch < CONFIG["warmup_epochs"]:
        reg_weight = 0.0
    else:
        # 5轮内拉满
        progress = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 5)
        reg_weight = CONFIG["reg_coeff"] * progress
    
    total_loss = loss_nll + reg_weight * loss_reg
    return total_loss, loss_nll.item(), loss_reg.item()

# ==============================================================================
# 3. 模型定义 (Gated Fusion)
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
        
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
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
        z = self.gate_net(torch.cat([c_vec, i_vec], dim=1))
        fused_vec = z * c_vec + (1 - z) * i_vec
        a_vec = self.algo_embed(ax)
        
        out = self.head(self.hidden(torch.cat([fused_vec, i_vec, a_vec], dim=1)))
        
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + 0.1
        alpha = F.softplus(out[:, 2]) + 1.1
        beta = F.softplus(out[:, 3]) + 1e-6
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 4. 数据加载
# ==============================================================================
class CTSDataset(Dataset):
    def __init__(self, cx, ix, ax, y):
        self.cx, self.ix, self.ax, self.y = torch.FloatTensor(cx), torch.FloatTensor(ix), torch.LongTensor(ax), torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

def load_data():
    print(f"🔄 读取数据: {CONFIG['data_path']} ...")
    if not os.path.exists(CONFIG['data_path']):
        print(f"❌ 错误: 找不到文件 {CONFIG['data_path']}")
        return None

    try:
        df_exp = pd.read_excel(CONFIG["data_path"])
        df_feat = pd.read_csv(CONFIG["feature_path"])
        
        rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
        df_exp = df_exp.rename(columns=rename_map)
        if 'total_time' not in df_exp.columns: 
            cols = [c for c in df_exp.columns if 'total_tim' in c]
            if cols: df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
        df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
        df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
        cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
        cols_i = [c for c in target_cols if c in df.columns]
        
        Xc = StandardScaler().fit_transform(df[cols_c].values)
        Xi = StandardScaler().fit_transform(df[cols_i].values)
        enc = LabelEncoder()
        Xa = enc.fit_transform(df['algo_name'].values)
        y = np.log1p(df['total_time'].values)
        
        with open('preprocessing_objects.pkl', 'wb') as f:
            pickle.dump({'scaler_c': StandardScaler().fit(df[cols_c].values), 
                         'scaler_i': StandardScaler().fit(df[cols_i].values), 
                         'enc': enc}, f)
        
        return Xc, Xi, Xa, y, enc, len(cols_c), len(cols_i)
    
    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        return None

# ==============================================================================
# 5. 训练主循环
# ==============================================================================
if __name__ == "__main__":
    data = load_data()
    if data:
        Xc, Xi, Xa, y, enc_algo, c_dim, i_dim = data
        N = len(y)
        idx = np.random.permutation(N)
        n_tr, n_val = int(N * 0.7), int(N * 0.15)
        
        tr_d = CTSDataset(Xc[idx[:n_tr]], Xi[idx[:n_tr]], Xa[idx[:n_tr]], y[idx[:n_tr]])
        val_d = CTSDataset(Xc[idx[n_tr:n_tr+n_val]], Xi[idx[n_tr:n_tr+n_val]], Xa[idx[n_tr:n_tr+n_val]], y[idx[n_tr:n_tr+n_val]])
        
        tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 训练开始 (策略: Strong Symmetric EUB)")
        
        model = CTSDualTowerModel(c_dim, i_dim, len(enc_algo.classes_)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
        
        best_corr = -1.0
        best_epoch = 0
        patience_counter = 0
        history = {'loss': [], 'corr': []}
        
        for epoch in range(CONFIG["epochs"]):
            model.train()
            t_loss = 0
            for cx, ix, ax, target in tr_loader:
                cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
                optimizer.zero_grad()
                loss, _, _ = evidential_loss(model(cx, ix, ax), target, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item()
            
            scheduler.step()
            
            model.eval()
            uncs, errs = [], []
            with torch.no_grad():
                for cx, ix, ax, target in val_loader:
                    cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
                    preds = model(cx, ix, ax)
                    gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                    
                    unc = beta / (v * (alpha - 1))
                    err = torch.abs(torch.expm1(gamma) - torch.expm1(target))
                    uncs.extend(unc.cpu().numpy()); errs.extend(err.cpu().numpy())
            
            try: corr, _ = spearmanr(uncs, errs)
            except: corr = 0.0
            if np.isnan(corr): corr = 0.0
            
            history['loss'].append(t_loss/len(tr_loader))
            history['corr'].append(corr)
            
            print(f"Epoch {epoch+1:03d} | Loss: {history['loss'][-1]:.4f} | Val Corr: {corr:.4f}", end="")
            
            if corr > best_corr:
                best_corr = corr
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_corr': best_corr,
                    'epoch': epoch,
                    'config': CONFIG
                }, CONFIG["model_save_path"])
                print(f" 🌟 New Best!")
            else:
                patience_counter += 1
                print(f" (Patience: {patience_counter}/{CONFIG['patience']})")
                
            if patience_counter >= CONFIG["patience"]:
                print(f"\n⏹️ 触发早停机制！")
                break
        
        print(f"\n✅ 训练结束。最佳模型: {CONFIG['model_save_path']} (Corr={best_corr:.4f})")
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Strong EUB Loss')
        plt.title('训练损失')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['corr'], color='#ff7f0e', label='Val Corr')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch+1}')
        plt.title('验证集相关性')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_result_strong.png')
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from scipy.stats import spearmanr
# import pickle
# import random
# import math

# # --- 1. 全局设置 ---
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(42)

# CONFIG = {
#     "kl_coeff": 1.0,           # KL散度正则化系数，防止不确定性发散
#     "annealing_epochs": 50,    # KL系数退火周期，前期专注于回归准确度
#     "lr": 0.0008,              # 略微降低学习率以配合Gated结构
#     "epochs": 200,             # 训练轮次
#     "data_path": "cts_data.xlsx",
#     "feature_path": "image_features_database.csv",
#     "batch_size": 128,         # 显存允许的话，大Batch有助于EDL收敛
#     "embed_dim": 32,           # Embedding维度
#     "model_save_path": "cts_best_model_gated.pth",
#     "weight_decay": 1e-4,      # L2正则化
# }

# # --- 2. 证据深度学习 (EDL) 核心损失函数 ---
# def nig_nll_loss(y, gamma, v, alpha, beta):
#     # 负对数似然损失 (NLL) - 拟合观测数据
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def nig_reg_loss(y, gamma, v, alpha, beta):
#     # 正则化损失 - 惩罚错误的自信
#     error = torch.abs(y - gamma)
#     evidence = 2 * v + alpha
#     return (error * evidence).mean()

# def evidential_loss(pred, target, epoch):
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
    
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = nig_reg_loss(target, gamma, v, alpha, beta)
    
#     # KL退火策略：让模型先学准(回归)，再学稳(不确定性)
#     annealing_coef = min(1.0, epoch / CONFIG["annealing_epochs"])
    
#     total_loss = loss_nll + CONFIG["kl_coeff"] * annealing_coef * loss_reg
#     return total_loss, loss_nll.item(), loss_reg.item()

# # --- 3. 模型定义 (Gated Fusion Version) ---
# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         # 优化点1：加入LayerNorm，稳定输入分布
#         self.norm = nn.LayerNorm(embed_dim)
#         nn.init.xavier_uniform_(self.weights)
#         nn.init.zeros_(self.biases)

#     def forward(self, x):
#         # 类似FTTransformer的特征Token化
#         tokens = x.unsqueeze(-1) * self.weights + self.biases
#         return self.norm(tokens)

# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         # 优化点2：标准的Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
#             batch_first=True, dropout=0.1, activation="gelu" # 使用GELU
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         tokens = torch.cat((cls_tokens, tokens), dim=1)
#         out = self.transformer(tokens)
#         return out[:, 0, :] # 只取 CLS Token 作为塔的输出

# class CTSDualTowerModel(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         # 优化点3：门控融合机制 (Gated Fusion)
#         # 学习一个权重 z，动态决定更信任 Client 还是 Image
#         self.gate_net = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.Sigmoid()
#         )
        
#         # 优化点4：增强的回归头 (GELU + Dropout)
#         self.hidden = nn.Sequential(
#             nn.Linear(embed_dim * 3, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.GELU()
#         )
#         self.head = nn.Linear(32, 4) # 输出4个EDL参数

#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx) # Client 特征
#         i_vec = self.image_tower(ix)  # Image 特征
        
#         # --- 门控融合核心逻辑 ---
#         # z 是一个 (Batch, Dim) 的权重向量，0~1之间
#         # 如果 z 接近 1，说明当前更加关注网络环境；反之关注镜像本身
#         z = self.gate_net(torch.cat([c_vec, i_vec], dim=1))
#         fused_vec = z * c_vec + (1 - z) * i_vec
        
#         a_vec = self.algo_embed(ax)
#         # 将融合特征、原始镜像特征、算法特征拼接
#         combined = torch.cat([fused_vec, i_vec, a_vec], dim=1)
        
#         hidden = self.hidden(combined)
#         out = self.head(hidden)
        
#         # 激活函数确保参数满足分布要求
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1
#         alpha = F.softplus(out[:, 2]) + 1.1 # 保证 alpha > 1
#         beta = F.softplus(out[:, 3]) + 1e-6
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # --- 4. 数据处理 (保持不变) ---
# class CTSDataset(Dataset):
#     def __init__(self, client_x, image_x, algo_x, y):
#         self.cx = torch.FloatTensor(client_x)
#         self.ix = torch.FloatTensor(image_x)
#         self.ax = torch.LongTensor(algo_x)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

# def load_and_process_data():
#     print(f"🔄 读取数据...")
#     # 模拟读取逻辑，请确保路径正确
#     try:
#         df_exp = pd.read_excel(CONFIG["data_path"])
#         df_feat = pd.read_csv(CONFIG["feature_path"])
#     except:
#         print("❌ 文件未找到，请检查 CONFIG 中的路径")
#         return None, None, None, None, None

#     # 列名映射 (根据你提供的截图调整)
#     rename_map = {
#         "image": "image_name", "method": "algo_name",
#         "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
#         "mem_limit": "mem_limit_mb"
#     }
#     df_exp = df_exp.rename(columns=rename_map)
#     if 'total_time' not in df_exp.columns:
#         possible = [c for c in df_exp.columns if 'total_tim' in c]
#         if possible: df_exp = df_exp.rename(columns={possible[0]: 'total_time'})
    
#     # 过滤无效数据
#     df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#     df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    
#     col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
    
#     # 预处理
#     scaler_c = StandardScaler(); X_client = scaler_c.fit_transform(df[col_client].values)
#     scaler_i = StandardScaler(); X_image = scaler_i.fit_transform(df[col_image].values)
#     enc_algo = LabelEncoder(); X_algo = enc_algo.fit_transform(df['algo_name'].values)
#     y_target = np.log1p(df['total_time'].values) # Log变换平滑长尾分布
    
#     return X_client, X_image, X_algo, y_target, enc_algo

# # --- 5. 训练主循环 ---
# if __name__ == "__main__":
#     Xc, Xi, Xa, y, enc_algo = load_and_process_data()
    
#     if Xc is not None:
#         # 划分数据集
#         N = len(y)
#         indices = np.random.permutation(N)
#         n_train, n_val = int(N * 0.7), int(N * 0.15)
#         train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
        
#         train_loader = DataLoader(CTSDataset(Xc[train_idx], Xi[train_idx], Xa[train_idx], y[train_idx]), 
#                                   batch_size=CONFIG["batch_size"], shuffle=True)
#         val_loader = DataLoader(CTSDataset(Xc[val_idx], Xi[val_idx], Xa[val_idx], y[val_idx]), 
#                                 batch_size=CONFIG["batch_size"])
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🚀 设备: {device} | 训练集: {len(train_idx)} | 验证集: {len(val_idx)}")
        
#         model = CTSDualTowerModel(
#             client_feats=Xc.shape[1], 
#             image_feats=Xi.shape[1], 
#             num_algos=len(enc_algo.classes_)
#         ).to(device)
        
#         # 优化点5：使用 CosineAnnealing 学习率调度
#         optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)
        
#         best_corr = -1.0
#         history = {'epoch': [], 'loss': [], 'val_corr': []}
        
#         for epoch in range(CONFIG["epochs"]):
#             model.train()
#             train_loss = 0
#             for cx, ix, ax, target in train_loader:
#                 cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 preds = model(cx, ix, ax)
#                 loss, _, _ = evidential_loss(preds, target, epoch)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
#                 optimizer.step()
#                 train_loss += loss.item()
            
#             scheduler.step()
            
#             # 验证：计算 Spearman Correlation (不确定性 vs 误差)
#             model.eval()
#             all_unc, all_err = [], []
#             with torch.no_grad():
#                 for cx, ix, ax, target in val_loader:
#                     cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                     preds = model(cx, ix, ax)
                    
#                     gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
#                     pred_time = torch.expm1(gamma) # 还原 Log
#                     true_time = torch.expm1(target)
                    
#                     # 核心公式：不确定性 = beta / (v * (alpha - 1))
#                     uncertainty = beta / (v * (alpha - 1))
#                     error = torch.abs(pred_time - true_time)
                    
#                     all_unc.extend(uncertainty.cpu().numpy())
#                     all_err.extend(error.cpu().numpy())
            
#             # 只有当数据量足够且无NaN时计算Corr
#             try:
#                 corr, _ = spearmanr(all_unc, all_err)
#             except:
#                 corr = 0.0
                
#             history['epoch'].append(epoch)
#             history['loss'].append(train_loss / len(train_loader))
#             history['val_corr'].append(corr)
            
#             # 保存最佳模型 (以 Correlation 为准)
#             if corr > best_corr and epoch > 10:
#                 best_corr = corr
#                 torch.save(model.state_dict(), CONFIG["model_save_path"])
#                 print(f"Epoch {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val Corr: {corr:.4f} (New Best!)")
#             elif (epoch+1) % 10 == 0:
#                 print(f"Epoch {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val Corr: {corr:.4f}")
        
#         print(f"\n✅ 训练完成。最佳不确定性相关系数: {best_corr:.4f}")
        
#         # 绘制简单的训练曲线
#         plt.figure(figsize=(10, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(history['epoch'], history['loss'], label='Train Loss')
#         plt.title('Loss Curve')
#         plt.subplot(1, 2, 2)
#         plt.plot(history['epoch'], history['val_corr'], color='orange', label='Val Correlation')
#         plt.title('Uncertainty-Error Correlation')
#         plt.savefig('training_result.png')
#         print("📊 训练曲线已保存为 training_result.png")







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from scipy.stats import spearmanr  # [修改] 用Spearman更稳健


# # [新增] 固定随机种子
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     import random
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# set_seed(42)


# CONFIG = {
#     "kl_coeff": 1.5,
#     "annealing_epochs": 150,
#     "lr": 0.001,
#     "epochs": 300,
#     "data_path": "cts_data.xlsx",         
#     "feature_path": "image_features_database.csv",
#     "batch_size": 64,
#     "embed_dim": 32,
#     "model_save_path": "cts_best_model_fixed_v3.pth",
#     "weight_decay": 1e-4,  # [新增] AdamW的weight decay
# }


# # 路径检查
# if not os.path.exists(CONFIG["data_path"]):
#     if os.path.exists(f"../{CONFIG['data_path']}"):
#         CONFIG["data_path"] = f"../{CONFIG['data_path']}"
#         CONFIG["feature_path"] = f"../{CONFIG['feature_path']}"
#         print(f"📂 自动切换数据路径到上一级: {CONFIG['data_path']}")

# # ==============================================================================
# # 🌟 修复后的证据深度学习损失函数
# # ==============================================================================

# def nig_nll_loss(y, gamma, v, alpha, beta):
#     """负对数似然损失"""
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def nig_reg_loss(y, gamma, v, alpha, beta):
#     """正则化损失：惩罚错误且自信"""
#     error = torch.abs(y - gamma)
#     evidence = 2 * v + alpha
#     return (error * evidence).mean()

# def evidential_loss(pred, target, epoch, lambda_coef=CONFIG["kl_coeff"]):
#     """总损失 = NLL + 正则化"""
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
    
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = nig_reg_loss(target, gamma, v, alpha, beta)
    
#     # [修复] 平方退火
#     if epoch < CONFIG["annealing_epochs"]:
#         annealing_coef = (epoch / CONFIG["annealing_epochs"]) ** 2
#     else:
#         annealing_coef = 1.0
    
#     total_loss = loss_nll + lambda_coef * annealing_coef * loss_reg
    
#     return total_loss, loss_nll.item(), loss_reg.item(), annealing_coef

# # ==============================================================================
# # 2. 模型定义 ([关键修复] 数值约束)
# # ==============================================================================

# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         nn.init.xavier_uniform_(self.weights)
#         nn.init.zeros_(self.biases)

#     def forward(self, x):
#         return x.unsqueeze(-1) * self.weights + self.biases

# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
#             batch_first=True, dropout=0.1
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         batch_size = x.shape[0]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         tokens = torch.cat((cls_tokens, tokens), dim=1)
#         out = self.transformer(tokens)
#         return out[:, 0, :]

# class CTSDualTowerModel(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         fusion_input_dim = embed_dim * 3 
#         self.hidden = nn.Sequential(
#             nn.Linear(fusion_input_dim, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
#         self.head = nn.Linear(64, 4) 

#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx)
#         i_vec = self.image_tower(ix)
#         a_vec = self.algo_embed(ax)
#         combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
#         hidden = self.hidden(combined)
#         out = self.head(hidden)
        
#         # [关键修复] 更强的数值约束
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1        # [修复] 最小0.1
#         alpha = F.softplus(out[:, 2]) + 1.1    # [修复] 最小1.1，(alpha-1)>=0.1
#         beta = F.softplus(out[:, 3]) + 1e-6
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 3. 数据加载
# # ==============================================================================

# def load_data():
#     print(f"🔄 1. 正在读取数据: {CONFIG['data_path']} ...")
#     try:
#         df_exp = pd.read_excel(CONFIG["data_path"])
#     except ImportError:
#         print("❌ 读取失败！请运行 'pip install openpyxl'")
#         exit(1)

#     rename_map = {
#         "image": "image_name", "method": "algo_name",
#         "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
#         "mem_limit": "mem_limit_mb"
#     }
#     df_exp = df_exp.rename(columns=rename_map)
    
#     if 'total_time' not in df_exp.columns:
#         possible_cols = [c for c in df_exp.columns if 'total_tim' in c]
#         if possible_cols: df_exp = df_exp.rename(columns={possible_cols[0]: 'total_time'})

#     df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
    
#     if 'mem_limit_mb' not in df_exp.columns: 
#         df_exp['mem_limit_mb'] = 1024.0
    
#     print(f"🔄 2. 读取镜像特征: {CONFIG['feature_path']} ...")
#     df_feat = pd.read_csv(CONFIG["feature_path"])
    
#     df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
#     print(f"✅ 数据加载完成，样本数: {len(df)}")
#     return df

# class CTSDataset(Dataset):
#     def __init__(self, client_x, image_x, algo_x, y):
#         self.cx = torch.FloatTensor(client_x)
#         self.ix = torch.FloatTensor(image_x)
#         self.ax = torch.LongTensor(algo_x)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

# # ==============================================================================
# # 4. 主训练流程 ([关键修复] 三分数据集 + 原始空间相关性)
# # ==============================================================================

# if __name__ == "__main__":
#     # --- Step 1: 准备数据 ---
#     df = load_data()
    
#     col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
    
#     scaler_c = StandardScaler()
#     X_client = scaler_c.fit_transform(df[col_client].values)
    
#     scaler_i = StandardScaler()
#     X_image = scaler_i.fit_transform(df[col_image].values)
    
#     enc_algo = LabelEncoder()
#     X_algo = enc_algo.fit_transform(df['algo_name'].values)
    
#     y_target = np.log1p(df['total_time'].values)

#     # [关键修复] 三分数据集：70% train / 15% val / 15% test
#     Xc_temp, Xc_test, Xi_temp, Xi_test, Xa_temp, Xa_test, y_temp, y_test = train_test_split(
#         X_client, X_image, X_algo, y_target, test_size=0.3, random_state=42
#     )
#     Xc_train, Xc_val, Xi_train, Xi_val, Xa_train, Xa_val, y_train, y_val = train_test_split(
#         Xc_temp, Xi_temp, Xa_temp, y_temp, test_size=0.5, random_state=42  # 0.5 * 0.3 = 0.15
#     )
    
#     print(f"\n📊 数据集划分:")
#     print(f"   训练集: {len(y_train)} 样本 (70%)")
#     print(f"   验证集: {len(y_val)} 样本 (15%)")
#     print(f"   测试集: {len(y_test)} 样本 (15%)")
    
#     train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train), 
#                               batch_size=CONFIG["batch_size"], shuffle=True)
#     val_loader = DataLoader(CTSDataset(Xc_val, Xi_val, Xa_val, y_val), 
#                             batch_size=CONFIG["batch_size"])
#     test_loader = DataLoader(CTSDataset(Xc_test, Xi_test, Xa_test, y_test), 
#                              batch_size=CONFIG["batch_size"])
    
#     # --- Step 2: 模型初始化 ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n🖥️ 训练设备: {device}")
    
#     model = CTSDualTowerModel(
#         client_feats=len(col_client),
#         image_feats=len(col_image),
#         num_algos=len(enc_algo.classes_)
#     ).to(device)
    
#     # [修改] AdamW + weight decay
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
#     # --- Step 3: 训练 ---
#     print(f"\n🚀 开始训练...")
#     print(f"配置: epochs={CONFIG['epochs']}, kl_coeff={CONFIG['kl_coeff']}")
    
#     best_val_loss = float('inf')
#     best_epoch = 0
#     patience = 30  # [新增] 早停耐心
#     patience_counter = 0
    
#     history = {'epoch': [], 'train_total': [], 'train_nll': [], 'train_reg': [], 
#                'val_nll': [], 'val_corr': []}  # [修改] val_corr
    
#     for epoch in range(CONFIG["epochs"]):
#         model.train()
#         train_total = train_nll = train_reg = 0
        
#         for cx, ix, ax, y in train_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             optimizer.zero_grad()
            
#             preds = model(cx, ix, ax)
#             loss, nll, reg, anneal = evidential_loss(preds, y, epoch)
            
#             loss.backward()
#             optimizer.step()
            
#             train_total += loss.item()
#             train_nll += nll
#             train_reg += reg
        
#         # [关键修复] 在验证集上评估（不是测试集）
#         model.eval()
#         val_nll = 0
#         all_uncertainties = []
#         all_errors = []  # [修改] 原始空间误差
        
#         with torch.no_grad():
#             for cx, ix, ax, y in val_loader:
#                 cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#                 preds = model(cx, ix, ax)
#                 gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
                
#                 val_nll += nig_nll_loss(y, gamma, v, alpha, beta).item()
                
#                 # [关键修复] 在原始时间空间计算误差
#                 pred_time = np.expm1(gamma.cpu().numpy())  # 秒
#                 true_time = np.expm1(y.cpu().numpy())      # 秒
#                 error = np.abs(pred_time - true_time)      # 秒
                
#                 uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
#                 all_uncertainties.extend(uncertainty)
#                 all_errors.extend(error)
        
#         # [修改] Spearman相关性
#         try:
#             corr, _ = spearmanr(all_uncertainties, all_errors)
#         except:
#             corr = 0
        
#         avg_train_total = train_total / len(train_loader)
#         avg_train_nll = train_nll / len(train_loader)
#         avg_train_reg = train_reg / len(train_loader)
#         avg_val_nll = val_nll / len(val_loader)
        
#         history['epoch'].append(epoch)
#         history['train_total'].append(avg_train_total)
#         history['train_nll'].append(avg_train_nll)
#         history['train_reg'].append(avg_train_reg)
#         history['val_nll'].append(avg_val_nll)
#         history['val_corr'].append(corr)
        
#         if (epoch + 1) % 20 == 0:
#             print(f"Epoch {epoch+1:03d} | "
#                   f"Train: {avg_train_total:.3f} | "
#                   f"Val NLL: {avg_val_nll:.3f} | "
#                   f"Val Corr: {corr:+.3f}")
        
#         # [新增] 早停：用验证集选择最佳模型
#         if avg_val_nll < best_val_loss:
#             best_val_loss = avg_val_nll
#             best_epoch = epoch
#             patience_counter = 0
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_val_loss': best_val_loss,
#                 'config': CONFIG,
#                 'scaler_c': scaler_c,
#                 'scaler_i': scaler_i,
#                 'enc_algo': enc_algo,
#                 'col_client': col_client,
#                 'col_image': col_image,
#             }, CONFIG["model_save_path"])
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"\n⏹️ 早停触发！连续{patience}轮无改善")
#                 break

#     print(f"\n💾 训练结束！最佳Val NLL: {best_val_loss:.4f} (Epoch {best_epoch})")
    
#     # --- Step 4: 最终测试集评估（只跑一次）---
#     print("\n🔮 最终测试集评估:")
#     checkpoint = torch.load(CONFIG["model_save_path"], weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     test_nll = 0
#     all_test_unc = []
#     all_test_err = []
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for cx, ix, ax, y in test_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             preds = model(cx, ix, ax)
#             gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            
#             test_nll += nig_nll_loss(y, gamma, v, alpha, beta).item()
            
#             # 原始空间
#             pred_time = np.expm1(gamma.cpu().numpy())
#             true_time = np.expm1(y.cpu().numpy())
#             error = np.abs(pred_time - true_time)
            
#             uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
#             all_test_unc.extend(uncertainty)
#             all_test_err.extend(error)
#             all_preds.extend(pred_time)
#             all_targets.extend(true_time)
    
#     # 测试集指标
#     final_corr, _ = spearmanr(all_test_unc, all_test_err)
#     rmse = np.sqrt(np.mean((np.array(all_targets) - np.array(all_preds))**2))
    
#     print(f"\n{'='*60}")
#     print(f"📊 测试集最终指标:")
#     print(f"   Test NLL: {test_nll/len(test_loader):.4f}")
#     print(f"   RMSE: {rmse:.4f} (秒)")
#     print(f"   Uncertainty-Error Corr: {final_corr:+.3f}")
#     print(f"   不确定性范围: [{np.min(all_test_unc):.3f}, {np.max(all_test_unc):.3f}]")
#     print(f"{'='*60}")
    
#     # --- Step 5: 保存scaler和特征信息（用于后续画图）---
#     print(f"\n📦 保存预处理信息...")
#     import pickle
#     with open('preprocessing_info.pkl', 'wb') as f:
#         pickle.dump({
#             'scaler_c': scaler_c,
#             'scaler_i': scaler_i,
#             'enc_algo': enc_algo,
#             'col_client': col_client,
#             'col_image': col_image,
#         }, f)
    
#     # --- Step 6: 绘图 ---
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
#     axes[0,0].plot(history['epoch'], history['train_total'], label='Train')
#     axes[0,0].plot(history['epoch'], history['val_nll'], label='Val NLL')
#     axes[0,0].set_title('Loss Curves')
#     axes[0,0].legend()
#     axes[0,0].grid(True, alpha=0.3)
    
#     axes[0,1].plot(history['epoch'], history['train_nll'], label='NLL')
#     axes[0,1].plot(history['epoch'], history['train_reg'], label='Reg')
#     axes[0,1].set_title('NLL vs Regularization')
#     axes[0,1].legend()
#     axes[0,1].grid(True, alpha=0.3)
    
#     axes[1,0].plot(history['epoch'], history['val_corr'], 'g-', linewidth=2)
#     axes[1,0].axhline(y=0, color='r', linestyle='--')
#     axes[1,0].set_title('Validation: Uncertainty-Error Correlation')
#     axes[1,0].set_ylabel('Spearman Correlation')
#     axes[1,0].grid(True, alpha=0.3)
    
#     axes[1,1].scatter(all_test_unc, all_test_err, alpha=0.5, s=10)
#     axes[1,1].set_xlabel('Uncertainty')
#     axes[1,1].set_ylabel('Absolute Error (seconds)')
#     axes[1,1].set_title(f'Test: Uncertainty vs Error (Corr={final_corr:.3f})')
#     axes[1,1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('training_diagnostics_v3.png', dpi=150)
#     print("\n📊 训练诊断图已保存")



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from scipy.stats import spearmanr
# import pickle


# # 固定随机种子
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     import random
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# set_seed(42)


# CONFIG = {
#     "kl_coeff": 1.5,
#     "annealing_epochs": 150,
#     "lr": 0.001,
#     "epochs": 300,
#     "data_path": "cts_data.xlsx",
#     "feature_path": "image_features_database.csv",
#     "batch_size": 64,
#     "embed_dim": 32,
#     "model_save_path": "cts_best_model_final.pth",
#     "weight_decay": 1e-4,
# }


# # 路径检查
# if not os.path.exists(CONFIG["data_path"]):
#     if os.path.exists(f"../{CONFIG['data_path']}"):
#         CONFIG["data_path"] = f"../{CONFIG['data_path']}"
#         CONFIG["feature_path"] = f"../{CONFIG['feature_path']}"

# # ==============================================================================
# # 损失函数
# # ==============================================================================

# def nig_nll_loss(y, gamma, v, alpha, beta):
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()


# def nig_reg_loss(y, gamma, v, alpha, beta):
#     error = torch.abs(y - gamma)
#     evidence = 2 * v + alpha
#     return (error * evidence).mean()


# def evidential_loss(pred, target, epoch, lambda_coef=CONFIG["kl_coeff"]):
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
    
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = nig_reg_loss(target, gamma, v, alpha, beta)
    
#     if epoch < CONFIG["annealing_epochs"]:
#         annealing_coef = (epoch / CONFIG["annealing_epochs"]) ** 2
#     else:
#         annealing_coef = 1.0
    
#     total_loss = loss_nll + lambda_coef * annealing_coef * loss_reg
#     return total_loss, loss_nll.item(), loss_reg.item(), annealing_coef

# # ==============================================================================
# # 模型定义（数值约束修复）
# # ==============================================================================

# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         nn.init.xavier_uniform_(self.weights)
#         nn.init.zeros_(self.biases)

#     def forward(self, x):
#         return x.unsqueeze(-1) * self.weights + self.biases


# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
#             batch_first=True, dropout=0.1
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         batch_size = x.shape[0]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         tokens = torch.cat((cls_tokens, tokens), dim=1)
#         out = self.transformer(tokens)
#         return out[:, 0, :]


# class CTSDualTowerModel(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         fusion_input_dim = embed_dim * 3
#         self.hidden = nn.Sequential(
#             nn.Linear(fusion_input_dim, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
#         self.head = nn.Linear(64, 4)

#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx)
#         i_vec = self.image_tower(ix)
#         a_vec = self.algo_embed(ax)
#         combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
#         hidden = self.hidden(combined)
#         out = self.head(hidden)
        
#         # 数值约束
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1
#         alpha = F.softplus(out[:, 2]) + 1.1
#         beta = F.softplus(out[:, 3]) + 1e-6
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)


# class CTSDataset(Dataset):
#     def __init__(self, client_x, image_x, algo_x, y):
#         self.cx = torch.FloatTensor(client_x)
#         self.ix = torch.FloatTensor(image_x)
#         self.ax = torch.LongTensor(algo_x)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]


# # ==============================================================================
# # 主流程
# # ==============================================================================

# if __name__ == "__main__":
#     # 加载数据
#     print(f"🔄 1. 正在读取数据: {CONFIG['data_path']} ...")
#     df_exp = pd.read_excel(CONFIG["data_path"])
#     df_feat = pd.read_csv(CONFIG["feature_path"])
    
#     rename_map = {
#         "image": "image_name", "method": "algo_name",
#         "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
#         "mem_limit": "mem_limit_mb"
#     }
#     df_exp = df_exp.rename(columns=rename_map)
#     if 'total_time' not in df_exp.columns:
#         possible_cols = [c for c in df_exp.columns if 'total_tim' in c]
#         if possible_cols: df_exp = df_exp.rename(columns={possible_cols[0]: 'total_time'})
#     df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#     if 'mem_limit_mb' not in df_exp.columns:
#         df_exp['mem_limit_mb'] = 1024.0
    
#     df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
#     print(f"✅ 数据加载完成，样本数: {len(df)}")
    
#     # 特征
#     col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
    
#     scaler_c = StandardScaler()
#     X_client = scaler_c.fit_transform(df[col_client].values)
#     scaler_i = StandardScaler()
#     X_image = scaler_i.fit_transform(df[col_image].values)
#     enc_algo = LabelEncoder()
#     X_algo = enc_algo.fit_transform(df['algo_name'].values)
#     y_target = np.log1p(df['total_time'].values)
    
#     # [关键修复] 精确70/15/15划分
#     n_total = len(y_target)
#     indices = np.random.permutation(n_total)
    
#     n_train = int(n_total * 0.70)
#     n_val = int(n_total * 0.15)
#     # n_test = n_total - n_train - n_val  # 剩余给测试
    
#     train_idx = indices[:n_train]
#     val_idx = indices[n_train:n_train+n_val]
#     test_idx = indices[n_train+n_val:]
    
#     Xc_train, Xi_train, Xa_train, y_train = X_client[train_idx], X_image[train_idx], X_algo[train_idx], y_target[train_idx]
#     Xc_val, Xi_val, Xa_val, y_val = X_client[val_idx], X_image[val_idx], X_algo[val_idx], y_target[val_idx]
#     Xc_test, Xi_test, Xa_test, y_test = X_client[test_idx], X_image[test_idx], X_algo[test_idx], y_target[test_idx]
    
#     print(f"\n📊 数据集划分:")
#     print(f"   训练集: {len(y_train)} 样本 ({len(y_train)/n_total*100:.1f}%)")
#     print(f"   验证集: {len(y_val)} 样本 ({len(y_val)/n_total*100:.1f}%)")
#     print(f"   测试集: {len(y_test)} 样本 ({len(y_test)/n_total*100:.1f}%)")
    
#     train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
#     val_loader = DataLoader(CTSDataset(Xc_val, Xi_val, Xa_val, y_val), batch_size=CONFIG["batch_size"])
#     test_loader = DataLoader(CTSDataset(Xc_test, Xi_test, Xa_test, y_test), batch_size=CONFIG["batch_size"])
    
#     # 模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n🖥️ 训练设备: {device}")
    
#     model = CTSDualTowerModel(
#         client_feats=len(col_client),
#         image_feats=len(col_image),
#         num_algos=len(enc_algo.classes_)
#     ).to(device)
    
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
#     # 训练
#     print(f"\n🚀 开始训练...")
#     best_val_loss = float('inf')
#     best_epoch = 0
#     patience = 30
#     patience_counter = 0
    
#     history = {'epoch': [], 'train_total': [], 'val_nll': [], 'val_corr': []}
    
#     for epoch in range(CONFIG["epochs"]):
#         model.train()
#         train_total = 0
        
#         for cx, ix, ax, y in train_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             optimizer.zero_grad()
#             preds = model(cx, ix, ax)
#             loss, nll, reg, anneal = evidential_loss(preds, y, epoch)
#             loss.backward()
#             optimizer.step()
#             train_total += loss.item()
        
#         # 验证
#         model.eval()
#         val_nll = 0
#         all_unc, all_err = [], []
        
#         with torch.no_grad():
#             for cx, ix, ax, y in val_loader:
#                 cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#                 preds = model(cx, ix, ax)
#                 gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
                
#                 val_nll += nig_nll_loss(y, gamma, v, alpha, beta).item()
                
#                 # 原始空间
#                 pred_time = np.expm1(gamma.cpu().numpy())
#                 true_time = np.expm1(y.cpu().numpy())
#                 error = np.abs(pred_time - true_time)
#                 uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
                
#                 all_unc.extend(uncertainty)
#                 all_err.extend(error)
        
#         try:
#             corr, _ = spearmanr(all_unc, all_err)
#         except:
#             corr = 0
        
#         avg_train = train_total / len(train_loader)
#         avg_val = val_nll / len(val_loader)
        
#         history['epoch'].append(epoch)
#         history['train_total'].append(avg_train)
#         history['val_nll'].append(avg_val)
#         history['val_corr'].append(corr)
        
#         if (epoch + 1) % 20 == 0:
#             print(f"Epoch {epoch+1:03d} | Train: {avg_train:.3f} | Val NLL: {avg_val:.3f} | Corr: {corr:+.3f}")
        
#         # 早停
#         if avg_val < best_val_loss:
#             best_val_loss = avg_val
#             best_epoch = epoch
#             patience_counter = 0
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'best_val_loss': best_val_loss,
#                 'scaler_c': scaler_c,
#                 'scaler_i': scaler_i,
#                 'enc_algo': enc_algo,
#                 'col_client': col_client,
#                 'col_image': col_image,
#             }, CONFIG["model_save_path"])
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"\n⏹️ 早停触发！")
#                 break
    
#     print(f"\n💾 最佳Val NLL: {best_val_loss:.4f} (Epoch {best_epoch})")
    
#     # 测试
#     print("\n🔮 最终测试集评估:")
#     checkpoint = torch.load(CONFIG["model_save_path"], weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     test_nll = 0
#     all_test_unc, all_test_err = [], []
    
#     with torch.no_grad():
#         for cx, ix, ax, y in test_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             preds = model(cx, ix, ax)
#             gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            
#             test_nll += nig_nll_loss(y, gamma, v, alpha, beta).item()
            
#             pred_time = np.expm1(gamma.cpu().numpy())
#             true_time = np.expm1(y.cpu().numpy())
#             error = np.abs(pred_time - true_time)
#             uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
            
#             all_test_unc.extend(uncertainty)
#             all_test_err.extend(error)
    
#     final_corr, _ = spearmanr(all_test_unc, all_test_err)
#     rmse = np.sqrt(np.mean(np.array(all_test_err)**2))
    
#     print(f"\n{'='*60}")
#     print(f"📊 测试集最终指标:")
#     print(f"   Test NLL: {test_nll/len(test_loader):.4f}")
#     print(f"   RMSE: {rmse:.4f} (秒)")
#     print(f"   Uncertainty-Error Corr: {final_corr:+.3f}")
#     print(f"   不确定性范围: [{np.min(all_test_unc):.3f}, {np.max(all_test_unc):.3f}]")
#     print(f"{'='*60}")
    
#     # 绘图
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     axes[0,0].plot(history['epoch'], history['train_total'], label='Train')
#     axes[0,0].plot(history['epoch'], history['val_nll'], label='Val')
#     axes[0,0].set_title('Loss Curves')
#     axes[0,0].legend()
#     axes[0,0].grid(True, alpha=0.3)
    
#     axes[1,0].plot(history['epoch'], history['val_corr'], 'g-', linewidth=2)
#     axes[1,0].axhline(y=0, color='r', linestyle='--')
#     axes[1,0].set_title('Validation Correlation')
#     axes[1,0].set_ylabel('Spearman')
#     axes[1,0].grid(True, alpha=0.3)
    
#     axes[1,1].scatter(all_test_unc, all_test_err, alpha=0.5, s=10)
#     axes[1,1].set_xlabel('Uncertainty')
#     axes[1,1].set_ylabel('Error (seconds)')
#     axes[1,1].set_title(f'Test: Corr={final_corr:.3f}')
#     axes[1,1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('final_results.png', dpi=150)
#     print("\n📊 结果图已保存")