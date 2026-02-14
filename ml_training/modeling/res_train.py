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
# 1. 超参数配置 (稳定版)
# ==============================================================================
CONFIG = {
    "lr": 0.0005,              
    "weight_decay": 1e-4,      
    "epochs": 200,             
    "patience": 15,           
    "batch_size": 128,         
    "embed_dim": 32,           
    
    # 正则化参数（建议训练时观察loss_nll和loss_reg的量级，适当调整）
    "reg_coeff": 1.0,          
    "warmup_epochs": 3,        
    
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "model_save_path": "cts_final_strong.pth",
}

# ==============================================================================
# 2. 损失函数：Symmetric Strong EUB（保持不变）
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
    对称保真度正则项：强制误差/标准差趋近1，同时惩罚过度自信和过度保守
    """
    error = torch.abs(y - gamma)
    
    # 计算标准差（移除 +1e-6 分母保护，因为 alpha>1 已确保）
    var = beta / (v * (alpha - 1))
    std = torch.sqrt(var + 1e-6)
    
    raw_ratio = error / (std + 1e-6)
    ratio = torch.clamp(raw_ratio, max=5.0)
    
    penalty = (ratio - 1.0) ** 2
    
    # 证据截断
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
        progress = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 5)
        reg_weight = CONFIG["reg_coeff"] * progress
    
    total_loss = loss_nll + reg_weight * loss_reg
    return total_loss, loss_nll.item(), loss_reg.item()

# ==============================================================================
# 3. 模型定义（移除门控，改为直接拼接）
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
    """
    双塔Transformer模型（无门控，直接拼接）
    - 客户端特征塔 + 镜像特征塔 → 特征向量拼接
    - 算法嵌入
    - 拼接后送入MLP预测NIG分布参数
    """
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
        super().__init__()
        self.client_tower = TransformerTower(client_feats, embed_dim)
        self.image_tower = TransformerTower(image_feats, embed_dim)
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        # 隐藏层（输入维度：client_vec + image_vec + algo_vec = embed_dim*3）
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
        # 提取特征向量
        c_vec = self.client_tower(cx)   # [batch, embed_dim]
        i_vec = self.image_tower(ix)    # [batch, embed_dim]
        a_vec = self.algo_embed(ax)     # [batch, embed_dim]
        
        # 直接拼接客户端和镜像特征（取消门控）
        fused_vec = torch.cat([c_vec, i_vec], dim=1)  # [batch, embed_dim*2]
        
        # 与算法向量拼接
        combined = torch.cat([fused_vec, a_vec], dim=1)  # [batch, embed_dim*3]
        
        out = self.head(self.hidden(combined))
        
        # 约束NIG参数
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + 0.1
        alpha = F.softplus(out[:, 2]) + 1.1   # 确保 alpha > 1
        beta = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 4. 数据加载（修复scaler保存错误，增加测试集划分）
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
        
        # 列名标准化
        rename_map = {
            "image": "image_name", 
            "method": "algo_name", 
            "network_bw": "bandwidth_mbps", 
            "network_delay": "network_rtt", 
            "mem_limit": "mem_limit_mb"
        }
        df_exp = df_exp.rename(columns=rename_map)
        
        # 兼容total_time列名
        if 'total_time' not in df_exp.columns: 
            cols = [c for c in df_exp.columns if 'total_tim' in c]
            if cols: 
                df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
        df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
        df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
        # 客户端特征
        cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        # 镜像特征（仅保留存在的列）
        target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
                       'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
        cols_i = [c for c in target_cols if c in df.columns]
        
        # ✅ 修复1：正确保存已拟合的 scaler，而不是重新fit
        scaler_c = StandardScaler().fit(df[cols_c].values)
        Xc = scaler_c.transform(df[cols_c].values)
        
        scaler_i = StandardScaler().fit(df[cols_i].values)
        Xi = scaler_i.transform(df[cols_i].values)
        
        enc = LabelEncoder()
        Xa = enc.fit_transform(df['algo_name'].values)
        y = np.log1p(df['total_time'].values)
        
        # 保存预处理对象
        with open('preprocessing_objects.pkl', 'wb') as f:
            pickle.dump({
                'scaler_c': scaler_c, 
                'scaler_i': scaler_i, 
                'enc': enc
            }, f)
        
        print(f"✅ 数据加载成功，总样本数: {len(y)}")
        return Xc, Xi, Xa, y, enc, len(cols_c), len(cols_i)
    
    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# 5. 训练主循环（增加独立测试集）
# ==============================================================================
if __name__ == "__main__":
    data = load_data()
    if data:
        Xc, Xi, Xa, y, enc_algo, c_dim, i_dim = data
        N = len(y)
        idx = np.random.permutation(N)
        
        # ✅ 修复2：划分训练(70%)、验证(15%)、测试(15%)
        n_tr = int(N * 0.7)
        n_val = int(N * 0.15)
        n_te = N - n_tr - n_val
        
        tr_idx = idx[:n_tr]
        val_idx = idx[n_tr:n_tr+n_val]
        te_idx = idx[n_tr+n_val:]
        
        print(f"📊 数据集划分: 训练 {len(tr_idx)} 条, 验证 {len(val_idx)} 条, 测试 {len(te_idx)} 条")
        
        # 创建数据集
        tr_d = CTSDataset(Xc[tr_idx], Xi[tr_idx], Xa[tr_idx], y[tr_idx])
        val_d = CTSDataset(Xc[val_idx], Xi[val_idx], Xa[val_idx], y[val_idx])
        te_d = CTSDataset(Xc[te_idx], Xi[te_idx], Xa[te_idx], y[te_idx])
        
        tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
        te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"])  # 仅用于最终评估
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 使用设备: {device}")
        
        model = CTSDualTowerModel(c_dim, i_dim, len(enc_algo.classes_)).to(device)
        print(f"📦 模型结构:\n{model}")
        
        optimizer = optim.AdamW(model.parameters(), 
                               lr=CONFIG["lr"], 
                               weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
        
        best_corr = -1.0
        best_epoch = 0
        patience_counter = 0
        history = {'loss': [], 'corr': [], 'test_corr': []}
        
        for epoch in range(CONFIG["epochs"]):
            # ---------- 训练 ----------
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
            
            # ---------- 验证 ----------
            model.eval()
            uncs, errs = [], []
            with torch.no_grad():
                for cx, ix, ax, target in val_loader:
                    cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
                    preds = model(cx, ix, ax)
                    gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                    
                    # 不确定性度量（方差）
                    unc = beta / (v * (alpha - 1))
                    # 绝对误差（原始尺度）
                    err = torch.abs(torch.expm1(gamma) - torch.expm1(target))
                    uncs.extend(unc.cpu().numpy())
                    errs.extend(err.cpu().numpy())
            
            try:
                corr, _ = spearmanr(uncs, errs)
                corr = corr if not np.isnan(corr) else 0.0
            except:
                corr = 0.0
            
            history['loss'].append(t_loss/len(tr_loader))
            history['corr'].append(corr)
            
            # ---------- 早停与模型保存 ----------
            print(f"Epoch {epoch+1:03d} | Loss: {history['loss'][-1]:.4f} | Val Corr: {corr:.4f}", end="")
            
            if corr > best_corr:
                best_corr = corr
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_corr': best_corr,
                    'epoch': epoch,
                    'config': CONFIG
                }, CONFIG["model_save_path"])
                print(f" 🌟 新最佳模型 (Corr={best_corr:.4f})")
            else:
                patience_counter += 1
                print(f" (耐心: {patience_counter}/{CONFIG['patience']})")
                
            if patience_counter >= CONFIG["patience"]:
                print(f"\n⏹️ 触发早停，停止训练。")
                break
        
        # ---------- 最终测试（加载最佳模型）----------
        print("\n🔍 加载最佳模型进行测试集评估...")
        checkpoint = torch.load(CONFIG["model_save_path"])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_uncs, test_errs = [], []
        with torch.no_grad():
            for cx, ix, ax, target in te_loader:
                cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
                preds = model(cx, ix, ax)
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
                unc = beta / (v * (alpha - 1))
                err = torch.abs(torch.expm1(gamma) - torch.expm1(target))
                test_uncs.extend(unc.cpu().numpy())
                test_errs.extend(err.cpu().numpy())
        
        try:
            test_corr, _ = spearmanr(test_uncs, test_errs)
            test_corr = test_corr if not np.isnan(test_corr) else 0.0
        except:
            test_corr = 0.0
        
        print(f"✅ 测试集 Spearman 相关系数: {test_corr:.4f}")
        
        # ---------- 训练曲线可视化 ----------
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['corr'], color='#ff7f0e', label='Validation Corr')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch+1}')
        plt.title('验证集 Spearman 相关性')
        plt.xlabel('Epoch')
        plt.ylabel('Spearman ρ')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.scatter(test_uncs, test_errs, alpha=0.5, s=10)
        plt.xlabel('预测不确定性 (方差)')
        plt.ylabel('绝对预测误差 (秒)')
        plt.title(f'测试集: 不确定性 vs 误差 (ρ={test_corr:.3f})')
        plt.xscale('log')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('training_result_strong.png', dpi=150)
        plt.show()
        
        print(f"\n🎉 训练完成！最佳模型: {CONFIG['model_save_path']}")
        print(f"   最佳验证 Corr: {best_corr:.4f} (Epoch {best_epoch+1})")
        print(f"   测试集 Corr: {test_corr:.4f}")