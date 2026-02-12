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
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import pickle


CONFIG = {
    "kl_coeff": 1.5,
    "annealing_epochs": 150,
    "lr": 1e-3,
    "epochs": 200,           # [修改] 减少轮数，防止过拟合
    "data_path": "cts_data.xlsx",         
    "feature_path": "image_features_database.csv",
    "batch_size": 64,
    "embed_dim": 32,         # [修改] 回到32，减少参数量
    "model_save_path": "cts_best_model_transformer_fix.pth",
    "scaler_save_path": "cts_scalers_transformer_fix.pkl",
    "weight_decay": 1e-3,    # [修改] 增大weight decay
    "grad_clip": 5.0,
    "use_spearman": True,
    "use_cross_features": True,
    "temperature_calibration": True,
    "patience": 30,          # [新增] 早停耐心值
}


def create_cross_features(df):
    df_new = df.copy()
    df_new['transfer_time_est'] = df_new['total_size_mb'] / (df_new['bandwidth_mbps'] + 1e-6)
    df_new['compute_density'] = df_new['layer_count'] * df_new['avg_layer_entropy']
    df_new['network_quality'] = df_new['bandwidth_mbps'] / (df_new['network_rtt'] + 1e-6)
    return df_new


def load_data():
    print(f"🔄 1. 正在读取数据: {CONFIG['data_path']} ...")
    try:
        df_exp = pd.read_excel(CONFIG["data_path"])
    except ImportError:
        print("❌ 读取失败！请运行 'pip install openpyxl'")
        exit(1)

    rename_map = {
        "image": "image_name", "method": "algo_name",
        "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
        "mem_limit": "mem_limit_mb"
    }
    df_exp = df_exp.rename(columns=rename_map)
    
    if 'total_time' not in df_exp.columns:
        possible_cols = [c for c in df_exp.columns if 'total_tim' in c]
        if possible_cols: df_exp = df_exp.rename(columns={possible_cols[0]: 'total_time'})

    df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
    if 'mem_limit_mb' not in df_exp.columns: 
        df_exp['mem_limit_mb'] = 1024.0
    
    print(f"🔄 2. 读取镜像特征: {CONFIG['feature_path']} ...")
    df_feat = pd.read_csv(CONFIG["feature_path"])
    df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    
    if CONFIG["use_cross_features"]:
        print("🔄 3. 创建物理启发交叉特征...")
        df = create_cross_features(df)
    
    print(f"✅ 数据加载完成，样本数: {len(df)}, 镜像数: {df['image_name'].nunique()}")
    return df


def split_proportional(df, test_size=0.15, val_size=0.15, random_state=42):
    """比例一致划分"""
    print(f"\n📊 比例一致划分 (每个镜像: 训练{1-test_size-val_size:.0%}/验证{val_size:.0%}/测试{test_size:.0%})...")
    
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for img_name in df['image_name'].unique():
        img_df = df[df['image_name'] == img_name].copy()
        n_total = len(img_df)
        
        n_test = max(1, int(n_total * test_size))
        n_val = max(1, int(n_total * val_size))
        n_train = n_total - n_test - n_val
        
        if n_train < 1:
            print(f"   ⚠️ 镜像 {img_name} 样本太少({n_total})，全部划入训练集")
            train_dfs.append(img_df)
            continue
        
        img_df = img_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        test = img_df.iloc[:n_test]
        val = img_df.iloc[n_test:n_test+n_val]
        train = img_df.iloc[n_test+n_val:]
        
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"\n📊 汇总:")
    print(f"   训练集: {len(train_df):5d} 样本 ({len(train_df)/len(df)*100:.1f}%) | {train_df['image_name'].nunique()} 镜像")
    print(f"   验证集: {len(val_df):5d} 样本 ({len(val_df)/len(df)*100:.1f}%) | {val_df['image_name'].nunique()} 镜像")
    print(f"   测试集: {len(test_df):5d} 样本 ({len(test_df)/len(df)*100:.1f}%) | {test_df['image_name'].nunique()} 镜像")
    
    return train_df, val_df, test_df


def analyze_split_quality(train_df, val_df, test_df):
    print("\n🔍 划分质量分析:")
    for name, df in [("训练", train_df), ("验证", val_df), ("测试", test_df)]:
        print(f"   {name}: 均值={df['total_time'].mean():7.2f}s | std={df['total_time'].std():7.2f}s")


# ==============================================================================
# Transformer双塔（原版结构，embed_dim=32）
# ==============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights + self.biases


class TransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        tokens = self.tokenizer(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        out = self.transformer(tokens)
        return out[:, 0, :]


class CTSDualTowerModel(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
        super().__init__()
        self.client_tower = TransformerTower(client_feats, embed_dim)
        self.image_tower = TransformerTower(image_feats, embed_dim)
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        fusion_input_dim = embed_dim * 3 
        self.hidden = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(64, 4)

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        hidden = self.hidden(combined)
        out = self.head(hidden)
        
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.01 + 1e-6
        beta = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)


# ==============================================================================
# 损失函数
# ==============================================================================

def nig_nll_loss(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()


def nig_reg_loss(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    evidence = 2 * v + alpha
    return (error * evidence).mean()


def evidential_loss(pred, target, epoch, lambda_coef=CONFIG["kl_coeff"]):
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    
    loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
    loss_reg = nig_reg_loss(target, gamma, v, alpha, beta)
    
    if epoch < CONFIG["annealing_epochs"]:
        annealing_coef = (epoch / CONFIG["annealing_epochs"]) ** 2
    else:
        annealing_coef = 1.0
    
    total_loss = loss_nll + lambda_coef * annealing_coef * loss_reg
    return total_loss, loss_nll.item(), loss_reg.item(), annealing_coef


class CTSDataset(Dataset):
    def __init__(self, client_x, image_x, algo_x, y):
        self.cx = torch.FloatTensor(client_x)
        self.ix = torch.FloatTensor(image_x)
        self.ax = torch.LongTensor(algo_x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]


def find_temperature(model, val_loader, device, n_samples=500):
    print("\n🌡️ 开始温度校准...")
    model.eval()
    all_uncertainties, all_errors = [], []
    
    with torch.no_grad():
        for i, (cx, ix, ax, y) in enumerate(val_loader):
            if i * val_loader.batch_size >= n_samples:
                break
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            
            uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
            error = torch.abs(y - gamma).cpu().numpy()
            all_uncertainties.extend(uncertainty)
            all_errors.extend(error)
    
    best_T, best_corr = 1.0, -1
    for T in np.linspace(0.5, 2.0, 31):
        scaled_unc = np.array(all_uncertainties) * T
        try:
            corr, _ = spearmanr(scaled_unc, all_errors)
            if corr > best_corr:
                best_corr = corr
                best_T = T
        except:
            continue
    
    print(f"   最优温度 T={best_T:.3f}, 校准后 Corr={best_corr:.3f}")
    return best_T, best_corr


# ==============================================================================
# 主训练流程
# ==============================================================================

if __name__ == "__main__":
    df = load_data()
    
    col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb', 'network_quality']
    col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio',
                 'transfer_time_est', 'compute_density']
    
    train_df, val_df, test_df = split_proportional(df, test_size=0.15, val_size=0.15)
    analyze_split_quality(train_df, val_df, test_df)
    
    def prepare_features(df_subset, scaler_c=None, scaler_i=None, enc_algo=None, fit=True):
        if fit:
            scaler_c, scaler_i, enc_algo = StandardScaler(), StandardScaler(), LabelEncoder()
            X_client = scaler_c.fit_transform(df_subset[col_client].values)
            X_image = scaler_i.fit_transform(df_subset[col_image].values)
            X_algo = enc_algo.fit_transform(df_subset['algo_name'].values)
        else:
            X_client = scaler_c.transform(df_subset[col_client].values)
            X_image = scaler_i.transform(df_subset[col_image].values)
            X_algo = enc_algo.transform(df_subset['algo_name'].values)
        y_target = np.log1p(df_subset['total_time'].values)
        return X_client, X_image, X_algo, y_target, scaler_c, scaler_i, enc_algo
    
    Xc_train, Xi_train, Xa_train, y_train, scaler_c, scaler_i, enc_algo = prepare_features(train_df, fit=True)
    Xc_val, Xi_val, Xa_val, y_val, _, _, _ = prepare_features(val_df, scaler_c, scaler_i, enc_algo, fit=False)
    Xc_test, Xi_test, Xa_test, y_test, _, _, _ = prepare_features(test_df, scaler_c, scaler_i, enc_algo, fit=False)
    
    print(f"\n📊 特征维度: client={len(col_client)}, image={len(col_image)}")
    
    train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train), 
                              batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(CTSDataset(Xc_val, Xi_val, Xa_val, y_val), 
                            batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(CTSDataset(Xc_test, Xi_test, Xa_test, y_test), 
                             batch_size=CONFIG["batch_size"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ 训练设备: {device}")
    
    model = CTSDualTowerModel(
        client_feats=len(col_client),
        image_feats=len(col_image),
        num_algos=len(enc_algo.classes_),
        embed_dim=CONFIG["embed_dim"]
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,} (比之前少{n_params-367876:,})")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    
    print(f"\n🚀 开始训练 (Transformer + 早停)...")
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'epoch': [], 'train_total': [], 'train_nll': [], 'train_reg': [], 
               'val_nll': [], 'uncertainty_corr': [], 'lr': []}
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_total = train_nll = train_reg = 0
        
        for cx, ix, ax, y in train_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            optimizer.zero_grad()
            
            preds = model(cx, ix, ax)
            loss, nll, reg, anneal = evidential_loss(preds, y, epoch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            
            train_total += loss.item()
            train_nll += nll
            train_reg += reg
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 验证
        model.eval()
        val_nll = 0
        all_uncertainties, all_errors = [], []
        
        with torch.no_grad():
            for cx, ix, ax, y in val_loader:
                cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
                preds = model(cx, ix, ax)
                gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
                
                val_nll += nig_nll_loss(y, gamma, v, alpha, beta).item()
                
                uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
                error = torch.abs(y - gamma).cpu().numpy()
                all_uncertainties.extend(uncertainty)
                all_errors.extend(error)
        
        try:
            corr, _ = spearmanr(all_uncertainties, all_errors)
        except:
            corr = 0
        
        avg_train_total = train_total / len(train_loader)
        avg_train_nll = train_nll / len(train_loader)
        avg_train_reg = train_reg / len(train_loader)
        avg_val_nll = val_nll / len(val_loader)
        
        history['epoch'].append(epoch)
        history['train_total'].append(avg_train_total)
        history['train_nll'].append(avg_train_nll)
        history['train_reg'].append(avg_train_reg)
        history['val_nll'].append(avg_val_nll)
        history['uncertainty_corr'].append(corr)
        history['lr'].append(current_lr)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_total:.3f} | "
                  f"Val NLL: {avg_val_nll:.3f} | Corr: {corr:+.3f} | LR: {current_lr:.2e}")
        
        # [新增] 早停检查
        if avg_val_nll < best_loss:
            best_loss = avg_val_nll
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': CONFIG,
            }, CONFIG["model_save_path"])
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n⏹️ 早停触发！连续{CONFIG['patience']}轮无改善")
                break

    print(f"\n💾 训练结束！最佳Val NLL: {best_loss:.4f} (Epoch {best_epoch})")
    
    # 保存scaler
    with open(CONFIG["scaler_save_path"], 'wb') as f:
        pickle.dump({
            'scaler_c': scaler_c, 'scaler_i': scaler_i, 'enc_algo': enc_algo,
            'col_client': col_client, 'col_image': col_image
        }, f)
    
    # 加载最佳模型进行校准和测试
    checkpoint = torch.load(CONFIG["model_save_path"], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    best_T, calibrated_corr = find_temperature(model, val_loader, device)
    CONFIG["best_temperature"] = best_T
    
    # 绘图（略，与之前相同）
    
    # 测试集评估
    print("\n🔮 最终测试集评估:")
    model.eval()
    test_nll = 0
    all_test_unc, all_test_err, all_preds, all_targets = [], [], [], []
    
    with torch.no_grad():
        for cx, ix, ax, y in test_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            
            test_nll += nig_nll_loss(y, gamma, v, alpha, beta).item()
            
            uncertainty = (beta / (v * (alpha - 1))).cpu().numpy()
            error = torch.abs(y - gamma).cpu().numpy()
            all_test_unc.extend(uncertainty)
            all_test_err.extend(error)
            all_preds.extend(gamma.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_test_unc = np.array(all_test_unc) * best_T
    final_corr, _ = spearmanr(all_test_unc, all_test_err)
    rmse = np.sqrt(np.mean((np.expm1(all_targets) - np.expm1(all_preds))**2))
    
    print(f"\n{'='*60}")
    print(f"📊 测试集最终指标:")
    print(f"   Test NLL: {test_nll/len(test_loader):.4f}")
    print(f"   RMSE: {rmse:.4f} (秒)")
    print(f"   Uncertainty-Error Corr: {final_corr:+.3f}")
    print(f"   最佳模型Epoch: {best_epoch}")
    print(f"{'='*60}")