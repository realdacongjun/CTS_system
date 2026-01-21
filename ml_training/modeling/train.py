import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. 配置区域 (Hyperparameters)
# ==============================================================================
CONFIG = {
    "data_path": "cts_data.xlsx",         
    "feature_path": "image_features_database.csv",
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 200,             # 增加训练轮数，让模型充分训练
    "embed_dim": 32,
    "kl_coeff": 0.15,          # [调整] 增加正则化权重，让不确定性更准确
    "model_save_path": "cts_best_model_full.pth" 
}

# 路径检查与自动修正 (解决你的路径烦恼)
# 如果当前目录下找不到，尝试去上一级目录找
if not os.path.exists(CONFIG["data_path"]):
    if os.path.exists(f"../{CONFIG['data_path']}"):
        CONFIG["data_path"] = f"../{CONFIG['data_path']}"
        CONFIG["feature_path"] = f"../{CONFIG['feature_path']}"
        print(f"📂 自动切换数据路径到上一级: {CONFIG['data_path']}")

# ==============================================================================
# 🌟 核心新增: 证据深度学习损失函数 (NIG Loss)
# ==============================================================================
# 参考文献: Deep Evidential Regression (Amini et al., NeurIPS 2020)
def nig_nll_loss(y, gamma, v, alpha, beta):
    """计算负对数似然损失 (NLL): 让预测值(gamma)接近真实值(y)"""
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_blambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean()

def nig_reg_loss(y, gamma, v, alpha, beta):
    """计算正则化损失: 惩罚模型在预测错误时还盲目自信"""
    error = torch.abs(y - gamma)
    evidence = 2 * v + alpha
    return (error * evidence).mean()

def evidential_loss(pred, target, epoch, total_epochs, lambda_coef=CONFIG["kl_coeff"]):
    """总损失 = NLL + 动态权重的正则项"""
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    
    loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
    loss_reg = nig_reg_loss(target, gamma, v, alpha, beta)
    
    # 动态调整正则化系数 (Annealing): 前期关注拟合，后期关注不确定性校准
    annealing_coef = min(1.0, epoch / (total_epochs * 0.15))  # [调整] 15%的训练轮数用于退火
    
    return loss_nll + lambda_coef * annealing_coef * loss_reg

# ==============================================================================
# 2. 模型定义 (必须与 cags_run.py 保持一致)
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
        # [修改点] 输出层 4 个神经元 (Gamma, v, Alpha, Beta)
        self.head = nn.Linear(64, 4) 

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        hidden = self.hidden(combined)
        out = self.head(hidden)
        
        # [修改点] 施加数学约束 (Softplus)
        gamma = out[:, 0]
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 3. 数据处理与加载
# ==============================================================================
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
    
    if 'mem_limit_mb' not in df_exp.columns: df_exp['mem_limit_mb'] = 1024.0
    
    print(f"🔄 2. 读取镜像特征: {CONFIG['feature_path']} ...")
    df_feat = pd.read_csv(CONFIG["feature_path"])
    
    df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    print(f"✅ 数据加载完成，样本数: {len(df)}")
    return df

class CTSDataset(Dataset):
    def __init__(self, client_x, image_x, algo_x, y):
        self.cx = torch.FloatTensor(client_x)
        self.ix = torch.FloatTensor(image_x)
        self.ax = torch.LongTensor(algo_x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

# ==============================================================================
# 4. 主训练流程 (含 EDL 训练逻辑)
# ==============================================================================
if __name__ == "__main__":
    # --- Step 1: 准备数据 ---
    df = load_data()
    
    col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
    col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
    
    scaler_c = StandardScaler()
    X_client = scaler_c.fit_transform(df[col_client].values)
    
    scaler_i = StandardScaler()
    X_image = scaler_i.fit_transform(df[col_image].values)
    
    enc_algo = LabelEncoder()
    X_algo = enc_algo.fit_transform(df['algo_name'].values)
    
    y_target = np.log1p(df['total_time'].values) # Log 变换

    Xc_train, Xc_test, Xi_train, Xi_test, Xa_train, Xa_test, y_train, y_test = train_test_split(
        X_client, X_image, X_algo, y_target, test_size=0.2, random_state=42
    )
    
    train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(CTSDataset(Xc_test, Xi_test, Xa_test, y_test), batch_size=CONFIG["batch_size"])
    
    # --- Step 2: 模型初始化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 训练设备: {device}")
    
    model = CTSDualTowerModel(
        client_feats=len(col_client),
        image_feats=len(col_image),
        num_algos=len(enc_algo.classes_)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    # --- Step 3: 训练 (EDL Loop) ---
    print(f"\n🚀 开始训练 (证据深度学习版 - Uncertainty Aware)...")
    best_loss = float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        
        for cx, ix, ax, y in train_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 前向传播 (输出4个参数)
            preds = model(cx, ix, ax)
            
            # 计算 EDL 损失 (NLL + Regularization)
            loss = evidential_loss(preds, y, epoch, CONFIG["epochs"])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证集 (只看 NLL 即可，验证预测准不准)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cx, ix, ax, y in test_loader:
                cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
                preds = model(cx, ix, ax)
                gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
                # 验证集不需要正则项，只算 NLL
                val_loss += nig_nll_loss(y, gamma, v, alpha, beta).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val NLL: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), CONFIG["model_save_path"])

    print(f"\n💾 训练结束！模型保存至: {os.path.abspath(CONFIG['model_save_path'])}")
    
    # --- Step 4: 演示 (含不确定性) ---
    print("\n🔮 预测效果与不确定性演示:")
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    model.eval()
    
    with torch.no_grad():
        cx, ix, ax, y = next(iter(test_loader))
        cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
        
        # 预测
        preds = model(cx, ix, ax)
        gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        
        print(f"{'算法':<12} | {'预测(s)':<10} | {'不确定性(U)':<12} | {'真实(s)':<10}")
        print("-" * 60)
        
        for i in range(5):
            pred_s = np.expm1(gamma[i].item())
            real_s = np.expm1(y[i].item())
            
            # 计算不确定性: Aleatoric + Epistemic
            # Uncertainty = Beta / (v * (Alpha - 1))
            uncertainty = beta[i] / (v[i] * (alpha[i] - 1))
            
            algo = enc_algo.inverse_transform([ax[i].item()])[0]
            
            print(f"{algo:<12} | {pred_s:<10.2f} | {uncertainty.item():<12.4f} | {real_s:<10.2f}")
    
    print("-" * 60)
    print("✅ 注意: '不确定性(U)' 越大，代表模型对该预测越没把握 (CAGS 将因此触发风险放大机制)。")