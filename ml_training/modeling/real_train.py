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
# import platform
# import matplotlib

# # ==============================================================================
# # 0. 基础配置
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False 

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(42)

# # ==============================================================================
# # 1. 超参数配置 (稳定版)
# # ==============================================================================
# CONFIG = {
#     "lr": 0.0005,              
#     "weight_decay": 1e-4,      
#     "epochs": 200,             
#     "patience": 15,           
#     "batch_size": 128,         
#     "embed_dim": 32,           
    
#     # 正则化参数（建议训练时观察loss_nll和loss_reg的量级，适当调整）
#     "reg_coeff": 1.0,          
#     "warmup_epochs": 3,        
    
#     "data_path": "cts_data.xlsx",
#     "feature_path": "image_features_database.csv",
#     "model_save_path": "cts_final_strong.pth",
# }

# # ==============================================================================
# # 2. 损失函数：Symmetric Strong EUB（保持不变）
# # ==============================================================================
# def nig_nll_loss(y, gamma, v, alpha, beta):
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def strong_eub_reg_loss(y, gamma, v, alpha, beta):
#     """
#     对称保真度正则项：强制误差/标准差趋近1，同时惩罚过度自信和过度保守
#     """
#     error = torch.abs(y - gamma)
    
#     # 计算标准差（移除 +1e-6 分母保护，因为 alpha>1 已确保）
#     var = beta / (v * (alpha - 1))
#     std = torch.sqrt(var + 1e-6)
    
#     raw_ratio = error / (std + 1e-6)
#     ratio = torch.clamp(raw_ratio, max=5.0)
    
#     penalty = (ratio - 1.0) ** 2
    
#     # 证据截断
#     evidence = torch.clamp(2 * v + alpha, max=20.0)
#     reg = penalty * torch.log1p(evidence)
    
#     return reg.mean()

# def evidential_loss(pred, target, epoch):
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
    
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = strong_eub_reg_loss(target, gamma, v, alpha, beta)
    
#     if epoch < CONFIG["warmup_epochs"]:
#         reg_weight = 0.0
#     else:
#         progress = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 5)
#         reg_weight = CONFIG["reg_coeff"] * progress
    
#     total_loss = loss_nll + reg_weight * loss_reg
#     return total_loss, loss_nll.item(), loss_reg.item()

# # ==============================================================================
# # 3. 模型定义（移除门控，改为直接拼接）
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
#     """
#     双塔Transformer模型（无门控，直接拼接）
#     - 客户端特征塔 + 镜像特征塔 → 特征向量拼接
#     - 算法嵌入
#     - 拼接后送入MLP预测NIG分布参数
#     """
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         # 隐藏层（输入维度：client_vec + image_vec + algo_vec = embed_dim*3）
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
#         # 提取特征向量
#         c_vec = self.client_tower(cx)   # [batch, embed_dim]
#         i_vec = self.image_tower(ix)    # [batch, embed_dim]
#         a_vec = self.algo_embed(ax)     # [batch, embed_dim]
        
#         # 直接拼接客户端和镜像特征（取消门控）
#         fused_vec = torch.cat([c_vec, i_vec], dim=1)  # [batch, embed_dim*2]
        
#         # 与算法向量拼接
#         combined = torch.cat([fused_vec, a_vec], dim=1)  # [batch, embed_dim*3]
        
#         out = self.head(self.hidden(combined))
        
#         # 约束NIG参数
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1
#         alpha = F.softplus(out[:, 2]) + 1.1   # 确保 alpha > 1
#         beta = F.softplus(out[:, 3]) + 1e-6
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 4. 数据加载（修复scaler保存错误，增加测试集划分）
# # ==============================================================================
# class CTSDataset(Dataset):
#     def __init__(self, cx, ix, ax, y):
#         self.cx = torch.FloatTensor(cx)
#         self.ix = torch.FloatTensor(ix)
#         self.ax = torch.LongTensor(ax)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): 
#         return len(self.y)
#     def __getitem__(self, idx): 
#         return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

# def load_data():
#     print(f"🔄 读取数据: {CONFIG['data_path']} ...")
#     if not os.path.exists(CONFIG['data_path']):
#         print(f"❌ 错误: 找不到文件 {CONFIG['data_path']}")
#         return None

#     try:
#         df_exp = pd.read_excel(CONFIG["data_path"])
#         df_feat = pd.read_csv(CONFIG["feature_path"])
        
#         # 列名标准化
#         rename_map = {
#             "image": "image_name", 
#             "method": "algo_name", 
#             "network_bw": "bandwidth_mbps", 
#             "network_delay": "network_rtt", 
#             "mem_limit": "mem_limit_mb"
#         }
#         df_exp = df_exp.rename(columns=rename_map)
        
#         # 兼容total_time列名
#         if 'total_time' not in df_exp.columns: 
#             cols = [c for c in df_exp.columns if 'total_tim' in c]
#             if cols: 
#                 df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
#         df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#         df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
#         # 客户端特征
#         cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#         # 镜像特征（仅保留存在的列）
#         target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
#                        'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#         cols_i = [c for c in target_cols if c in df.columns]
        
#         # ✅ 修复1：正确保存已拟合的 scaler，而不是重新fit
#         scaler_c = StandardScaler().fit(df[cols_c].values)
#         Xc = scaler_c.transform(df[cols_c].values)
        
#         scaler_i = StandardScaler().fit(df[cols_i].values)
#         Xi = scaler_i.transform(df[cols_i].values)
        
#         enc = LabelEncoder()
#         Xa = enc.fit_transform(df['algo_name'].values)
#         y = np.log1p(df['total_time'].values)
        
#         # 保存预处理对象
#         with open('preprocessing_objects.pkl', 'wb') as f:
#             pickle.dump({
#                 'scaler_c': scaler_c, 
#                 'scaler_i': scaler_i, 
#                 'enc': enc
#             }, f)
        
#         print(f"✅ 数据加载成功，总样本数: {len(y)}")
#         return Xc, Xi, Xa, y, enc, len(cols_c), len(cols_i)
    
#     except Exception as e:
#         print(f"❌ 数据处理出错: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # ==============================================================================
# # 5. 训练主循环（增加独立测试集）
# # ==============================================================================
# if __name__ == "__main__":
#     data = load_data()
#     if data:
#         Xc, Xi, Xa, y, enc_algo, c_dim, i_dim = data
#         N = len(y)
#         idx = np.random.permutation(N)
        
#         # ✅ 修复2：划分训练(70%)、验证(15%)、测试(15%)
#         n_tr = int(N * 0.7)
#         n_val = int(N * 0.15)
#         n_te = N - n_tr - n_val
        
#         tr_idx = idx[:n_tr]
#         val_idx = idx[n_tr:n_tr+n_val]
#         te_idx = idx[n_tr+n_val:]
        
#         print(f"📊 数据集划分: 训练 {len(tr_idx)} 条, 验证 {len(val_idx)} 条, 测试 {len(te_idx)} 条")
        
#         # 创建数据集
#         tr_d = CTSDataset(Xc[tr_idx], Xi[tr_idx], Xa[tr_idx], y[tr_idx])
#         val_d = CTSDataset(Xc[val_idx], Xi[val_idx], Xa[val_idx], y[val_idx])
#         te_d = CTSDataset(Xc[te_idx], Xi[te_idx], Xa[te_idx], y[te_idx])
        
#         tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
#         val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
#         te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"])  # 仅用于最终评估
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🚀 使用设备: {device}")
        
#         model = CTSDualTowerModel(c_dim, i_dim, len(enc_algo.classes_)).to(device)
#         print(f"📦 模型结构:\n{model}")
        
#         optimizer = optim.AdamW(model.parameters(), 
#                                lr=CONFIG["lr"], 
#                                weight_decay=CONFIG["weight_decay"])
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
        
#         best_corr = -1.0
#         best_epoch = 0
#         patience_counter = 0
#         history = {'loss': [], 'corr': [], 'test_corr': []}
        
#         for epoch in range(CONFIG["epochs"]):
#             # ---------- 训练 ----------
#             model.train()
#             t_loss = 0
#             for cx, ix, ax, target in tr_loader:
#                 cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 loss, _, _ = evidential_loss(model(cx, ix, ax), target, epoch)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 t_loss += loss.item()
            
#             scheduler.step()
            
#             # ---------- 验证 ----------
#             model.eval()
#             uncs, errs = [], []
#             with torch.no_grad():
#                 for cx, ix, ax, target in val_loader:
#                     cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                     preds = model(cx, ix, ax)
#                     gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                    
#                     # 不确定性度量（方差）
#                     unc = beta / (v * (alpha - 1))
#                     # 绝对误差（原始尺度）
#                     err = torch.abs(torch.expm1(gamma) - torch.expm1(target))
#                     uncs.extend(unc.cpu().numpy())
#                     errs.extend(err.cpu().numpy())
            
#             try:
#                 corr, _ = spearmanr(uncs, errs)
#                 corr = corr if not np.isnan(corr) else 0.0
#             except:
#                 corr = 0.0
            
#             history['loss'].append(t_loss/len(tr_loader))
#             history['corr'].append(corr)
            
#             # ---------- 早停与模型保存 ----------
#             print(f"Epoch {epoch+1:03d} | Loss: {history['loss'][-1]:.4f} | Val Corr: {corr:.4f}", end="")
            
#             if corr > best_corr:
#                 best_corr = corr
#                 best_epoch = epoch
#                 patience_counter = 0
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'best_corr': best_corr,
#                     'epoch': epoch,
#                     'config': CONFIG
#                 }, CONFIG["model_save_path"])
#                 print(f" 🌟 新最佳模型 (Corr={best_corr:.4f})")
#             else:
#                 patience_counter += 1
#                 print(f" (耐心: {patience_counter}/{CONFIG['patience']})")
                
#             if patience_counter >= CONFIG["patience"]:
#                 print(f"\n⏹️ 触发早停，停止训练。")
#                 break
        
#         # ---------- 最终测试（加载最佳模型）----------
#         print("\n🔍 加载最佳模型进行测试集评估...")
#         checkpoint = torch.load(CONFIG["model_save_path"])
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
        
#         test_uncs, test_errs = [], []
#         with torch.no_grad():
#             for cx, ix, ax, target in te_loader:
#                 cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                 preds = model(cx, ix, ax)
#                 gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
#                 unc = beta / (v * (alpha - 1))
#                 err = torch.abs(torch.expm1(gamma) - torch.expm1(target))
#                 test_uncs.extend(unc.cpu().numpy())
#                 test_errs.extend(err.cpu().numpy())
        
#         try:
#             test_corr, _ = spearmanr(test_uncs, test_errs)
#             test_corr = test_corr if not np.isnan(test_corr) else 0.0
#         except:
#             test_corr = 0.0
        
#         print(f"✅ 测试集 Spearman 相关系数: {test_corr:.4f}")
        
#         # ---------- 训练曲线可视化 ----------
#         plt.figure(figsize=(15, 5))
        
#         plt.subplot(1, 3, 1)
#         plt.plot(history['loss'], label='Training Loss')
#         plt.title('训练损失曲线')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.subplot(1, 3, 2)
#         plt.plot(history['corr'], color='#ff7f0e', label='Validation Corr')
#         plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch+1}')
#         plt.title('验证集 Spearman 相关性')
#         plt.xlabel('Epoch')
#         plt.ylabel('Spearman ρ')
#         plt.legend()
        
#         plt.subplot(1, 3, 3)
#         plt.scatter(test_uncs, test_errs, alpha=0.5, s=10)
#         plt.xlabel('预测不确定性 (方差)')
#         plt.ylabel('绝对预测误差 (秒)')
#         plt.title(f'测试集: 不确定性 vs 误差 (ρ={test_corr:.3f})')
#         plt.xscale('log')
#         plt.yscale('log')
        
#         plt.tight_layout()
#         plt.savefig('training_result_strong.png', dpi=150)
#         plt.show()
        
#         print(f"\n🎉 训练完成！最佳模型: {CONFIG['model_save_path']}")
#         print(f"   最佳验证 Corr: {best_corr:.4f} (Epoch {best_epoch+1})")
#         print(f"   测试集 Corr: {test_corr:.4f}")

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
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.stats import spearmanr
# import pickle
# import random
# import math
# import platform
# import matplotlib

# # ==============================================================================
# # 0. 基础配置
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False 

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(42)

# # ==============================================================================
# # 1. 超参数配置 (稳定版)
# # ==============================================================================
# CONFIG = {
#     "lr": 0.0005,              
#     "weight_decay": 1e-4,      
#     "epochs": 200,             
#     "patience": 15,           
#     "batch_size": 128,         
#     "embed_dim": 32,           
    
#     # 正则化参数（建议训练时观察loss_nll和loss_reg的量级，适当调整）
#     "reg_coeff": 1.0,          
#     "warmup_epochs": 3,        
    
#     "data_path": "cts_data.xlsx",
#     "feature_path": "image_features_database.csv",
#     "model_save_path": "cts_final_strong.pth",
# }

# # ==============================================================================
# # 2. 损失函数：Symmetric Strong EUB（保持不变）
# # ==============================================================================
# def nig_nll_loss(y, gamma, v, alpha, beta):
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def strong_eub_reg_loss(y, gamma, v, alpha, beta):
#     """
#     对称保真度正则项：强制误差/标准差趋近1，同时惩罚过度自信和过度保守
#     """
#     error = torch.abs(y - gamma)
    
#     # 计算标准差（移除 +1e-6 分母保护，因为 alpha>1 已确保）
#     var = beta / (v * (alpha - 1))
#     std = torch.sqrt(var + 1e-6)
    
#     raw_ratio = error / (std + 1e-6)
#     ratio = torch.clamp(raw_ratio, max=5.0)
    
#     penalty = (ratio - 1.0) ** 2
    
#     # 证据截断
#     evidence = torch.clamp(2 * v + alpha, max=20.0)
#     reg = penalty * torch.log1p(evidence)
    
#     return reg.mean()

# def evidential_loss(pred, target, epoch):
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
    
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = strong_eub_reg_loss(target, gamma, v, alpha, beta)
    
#     if epoch < CONFIG["warmup_epochs"]:
#         reg_weight = 0.0
#     else:
#         progress = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 5)
#         reg_weight = CONFIG["reg_coeff"] * progress
    
#     total_loss = loss_nll + reg_weight * loss_reg
#     return total_loss, loss_nll.item(), loss_reg.item()

# # ==============================================================================
# # 3. 模型定义（移除门控，改为直接拼接）
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
#     """
#     双塔Transformer模型（无门控，直接拼接）
#     - 客户端特征塔 + 镜像特征塔 → 特征向量拼接
#     - 算法嵌入
#     - 拼接后送入MLP预测NIG分布参数
#     """
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         # 隐藏层（输入维度：client_vec + image_vec + algo_vec = embed_dim*3）
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
#         # 提取特征向量
#         c_vec = self.client_tower(cx)   # [batch, embed_dim]
#         i_vec = self.image_tower(ix)    # [batch, embed_dim]
#         a_vec = self.algo_embed(ax)     # [batch, embed_dim]
        
#         # 直接拼接客户端和镜像特征（取消门控）
#         fused_vec = torch.cat([c_vec, i_vec], dim=1)  # [batch, embed_dim*2]
        
#         # 与算法向量拼接
#         combined = torch.cat([fused_vec, a_vec], dim=1)  # [batch, embed_dim*3]
        
#         out = self.head(self.hidden(combined))
        
#         # 约束NIG参数
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1
#         alpha = F.softplus(out[:, 2]) + 1.1   # 确保 alpha > 1
#         beta = F.softplus(out[:, 3]) + 1e-6
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 4. 数据加载（修复scaler保存错误，增加测试集划分）
# # ==============================================================================
# class CTSDataset(Dataset):
#     def __init__(self, cx, ix, ax, y):
#         self.cx = torch.FloatTensor(cx)
#         self.ix = torch.FloatTensor(ix)
#         self.ax = torch.LongTensor(ax)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): 
#         return len(self.y)
#     def __getitem__(self, idx): 
#         return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

# def load_data():
#     print(f"🔄 读取数据: {CONFIG['data_path']} ...")
#     if not os.path.exists(CONFIG['data_path']):
#         print(f"❌ 错误: 找不到文件 {CONFIG['data_path']}")
#         return None

#     try:
#         df_exp = pd.read_excel(CONFIG["data_path"])
#         df_feat = pd.read_csv(CONFIG["feature_path"])
        
#         # 列名标准化
#         rename_map = {
#             "image": "image_name", 
#             "method": "algo_name", 
#             "network_bw": "bandwidth_mbps", 
#             "network_delay": "network_rtt", 
#             "mem_limit": "mem_limit_mb"
#         }
#         df_exp = df_exp.rename(columns=rename_map)
        
#         # 兼容total_time列名
#         if 'total_time' not in df_exp.columns: 
#             cols = [c for c in df_exp.columns if 'total_tim' in c]
#             if cols: 
#                 df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
#         df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#         df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
#         # 客户端特征
#         cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#         # 镜像特征（仅保留存在的列）
#         target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
#                        'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#         cols_i = [c for c in target_cols if c in df.columns]
        
#         # ✅ 修复1：正确保存已拟合的 scaler，而不是重新fit
#         scaler_c = StandardScaler().fit(df[cols_c].values)
#         Xc = scaler_c.transform(df[cols_c].values)
        
#         scaler_i = StandardScaler().fit(df[cols_i].values)
#         Xi = scaler_i.transform(df[cols_i].values)
        
#         enc = LabelEncoder()
#         Xa = enc.fit_transform(df['algo_name'].values)
#         y = np.log1p(df['total_time'].values)
        
#         # 保存预处理对象
#         with open('preprocessing_objects.pkl', 'wb') as f:
#             pickle.dump({
#                 'scaler_c': scaler_c, 
#                 'scaler_i': scaler_i, 
#                 'enc': enc
#             }, f)
        
#         print(f"✅ 数据加载成功，总样本数: {len(y)}")
#         return Xc, Xi, Xa, y, enc, len(cols_c), len(cols_i)
    
#     except Exception as e:
#         print(f"❌ 数据处理出错: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # ==============================================================================
# # 5. 训练主循环（修改：学习率调度 + 测试集指标）
# # ==============================================================================
# if __name__ == "__main__":
#     data = load_data()
#     if data:
#         Xc, Xi, Xa, y, enc_algo, c_dim, i_dim = data
#         N = len(y)
#         idx = np.random.permutation(N)
        
#         # ✅ 划分训练(70%)、验证(15%)、测试(15%)
#         n_tr = int(N * 0.7)
#         n_val = int(N * 0.15)
#         n_te = N - n_tr - n_val
        
#         tr_idx = idx[:n_tr]
#         val_idx = idx[n_tr:n_tr+n_val]
#         te_idx = idx[n_tr+n_val:]
        
#         # 验证划分正确性
#         assert len(tr_idx) + len(val_idx) + len(te_idx) == N, "索引划分错误"
        
#         print(f"📊 数据集划分: 训练 {len(tr_idx)} 条, 验证 {len(val_idx)} 条, 测试 {len(te_idx)} 条")
        
#         # 创建数据集
#         tr_d = CTSDataset(Xc[tr_idx], Xi[tr_idx], Xa[tr_idx], y[tr_idx])
#         val_d = CTSDataset(Xc[val_idx], Xi[val_idx], Xa[val_idx], y[val_idx])
#         te_d = CTSDataset(Xc[te_idx], Xi[te_idx], Xa[te_idx], y[te_idx])
        
#         tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
#         val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
#         te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"])
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🚀 使用设备: {device}")
        
#         model = CTSDualTowerModel(c_dim, i_dim, len(enc_algo.classes_)).to(device)
#         print(f"📦 模型结构:\n{model}")
        
#         optimizer = optim.AdamW(model.parameters(), 
#                                lr=CONFIG["lr"], 
#                                weight_decay=CONFIG["weight_decay"])
        
#         # ✅ 修改1：使用 ReduceLROnPlateau 替代 CosineAnnealingLR
#         # 监控验证集 Spearman 相关系数，当指标不再提升时降低学习率
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, 
#             mode='max',           # 最大化 Spearman corr
#             factor=0.5,           # 学习率衰减因子
#             patience=5,           # 等待5个epoch
#             verbose=True,         # 打印学习率调整信息
#             min_lr=1e-6           # 最小学习率
#         )
        
#         best_corr = -1.0
#         best_epoch = 0
#         patience_counter = 0
#         history = {'loss': [], 'corr': [], 'test_corr': []}
        
#         for epoch in range(CONFIG["epochs"]):
#             # ---------- 训练 ----------
#             model.train()
#             t_loss = 0
#             for cx, ix, ax, target in tr_loader:
#                 cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 loss, _, _ = evidential_loss(model(cx, ix, ax), target, epoch)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 t_loss += loss.item()
            
#             # ---------- 验证 ----------
#             model.eval()
#             val_preds, val_targets = [], []
#             val_uncs, val_errs = [], []
            
#             with torch.no_grad():
#                 for cx, ix, ax, target in val_loader:
#                     cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                     preds = model(cx, ix, ax)
#                     gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                    
#                     # 不确定性度量（方差）
#                     unc = beta / (v * (alpha - 1))
#                     # 绝对误差（原始尺度）
#                     err = torch.abs(torch.expm1(gamma) - torch.expm1(target))
                    
#                     val_uncs.extend(unc.cpu().numpy())
#                     val_errs.extend(err.cpu().numpy())
#                     val_preds.extend(torch.expm1(gamma).cpu().numpy())
#                     val_targets.extend(torch.expm1(target).cpu().numpy())
            
#             try:
#                 corr, _ = spearmanr(val_uncs, val_errs)
#                 corr = corr if not np.isnan(corr) else 0.0
#             except:
#                 corr = 0.0
            
#             history['loss'].append(t_loss/len(tr_loader))
#             history['corr'].append(corr)
            
#             # ✅ 修改1续：根据验证指标调整学习率
#             scheduler.step(corr)
            
#             # ---------- 早停与模型保存 ----------
#             print(f"Epoch {epoch+1:03d} | Loss: {history['loss'][-1]:.4f} | Val Corr: {corr:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}", end="")
            
#             if corr > best_corr:
#                 best_corr = corr
#                 best_epoch = epoch
#                 patience_counter = 0
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'best_corr': best_corr,
#                     'epoch': epoch,
#                     'config': CONFIG
#                 }, CONFIG["model_save_path"])
#                 print(f" 🌟 新最佳模型 (Corr={best_corr:.4f})")
#             else:
#                 patience_counter += 1
#                 print(f" (耐心: {patience_counter}/{CONFIG['patience']})")
                
#             if patience_counter >= CONFIG["patience"]:
#                 print(f"\n⏹️ 触发早停，停止训练。")
#                 break
        
#         # ---------- 最终测试（加载最佳模型）----------
#         print("\n🔍 加载最佳模型进行测试集评估...")
#         checkpoint = torch.load(CONFIG["model_save_path"])
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
        
#         # ✅ 修改2：增加 MAE 和 RMSE 指标计算
#         test_uncs, test_errs = [], []
#         test_preds, test_targets = [], []  # 用于MAE/RMSE
        
#         with torch.no_grad():
#             for cx, ix, ax, target in te_loader:
#                 cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                 preds = model(cx, ix, ax)
#                 gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
#                 # 转换回原始时间尺度（秒）
#                 pred_time = torch.expm1(gamma)
#                 true_time = torch.expm1(target)
                
#                 # 不确定性
#                 unc = beta / (v * (alpha - 1))
#                 # 误差
#                 err = torch.abs(pred_time - true_time)
                
#                 test_uncs.extend(unc.cpu().numpy())
#                 test_errs.extend(err.cpu().numpy())
#                 test_preds.extend(pred_time.cpu().numpy())
#                 test_targets.extend(true_time.cpu().numpy())
        
#         # 转换为numpy数组
#         test_preds = np.array(test_preds)
#         test_targets = np.array(test_targets)
#         test_errs = np.array(test_errs)
        
#         # 计算各项指标
#         try:
#             test_corr, _ = spearmanr(test_uncs, test_errs)
#             test_corr = test_corr if not np.isnan(test_corr) else 0.0
#         except:
#             test_corr = 0.0
        
#         # ✅ 修改2续：计算 MAE 和 RMSE
#         test_mae = mean_absolute_error(test_targets, test_preds)
#         test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        
#         print(f"\n{'='*50}")
#         print(f"📊 测试集评估结果:")
#         print(f"{'='*50}")
#         print(f"✅ Spearman 相关系数 (不确定性 vs 误差): {test_corr:.4f}")
#         print(f"✅ MAE  (平均绝对误差): {test_mae:.4f} 秒")
#         print(f"✅ RMSE (均方根误差):   {test_rmse:.4f} 秒")
#         print(f"{'='*50}")
        
#         # ---------- 训练曲线可视化 ----------
#         plt.figure(figsize=(15, 5))
        
#         plt.subplot(1, 3, 1)
#         plt.plot(history['loss'], label='Training Loss')
#         plt.title('训练损失曲线')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.subplot(1, 3, 2)
#         plt.plot(history['corr'], color='#ff7f0e', label='Validation Corr')
#         plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch+1}')
#         plt.title('验证集 Spearman 相关性')
#         plt.xlabel('Epoch')
#         plt.ylabel('Spearman ρ')
#         plt.legend()
        
#         plt.subplot(1, 3, 3)
#         plt.scatter(test_uncs, test_errs, alpha=0.5, s=10)
#         plt.xlabel('预测不确定性 (方差)')
#         plt.ylabel('绝对预测误差 (秒)')
#         plt.title(f'测试集: 不确定性 vs 误差 (ρ={test_corr:.3f})')
#         plt.xscale('log')
#         plt.yscale('log')
        
#         plt.tight_layout()
#         plt.savefig('training_result_strong.png', dpi=150)
#         plt.show()
        
#         print(f"\n🎉 训练完成！最佳模型: {CONFIG['model_save_path']}")
#         print(f"   最佳验证 Corr: {best_corr:.4f} (Epoch {best_epoch+1})")
#         print(f"   测试集 Corr: {test_corr:.4f}")
#         print(f"   测试集 MAE:  {test_mae:.4f} 秒")
#         print(f"   测试集 RMSE: {test_rmse:.4f} 秒")




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
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.stats import spearmanr
# from scipy.stats import mstats  # 用于Winsorizing
# import pickle
# import random
# import math
# import platform
# import matplotlib
# from collections import Counter

# # ==============================================================================
# # 0. 基础配置
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False 

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(42)

# # ==============================================================================
# # 1. 超参数配置
# # ==============================================================================
# CONFIG = {
#     "lr": 0.0003,
#     "weight_decay": 1e-4,
#     "epochs": 200,
#     "patience": 30,
#     "batch_size": 128,
#     "embed_dim": 32,
#     "reg_coeff": 2.0,
#     "warmup_epochs": 5,
    
#     "data_path": "cts_data.xlsx",
#     "feature_path": "image_features_database.csv",
#     "model_save_path": "cts_final_mape.pth",
    
#     "mape_weight": 0.5,
#     "corr_weight": 0.3,
#     "ece_weight": 0.2,
#     "ema_alpha": 0.9,
    
#     # Winsorizing参数（截尾而非删除）
#     "winsorize_limits": 0.05,  # 上下各5%截尾
# }

# # ==============================================================================
# # 2. 损失函数
# # ==============================================================================
# def nig_nll_loss(y, gamma, v, alpha, beta):
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def strong_eub_reg_loss(y, gamma, v, alpha, beta):
#     error = torch.abs(y - gamma)
#     var = beta / (v * (alpha - 1))
#     std = torch.sqrt(var + 1e-6)
#     raw_ratio = error / (std + 1e-6)
#     ratio = torch.clamp(raw_ratio, max=5.0)
#     penalty = (ratio - 1.0) ** 2
#     evidence = torch.clamp(2 * v + alpha, max=20.0)
#     reg = penalty * torch.log1p(evidence)
#     return reg.mean()

# def evidential_loss(pred, target, epoch):
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = strong_eub_reg_loss(target, gamma, v, alpha, beta)
    
#     if epoch < CONFIG["warmup_epochs"]:
#         reg_weight = 0.0
#     else:
#         if epoch < 20:
#             progress = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 10)
#             reg_weight = CONFIG["reg_coeff"] * progress
#         else:
#             reg_weight = CONFIG["reg_coeff"]
    
#     total_loss = loss_nll + reg_weight * loss_reg
#     return total_loss, loss_nll.item(), loss_reg.item()

# # ==============================================================================
# # 3. 模型定义
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
# # 4. 数据加载（保留所有数据，仅统计通知）
# # ==============================================================================
# class CTSDataset(Dataset):
#     def __init__(self, cx, ix, ax, y):
#         self.cx = torch.FloatTensor(cx)
#         self.ix = torch.FloatTensor(ix)
#         self.ax = torch.LongTensor(ax)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): 
#         return len(self.y)
#     def __getitem__(self, idx): 
#         return self.cx[idx], self.ix[idx], self.ax[idx], self.y[idx]

# def load_data():
#     print(f"🔄 读取数据: {CONFIG['data_path']} ...")
#     if not os.path.exists(CONFIG['data_path']):
#         print(f"❌ 错误: 找不到文件 {CONFIG['data_path']}")
#         return None

#     try:
#         df_exp = pd.read_excel(CONFIG["data_path"])
#         df_feat = pd.read_csv(CONFIG["feature_path"])
        
#         rename_map = {
#             "image": "image_name", 
#             "method": "algo_name", 
#             "network_bw": "bandwidth_mbps", 
#             "network_delay": "network_rtt", 
#             "mem_limit": "mem_limit_mb"
#         }
#         df_exp = df_exp.rename(columns=rename_map)
        
#         if 'total_time' not in df_exp.columns: 
#             cols = [c for c in df_exp.columns if 'total_tim' in c]
#             if cols: 
#                 df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
#         df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#         df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
#         # ✅ 统计极小值而非过滤
#         tiny_samples = (df['total_time'] < 0.5).sum()
#         tiny_ratio = tiny_samples / len(df) * 100
#         print(f"  极小值样本统计: {tiny_samples} 条 (<0.5s, {tiny_ratio:.2f}%)")
        
#         cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#         target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
#                        'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#         cols_i = [c for c in target_cols if c in df.columns]
        
#         Xc_raw = df[cols_c].values
#         Xi_raw = df[cols_i].values
#         y_raw = np.log1p(df['total_time'].values)
#         algo_names_raw = df['algo_name'].values
        
#         print(f"✅ 数据加载成功，总样本数: {len(y_raw)}")
#         print(f"   时间范围: [{df['total_time'].min():.2f}s, {df['total_time'].max():.2f}s]")
#         print(f"   时间中位数: {df['total_time'].median():.2f}s")
#         print(f"   客户端特征: {cols_c}")
#         print(f"   镜像特征: {cols_i}")
        
#         return Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i
        
#     except Exception as e:
#         print(f"❌ 数据处理出错: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # ==============================================================================
# # 5. 评估指标（双MAPE + Winsorizing）
# # ==============================================================================
# def calculate_mape(y_true, y_pred, epsilon=1e-8):
#     """
#     传统MAPE（用于对比）
#     """
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
#     return mape

# def calculate_smape(y_true, y_pred, epsilon=1e-8):
#     """
#     对称MAPE（主要指标）
#     """
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     numerator = 2 * np.abs(y_true - y_pred)
#     denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
#     smape = np.mean(numerator / denominator) * 100
#     return smape

# def winsorize_array(arr, limits=0.05):
#     """
#     Winsorizing：截尾极端值而非删除
#     """
#     return mstats.winsorize(arr, limits=[limits, limits]).data

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

# class MetricsNormalizer:
#     """
#     EMA平滑的动态归一化器（更早启动）
#     """
#     def __init__(self, window_size=5, ema_alpha=0.9):  # window_size从10改为5
#         self.smape_history = []
#         self.ece_history = []
#         self.window_size = window_size
#         self.ema_alpha = ema_alpha
#         self.smape_scale = 100.0
#         self.ece_scale = 50.0
        
#     def update(self, smape, ece):
#         self.smape_history.append(smape)
#         self.ece_history.append(ece)
        
#         # 更早开始更新（5轮后）
#         if len(self.smape_history) >= self.window_size:
#             recent_smape = self.smape_history[-self.window_size:]
#             recent_ece = self.ece_history[-self.window_size:]
            
#             new_smape_scale = np.percentile(recent_smape, 90)
#             new_ece_scale = np.percentile(recent_ece, 90)
            
#             # EMA平滑
#             self.smape_scale = self.ema_alpha * self.smape_scale + (1 - self.ema_alpha) * new_smape_scale
#             self.ece_scale = self.ema_alpha * self.ece_scale + (1 - self.ema_alpha) * new_ece_scale
            
#     def normalize(self, smape, corr, ece):
#         smape_norm = smape / (self.smape_scale + 1e-8)
#         ece_norm = ece / (self.ece_scale + 1e-8)
#         corr_norm = (1.0 - corr) / 2.0
        
#         return smape_norm, corr_norm, ece_norm

# def calculate_composite_score(smape_norm, corr_norm, ece_norm, config):
#     """计算综合得分（越低越好）"""
#     score = (config["mape_weight"] * smape_norm + 
#              config["corr_weight"] * corr_norm + 
#              config["ece_weight"] * ece_norm)
#     return score

# # ==============================================================================
# # 6. 训练主循环
# # ==============================================================================
# if __name__ == "__main__":
#     data = load_data()
#     if data is None:
#         exit(1)
        
#     Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i = data
#     N = len(y_raw)
    
#     # 划分索引
#     idx = np.random.permutation(N)
#     n_tr = int(N * 0.7)
#     n_val = int(N * 0.15)
    
#     tr_idx = idx[:n_tr]
#     val_idx = idx[n_tr:n_tr+n_val]
#     te_idx = idx[n_tr+n_val:]
    
#     print(f"📊 数据集划分: 训练 {len(tr_idx)} 条, 验证 {len(val_idx)} 条, 测试 {len(te_idx)} 条")
    
#     # 数据标准化
#     scaler_c = StandardScaler().fit(Xc_raw[tr_idx])
#     scaler_i = StandardScaler().fit(Xi_raw[tr_idx])
    
#     Xc_train = scaler_c.transform(Xc_raw[tr_idx])
#     Xc_val = scaler_c.transform(Xc_raw[val_idx])
#     Xc_test = scaler_c.transform(Xc_raw[te_idx])
    
#     Xi_train = scaler_i.transform(Xi_raw[tr_idx])
#     Xi_val = scaler_i.transform(Xi_raw[val_idx])
#     Xi_test = scaler_i.transform(Xi_raw[te_idx])
    
#     # LabelEncoder
#     enc = LabelEncoder()
#     enc.fit(algo_names_raw[tr_idx])
    
#     class_counts = Counter(algo_names_raw[tr_idx])
#     most_common_class = class_counts.most_common(1)[0][0]
#     default_idx = enc.transform([most_common_class])[0]
#     print(f"   默认算法: {most_common_class} (索引 {default_idx})")
    
#     def safe_transform(encoder, labels, default):
#         known_classes = set(encoder.classes_)
#         transformed = []
#         unknown_count = 0
#         for label in labels:
#             if label in known_classes:
#                 transformed.append(encoder.transform([label])[0])
#             else:
#                 transformed.append(default)
#                 unknown_count += 1
#         if unknown_count > 0:
#             print(f"   警告: {unknown_count} 个未知类别已映射为 {most_common_class}")
#         return np.array(transformed)
    
#     Xa_train = enc.transform(algo_names_raw[tr_idx])
#     Xa_val = safe_transform(enc, algo_names_raw[val_idx], default_idx)
#     Xa_test = safe_transform(enc, algo_names_raw[te_idx], default_idx)
    
#     y_train = y_raw[tr_idx]
#     y_val = y_raw[val_idx]
#     y_test = y_raw[te_idx]
    
#     # 保存预处理对象
#     with open('preprocessing_objects.pkl', 'wb') as f:
#         pickle.dump({
#             'scaler_c': scaler_c, 
#             'scaler_i': scaler_i, 
#             'enc': enc,
#             'cols_c': cols_c,
#             'cols_i': cols_i,
#             'default_algo_idx': default_idx,
#             'most_common_algo': most_common_class
#         }, f)
    
#     # 创建数据集
#     tr_d = CTSDataset(Xc_train, Xi_train, Xa_train, y_train)
#     val_d = CTSDataset(Xc_val, Xi_val, Xa_val, y_val)
#     te_d = CTSDataset(Xc_test, Xi_test, Xa_test, y_test)
    
#     tr_loader = DataLoader(tr_d, batch_size=CONFIG["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_d, batch_size=CONFIG["batch_size"])
#     te_loader = DataLoader(te_d, batch_size=CONFIG["batch_size"])
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🚀 使用设备: {device}")
    
#     model = CTSDualTowerModel(len(cols_c), len(cols_i), len(enc.classes_)).to(device)
#     print(f"📦 模型结构:\n{model}")
    
#     optimizer = optim.AdamW(model.parameters(), 
#                            lr=CONFIG["lr"], 
#                            weight_decay=CONFIG["weight_decay"])
    
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 
#         mode='min',
#         factor=0.5,           
#         patience=10,
#         verbose=True,         
#         min_lr=1e-6           
#     )
    
#     # EMA归一化器（更早启动）
#     normalizer = MetricsNormalizer(window_size=5, ema_alpha=CONFIG["ema_alpha"])
    
#     best_score = float('inf')
#     best_epoch = 0
#     patience_counter = 0
#     history = {
#         'loss': [], 'mae': [], 'smape': [], 'mape': [], 'rmse': [], 'corr': [], 'ece': [],'score': [], 'smape_scale': [], 'ece_scale': []
#     }
    
#     for epoch in range(CONFIG["epochs"]):
#         # ---------- 训练 ----------
#         model.train()
#         t_loss = 0
#         for cx, ix, ax, target in tr_loader:
#             cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#             optimizer.zero_grad()
#             loss, _, _ = evidential_loss(model(cx, ix, ax), target, epoch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             t_loss += loss.item()
        
#         # ---------- 验证 ----------
#         model.eval()
#         val_preds, val_targets = [], []
#         val_uncs, val_errs = [], []
        
#         with torch.no_grad():
#             for cx, ix, ax, target in val_loader:
#                 cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#                 preds = model(cx, ix, ax)
#                 gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
#                 pred_time = torch.expm1(gamma)
#                 true_time = torch.expm1(target)
                
#                 var = beta / (v * (alpha - 1))
#                 unc = torch.sqrt(var + 1e-6)
#                 err = torch.abs(pred_time - true_time)
                
#                 val_preds.extend(pred_time.cpu().numpy())
#                 val_targets.extend(true_time.cpu().numpy())
#                 val_uncs.extend(unc.cpu().numpy())
#                 val_errs.extend(err.cpu().numpy())
        
#         val_preds = np.array(val_preds)
#         val_targets = np.array(val_targets)
#         val_uncs = np.array(val_uncs)
#         val_errs = np.array(val_errs)
        
#         # ✅ Winsorizing极端误差（避免MAPE被离群点主导）
#         val_errs_winsorized = winsorize_array(val_errs, limits=CONFIG["winsorize_limits"])
        
#         # 计算指标（双MAPE）
#         val_mae = mean_absolute_error(val_targets, val_preds)
#         val_smape = calculate_smape(val_targets, val_preds)
#         val_mape = calculate_mape(val_targets, val_preds)  # 传统MAPE用于对比
#         val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
#         val_corr, _ = spearmanr(val_uncs, val_errs)
#         val_corr = val_corr if not np.isnan(val_corr) else 0.0
#         val_ece = calculate_ece_quantile(val_errs_winsorized, val_uncs)  # 使用Winsorized误差
        
#         # 动态归一化
#         normalizer.update(val_smape, val_ece)
#         smape_norm, corr_norm, ece_norm = normalizer.normalize(val_smape, val_corr, val_ece)
#         val_score = calculate_composite_score(smape_norm, corr_norm, ece_norm, CONFIG)
        
#         # 记录历史
#         history['loss'].append(t_loss/len(tr_loader))
#         history['mae'].append(val_mae)
#         history['smape'].append(val_smape)
#         history['mape'].append(val_mape)
#         history['rmse'].append(val_rmse)
#         history['corr'].append(val_corr)
#         history['ece'].append(val_ece)
#         history['score'].append(val_score)
#         history['smape_scale'].append(normalizer.smape_scale)
#         history['ece_scale'].append(normalizer.ece_scale)
        
#         scheduler.step(val_score)
        
#         # ---------- 模型保存 ----------
#         print(f"Epoch {epoch+1:03d} | Loss: {history['loss'][-1]:.4f} | "
#               f"MAE: {val_mae:.2f}s | sMAPE: {val_smape:.2f}% | MAPE: {val_mape:.2f}% | "
#               f"RMSE: {val_rmse:.2f}s | Corr: {val_corr:.4f} | ECE: {val_ece:.4f} | "
#               f"Score: {val_score:.4f} | "
#               f"Scale: {normalizer.smape_scale:.1f}/{normalizer.ece_scale:.1f} | "
#               f"LR: {optimizer.param_groups[0]['lr']:.6f}", end="")
        
#         if val_score < best_score:
#             best_score = val_score
#             best_epoch = epoch
#             patience_counter = 0
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_score': best_score,
#                 'best_smape': val_smape,
#                 'best_mape': val_mape,
#                 'best_corr': val_corr,
#                 'best_ece': val_ece,
#                 'epoch': epoch,
#                 'config': CONFIG,
#                 'normalizer_stats': {
#                     'smape_scale': normalizer.smape_scale,
#                     'ece_scale': normalizer.ece_scale
#                 }
#             }, CONFIG["model_save_path"])
#             print(f" 🌟 新最佳模型 (sMAPE={val_smape:.2f}%, Corr={val_corr:.3f})")
#         else:
#             patience_counter += 1
#             print(f" (耐心: {patience_counter}/{CONFIG['patience']})")
            
#         if patience_counter >= CONFIG["patience"]:
#             print(f"\n⏹️ 触发早停，停止训练。")
#             break
    
#     # ---------- 最终测试 ----------
#     print("\n🔍 加载最佳模型进行测试集评估...")
#     checkpoint = torch.load(CONFIG["model_save_path"])
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     test_uncs, test_errs = [], []
#     test_preds, test_targets = [], []
    
#     with torch.no_grad():
#         for cx, ix, ax, target in te_loader:
#             cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
#             preds = model(cx, ix, ax)
#             gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            
#             pred_time = torch.expm1(gamma)
#             true_time = torch.expm1(target)
            
#             var = beta / (v * (alpha - 1))
#             unc = torch.sqrt(var + 1e-6)
#             err = torch.abs(pred_time - true_time)
            
#             test_uncs.extend(unc.cpu().numpy())
#             test_errs.extend(err.cpu().numpy())
#             test_preds.extend(pred_time.cpu().numpy())
#             test_targets.extend(true_time.cpu().numpy())
    
#     test_preds = np.array(test_preds)
#     test_targets = np.array(test_targets)
#     test_errs = np.array(test_errs)
#     test_uncs = np.array(test_uncs)
    
#     # Winsorizing
#     test_errs_winsorized = winsorize_array(test_errs, limits=CONFIG["winsorize_limits"])
    
#     test_mae = mean_absolute_error(test_targets, test_preds)
#     test_smape = calculate_smape(test_targets, test_preds)
#     test_mape = calculate_mape(test_targets, test_preds)
#     test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
#     test_corr, _ = spearmanr(test_uncs, test_errs)
#     test_corr = test_corr if not np.isnan(test_corr) else 0.0
#     test_ece = calculate_ece_quantile(test_errs_winsorized, test_uncs)
    
#     print(f"\n{'='*60}")
#     print(f"📊 测试集评估结果:")
#     print(f"{'='*60}")
#     print(f"✅ MAE    (平均绝对误差):    {test_mae:.4f} 秒")
#     print(f"✅ sMAPE  (对称平均绝对百分比误差): {test_smape:.2f}% ⭐")
#     print(f"✅ MAPE   (传统平均绝对百分比误差): {test_mape:.2f}% (对比用)")
#     print(f"✅ RMSE   (均方根误差):      {test_rmse:.4f} 秒")
#     print(f"✅ Corr   (Spearman相关系数): {test_corr:.4f}")
#     print(f"✅ ECE    (期望校准误差):    {test_ece:.4f}")
#     print(f"{'='*60}")
    
#     # 可视化
#     fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
#     metrics = [
#         ('loss', '训练损失', 'Loss'),
#         ('mae', '验证集 MAE', 'MAE (seconds)'),
#         ('smape', '验证集 sMAPE', 'sMAPE (%)'),
#         ('mape', '验证集 MAPE', 'MAPE (%)'),
#         ('rmse', '验证集 RMSE', 'RMSE (seconds)'),
#         ('corr', '验证集 Spearman Corr', 'Correlation'),
#         ('ece', '验证集 ECE', 'ECE'),
#         ('score', '验证集综合得分', 'Composite Score'),
#         ('smape_scale', 'sMAPE归一化尺度', 'Scale'),
#         ('ece_scale', 'ECE归一化尺度', 'Scale')
#     ]
    
#     for idx, (key, title, ylabel) in enumerate(metrics):
#         row, col = idx // 5, idx % 5
#         axes[row, col].plot(history[key], label=f'{key.upper()}')
#         axes[row, col].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch+1}')
#         axes[row, col].set_title(title)
#         axes[row, col].set_xlabel('Epoch')
#         axes[row, col].set_ylabel(ylabel)
#         axes[row, col].legend()
#         axes[row, col].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('training_metrics_final.png', dpi=150)
#     plt.show()
    
#     print(f"\n🎉 训练完成！")
#     print(f"   最佳模型: {CONFIG['model_save_path']}")
#     print(f"   最佳验证 Score: {best_score:.4f} (Epoch {best_epoch+1})")
#     print(f"   测试集 sMAPE: {test_smape:.2f}% ⭐")
#     print(f"   测试集 MAPE:  {test_mape:.2f}%")
#     print(f"   测试集 Corr:  {test_corr:.4f}")
    
#     # 保存详细结果
#     results = {
#         'test_mae': float(test_mae),
#         'test_smape': float(test_smape),
#         'test_mape': float(test_mape),
#         'test_rmse': float(test_rmse),
#         'test_corr': float(test_corr),
#         'test_ece': float(test_ece),
#         'best_epoch': int(best_epoch),
#         'best_score': float(best_score)
#     }
    
#     with open('test_results.json', 'w') as f:
#         import json
#         json.dump(results, f, indent=2)
    
#     print(f"\n   详细结果已保存至 test_results.json")


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
    "epochs": 150,
    "patience": 30,
    "batch_size": 64,
    
    "embed_dim": 16,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 32,
    
    # EDL参数（关键修复）
    "reg_coeff": 2.0,        # 从1.0增加到2.0
    "warmup_epochs": 3,
    "alpha_init": 1.5,       # 从2.0降到1.5，允许更大方差
    "beta_init": 1.0,        # 新增：从1e-6增加到1.0，显著增大初始方差
    
    "v_init": 0.5,           # 新增：从0.1增加到0.5，降低初始精度
    
    # PICP约束（新增）
    "picp_target": 0.8,      # 目标覆盖率80%
    "picp_weight": 1.0,      # PICP损失权重
    
    # 数据路径
    "data_path": "cts_data.xlsx",
    "feature_path": "image_features_database.csv",
    "model_save_path": f"cts_fixed_{datetime.now().strftime('%m%d_%H%M')}_seed{SEED}.pth",
    
    # 评估权重
    "mape_weight": 0.4,      # 降低精度权重，增加不确定性权重
    "corr_weight": 0.3,
    "ece_weight": 0.3,       # 增加ECE权重
    
    "ema_alpha": 0.9,
    "winsorize_limits": 0.05,
    "use_winsorized_for_selection": False,
    
    "use_mixup": True,
    "mixup_alpha": 0.2,
}

# ==============================================================================
# 2. 模型架构（修复初始化）
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
        # 关键修复：调整初始化参数，使初始方差更大
        v = F.softplus(out[:, 1]) + self.v_init           # 0.5（降低精度）
        alpha = F.softplus(out[:, 2]) + self.alpha_init   # 1.5（允许更大方差）
        # beta = F.softplus(out[:, 3]) + self.beta_init     # 1.0（显著增大方差）
        beta = F.softplus(out[:, 3]) + 5.0  # 强行拉宽初始预测区间
        
        return torch.stack([gamma, v, alpha, beta], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ==============================================================================
# 3. 损失函数（新增PICP约束）
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

def picp_loss(y_true, y_pred, uncertainties, target_coverage=0.8):
    """
    预测区间覆盖损失：强制达到目标覆盖率
    """
    z = norm.ppf((1 + target_coverage) / 2)
    lower = y_pred - z * uncertainties
    upper = y_pred + z * uncertainties
    
    # 是否被覆盖
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    actual_coverage = covered.mean()
    
    # 关键：如果覆盖率低于目标，施加较大惩罚
    # 使用不对称惩罚：低覆盖率惩罚更重
    if actual_coverage < target_coverage:
        coverage_error = (target_coverage - actual_coverage) ** 2 * 2.0  # 低覆盖加重惩罚
    else:
        coverage_error = (actual_coverage - target_coverage) ** 2 * 0.5  # 高覆盖轻惩罚
    
    # 同时惩罚过窄的区间
    interval_width = (upper - lower).mean()
    target_width = torch.abs(y_true - y_pred).mean().detach() * 2  # 期望宽度为2倍MAE
    width_penalty = F.relu(target_width - interval_width) / (target_width + 1e-6)
    
    return coverage_error + 0.2 * width_penalty, actual_coverage.item()

def evidential_loss(pred, target, epoch, config):
    """
    组合损失：NLL + EUB + PICP
    """
    gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    target = target.view(-1)
    
    # 基础EDL损失
    loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
    loss_reg = improved_eub_loss(target, gamma, v, alpha, beta)
    
    # 计算不确定性
    var = beta / (v * (alpha - 1) + 1e-6)
    uncertainties = torch.sqrt(var + 1e-6)
    
    # PICP损失（在原始时间空间计算）
    loss_picp = 0.0
    picp_val = 0.0
    if epoch >= config["warmup_epochs"]:
        y_true_time = torch.expm1(target)
        y_pred_time = torch.expm1(gamma)
        loss_picp, picp_val = picp_loss(
            y_true_time, y_pred_time, uncertainties, 
            config["picp_target"]
        )
        loss_picp = config["picp_weight"] * loss_picp
    
    # 动态权重
    if epoch < config["warmup_epochs"]:
        reg_weight = 0.0
        picp_weight = 0.0
    else:
        progress = min(1.0, (epoch - config["warmup_epochs"]) / 10)
        reg_weight = config["reg_coeff"] * progress
        picp_weight = config["picp_weight"] * progress
    
    total_loss = loss_nll + reg_weight * loss_reg + picp_weight * loss_picp
    
    return total_loss, loss_nll.item(), loss_reg.item(), picp_val, uncertainties.mean().item()

# ==============================================================================
# 4. 数据加载（与之前相同，略）
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
# 5. 评估指标（与之前相同）
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
    """递归转换numpy类型为Python原生类型"""
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

def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8):
    """
    在验证集上学习最优缩放因子，使PICP达到目标覆盖率
    这是解决不确定性低估的 guaranteed 有效方法
    """
    def picp_with_scale(s):
        lower = y_pred - s * unc_raw
        upper = y_pred + s * unc_raw
        return np.mean((y_true >= lower) & (y_true <= upper))
    
    # 边界检查
    picp_at_01 = picp_with_scale(0.1)
    picp_at_10 = picp_with_scale(10.0)
    
    if picp_at_01 >= target_coverage:
        return 0.1  # 已经很宽了
    elif picp_at_10 <= target_coverage:
        return 10.0  # 即使最大缩放也达不到，返回最大值
    else:
        # 二分查找最优缩放因子
        try:
            s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, 0.1, 10.0)
            return s_opt
        except:
            # 备用：网格搜索
            scales = np.linspace(0.1, 10, 200)
            picps = [picp_with_scale(s) for s in scales]
            best_idx = np.argmin(np.abs(np.array(picps) - target_coverage))
            return scales[best_idx]


class MetricsNormalizer:
    def __init__(self, window_size=5, ema_alpha=0.9):
        self.smape_history = []
        self.ece_history = []
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.smape_scale = 100.0
        self.ece_scale = 50.0
        
    def update(self, smape, ece):
        self.smape_history.append(smape)
        self.ece_history.append(ece)
        
        if len(self.smape_history) >= self.window_size:
            recent_smape = self.smape_history[-self.window_size:]
            recent_ece = self.ece_history[-self.window_size:]
            
            new_smape_scale = np.percentile(recent_smape, 90)
            new_ece_scale = np.percentile(recent_ece, 90)
            
            self.smape_scale = self.ema_alpha * self.smape_scale + (1 - self.ema_alpha) * new_smape_scale
            self.ece_scale = self.ema_alpha * self.ece_scale + (1 - self.ema_alpha) * new_ece_scale
            
    def normalize(self, smape, corr, ece):
        smape_norm = smape / (self.smape_scale + 1e-8)
        ece_norm = ece / (self.ece_scale + 1e-8)
        corr_norm = (1.0 - corr) / 2.0
        return smape_norm, corr_norm, ece_norm

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
    
    normalizer = MetricsNormalizer(window_size=5, ema_alpha=CONFIG["ema_alpha"])
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
        for cx, ix, ax, target in tr_loader:
            cx, ix, ax, target = cx.to(device), ix.to(device), ax.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(cx, ix, ax)
            loss, nll, reg, picp, mean_unc = evidential_loss(pred, target, epoch, CONFIG)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(tr_loader)
        
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
        
        # 计算指标
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_smape = calculate_smape(val_targets, val_preds)
        val_mape = calculate_mape(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_corr = spearmanr(val_uncs, val_errs)[0]
        val_corr = 0.0 if np.isnan(val_corr) else val_corr
        
        val_ece = calculate_ece_quantile(val_errs, val_uncs)
        val_picp, val_mpiw = calculate_picp_mpiw(val_targets, val_preds, val_uncs, 0.8)
        
        # 动态归一化与得分
        normalizer.update(val_smape, val_ece)
        smape_n, corr_n, ece_n = normalizer.normalize(val_smape, val_corr, val_ece)
        val_score = (CONFIG["mape_weight"] * smape_n + 
                    CONFIG["corr_weight"] * corr_n + 
                    CONFIG["ece_weight"] * ece_n)
        
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
              f"PICP:{val_picp:.1f}% MPIW:{val_mpiw:.1f}s | "
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
                'normalizer_stats': {'smape_scale': normalizer.smape_scale, 
                                    'ece_scale': normalizer.ece_scale}
            }, CONFIG["model_save_path"])
            print(f" ⭐ BEST")
        else:
            patience_counter += 1
            print(f" (pat:{patience_counter}/{CONFIG['patience']})")
        
        if patience_counter >= CONFIG["patience"]:
            print(f"\n⏹️ 早停触发 (Best Epoch {best_metrics['epoch']+1})")
            break
    
    print("\n🔍 加载最佳模型进行测试与校准...")
    checkpoint = torch.load(CONFIG["model_save_path"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 先在验证集上收集预测和不确定性（用于校准）
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
    
    # 关键：在验证集上学习缩放因子
    print(f"   原始验证集PICP: {calculate_picp_mpiw(val_targets_cal, val_preds_cal, val_uncs_cal, 0.8)[0]:.1f}%")
    scale_factor = post_hoc_calibration(val_targets_cal, val_preds_cal, val_uncs_cal, target_coverage=0.8)
    print(f"   学习到的缩放因子: {scale_factor:.3f}")
    
    # 在测试集上应用缩放
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
    
    # 应用校准后的不确定性
    test_uncs_calibrated = test_uncs_raw * scale_factor
    
    # 计算原始和校准后的指标
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_smape = calculate_smape(test_targets, test_preds)
    test_mape = calculate_mape(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_corr = spearmanr(test_uncs_raw, np.abs(test_targets - test_preds))[0]
    test_corr = 0.0 if np.isnan(test_corr) else test_corr
    
    # 原始PICP/MPIW
    picp_raw, mpiw_raw = calculate_picp_mpiw(test_targets, test_preds, test_uncs_raw, 0.8)
    # 校准后PICP/MPIW
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
    
    # 保存结果（使用修复后的序列化）
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
    
    # 关键：转换所有numpy类型
    results = convert_to_native(results)
    
    result_path = f'results_calibrated_{datetime.now().strftime("%m%d_%H%M")}_seed{SEED}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存至: {result_path}")
    return results

def plot_training_history(history, best_epoch):
    """绘制训练历史（2x4布局）"""
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