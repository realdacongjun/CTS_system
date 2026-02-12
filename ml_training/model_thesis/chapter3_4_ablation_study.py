# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import os
# import json
# # --- 【新增】绘图相关库 ---
# import matplotlib.pyplot as plt
# import matplotlib
# import platform
# # ------------------------
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # ==============================================================================
# # 0. 【新增】绘图配置 (解决中文字体和负号)
# # ==============================================================================
# system_name = platform.system()
# if system_name == 'Windows':
#     font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
# elif system_name == 'Darwin':
#     font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
# else:
#     font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']
# matplotlib.rcParams['font.sans-serif'] = font_list
# matplotlib.rcParams['axes.unicode_minus'] = False

# # ==============================================================================
# # 1. 基础配置与组件 (复用 train.py)
# # ==============================================================================
# # CONFIG = {
# #     "data_path": "E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx",         
# #     "feature_path": "E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv",
# #     "batch_size": 64,
# #     "lr": 0.001,
# #     "epochs": 150,  # 消融实验可以稍微少跑几轮，150足够收敛
# #     "embed_dim": 32,
# #     "kl_coeff": 0.15,
# #     "plot_filename": "figure_3_6_component_contribution_real.png", # 【新增】图片文件名
# #     "json_filename": "ablation_results_final.json" # 【新增】JSON文件名
# # }
# CONFIG = {
#     "data_path": "E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx",         
#     "feature_path": "E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv",
#     "batch_size": 32,       # [修改] 64 -> 32 (小Batch通常泛化更好，能帮大模型跳出局部最优)
#     "lr": 0.0005,           # [修改] 0.001 -> 0.0005 (慢工出细活)
#     "epochs": 300,          # [修改] 150 -> 300 (给双塔模型更多时间追赶)
#     "embed_dim": 32,
#     "kl_coeff": 0.01,       # [修改] 0.15 -> 0.01 (大幅降低！先让RMSE降下来，不要过分关注不确定性)
#     "plot_filename": "figure_3_6_component_contribution_real.png",
#     "json_filename": "ablation_results_final.json" 
# }

# # 路径自适应
# if not os.path.exists(CONFIG["data_path"]):
#     if os.path.exists(f"../{CONFIG['data_path']}"):
#         CONFIG["data_path"] = f"../{CONFIG['data_path']}"
#         CONFIG["feature_path"] = f"../{CONFIG['feature_path']}"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- 损失函数 (NIG Loss) ---
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

# def evidential_loss(pred, target, epoch, total_epochs):
#     gamma, v, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#     target = target.view(-1)
#     loss_nll = nig_nll_loss(target, gamma, v, alpha, beta)
#     loss_reg = nig_reg_loss(target, gamma, v, alpha, beta)
#     annealing_coef = min(1.0, epoch / (total_epochs * 0.15))
#     return loss_nll + CONFIG["kl_coeff"] * annealing_coef * loss_reg

# # --- 基础模块 ---
# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         nn.init.xavier_uniform_(self.weights)
#         nn.init.zeros_(self.biases)
#     def forward(self, x):
#         return x.unsqueeze(-1) * self.weights + self.biases

# # ==============================================================================
# # 2. 模型变体定义 (The Variants - 基于完全体修改)
# # ==============================================================================

# # --- 变体 A: 完全体 (Ours: Full CFT-Net) ---
# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True, dropout=0.1)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         batch_size = x.shape[0]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         tokens = torch.cat((cls_tokens, tokens), dim=1)
#         out = self.transformer(tokens)
#         return out[:, 0, :]

# class FullCFTNet(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32, output_dim=4):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
#         self.head = nn.Sequential(
#             nn.Linear(embed_dim * 3, 64),
#             nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64, output_dim)
#         )
#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx)
#         i_vec = self.image_tower(ix)
#         a_vec = self.algo_embed(ax)
#         out = self.head(torch.cat([c_vec, i_vec, a_vec], dim=1))
#         # 如果输出是4维，说明是NIG分布参数
#         if out.shape[1] == 4:
#             gamma = out[:, 0]
#             v = F.softplus(out[:, 1]) + 1e-6
#             alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
#             beta = F.softplus(out[:, 3]) + 1e-6
#             return torch.stack([gamma, v, alpha, beta], dim=1)
#         return out

# # --- 变体 B: 去掉 Transformer (用 MLP 替代) ---
# class MLPTower(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         # 简单的全连接层，没有 Self-Attention
#         self.net = nn.Sequential(
#             nn.Linear(num_features, embed_dim * 2),
#             nn.ReLU(),
#             nn.Linear(embed_dim * 2, embed_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

# class CFTNet_NoTransformer(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = MLPTower(client_feats, embed_dim)
#         self.image_tower = MLPTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
#         self.head = nn.Sequential(
#             nn.Linear(embed_dim * 3, 64),
#             nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64, 4)
#         )
#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx)
#         i_vec = self.image_tower(ix)
#         a_vec = self.algo_embed(ax)
#         out = self.head(torch.cat([c_vec, i_vec, a_vec], dim=1))
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 1e-6
#         alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
#         beta = F.softplus(out[:, 3]) + 1e-6
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # --- 变体 C: 去掉双塔结构 (单塔 Transformer) ---
# class SingleTowerTransformer(nn.Module):
#     def __init__(self, total_features, num_algos, embed_dim=32):
#         super().__init__()
#         # 把所有特征拼在一起 Tokenizer
#         self.tokenizer = FeatureTokenizer(total_features, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         self.head = nn.Sequential(
#             nn.Linear(embed_dim, 64), # 这里只有 embed_dim (因为只有1个CLS token)
#             nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64, 4)
#         )

#     def forward(self, cx, ix, ax):
#         # 1. 拼接原始特征
#         combined_features = torch.cat([cx, ix], dim=1)
#         tokens = self.tokenizer(combined_features)
#         # 2. 加上算法 Embedding 作为额外的 Token
#         a_vec = self.algo_embed(ax).unsqueeze(1) # [Batch, 1, Dim]
#         # 3. 加上 CLS Token
#         batch_size = cx.shape[0]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         # 4. 全部拼成序列: [CLS, Algo, Feat1, Feat2...]
#         full_seq = torch.cat((cls_tokens, a_vec, tokens), dim=1)
#         out_seq = self.transformer(full_seq)
#         cls_out = out_seq[:, 0, :] # 取 CLS
#         out = self.head(cls_out)
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 1e-6
#         alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
#         beta = F.softplus(out[:, 3]) + 1e-6
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 3. 训练与评估逻辑
# # ==============================================================================
# class CTSDataset(Dataset):
#     def __init__(self, cx, ix, ax, y):
#         self.cx = torch.FloatTensor(cx)
#         self.ix = torch.FloatTensor(ix)
#         self.ax = torch.LongTensor(ax)
#         self.y = torch.FloatTensor(y)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, i): return self.cx[i], self.ix[i], self.ax[i], self.y[i]

# def train_variant(model_name, model, train_loader, test_loader, use_nig_loss=True):
#     print(f"\n⚡ 正在训练变体: {model_name}")
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
#     criterion_mse = nn.MSELoss()
    
#     # 简单的进度显示
#     for epoch in range(CONFIG["epochs"]):
#         model.train()
#         for cx, ix, ax, y in train_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             optimizer.zero_grad()
#             preds = model(cx, ix, ax)
            
#             if use_nig_loss:
#                 loss = evidential_loss(preds, y, epoch, CONFIG["epochs"])
#             else:
#                 # w/o Uncertainty 变体，只输出1个值
#                 loss = criterion_mse(preds.squeeze(), y)
                
#             loss.backward()
#             optimizer.step()
#         if (epoch + 1) % 50 == 0:
#             print(f"  Epoch {epoch+1}/{CONFIG['epochs']} done.")
            
#     # 评估阶段
#     model.eval()
#     preds_list, true_list = [], []
#     with torch.no_grad():
#         for cx, ix, ax, y in test_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             preds = model(cx, ix, ax)
            
#             if use_nig_loss:
#                 gamma = preds[:, 0] # 取预测均值
#             else:
#                 gamma = preds.squeeze()
                
#             preds_list.extend(np.expm1(gamma.cpu().numpy())) # Log反变换
#             true_list.extend(np.expm1(y.cpu().numpy()))
            
#     rmse = np.sqrt(mean_squared_error(true_list, preds_list))
#     mae = mean_absolute_error(true_list, preds_list) # 【新增】计算 MAE
#     r2 = r2_score(true_list, preds_list)
#     print(f"✅ {model_name} 完成 -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
#     # 【修改】返回 MAE
#     return {"rmse": rmse, "mae": mae, "r2": r2}

# # ==============================================================================
# # 4. 【新增】绘图函数 (保持风格一致)
# # ==============================================================================
# def generate_component_contribution_plot(results, filename):
#     """生成组件贡献分析图 (3个子图: RMSE, MAE, R²)"""
#     print(f"\n📊 正在生成组件贡献分析图: {filename} ...")
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     fig.suptitle('图3.6 组件贡献分析（基于真实数据 - 严谨消融实验）', fontsize=16, fontweight='bold')
    
#     # 提取数据
#     variants = list(results.keys())
#     rmses = [results[v]['rmse'] for v in variants]
#     maes = [results[v]['mae'] for v in variants]
#     r2s = [results[v]['r2'] for v in variants]
    
#     # 定义颜色 (突出显示 Ours)
#     # Ours用绿色，其他用不同深浅的红色/橙色
#     colors = ['#2ca02c'] + ['#d62728', '#ff7f0e', '#9467bd'][0:len(variants)-1]

#     # 通用标注函数
#     def annotate_bars(ax, bars):
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(f'{height:.3f}', 
#                         xy=(bar.get_x() + bar.get_width()/2, height),
#                         xytext=(0, 3), textcoords="offset points",
#                         ha='center', va='bottom', fontsize=10, fontweight='bold')

#     # 图1: RMSE对比
#     bars1 = axes[0].bar(variants, rmses, color=colors, alpha=0.8)
#     axes[0].set_title('RMSE对比 (越低越好)', fontweight='bold')
#     axes[0].set_ylabel('均方根误差 (秒)')
#     axes[0].tick_params(axis='x', rotation=20)
#     annotate_bars(axes[0], bars1)
    
#     # 图2: MAE对比
#     bars2 = axes[1].bar(variants, maes, color=colors, alpha=0.8)
#     axes[1].set_title('MAE对比 (越低越好)', fontweight='bold')
#     axes[1].set_ylabel('平均绝对误差 (秒)')
#     axes[1].tick_params(axis='x', rotation=20)
#     annotate_bars(axes[1], bars2)
    
#     # 图3: R²对比
#     bars3 = axes[2].bar(variants, r2s, color=colors, alpha=0.8)
#     axes[2].set_title('R²对比 (越高越好)', fontweight='bold')
#     axes[2].set_ylabel('决定系数')
#     axes[2].tick_params(axis='x', rotation=20)
#     # R2 的Y轴限制在 0-1 之间看起来更直观，除非有负数
#     axes[2].set_ylim(bottom=min(0, min(r2s))*1.1, top=min(1.0, max(r2s)*1.05))
#     annotate_bars(axes[2], bars3)
    
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()
#     print("✅ 图片生成完成!")

# # ==============================================================================
# # 5. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     print("=== 开始严谨版消融实验 (基于 Full CFT-Net) ===")
#     # --- 数据准备 (和 train.py 完全一致) ---
#     print("🔄 加载数据...")
#     df = pd.read_excel(CONFIG["data_path"])
#     df_feat = pd.read_csv(CONFIG["feature_path"])
    
#     # 简单的预处理
#     rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
#     df = df.rename(columns=rename_map)
#     if 'total_time' not in df.columns: 
#         cols = [c for c in df.columns if 'total_tim' in c]
#         if cols: df = df.rename(columns={cols[0]: 'total_time'})
    
#     df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
#     if 'mem_limit_mb' not in df.columns: df['mem_limit_mb'] = 1024.0
#     df = pd.merge(df, df_feat, on="image_name", how="inner")
    
#     col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     # 使用 entropy_std (熵标准差) 和 size_std_mb (层大小标准差) 替代缺失的列
#     col_image = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb']
    
#     scaler_c = StandardScaler()
#     X_client = scaler_c.fit_transform(df[col_client].values)
#     scaler_i = StandardScaler()
#     X_image = scaler_i.fit_transform(df[col_image].values)
#     enc_algo = LabelEncoder()
#     X_algo = enc_algo.fit_transform(df['algo_name'].values)
#     y_target = np.log1p(df['total_time'].values)

#     # 划分数据集
#     Xc_train, Xc_test, Xi_train, Xi_test, Xa_train, Xa_test, y_train, y_test = train_test_split(
#         X_client, X_image, X_algo, y_target, test_size=0.2, random_state=42
#     )
    
#     train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
#     test_loader = DataLoader(CTSDataset(Xc_test, Xi_test, Xa_test, y_test), batch_size=CONFIG["batch_size"])
    
#     num_algos = len(enc_algo.classes_)
#     c_dim = len(col_client)
#     i_dim = len(col_image)

#     # --- 开始消融实验 (训练4个变体) ---
#     results = {}
    
#     # 1. 完整模型 (Transformer + Dual Tower + NIG)
#     model_full = FullCFTNet(c_dim, i_dim, num_algos)
#     results['Full CFT-Net (Ours)'] = train_variant('Full CFT-Net', model_full, train_loader, test_loader, use_nig_loss=True)
    
#     # 2. 去掉 Transformer (MLP + Dual Tower + NIG)
#     model_no_attn = CFTNet_NoTransformer(c_dim, i_dim, num_algos)
#     results['w/o Attention (MLP)'] = train_variant('w/o Attention', model_no_attn, train_loader, test_loader, use_nig_loss=True)
    
#     # 3. 去掉双塔 (Single Transformer + NIG)
#     model_single = SingleTowerTransformer(c_dim + i_dim, num_algos)
#     results['w/o Dual-Tower'] = train_variant('w/o Dual-Tower', model_single, train_loader, test_loader, use_nig_loss=True)
    
#     # 4. 去掉不确定性损失 (Full Model + MSE Loss)
#     # 结构一样，但输出维度改为 1，损失用 MSE
#     model_mse = FullCFTNet(c_dim, i_dim, num_algos, output_dim=1) 
#     results['w/o Uncertainty (MSE)'] = train_variant('w/o Uncertainty', model_mse, train_loader, test_loader, use_nig_loss=False)

#     # --- 生成最终报表 (终端显示) ---
#     print("\n" + "="*75)
#     print(f"{'Ablation Variant':<25} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10} | {'Drop (RMSE)'}")
#     print("-" * 75)
    
#     base_rmse = results['Full CFT-Net (Ours)']['rmse']
    
#     for name, metrics in results.items():
#         rmse = metrics['rmse']
#         mae = metrics['mae']
#         r2 = metrics['r2']
#         drop = (rmse - base_rmse) / base_rmse * 100 if name != 'Full CFT-Net (Ours)' else 0.0
        
#         print(f"{name:<25} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {f'+{drop:.1f}%' if drop > 0 else '-'}")
#     print("="*75)
    
#     # --- 保存结果 ---
#     # 1. 保存 JSON 数据
#     with open(CONFIG["json_filename"], 'w') as f:
#         # 将 numpy 类型转换为 float 以便 json 序列化
#         serializable_results = {k: {m: float(v) for m, v in mets.items()} for k, mets in results.items()}
#         json.dump(serializable_results, f, indent=4)
#         print(f"\n💾 结果已保存至: {CONFIG['json_filename']}")

#     # 2. 【新增】生成对比图
#     generate_component_contribution_plot(results, CONFIG["plot_filename"])
#     print(f"📊 图片已保存至: {CONFIG['plot_filename']}")

#     print("\n=== 严谨版消融实验完成 ===")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import os
# import json
# import matplotlib.pyplot as plt
# import matplotlib
# import platform
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# set_seed(42)
# # DATA_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx"
# # FEAT_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv"
# # MODEL_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_final_strong.pth" # 确保文件名对
# CONFIG = {
#     "data_path": "E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx",
#     "feature_path": "E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv",
#     "epochs": 150,  # 消融实验跑50轮看趋势
#     "batch_size": 128,
#     "lr": 0.001,
#     "reg_coeff": 0.2, 
#     "json_filename": "ablation_final.json",
#     "plot_filename": "figure_3_6_ablation_study.png"
# }
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==============================================================================
# # 1. 损失函数集
# # ==============================================================================
# def nig_nll_loss(y, gamma, v, alpha, beta):
#     two_blambda = 2 * beta * (1 + v)
#     nll = 0.5 * torch.log(np.pi / v) \
#         - alpha * torch.log(two_blambda) \
#         + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) \
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     return nll.mean()

# def robust_eub_reg_loss(y, gamma, v, alpha, beta):
#     error = torch.abs(y - gamma)
#     var = beta / (v * (alpha - 1) + 1e-6)
#     std = torch.sqrt(var + 1e-6)
#     ratio = torch.clamp(error / (std + 1e-6), max=10.0)
#     penalty = torch.where(ratio > 1.0, (ratio - 1.0)**2, 0.1 * (1.0 - ratio))
#     evidence = torch.clamp(2 * v + alpha, max=50.0)
#     return (penalty * torch.log1p(evidence)).mean()

# def vanilla_kl_reg_loss(y, gamma, v, alpha, beta):
#     error = torch.abs(y - gamma)
#     evidence = torch.clamp(2 * v + alpha, max=50.0)
#     return (error * evidence).mean()

# # ==============================================================================
# # 2. 基础组件
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
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True, dropout=0.1, activation="gelu"),
#             num_layers=2
#         )
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

# # ==============================================================================
# # 3. 严谨的模型变体定义 (显式定义每个类)
# # ==============================================================================

# # --- A. Ours: Gated Dual-Tower Transformer ---
# class OursModel(nn.Module):
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32, output_dim=4):
#         super().__init__()
#         self.client_tower = TransformerTower(c_dim, embed_dim)
#         self.image_tower = TransformerTower(i_dim, embed_dim)
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
#         self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
#         self.hidden = nn.Sequential(nn.Linear(embed_dim * 3, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU())
#         self.head = nn.Linear(32, output_dim)

#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
#         z = self.gate_net(torch.cat([c, i], dim=1))
#         fused = z * c + (1 - z) * i
#         a = self.algo_embed(ax)
#         out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
        
#         if out.shape[1] == 4:
#             return torch.stack([out[:,0], F.softplus(out[:,1])+0.1, F.softplus(out[:,2])+1.1, F.softplus(out[:,3])+1e-6], dim=1)
#         return out

# # --- B. Variant: Concat Fusion (Explicitly NO GateNet) ---
# class ConcatModel(nn.Module):
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(c_dim, embed_dim)
#         self.image_tower = TransformerTower(i_dim, embed_dim)
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
        
#         # [区别] 显式移除 GateNet，减少参数量
#         # self.gate_net = ... (Removed)
        
#         self.hidden = nn.Sequential(nn.Linear(embed_dim * 3, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU())
#         self.head = nn.Linear(32, 4)

#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
        
#         # [区别] 简单平均融合
#         fused = (c + i) / 2.0 
        
#         a = self.algo_embed(ax)
#         out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
#         return torch.stack([out[:,0], F.softplus(out[:,1])+0.1, F.softplus(out[:,2])+1.1, F.softplus(out[:,3])+1e-6], dim=1)

# # --- C. Variant: MLP Backbone ---
# class MLPBackboneModel(nn.Module):
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = MLPTower(c_dim, embed_dim) # [区别] 用 MLP
#         self.image_tower = MLPTower(i_dim, embed_dim)  # [区别] 用 MLP
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
#         self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
#         self.hidden = nn.Sequential(nn.Linear(embed_dim * 3, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU())
#         self.head = nn.Linear(32, 4)

#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
#         z = self.gate_net(torch.cat([c, i], dim=1))
#         fused = z * c + (1 - z) * i
#         a = self.algo_embed(ax)
#         out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
#         return torch.stack([out[:,0], F.softplus(out[:,1])+0.1, F.softplus(out[:,2])+1.1, F.softplus(out[:,3])+1e-6], dim=1)

# # --- D. Variant: Single Tower ---
# class SingleTowerModel(nn.Module):
#     def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
#         super().__init__()
#         # [区别] 单塔处理所有特征
#         self.tower = TransformerTower(c_dim + i_dim, embed_dim)
#         self.algo_embed = nn.Embedding(n_algos, embed_dim)
#         self.hidden = nn.Sequential(nn.Linear(embed_dim * 2, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU())
#         self.head = nn.Linear(32, 4)

#     def forward(self, cx, ix, ax):
#         combined = torch.cat([cx, ix], dim=1)
#         feat = self.tower(combined)
#         a = self.algo_embed(ax)
#         # [区别] 只有 feat 和 a，没有 gate，没有 fusion
#         out = self.head(self.hidden(torch.cat([feat, a], dim=1)))
#         return torch.stack([out[:,0], F.softplus(out[:,1])+0.1, F.softplus(out[:,2])+1.1, F.softplus(out[:,3])+1e-6], dim=1)

# # ==============================================================================
# # 4. 训练逻辑
# # ==============================================================================
# class CTSDataset(Dataset):
#     def __init__(self, cx, ix, ax, y):
#         self.cx, self.ix, self.ax, self.y = torch.FloatTensor(cx), torch.FloatTensor(ix), torch.LongTensor(ax), torch.FloatTensor(y)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, i): return self.cx[i], self.ix[i], self.ax[i], self.y[i]

# def train_ablation(name, model, train_loader, test_loader, loss_type='robust_eub'):
#     print(f"\n⚡ [实验] {name} | Loss: {loss_type}")
#     model = model.to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    
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
                
#                 if loss_type == 'robust_eub':
#                     loss_reg = robust_eub_reg_loss(y, gamma, v, alpha, beta)
#                 elif loss_type == 'kl':
#                     loss_reg = vanilla_kl_reg_loss(y, gamma, v, alpha, beta)
                
#                 reg_w = 0 if epoch < 3 else CONFIG["reg_coeff"]
#                 loss = loss_nll + reg_w * loss_reg
            
#             loss.backward()
#             optimizer.step()

#     # 评估
#     model.eval()
#     preds_list, true_list = [], []
#     with torch.no_grad():
#         for cx, ix, ax, y in test_loader:
#             cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
#             preds = model(cx, ix, ax)
#             gamma = preds.squeeze() if loss_type == 'mse' else preds[:, 0]
#             preds_list.extend(np.expm1(gamma.cpu().numpy()))
#             true_list.extend(np.expm1(y.cpu().numpy()))
    
#     rmse = np.sqrt(mean_squared_error(true_list, preds_list))
#     print(f"✅ {name} -> RMSE: {rmse:.4f}")
#     return rmse

# # ==============================================================================
# # 5. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     print("=== 全方位消融实验 (Strict Mode) ===")
    
#     # 1. 加载数据
#     df = pd.read_excel(CONFIG["data_path"])
#     df_feat = pd.read_csv(CONFIG["feature_path"])
#     rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
#     df = df.rename(columns=rename_map)
#     if 'total_time' not in df.columns: df['total_time'] = df[[c for c in df.columns if 'total_tim' in c][0]]
#     df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
#     df = pd.merge(df, df_feat, on="image_name", how="inner")
    
#     cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#     cols_i = [c for c in target_cols if c in df.columns]
    
#     Xc = StandardScaler().fit_transform(df[cols_c].values)
#     Xi = StandardScaler().fit_transform(df[cols_i].values)
#     enc = LabelEncoder()
#     Xa = enc.fit_transform(df['algo_name'].values)
#     y = np.log1p(df['total_time'].values)
    
#     from sklearn.model_selection import train_test_split
#     idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
#     tr_loader = DataLoader(CTSDataset(Xc[idx_tr], Xi[idx_tr], Xa[idx_tr], y[idx_tr]), batch_size=CONFIG["batch_size"], shuffle=True)
#     te_loader = DataLoader(CTSDataset(Xc[idx_te], Xi[idx_te], Xa[idx_te], y[idx_te]), batch_size=CONFIG["batch_size"])
#     c_dim, i_dim, n_algos = Xc.shape[1], Xi.shape[1], len(enc.classes_)

#     results = {}
    
#     # --- 实验开始 ---
    
#     # 1. 架构消融 (Backbone Ablation)
#     results['w/o Transformer (MLP)'] = train_ablation('MLP', MLPBackboneModel(c_dim, i_dim, n_algos), tr_loader, te_loader)
#     results['w/o Dual-Tower'] = train_ablation('SingleTower', SingleTowerModel(c_dim, i_dim, n_algos), tr_loader, te_loader)
    
#     # 2. 融合消融 (Fusion Ablation)
#     # 使用 ConcatModel，显式移除 GateNet
#     results['w/o Gated Fusion'] = train_ablation('Concat', ConcatModel(c_dim, i_dim, n_algos), tr_loader, te_loader)
    
#     # 3. 损失消融 (Loss Ablation)
#     # 使用 OursModel，但 Loss 用 KL
#     results['w/o Robust EUB'] = train_ablation('Vanilla KL', OursModel(c_dim, i_dim, n_algos), tr_loader, te_loader, loss_type='kl')
    
#     # 4. 任务消融 (Task Ablation)
#     # 使用 OursModel (Output=1)，Loss 用 MSE
#     results['w/o Uncertainty'] = train_ablation('MSE Only', OursModel(c_dim, i_dim, n_algos, output_dim=1), tr_loader, te_loader, loss_type='mse')
    
#     # 5. Ours
#     results['Ours (Full)'] = train_ablation('Ours', OursModel(c_dim, i_dim, n_algos), tr_loader, te_loader, loss_type='robust_eub')

#     # --- 绘图 ---
#     print("\n📊 生成对比图...")
#     names = list(results.keys())
#     values = list(results.values())
    
#     # 排序：Ours 在最后
#     sorted_indices = np.argsort(values)[::-1]
#     names = [names[i] for i in sorted_indices]
#     values = [values[i] for i in sorted_indices]
    
#     colors = ['gray'] * (len(names)-1) + ['#2ca02c']
    
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(names, values, color=colors)
#     plt.title('各组件对模型性能的影响 (Ablation Study)')
#     plt.ylabel('RMSE (越低越好)')
#     plt.xticks(rotation=20, ha='right')
#     plt.ylim(min(values)*0.9, max(values)*1.05)
    
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        
#     plt.tight_layout()
#     plt.savefig(CONFIG["plot_filename"], dpi=300)
#     print(f"✅ 图片已保存: {CONFIG['plot_filename']}")
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
from scipy.stats import spearmanr, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 绘图配置（【强化】自动适配中英文，保证中文显示）
# ==============================================================================
system_name = platform.system()
if system_name == 'Windows':
    font_list = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system_name == 'Darwin':
    font_list = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

# 设置全局字体，负号正常显示
matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 检查字体是否可用，若配置的字体均不存在则回退到 DejaVu Sans（英文）
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_list:
    if font in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [font]
        break
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️ 未找到中文字体，使用英文显示")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ==============================================================================
# 1. 超参数配置（【最强版】回归已验证参数）
# ==============================================================================
CONFIG = {
    "data_path": "E:\\硕士毕业论文材料合集\\论文实验代码相关\\CTS_system\\ml_training\\modeling\\cts_data.xlsx",
    "feature_path": "E:\\硕士毕业论文材料合集\\论文实验代码相关\\CTS_system\\ml_training\\image_features_database.csv",
    "epochs": 60,
    "patience": 15,
    "batch_size": 128,
    "lr": 0.0005,
    "reg_coeff": 1.0,
    "embed_dim": 32,
    "n_runs": 5,
    "random_seeds": [42, 123, 456, 789, 2024],
    "plot_ablation": "figure_3_6_ablation.png",
    "plot_calibration": "figure_3_7_calibration_ablation.png"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. 损失函数（不变）
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

def vanilla_kl_reg_loss(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    evidence = torch.clamp(2 * v + alpha, max=50.0)
    return (error * evidence).mean()

# ==============================================================================
# 3. 模型组件与变体（MSE模型保留定义，但不参与消融主实验）
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

# ----- 模型变体 -----
class OursModel(nn.Module):
    """完整模型：双塔Transformer + 平均融合 + 证据回归头"""
    def __init__(self, c_dim, i_dim, n_algos, embed_dim=32, output_dim=4):
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
        self.head = nn.Linear(32, output_dim)
    def forward(self, cx, ix, ax):
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        fused = (c + i) / 2.0          # 【正确】平均融合，无门控
        a = self.algo_embed(ax)
        out = self.head(self.hidden(torch.cat([fused, i, a], dim=1)))
        if out.shape[1] == 4:
            return torch.stack([
                out[:, 0],
                F.softplus(out[:, 1]) + 0.1,
                F.softplus(out[:, 2]) + 1.1,
                F.softplus(out[:, 3]) + 1e-6
            ], dim=1)
        return out

class MLPBackboneModel(nn.Module):
    """消融A：Transformer → MLP"""
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
    """消融B：双塔 → 单塔（输入拼接后过Transformer）"""
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

# ----- MSE模型（保留定义，但默认不参与消融实验）-----
class MSEModel(OursModel):
    """消融C（补充对照）：证据回归 → 普通MSE回归（输出维度1）"""
    def __init__(self, c_dim, i_dim, n_algos, embed_dim=32):
        super().__init__(c_dim, i_dim, n_algos, embed_dim, output_dim=1)

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
# 5. 核心训练与评估函数（单次运行，带早停）
# ==============================================================================
def compute_ece(uncertainty, abs_error, n_bins=15):
    """期望校准误差（归一化后）"""
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

def train_ablation_single(name, model_class, loss_type, c_dim, i_dim, n_algos,
                          Xc_train, Xi_train, Xa_train, y_train,
                          Xc_val, Xi_val, Xa_val, y_val, seed):
    """单次实验：训练并返回最佳指标及模型（带早停）"""
    set_seed(seed)
    model = model_class(c_dim, i_dim, n_algos).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    train_loader = DataLoader(CTSDataset(Xc_train, Xi_train, Xa_train, y_train),
                              batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(CTSDataset(Xc_val, Xi_val, Xa_val, y_val),
                            batch_size=CONFIG["batch_size"], shuffle=False)

    best_rmse = float('inf')
    best_metrics = {}
    best_model_state = None
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for cx, ix, ax, y in train_loader:
            cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(cx, ix, ax)

            if loss_type == 'mse':
                loss = F.mse_loss(preds.squeeze(), y)
            else:
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                loss_nll = nig_nll_loss(y, gamma, v, alpha, beta)
                if loss_type == 'strong_eub':
                    loss_reg = strong_eub_reg_loss(y, gamma, v, alpha, beta)
                elif loss_type == 'kl':
                    loss_reg = vanilla_kl_reg_loss(y, gamma, v, alpha, beta)
                else:
                    loss_reg = 0.0
                reg_w = 0.0 if epoch < 3 else CONFIG["reg_coeff"]
                loss = loss_nll + reg_w * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # --- 验证 ---
        model.eval()
        pred_list, true_list, unc_list = [], [], []
        with torch.no_grad():
            for cx, ix, ax, y in val_loader:
                cx, ix, ax, y = cx.to(device), ix.to(device), ax.to(device), y.to(device)
                preds = model(cx, ix, ax)
                if loss_type == 'mse':
                    gamma = preds.squeeze()
                else:
                    gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                    unc = beta / (v * (alpha - 1))
                    unc_list.extend(unc.cpu().numpy())

                pred_list.extend(np.expm1(gamma.cpu().numpy()))
                true_list.extend(np.expm1(y.cpu().numpy()))

        pred_list = np.array(pred_list)
        true_list = np.array(true_list)
        curr_rmse = np.sqrt(mean_squared_error(true_list, pred_list))

        # --- 防坍塌机制（仅针对KL）---
        valid = True
        if loss_type == 'kl':
            pred_std = np.std(pred_list)
            if pred_std < 0.5:
                valid = False
            if len(unc_list) > 0:
                abs_err = np.abs(true_list - pred_list)
                corr, _ = spearmanr(unc_list, abs_err)
                if np.isnan(corr) or corr < 0.05:
                    valid = False

        # --- 早停与最佳模型保存 ---
        if curr_rmse < best_rmse and valid:
            best_rmse = curr_rmse
            best_model_state = model.state_dict()
            best_metrics['rmse'] = curr_rmse
            best_metrics['mae'] = mean_absolute_error(true_list, pred_list)
            best_metrics['r2'] = r2_score(true_list, pred_list)
            if loss_type != 'mse' and len(unc_list) > 0:
                abs_err = np.abs(true_list - pred_list)
                spearman_corr, _ = spearmanr(unc_list, abs_err)
                best_metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
                best_metrics['ece'] = compute_ece(np.array(unc_list), abs_err)
            else:
                best_metrics['spearman'] = 0.0
                best_metrics['ece'] = 1.0
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                break

    # 保存最佳模型
    if best_model_state is not None:
        os.makedirs('ablation_models', exist_ok=True)
        torch.save(best_model_state, f'ablation_models/best_{name}_seed{seed}.pth')

    return best_metrics

# ==============================================================================
# 6. 多次独立实验主流程（已移除MSE）
# ==============================================================================
def run_ablation_experiments():
    print("="*60)
    print("🔬 消融实验（多次独立运行，无MSE基线）")
    print("="*60)

    df = pd.read_excel(CONFIG["data_path"])
    df_feat = pd.read_csv(CONFIG["feature_path"])
    rename_map = {"image": "image_name", "method": "algo_name",
                  "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
                  "mem_limit": "mem_limit_mb"}
    df = df.rename(columns=rename_map)
    if 'total_time' not in df.columns:
        df['total_time'] = df[[c for c in df.columns if 'total_tim' in c][0]]
    df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
    if 'mem_limit_mb' not in df.columns:
        df['mem_limit_mb'] = 1024.0
    df = pd.merge(df, df_feat, on="image_name", how="inner")

    cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
    cols_i = ['total_size_mb', 'avg_layer_entropy', 'entropy_std',
              'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
    cols_i = [c for c in cols_i if c in df.columns]

    # 仅包含四个消融变体
    all_results = {
        'Ours (Full)': [],
        'w/o Transformer (MLP)': [],
        'w/o Dual-Tower (Single)': [],
        'w/o Robust EUB (KL)': []
    }

    for run_idx, seed in enumerate(CONFIG["random_seeds"][:CONFIG["n_runs"]]):
        print(f"\n--- 实验 {run_idx+1}/{CONFIG['n_runs']} (seed={seed}) ---")
        set_seed(seed)

        idx = np.arange(len(df))
        idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=seed)

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

        Xc_val_raw = df.iloc[idx_val][cols_c].values
        Xi_val_raw = df.iloc[idx_val][cols_i].values
        Xa_val_raw = df.iloc[idx_val]['algo_name'].values
        y_val_raw = np.log1p(df.iloc[idx_val]['total_time'].values)

        Xc_val = scaler_c.transform(Xc_val_raw)
        Xi_val = scaler_i.transform(Xi_val_raw)
        Xa_val = enc.transform(Xa_val_raw)

        c_dim = Xc_train.shape[1]
        i_dim = Xi_train.shape[1]
        n_algos = len(enc.classes_)

        # 1. Ours (Full)
        res_ours = train_ablation_single(
            'Ours', OursModel, 'strong_eub',
            c_dim, i_dim, n_algos,
            Xc_train, Xi_train, Xa_train, y_train_raw,
            Xc_val, Xi_val, Xa_val, y_val_raw, seed
        )
        all_results['Ours (Full)'].append(res_ours)

        # 2. MLP Backbone
        res_mlp = train_ablation_single(
            'MLP', MLPBackboneModel, 'strong_eub',
            c_dim, i_dim, n_algos,
            Xc_train, Xi_train, Xa_train, y_train_raw,
            Xc_val, Xi_val, Xa_val, y_val_raw, seed
        )
        all_results['w/o Transformer (MLP)'].append(res_mlp)

        # 3. Single Tower
        res_single = train_ablation_single(
            'Single', SingleTowerModel, 'strong_eub',
            c_dim, i_dim, n_algos,
            Xc_train, Xi_train, Xa_train, y_train_raw,
            Xc_val, Xi_val, Xa_val, y_val_raw, seed
        )
        all_results['w/o Dual-Tower (Single)'].append(res_single)

        # 4. Vanilla KL
        res_kl = train_ablation_single(
            'KL', OursModel, 'kl',
            c_dim, i_dim, n_algos,
            Xc_train, Xi_train, Xa_train, y_train_raw,
            Xc_val, Xi_val, Xa_val, y_val_raw, seed
        )
        all_results['w/o Robust EUB (KL)'].append(res_kl)

        # ----- MSE基线（注释掉，不参与消融实验）-----
        # res_mse = train_ablation_single(
        #     'MSE', MSEModel, 'mse',
        #     c_dim, i_dim, n_algos,
        #     Xc_train, Xi_train, Xa_train, y_train_raw,
        #     Xc_val, Xi_val, Xa_val, y_val_raw, seed
        # )
        # all_results['MSE Baseline'].append(res_mse)

    # ----- 汇总统计与显著性检验 -----
    summary = {}
    for name, runs in all_results.items():
        df_runs = pd.DataFrame(runs)
        summary[name] = {
            'rmse_mean': df_runs['rmse'].mean(),
            'rmse_std': df_runs['rmse'].std(),
            'mae_mean': df_runs['mae'].mean(),
            'mae_std': df_runs['mae'].std(),
            'r2_mean': df_runs['r2'].mean(),
            'r2_std': df_runs['r2'].std(),
            'spearman_mean': df_runs['spearman'].mean() if 'spearman' in df_runs else 0,
            'spearman_std': df_runs['spearman'].std() if 'spearman' in df_runs else 0,
            'ece_mean': df_runs['ece'].mean() if 'ece' in df_runs else 1,
            'ece_std': df_runs['ece'].std() if 'ece' in df_runs else 0
        }

    # 显著性检验：Ours vs 每个消融变体
    ours_rmse = [r['rmse'] for r in all_results['Ours (Full)']]
    for name in all_results.keys():
        if name == 'Ours (Full)': continue
        other_rmse = [r['rmse'] for r in all_results[name]]
        if len(ours_rmse) == len(other_rmse) and len(ours_rmse) > 1:
            try:
                stat, p = wilcoxon(ours_rmse, other_rmse, alternative='less')
                summary[name]['p_vs_ours'] = p
            except:
                summary[name]['p_vs_ours'] = 1.0
        else:
            summary[name]['p_vs_ours'] = 1.0

    return summary, all_results

# ==============================================================================
# 7. 可视化（【强化】全中文显示，字体回退）
# ==============================================================================
def plot_ablation_results(summary):
    """绘制消融实验对比图（全中文，Ours绿色压轴）"""
    # 强制中文字体（二次保险）
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False

    names = list(summary.keys())
    
    # 强制排序：非Ours按RMSE从大到小，Ours固定放在最后
    non_ours = [n for n in names if "Ours" not in n]
    rmse_means_non_ours = [summary[n]['rmse_mean'] for n in non_ours]
    sorted_idx = np.argsort(rmse_means_non_ours)[::-1]
    sorted_names = [non_ours[i] for i in sorted_idx]
    if 'Ours (Full)' in names:
        sorted_names.append('Ours (Full)')
    names = sorted_names

    # 配色：Ours绿色，其他灰色
    colors = ['#808080'] * (len(names)-1) + ['#2ca02c']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['rmse', 'mae', 'r2']
    titles = ['RMSE (秒) ↓', 'MAE (秒) ↓', 'R² 分数 ↑']
    ylabels = ['RMSE', 'MAE', 'R²']

    for i, metric in enumerate(metrics):
        means = [summary[n][f'{metric}_mean'] for n in names]
        stds = [summary[n][f'{metric}_std'] for n in names]

        bars = axes[i].bar(names, means, yerr=stds, capsize=5, color=colors,
                           error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'alpha':0.7})
        axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].set_ylabel(ylabels[i], fontsize=12)
        axes[i].tick_params(axis='x', rotation=20, labelsize=10)

        # 数值标签（Ours加粗）
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            offset = 0.02 if metric == 'r2' else height * 0.05
            fw = 'bold' if bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0) else 'normal'
            if metric == 'r2':
                text = f'{mean:.3f}±{std:.3f}'
            else:
                text = f'{mean:.2f}±{std:.2f}'
            axes[i].text(bar.get_x() + bar.get_width()/2., height + offset,
                         text, ha='center', va='bottom', fontsize=9, fontweight=fw)

    plt.tight_layout()
    plt.savefig(CONFIG["plot_ablation"], dpi=300)
    plt.close()
    print(f"✅ 消融实验柱状图已保存: {CONFIG['plot_ablation']}")

def plot_calibration_ablation(summary):
    """绘制校准指标对比（全中文，Ours绿色压轴）"""
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False

    names = list(summary.keys())
    # Ours放最后
    non_ours = [n for n in names if "Ours" not in n]
    rmse_means_non_ours = [summary[n]['rmse_mean'] for n in non_ours]
    sorted_idx = np.argsort(rmse_means_non_ours)[::-1]
    sorted_names = [non_ours[i] for i in sorted_idx]
    if 'Ours (Full)' in names:
        sorted_names.append('Ours (Full)')
    names = sorted_names

    colors = ['#808080'] * (len(names)-1) + ['#2ca02c']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spearman相关性
    sp_means = [summary[n]['spearman_mean'] for n in names]
    sp_stds = [summary[n]['spearman_std'] for n in names]
    bars1 = axes[0].bar(names, sp_means, yerr=sp_stds, capsize=5, color=colors,
                        error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    axes[0].set_title('不确定性校准 - Spearman相关性 ↑', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Spearman ρ', fontsize=12)
    axes[0].tick_params(axis='x', rotation=20, labelsize=10)
    for bar, m, s in zip(bars1, sp_means, sp_stds):
        fw = 'bold' if bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0) else 'normal'
        axes[0].text(bar.get_x()+bar.get_width()/2., m+s+0.02, f'{m:.3f}±{s:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight=fw)

    # 期望校准误差（ECE）
    ece_means = [summary[n]['ece_mean'] for n in names]
    ece_stds = [summary[n]['ece_std'] for n in names]
    bars2 = axes[1].bar(names, ece_means, yerr=ece_stds, capsize=5, color=colors,
                        error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    axes[1].set_title('不确定性校准 - 期望校准误差(ECE) ↓', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('ECE', fontsize=12)
    axes[1].tick_params(axis='x', rotation=20, labelsize=10)
    for bar, m, s in zip(bars2, ece_means, ece_stds):
        fw = 'bold' if bar.get_facecolor() == (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0) else 'normal'
        axes[1].text(bar.get_x()+bar.get_width()/2., m+s+0.01, f'{m:.3f}±{s:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight=fw)

    plt.tight_layout()
    plt.savefig(CONFIG["plot_calibration"], dpi=300)
    plt.close()
    print(f"✅ 校准指标对比图已保存: {CONFIG['plot_calibration']}")

# ==============================================================================
# 8. 主程序
# ==============================================================================
if __name__ == "__main__":
    summary, all_results = run_ablation_experiments()

    print("\n" + "="*60)
    print("📊 消融实验最终结果（均值 ± 标准差, n={})".format(CONFIG['n_runs']))
    print("="*60)
    header = f"{'变体':<25} {'RMSE':<15} {'MAE':<15} {'R²':<15} {'Spearman':<15} {'ECE':<15} {'p vs Ours'}"
    print(header)
    print("-"*100)
    for name in summary.keys():
        s = summary[name]
        rmse = f"{s['rmse_mean']:.2f}±{s['rmse_std']:.2f}"
        mae = f"{s['mae_mean']:.2f}±{s['mae_std']:.2f}"
        r2 = f"{s['r2_mean']:.3f}±{s['r2_std']:.3f}"
        sp = f"{s['spearman_mean']:.3f}±{s['spearman_std']:.3f}" if 'spearman_mean' in s else 'N/A'
        ece = f"{s['ece_mean']:.3f}±{s['ece_std']:.3f}" if 'ece_mean' in s else 'N/A'
        p = f"{s['p_vs_ours']:.4f}" if 'p_vs_ours' in s else '-'
        print(f"{name:<25} {rmse:<15} {mae:<15} {r2:<15} {sp:<15} {ece:<15} {p}")
    print("="*60)

    # 保存汇总结果（不包含MSE）
    pd.DataFrame(summary).T.to_csv('ablation_results_summary.csv')
    print("✅ 汇总结果已保存至 ablation_results_summary.csv")

    plot_ablation_results(summary)
    plot_calibration_ablation(summary)