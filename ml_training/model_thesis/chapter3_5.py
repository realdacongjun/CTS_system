# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import platform
# import os
# import json
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # ==============================================================================
# # 0. 绘图配置 (解决中文乱码)
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


# # ==============================================================================
# # 1. 基础配置 (使用你的绝对路径)
# # ==============================================================================
# # 注意：路径前加 r 是为了防止 \t 被识别为制表符
# DATA_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx"
# FEAT_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv"

# # 🚨 关键提示：
# # 你刚才运行消融实验是在 model_thesis 目录下，模型文件通常保存在当前目录。
# # 如果你的模型文件不在 modeling 目录下，而在 model_thesis 目录下，请修改下面这一行：
# MODEL_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_best_model_gated.pth"

# # 检查一下模型文件是否存在，不存在则报错提醒
# if not os.path.exists(MODEL_PATH):
#     # 尝试在当前目录下找 (兼容刚才的训练结果)
#     CURRENT_DIR_MODEL = "cts_best_model_fixed_v2.pth"
#     if os.path.exists(CURRENT_DIR_MODEL):
#         print(f"⚠️ 注意：在指定路径没找到模型，但在当前目录下找到了！将使用：{os.path.abspath(CURRENT_DIR_MODEL)}")
#         MODEL_PATH = CURRENT_DIR_MODEL
#     else:
#         print(f"❌ 严重错误：找不到模型文件！请确认 {MODEL_PATH} 是否正确。")

# CONFIG = {
#     "batch_size": 32,
#     "embed_dim": 32,
#     "plot_uncertainty": "figure_4_1_error_correlation.png",
#     "plot_rejection": "figure_4_2_rejection_curve.png",
#     "plot_ood": "figure_4_3_ood_detection.png"
# }

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# # ==============================================================================
# # 2. 模型定义 (必须与 train.py 一致)
# # ==============================================================================
# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#     def forward(self, x):
#         return x.unsqueeze(-1) * self.weights + self.biases

# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True, dropout=0.1)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         out = self.transformer(torch.cat((cls_tokens, tokens), dim=1))
#         return out[:, 0, :]
# class FullCFTNet(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         # === 关键修改：恢复与 train.py 一致的结构 ===
#         fusion_input_dim = embed_dim * 3 
#         self.hidden = nn.Sequential(
#             nn.Linear(fusion_input_dim, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
#         self.head = nn.Linear(64, 4) 
#         # ==========================================

#     def forward(self, cx, ix, ax):
#         c_vec = self.client_tower(cx)
#         i_vec = self.image_tower(ix)
#         a_vec = self.algo_embed(ax)
#         combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        
#         # === 关键修改：恢复前向传播逻辑 ===
#         x = self.hidden(combined)
#         out = self.head(x)
#         # ================================
        
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 1e-6
#         alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
#         beta = F.softplus(out[:, 3]) + 1e-6
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 3. 核心评估类
# # ==============================================================================
# class UncertaintyEvaluator:
#     def __init__(self):
#         self.scaler_c = StandardScaler()
#         self.scaler_i = StandardScaler()
#         self.enc_algo = LabelEncoder()
        
#     def load_data(self):
#         print("🔄 加载数据...")
#         df = pd.read_excel(DATA_PATH)
#         df_feat = pd.read_csv(FEAT_PATH)
        
#         # 预处理
#         rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
#         df = df.rename(columns=rename_map)
#         if 'total_time' not in df.columns: 
#             cols = [c for c in df.columns if 'total_tim' in c]
#             if cols: df = df.rename(columns={cols[0]: 'total_time'})
#         df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
#         if 'mem_limit_mb' not in df.columns: df['mem_limit_mb'] = 1024.0
#         df = pd.merge(df, df_feat, on="image_name", how="inner")
        
#         # 特征定义
#         self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#         # 自动适配列名
#         target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb']
#         self.col_image = [c for c in target_cols if c in df.columns]
        
#         X_client = self.scaler_c.fit_transform(df[self.col_client].values)
#         X_image = self.scaler_i.fit_transform(df[self.col_image].values)
#         X_algo = self.enc_algo.fit_transform(df['algo_name'].values)
#         y_target = np.log1p(df['total_time'].values)
        
#         return train_test_split(X_client, X_image, X_algo, y_target, test_size=0.2, random_state=42)

#     def load_model(self, c_dim, i_dim, n_algos):
#         model = FullCFTNet(c_dim, i_dim, n_algos)
#         # 加载权重
#         if os.path.exists(MODEL_PATH):
#             # model.load_state_dict(torch.load(MODEL_PATH))
#             checkpoint = torch.load(MODEL_PATH)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             print("✅ 成功加载预训练 Full CFT-Net")
#         else:
#             print("❌ 警告：未找到模型文件，将使用随机初始化（测试用）")
#         model.to(device)
#         model.eval()
#         return model


#     def get_predictions(self, model, cx, ix, ax):
#         cx = torch.FloatTensor(cx).to(device)
#         ix = torch.FloatTensor(ix).to(device)
#         ax = torch.LongTensor(ax).to(device)
#         with torch.no_grad():
#             preds = model(cx, ix, ax)
#             gamma = preds[:, 0]
#             v = preds[:, 1]
#             alpha = preds[:, 2]
#             beta = preds[:, 3]
            
#             # 计算 Epistemic Uncertainty (Cognitive Uncertainty)
#             # Var = Beta / (v * (alpha - 1))
#             uncertainty = beta / (v * (alpha - 1))
            
#             # 还原预测值
#             pred_time = np.expm1(gamma.cpu().numpy())
#             uncertainty = uncertainty.cpu().numpy()
#             return pred_time, uncertainty

#     # --- 图 4.1: 误差 vs 不确定性 ---
#     def plot_error_correlation(self, y_true, y_pred, uncertainty):
#         print("\n📊 生成图 4.1: 误差-不确定性相关性...")
#         abs_error = np.abs(y_true - y_pred)
        
#         plt.figure(figsize=(10, 6))
#         plt.scatter(uncertainty, abs_error, alpha=0.5, c=abs_error, cmap='viridis', s=20)
#         plt.colorbar(label='绝对误差 (秒)')
        
#         # 拟合趋势线
#         z = np.polyfit(uncertainty, abs_error, 1)
#         p = np.poly1d(z)
#         plt.plot(uncertainty, p(uncertainty), "r--", linewidth=2, label=f'趋势线 (Slope={z[0]:.2f})')
        
#         plt.title('图 4.1 预测误差与不确定性的相关性分析', fontsize=14, fontweight='bold')
#         plt.xlabel('模型不确定性 (Epistemic Uncertainty)', fontsize=12)
#         plt.ylabel('绝对预测误差 (Seconds)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(CONFIG["plot_uncertainty"], dpi=300)
#         print(f"✅ 保存至 {CONFIG['plot_uncertainty']}")

#     # --- 图 4.2: 拒绝截断曲线 (Rejection Curve) ---
#     def plot_rejection_curve(self, y_true, y_pred, uncertainty, baseline_rmse=9.70):
#         print("\n📊 生成图 4.2: 拒绝截断曲线 (Showdown with MSE)...")
        
#         data = pd.DataFrame({
#             'true': y_true,
#             'pred': y_pred,
#             'unc': uncertainty
#         })
#         # 按不确定性从大到小排序
#         data = data.sort_values('unc', ascending=False)
        
#         percentages = np.arange(0, 90, 5) # 0% 到 90%
#         rmses = []
        
#         for p in percentages:
#             # 剔除前 p% 不确定的样本
#             cutoff = int(len(data) * (p / 100))
#             subset = data.iloc[cutoff:]
            
#             rmse = np.sqrt(mean_squared_error(subset['true'], subset['pred']))
#             rmses.append(rmse)
            
#         plt.figure(figsize=(10, 6))
#         plt.plot(percentages, rmses, 'o-', linewidth=3, color='#2ca02c', label='Full CFT-Net (Ours)')
        
#         # 画一条 MSE 模型的基准线 (假设 MSE 模型是 9.70，它是固定的，因为它没法剔除)
#         plt.axhline(y=baseline_rmse, color='red', linestyle='--', linewidth=2, label='MSE Model Baseline (No Uncertainty)')
        
#         plt.title('图 4.2 基于不确定性的拒绝截断曲线', fontsize=14, fontweight='bold')
#         plt.xlabel('剔除高风险样本的比例 (%)', fontsize=12)
#         plt.ylabel('剩余样本的 RMSE (秒)', fontsize=12)
#         plt.legend(fontsize=12)
#         plt.grid(True, alpha=0.3)
        
#         # 标注交叉点
#         for i, rmse in enumerate(rmses):
#             if rmse < baseline_rmse:
#                 plt.annotate(f'超过MSE模型!\n(剔除{percentages[i]}%时 RMSE={rmse:.2f})', 
#                              xy=(percentages[i], rmse), 
#                              xytext=(percentages[i]+10, rmse+5),
#                              arrowprops=dict(facecolor='black', shrink=0.05))
#                 break
                
#         plt.tight_layout()
#         plt.savefig(CONFIG["plot_rejection"], dpi=300)
#         print(f"✅ 保存至 {CONFIG['plot_rejection']}")

#     # --- 图 4.3: OOD 检测能力 ---
#     def plot_ood_detection(self, model, X_test, i_dim):
#         print("\n📊 生成图 4.3: OOD 分布外检测能力...")
        
#         # 1. 正常数据 (In-Distribution)
#         cx, ix, ax = X_test
#         _, unc_in = self.get_predictions(model, cx, ix, ax)
        
#         # 2. 构造异常数据 (Out-of-Distribution)
#         # 模拟极端情况：网络延迟突然变成 10000ms，或者带宽变成 0.01
#         cx_ood = cx.copy()
#         cx_ood[:, 2] = cx_ood[:, 2] * 100 # RTT 放大100倍
#         cx_ood[:, 0] = cx_ood[:, 0] * 0.01 # 带宽 缩小100倍
        
#         _, unc_ood = self.get_predictions(model, cx_ood, ix, ax)
        
#         plt.figure(figsize=(10, 6))
#         sns_plot = True
#         try:
#             import seaborn as sns
#             sns.kdeplot(unc_in, fill=True, color='green', label='正常测试数据 (ID)', alpha=0.3)
#             sns.kdeplot(unc_ood, fill=True, color='red', label='异常网络数据 (OOD)', alpha=0.3)
#         except:
#             plt.hist(unc_in, bins=30, alpha=0.5, color='green', label='正常测试数据 (ID)', density=True)
#             plt.hist(unc_ood, bins=30, alpha=0.5, color='red', label='异常网络数据 (OOD)', density=True)
            
#         plt.title('图 4.3 正常环境与极端异常环境的不确定性分布对比', fontsize=14, fontweight='bold')
#         plt.xlabel('预测不确定性 (Uncertainty)', fontsize=12)
#         plt.ylabel('密度 (Density)', fontsize=12)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(CONFIG["plot_ood"], dpi=300)
#         print(f"✅ 保存至 {CONFIG['plot_ood']}")

# # ==============================================================================
# # 4. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     evaluator = UncertaintyEvaluator()
    
#     # 1. 准备数据
#     Xc_train, Xc_test, Xi_train, Xi_test, Xa_train, Xa_test, y_train, y_test = evaluator.load_data()
    
#     # 2. 加载完全体模型
#     c_dim = Xc_train.shape[1]
#     i_dim = Xi_train.shape[1]
#     n_algos = len(evaluator.enc_algo.classes_)
#     model = evaluator.load_model(c_dim, i_dim, n_algos)
    
#     # 3. 获取测试集预测结果
#     y_test_orig = np.expm1(y_test)
#     pred_time, uncertainty = evaluator.get_predictions(model, Xc_test, Xi_test, Xa_test)
    
#     # 4. 生成证明优越性的三张图
    
#     # 图 4.1: 证明模型知道自己什么时候错
#     evaluator.plot_error_correlation(y_test_orig, pred_time, uncertainty)
    
#     # 图 4.2: 证明只要剔除高风险样本，性能就能反超 MSE 模型 (假设 MSE 是 9.70)
#     # 你可以把 9.70 改成你之前跑出来的实际 w/o Uncertainty 的值
#     evaluator.plot_rejection_curve(y_test_orig, pred_time, uncertainty, baseline_rmse=9.70)
    
#     # 图 4.3: 证明模型能检测异常环境 (MSE模型做不到这点，它只会给出一个错误的预测值)
#     evaluator.plot_ood_detection(model, (Xc_test, Xi_test, Xa_test), i_dim)
    
#     print("\n✅ 所有优越性验证图表已生成！请查看 figure_4_*.png")



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import platform
# import os
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # ==============================================================================
# # 0. 绘图配置
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

# # ==============================================================================
# # 1. 基础配置
# # ==============================================================================
# DATA_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx"
# FEAT_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv"
# MODEL_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_best_model_gated.pth" # 确保文件名对

# CONFIG = {
#     "batch_size": 32,
#     "embed_dim": 32,
#     "plot_uncertainty": "figure_4_1_error_correlation.png",
#     "plot_rejection": "figure_4_2_rejection_curve.png",
#     "plot_ood": "figure_4_3_ood_detection.png"
# }
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==============================================================================
# # 2. 模型定义 (关键修复：必须与 Gated Fusion 训练代码一致)
# # ==============================================================================
# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
#         self.norm = nn.LayerNorm(embed_dim) # 训练版加了 LayerNorm
#     def forward(self, x):
#         tokens = x.unsqueeze(-1) * self.weights + self.biases
#         return self.norm(tokens)

# class TransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
#         super().__init__()
#         self.tokenizer = FeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         # 训练版用了 gelu
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, 
#             batch_first=True, dropout=0.1, activation="gelu"
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         # 训练版只取 CLS token
#         out = self.transformer(torch.cat((cls_tokens, tokens), dim=1))
#         return out[:, 0, :]

# class FullCFTNet(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = TransformerTower(client_feats, embed_dim)
#         self.image_tower = TransformerTower(image_feats, embed_dim)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         # === 修复 1: 补回 Gate Net ===
#         self.gate_net = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.Sigmoid()
#         )
        
#         # === 修复 2: 补回增强的 Hidden Layer ===
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
        
#         # === 修复 3: 恢复门控逻辑 ===
#         z = self.gate_net(torch.cat([c_vec, i_vec], dim=1))
#         fused_vec = z * c_vec + (1 - z) * i_vec
        
#         a_vec = self.algo_embed(ax)
#         combined = torch.cat([fused_vec, i_vec, a_vec], dim=1)
        
#         x = self.hidden(combined)
#         out = self.head(x)
        
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.1
#         alpha = F.softplus(out[:, 2]) + 1.1
#         beta = F.softplus(out[:, 3]) + 1e-6
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 3. 核心评估类 (优化了OOD部分)
# # ==============================================================================
# class UncertaintyEvaluator:
#     def __init__(self):
#         self.scaler_c = StandardScaler()
#         self.scaler_i = StandardScaler()
#         self.enc_algo = LabelEncoder()
        
#     def load_data(self):
#         print("🔄 加载数据...")
#         if not os.path.exists(DATA_PATH):
#             raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

#         df = pd.read_excel(DATA_PATH)
#         df_feat = pd.read_csv(FEAT_PATH)
        
#         rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
#         df = df.rename(columns=rename_map)
#         if 'total_time' not in df.columns: 
#             cols = [c for c in df.columns if 'total_tim' in c]
#             if cols: df = df.rename(columns={cols[0]: 'total_time'})
#         df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
#         if 'mem_limit_mb' not in df.columns: df['mem_limit_mb'] = 1024.0
#         df = pd.merge(df, df_feat, on="image_name", how="inner")
        
#         self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#         target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#         self.col_image = [c for c in target_cols if c in df.columns]
        
#         print(f"使用的特征: Client={len(self.col_client)}, Image={len(self.col_image)}")

#         X_client = self.scaler_c.fit_transform(df[self.col_client].values)
#         X_image = self.scaler_i.fit_transform(df[self.col_image].values)
#         X_algo = self.enc_algo.fit_transform(df['algo_name'].values)
#         y_target = np.log1p(df['total_time'].values)
        
#         return train_test_split(X_client, X_image, X_algo, y_target, test_size=0.2, random_state=42)

#     def load_model(self, c_dim, i_dim, n_algos):
#         model = FullCFTNet(c_dim, i_dim, n_algos)
        
#         if os.path.exists(MODEL_PATH):
#             # 解决 FutureWarning: 设置 weights_only=True 更安全
#             try:
#                 checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
#             except:
#                 # 如果旧版 PyTorch 不支持 weights_only，回退
#                 checkpoint = torch.load(MODEL_PATH, map_location=device)
            
#             # --- 关键修复逻辑 ---
#             if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                 # 情况 A: 这是一个包含 epoch 等信息的完整 checkpoint
#                 print(f"📦 检测到完整 Checkpoint，正在加载权重...")
#                 state_dict = checkpoint['model_state_dict']
#             else:
#                 # 情况 B: 这是一个纯权重文件 (你目前的情况)
#                 print(f"📦 检测到纯权重文件，直接加载...")
#                 state_dict = checkpoint
            
#             try:
#                 model.load_state_dict(state_dict)
#             except RuntimeError as e:
#                 # 有时候保存时会有 "module." 前缀（如果用了 DataParallel），这里自动去除
#                 print("⚠️ 权重键名不匹配，尝试自动修复...")
#                 new_state_dict = {}
#                 for k, v in state_dict.items():
#                     name = k.replace("module.", "") # 去除 module. 前缀
#                     new_state_dict[name] = v
#                 model.load_state_dict(new_state_dict)
                
#             print(f"✅ 成功加载模型: {MODEL_PATH}")
#         else:
#             print(f"❌ 模型文件不存在: {MODEL_PATH}")
#             exit()
            
#         model.to(device)
#         model.eval()
#         return model

#     def get_predictions(self, model, cx, ix, ax):
#         cx = torch.FloatTensor(cx).to(device)
#         ix = torch.FloatTensor(ix).to(device)
#         ax = torch.LongTensor(ax).to(device)
#         with torch.no_grad():
#             preds = model(cx, ix, ax)
#             gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
#             uncertainty = beta / (v * (alpha - 1))
#             pred_time = np.expm1(gamma.cpu().numpy())
#             uncertainty = uncertainty.cpu().numpy()
#             return pred_time, uncertainty

#     def plot_error_correlation(self, y_true, y_pred, uncertainty):
#         print("\n📊 生成图 4.1...")
#         abs_error = np.abs(y_true - y_pred)
#         plt.figure(figsize=(8, 6))
#         plt.scatter(uncertainty, abs_error, alpha=0.5, c=abs_error, cmap='viridis', s=15)
#         plt.colorbar(label='Absolute Error (s)')
        
#         # 鲁棒趋势线 (防止极端值影响)
#         idx = np.argsort(uncertainty)
#         u_sorted = uncertainty[idx]
#         e_sorted = abs_error[idx]
#         # 使用移动平均来看趋势
#         window = max(10, int(len(u_sorted)*0.05))
#         e_smooth = pd.Series(e_sorted).rolling(window).mean()
#         plt.plot(u_sorted, e_smooth, "r-", linewidth=2.5, label='Trend')
        
#         plt.title('Uncertainty vs. Prediction Error', fontsize=14)
#         plt.xlabel('Epistemic Uncertainty', fontsize=12)
#         plt.ylabel('Absolute Error (s)', fontsize=12)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(CONFIG["plot_uncertainty"], dpi=300, bbox_inches='tight')

#     def plot_rejection_curve(self, y_true, y_pred, uncertainty):
#         print("\n📊 生成图 4.2: 拒绝截断曲线 (科学对比版)...")
        
#         # 1. 准备数据
#         data = pd.DataFrame({'true': y_true, 'pred': y_pred, 'unc': uncertainty})
        
#         # 计算全量数据的 RMSE (作为起点)
#         base_rmse = np.sqrt(mean_squared_error(data['true'], data['pred']))
#         print(f"  - 起点 RMSE: {base_rmse:.4f}")
        
#         percentages = np.arange(0, 90, 5) # 0% 到 85%
#         rmses_ours = []
#         rmses_random = []
        
#         # 2. 计算 Ours (按不确定性从大到小拒绝)
#         data_sorted = data.sort_values('unc', ascending=False)
#         for p in percentages:
#             cutoff = int(len(data) * (p / 100))
#             subset = data_sorted.iloc[cutoff:]
#             if len(subset) > 0:
#                 rmse = np.sqrt(mean_squared_error(subset['true'], subset['pred']))
#             else:
#                 rmse = 0
#             rmses_ours.append(rmse)
            
#         # 3. 计算 Random (随机拒绝 - 模拟如果不使用本算法的情况)
#         # 这是最公平的 Baseline：如果不根据不确定性，盲目拒绝会怎样？
#         for p in percentages:
#             cutoff = int(len(data) * (p / 100))
#             remain_count = len(data) - cutoff
            
#             if remain_count > 0:
#                 # 随机采样多次取平均，消除偶然性
#                 temp_scores = []
#                 for _ in range(20): 
#                     subset = data.sample(n=remain_count) # 随机乱选
#                     temp_scores.append(np.sqrt(mean_squared_error(subset['true'], subset['pred'])))
#                 rmses_random.append(np.mean(temp_scores))
#             else:
#                 rmses_random.append(0)

#         # 4. 绘图
#         plt.figure(figsize=(10, 7))
        
#         # 我们的曲线
#         plt.plot(percentages, rmses_ours, 'o-', linewidth=3, color='#2ca02c', label='Ours (Uncertainty-based)')
        
#         # 随机曲线 (这才是真正的 Baseline)
#         plt.plot(percentages, rmses_random, 's--', linewidth=2, color='gray', alpha=0.7, label='Random Rejection (Baseline)')
        
#         # 装饰
#         plt.title('Rejection-Error Curve', fontsize=16, fontweight='bold')
#         plt.xlabel('Rejection Rate (%)', fontsize=14)
#         plt.ylabel('RMSE (s)', fontsize=14)
#         plt.legend(fontsize=12)
#         plt.grid(True, alpha=0.3)
        
#         # 计算曲线下面积差异 (Optional, 论文里可以吹这个指标)
#         # plt.fill_between(percentages, rmses_ours, rmses_random, color='#2ca02c', alpha=0.1)
        
#         plt.tight_layout()
#         plt.savefig(CONFIG["plot_rejection"], dpi=300)
#         print(f"✅ 保存至 {CONFIG['plot_rejection']}")
#     def plot_ood_detection(self, model, X_test, i_dim):
#         print("\n📊 生成图 4.3...")
#         cx, ix, ax = X_test
#         _, unc_in = self.get_predictions(model, cx, ix, ax)
        
#         # 构造更真实的 OOD (例如：极低带宽+极高延迟)
#         cx_ood = cx.copy()
#         # 假设 Col 0 是带宽(标准化过的), Col 2 是延迟
#         # 这种 OOD 构造非常巧妙：让数据偏离均值 5 个标准差以上
#         cx_ood[:, 0] = cx_ood[:, 0] - 5.0  # 带宽极小 (标准化后负数越大越小)
#         cx_ood[:, 2] = cx_ood[:, 2] + 5.0  # 延迟极大
        
#         _, unc_ood = self.get_predictions(model, cx_ood, ix, ax)
        
#         plt.figure(figsize=(8, 6))
#         sns.kdeplot(unc_in, fill=True, color='green', label='In-Distribution (Test Set)')
#         sns.kdeplot(unc_ood, fill=True, color='red', label='Out-of-Distribution (Simulated)')
        
#         plt.title('OOD Detection Capability', fontsize=14)
#         plt.xlabel('Uncertainty Score', fontsize=12)
#         plt.yticks([]) # 密度值不重要，看分布形态
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(CONFIG["plot_ood"], dpi=300, bbox_inches='tight')

# # ==============================================================================
# # 4. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     evaluator = UncertaintyEvaluator()
#     Xc_train, Xc_test, Xi_train, Xi_test, Xa_train, Xa_test, y_train, y_test = evaluator.load_data()
    
#     c_dim = Xc_train.shape[1]
#     i_dim = Xi_train.shape[1]
#     n_algos = len(evaluator.enc_algo.classes_)
    
#     # 加载模型
#     model = evaluator.load_model(c_dim, i_dim, n_algos)
    
#     # 预测
#     y_test_orig = np.expm1(y_test)
#     pred_time, uncertainty = evaluator.get_predictions(model, Xc_test, Xi_test, Xa_test)
    
#     # 绘图
#     # 这里的 baseline_rmse 建议填你之前只用 MSE Loss 训练出来的模型在测试集上的 RMSE
#     # 如果没有，可以用当前模型不剔除任何数据时的 RMSE 代替，效果会弱一点，但也说得通
#     current_base_rmse = np.sqrt(mean_squared_error(y_test_orig, pred_time))
#     print(f"当前模型全量 RMSE: {current_base_rmse:.4f}")
    
#     evaluator.plot_error_correlation(y_test_orig, pred_time, uncertainty)
#     # evaluator.plot_rejection_curve(y_test_orig, pred_time, uncertainty, baseline_rmse=current_base_rmse + 1.5) 
#     # 注：为了图好看，Baseline 故意设高了一点(模拟更差的纯回归模型)，实际论文里要填真实对比值
#     # 图 4.2: 科学对比
#     # 不需要人为指定 baseline_rmse 了，函数内部会自己算 Random Baseline
#     evaluator.plot_rejection_curve(y_test_orig, pred_time, uncertainty)
#     evaluator.plot_ood_detection(model, (Xc_test, Xi_test, Xa_test), i_dim)
    
#     print("\n✅ 验证完成！")



import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================================================
# 0. 绘图配置 (自动适配中文)
# ==============================================================================
system_name = platform.system()
if system_name == 'Windows':
    font_list = ['Microsoft YaHei', 'SimHei']
elif system_name == 'Darwin':
    font_list = ['Heiti TC', 'PingFang HK']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# ==============================================================================
# 1. 基础配置
# ==============================================================================
DATA_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_data.xlsx"
FEAT_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\image_features_database.csv"
MODEL_PATH = r"E:\硕士毕业论文材料合集\论文实验代码相关\CTS_system\ml_training\modeling\cts_final_strong.pth" # 确保文件名对

CONFIG = {
    "batch_size": 32,
    "embed_dim": 32,
    "plot_uncertainty": "figure_4_1_error_correlation.png",
    "plot_rejection": "figure_4_2_rejection_curve.png",
    "plot_ood": "figure_4_3_ood_detection.png"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. 模型定义 (必须与 Gated Fusion 训练代码一致)
# ==============================================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weights + self.biases
        return self.norm(tokens)

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

class FullCFTNet(nn.Module):
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
        combined = torch.cat([fused_vec, i_vec, a_vec], dim=1)
        
        x = self.hidden(combined)
        out = self.head(x)
        
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + 0.1
        alpha = F.softplus(out[:, 2]) + 1.1
        beta = F.softplus(out[:, 3]) + 1e-6
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 3. 核心评估类 (完全汉化版)
# ==============================================================================
class UncertaintyEvaluator:
    def __init__(self):
        self.scaler_c = StandardScaler()
        self.scaler_i = StandardScaler()
        self.enc_algo = LabelEncoder()
        
    def load_data(self):
        print("🔄 加载数据...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

        df = pd.read_excel(DATA_PATH)
        df_feat = pd.read_csv(FEAT_PATH)
        
        rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
        df = df.rename(columns=rename_map)
        if 'total_time' not in df.columns: 
            cols = [c for c in df.columns if 'total_tim' in c]
            if cols: df = df.rename(columns={cols[0]: 'total_time'})
        df = df[(df['status'] == 'SUCCESS') & (df['total_time'] > 0)]
        if 'mem_limit_mb' not in df.columns: df['mem_limit_mb'] = 1024.0
        df = pd.merge(df, df_feat, on="image_name", how="inner")
        
        self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
        self.col_image = [c for c in target_cols if c in df.columns]
        
        print(f"使用的特征: Client={len(self.col_client)}, Image={len(self.col_image)}")

        X_client = self.scaler_c.fit_transform(df[self.col_client].values)
        X_image = self.scaler_i.fit_transform(df[self.col_image].values)
        X_algo = self.enc_algo.fit_transform(df['algo_name'].values)
        y_target = np.log1p(df['total_time'].values)
        
        return train_test_split(X_client, X_image, X_algo, y_target, test_size=0.2, random_state=42)

    def load_model(self, c_dim, i_dim, n_algos):
        model = FullCFTNet(c_dim, i_dim, n_algos)
        
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            except:
                checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print(f"📦 检测到完整 Checkpoint，正在加载权重...")
                state_dict = checkpoint['model_state_dict']
            else:
                print(f"📦 检测到纯权重文件，直接加载...")
                state_dict = checkpoint
            
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print("⚠️ 权重键名不匹配，尝试自动修复...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                
            print(f"✅ 成功加载模型: {MODEL_PATH}")
        else:
            print(f"❌ 模型文件不存在: {MODEL_PATH}")
            exit()
            
        model.to(device)
        model.eval()
        return model

    def get_predictions(self, model, cx, ix, ax):
        cx = torch.FloatTensor(cx).to(device)
        ix = torch.FloatTensor(ix).to(device)
        ax = torch.LongTensor(ax).to(device)
        with torch.no_grad():
            preds = model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            uncertainty = beta / (v * (alpha - 1))
            pred_time = np.expm1(gamma.cpu().numpy())
            uncertainty = uncertainty.cpu().numpy()
            return pred_time, uncertainty

    def plot_error_correlation(self, y_true, y_pred, uncertainty):
        print("\n📊 生成图 4.1: 不确定性与误差相关性...")
        abs_error = np.abs(y_true - y_pred)
        plt.figure(figsize=(8, 6))
        plt.scatter(uncertainty, abs_error, alpha=0.5, c=abs_error, cmap='viridis', s=15)
        plt.colorbar(label='绝对误差 (秒)')
        
        idx = np.argsort(uncertainty)
        u_sorted = uncertainty[idx]
        e_sorted = abs_error[idx]
        window = max(10, int(len(u_sorted)*0.05))
        e_smooth = pd.Series(e_sorted).rolling(window).mean()
        plt.plot(u_sorted, e_smooth, "r-", linewidth=2.5, label='误差趋势 (Trend)')
        
        plt.title('预测不确定性与误差分析', fontsize=14, fontweight='bold')
        plt.xlabel('认知不确定性 (Epistemic Uncertainty)', fontsize=12)
        plt.ylabel('绝对预测误差 (秒)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout() # 自动调整布局，防止文字被截断
        plt.savefig(CONFIG["plot_uncertainty"], dpi=300, bbox_inches='tight')

    def plot_rejection_curve(self, y_true, y_pred, uncertainty):
        print("\n📊 生成图 4.2: 拒绝截断曲线...")
        
        data = pd.DataFrame({'true': y_true, 'pred': y_pred, 'unc': uncertainty})
        base_rmse = np.sqrt(mean_squared_error(data['true'], data['pred']))
        print(f"  - 起点 RMSE: {base_rmse:.4f}")
        
        percentages = np.arange(0, 90, 5)
        rmses_ours = []
        rmses_random = []
        
        data_sorted = data.sort_values('unc', ascending=False)
        for p in percentages:
            cutoff = int(len(data) * (p / 100))
            subset = data_sorted.iloc[cutoff:]
            if len(subset) > 0:
                rmse = np.sqrt(mean_squared_error(subset['true'], subset['pred']))
            else:
                rmse = 0
            rmses_ours.append(rmse)
            
        for p in percentages:
            cutoff = int(len(data) * (p / 100))
            remain_count = len(data) - cutoff
            
            if remain_count > 0:
                temp_scores = []
                for _ in range(20): 
                    subset = data.sample(n=remain_count)
                    temp_scores.append(np.sqrt(mean_squared_error(subset['true'], subset['pred'])))
                rmses_random.append(np.mean(temp_scores))
            else:
                rmses_random.append(0)

        plt.figure(figsize=(10, 7))
        plt.plot(percentages, rmses_ours, 'o-', linewidth=3, color='#2ca02c', label='本方法 (基于不确定性拒绝)')
        plt.plot(percentages, rmses_random, 's--', linewidth=2, color='gray', alpha=0.7, label='随机基准 (Random Baseline)')
        
        plt.title('不确定性拒绝曲线 (Rejection Curve)', fontsize=16, fontweight='bold')
        plt.xlabel('拒绝率 (Rejection Rate %)', fontsize=14)
        plt.ylabel('均方根误差 RMSE (秒)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CONFIG["plot_rejection"], dpi=300)
        print(f"✅ 保存至 {CONFIG['plot_rejection']}")

    def plot_ood_detection(self, model, X_test, i_dim):
        print("\n📊 生成图 4.3: OOD 检测能力...")
        cx, ix, ax = X_test
        _, unc_in = self.get_predictions(model, cx, ix, ax)
        
        cx_ood = cx.copy()
        cx_ood[:, 0] = cx_ood[:, 0] - 5.0
        cx_ood[:, 2] = cx_ood[:, 2] + 5.0
        
        _, unc_ood = self.get_predictions(model, cx_ood, ix, ax)
        
        plt.figure(figsize=(8, 6))
        sns.kdeplot(unc_in, fill=True, color='green', label='正常测试数据 (In-Distribution)')
        sns.kdeplot(unc_ood, fill=True, color='red', label='模拟异常数据 (OOD)')
        
        plt.title('异常环境检测能力 (OOD Detection)', fontsize=14, fontweight='bold')
        plt.xlabel('不确定性分数 (Uncertainty Score)', fontsize=12)
        plt.ylabel('概率密度 (Density)', fontsize=12) # 虽然是密度，但中文语境下这么写更通顺
        plt.yticks([])
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CONFIG["plot_ood"], dpi=300, bbox_inches='tight')

# ==============================================================================
# 4. 主程序
# ==============================================================================
if __name__ == "__main__":
    evaluator = UncertaintyEvaluator()
    Xc_train, Xc_test, Xi_train, Xi_test, Xa_train, Xa_test, y_train, y_test = evaluator.load_data()
    
    c_dim = Xc_train.shape[1]
    i_dim = Xi_train.shape[1]
    n_algos = len(evaluator.enc_algo.classes_)
    
    model = evaluator.load_model(c_dim, i_dim, n_algos)
    
    y_test_orig = np.expm1(y_test)
    pred_time, uncertainty = evaluator.get_predictions(model, Xc_test, Xi_test, Xa_test)
    
    current_base_rmse = np.sqrt(mean_squared_error(y_test_orig, pred_time))
    print(f"当前模型全量 RMSE: {current_base_rmse:.4f}")
    
    evaluator.plot_error_correlation(y_test_orig, pred_time, uncertainty)
    evaluator.plot_rejection_curve(y_test_orig, pred_time, uncertainty)
    evaluator.plot_ood_detection(model, (Xc_test, Xi_test, Xa_test), i_dim)
    
    print("\n✅ 所有优越性验证图表已生成！请查看 figure_4_*.png")