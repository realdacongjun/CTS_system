# """
# CFT-Net 完整对比评测脚本
# 生成用于论文的对比表格和雷达图（精度、风险感知、可靠性、轻量化）
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import os
# import time
# import pickle
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from scipy.stats import spearmanr, norm
# from scipy.optimize import brentq
# from collections import Counter
# import warnings
# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 0. 基础配置
# # ==============================================================================
# # 中文字体设置（根据系统调整）
# import platform
# system = platform.system()
# if system == 'Windows':
#     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# elif system == 'Darwin':
#     plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK']
# else:
#     plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
# plt.rcParams['axes.unicode_minus'] = False

# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# # 创建结果目录
# os.makedirs("evaluation_results", exist_ok=True)

# # ==============================================================================
# # 1. 模型定义（与训练时完全一致）
# # ==============================================================================
# class LightweightFeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.embeddings = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
#         self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
#         self.norm = nn.LayerNorm(embed_dim)
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         out = x * self.embeddings + self.bias
#         return self.norm(out)

# class LightweightTransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim=32, nhead=2):
#         super().__init__()
#         self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.encoder = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=32,
#             batch_first=True, dropout=0.1, activation="gelu"
#         )
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls = self.cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat([cls, tokens], dim=1)
#         out = self.encoder(x)
#         return out[:, 0, :]

# class CompactCFTNet(nn.Module):
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
#         super().__init__()
#         self.client_tower = LightweightTransformerTower(client_feats, embed_dim, nhead=2)
#         self.image_tower = LightweightTransformerTower(image_feats, embed_dim, nhead=2)
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
#         self.fusion = nn.Sequential(
#             nn.Linear(embed_dim * 3, 32),
#             nn.LayerNorm(32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 4)
#         )
#         self._init_weights()
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
#         a = self.algo_embed(ax)
#         fused = torch.cat([c, i, a], dim=-1)
#         out = self.fusion(fused)
#         gamma = out[:, 0]
#         v = F.softplus(out[:, 1]) + 0.5
#         alpha = F.softplus(out[:, 2]) + 1.5
#         beta = F.softplus(out[:, 3]) + 1.0
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 2. 评估指标函数
# # ==============================================================================
# def calculate_smape(y_true, y_pred):
#     """对称平均绝对百分比误差"""
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
#     smape = np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100
#     return smape

# def calculate_mape(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

# def calculate_picp_mpiw(y_true, y_pred, unc, confidence=0.8):
#     z = norm.ppf((1 + confidence) / 2)
#     lower = y_pred - z * unc
#     upper = y_pred + z * unc
#     picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
#     mpiw = np.mean(upper - lower)
#     return picp, mpiw

# def calculate_ece_quantile(errors, uncertainties, n_bins=10):
#     if len(errors) == 0:
#         return 0.0
#     quantiles = np.linspace(0, 100, n_bins + 1)
#     bin_edges = np.percentile(uncertainties, quantiles)
#     bin_edges[-1] += 1e-8
#     ece = 0.0
#     for i in range(n_bins):
#         in_bin = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
#         if i == n_bins - 1:
#             in_bin = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])
#         prop = in_bin.sum() / len(errors)
#         if prop > 0:
#             avg_unc = uncertainties[in_bin].mean()
#             avg_err = errors[in_bin].mean()
#             ece += np.abs(avg_err - avg_unc) * prop
#     return ece

# def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, search_range=(0.1, 50)):
#     def picp_with_scale(s):
#         lower = y_pred - s * unc_raw
#         upper = y_pred + s * unc_raw
#         return np.mean((y_true >= lower) & (y_true <= upper))
#     s_min, s_max = search_range
#     picp_min = picp_with_scale(s_min)
#     picp_max = picp_with_scale(s_max)
#     if picp_min >= target_coverage:
#         return s_min
#     elif picp_max <= target_coverage:
#         print(f"警告：最大缩放 {s_max} 仅能达到 PICP={picp_max:.1f}%，小于目标 {target_coverage*100}%")
#         return s_max
#     else:
#         try:
#             s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, s_min, s_max)
#             return s_opt
#         except:
#             scales = np.linspace(s_min, s_max, 200)
#             picps = [picp_with_scale(s) for s in scales]
#             best_idx = np.argmin(np.abs(np.array(picps) - target_coverage))
#             return scales[best_idx]

# # ==============================================================================
# # 3. 数据加载与预处理
# # ==============================================================================
# def load_preprocessing_objects():
#     with open('preprocessing_objects.pkl', 'rb') as f:
#         prep = pickle.load(f)
#     return prep

# def load_data():
#     df_exp = pd.read_excel("cts_data.xlsx")
#     df_feat = pd.read_csv("image_features_database.csv")
#     rename_map = {
#         "image": "image_name", "method": "algo_name",
#         "network_bw": "bandwidth_mbps", "network_delay": "network_rtt",
#         "mem_limit": "mem_limit_mb"
#     }
#     df_exp = df_exp.rename(columns=rename_map)
#     if 'total_time' not in df_exp.columns:
#         cols = [c for c in df_exp.columns if 'total_tim' in c]
#         if cols:
#             df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
#     df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
#     df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
#     cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
#     target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std',
#                    'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
#     cols_i = [c for c in target_cols if c in df.columns]
#     Xc_raw = df[cols_c].values
#     Xi_raw = df[cols_i].values
#     y_raw = np.log1p(df['total_time'].values)
#     algo_names_raw = df['algo_name'].values
#     return Xc_raw, Xi_raw, algo_names_raw, y_raw, cols_c, cols_i, df['total_time'].values

# # ==============================================================================
# # 4. 评估主类
# # ==============================================================================
# class ModelEvaluator:
#     def __init__(self, model_path, seed=42):
#         self.seed = seed
#         np.random.seed(seed)
#         self.prep = load_preprocessing_objects()
#         self.scaler_c = self.prep['scaler_c']
#         self.scaler_i = self.prep['scaler_i']
#         self.enc = self.prep['enc']
#         self.cols_c = self.prep.get('cols_c', ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb'])
#         self.cols_i = self.prep.get('cols_i', ['total_size_mb', 'avg_layer_entropy', 'layer_count', 'text_ratio', 'zero_ratio'])
#         self.default_algo = self.prep.get('most_common_algo', self.enc.classes_[0])
#         self.default_idx = self.enc.transform([self.default_algo])[0]
        
#         # 加载原始数据并划分
#         Xc_raw, Xi_raw, algo_names_raw, y_log, _, _, y_orig = load_data()
#         N = len(y_log)
#         idx = np.random.permutation(N)
#         n_tr = int(N * 0.7)
#         n_val = int(N * 0.15)
#         self.tr_idx = idx[:n_tr]
#         self.val_idx = idx[n_tr:n_tr+n_val]
#         self.te_idx = idx[n_tr+n_val:]
        
#         # 标准化
#         self.Xc_train = self.scaler_c.transform(Xc_raw[self.tr_idx])
#         self.Xc_val = self.scaler_c.transform(Xc_raw[self.val_idx])
#         self.Xc_test = self.scaler_c.transform(Xc_raw[self.te_idx])
#         self.Xi_train = self.scaler_i.transform(Xi_raw[self.tr_idx])
#         self.Xi_val = self.scaler_i.transform(Xi_raw[self.val_idx])
#         self.Xi_test = self.scaler_i.transform(Xi_raw[self.te_idx])
        
#         # 算法编码
#         def safe_transform(labels):
#             known = set(self.enc.classes_)
#             return np.array([self.enc.transform([l])[0] if l in known else self.default_idx for l in labels])
        
#         self.Xa_train = self.enc.transform(algo_names_raw[self.tr_idx])
#         self.Xa_val = safe_transform(algo_names_raw[self.val_idx])
#         self.Xa_test = safe_transform(algo_names_raw[self.te_idx])
        
#         self.y_train_log = y_log[self.tr_idx]
#         self.y_val_log = y_log[self.val_idx]
#         self.y_test_log = y_log[self.te_idx]
#         self.y_train_orig = y_orig[self.tr_idx]
#         self.y_val_orig = y_orig[self.val_idx]
#         self.y_test_orig = y_orig[self.te_idx]
        
#         # 为基线模型准备组合特征
#         self.X_train_comb = np.hstack([self.Xc_train, self.Xi_train, self.Xa_train.reshape(-1,1)])
#         self.X_val_comb = np.hstack([self.Xc_val, self.Xi_val, self.Xa_val.reshape(-1,1)])
#         self.X_test_comb = np.hstack([self.Xc_test, self.Xi_test, self.Xa_test.reshape(-1,1)])
        
#         print(f"数据划分: 训练 {len(self.tr_idx)} | 验证 {len(self.val_idx)} | 测试 {len(self.te_idx)}")
        
#         # 加载CFT-Net模型
#         self.cftnet = CompactCFTNet(len(self.cols_c), len(self.cols_i), len(self.enc.classes_)).to(device)
#         checkpoint = torch.load(model_path, map_location=device)
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         else:
#             state_dict = checkpoint
#         self.cftnet.load_state_dict(state_dict)
#         self.cftnet.eval()
#         print("CFT-Net 模型加载成功")
        
#         self.results = {}
    
#     def predict_cftnet(self, Xc, Xi, Xa):
#         """批量预测，返回预测值和原始不确定性"""
#         batch_size = 1024
#         n = len(Xc)
#         preds = []
#         uncs = []
#         with torch.no_grad():
#             for i in range(0, n, batch_size):
#                 cx = torch.FloatTensor(Xc[i:i+batch_size]).to(device)
#                 ix = torch.FloatTensor(Xi[i:i+batch_size]).to(device)
#                 ax = torch.LongTensor(Xa[i:i+batch_size]).to(device)
#                 out = self.cftnet(cx, ix, ax)
#                 gamma = out[:, 0]
#                 v = out[:, 1]
#                 alpha = out[:, 2]
#                 beta = out[:, 3]
#                 pred_time = torch.expm1(gamma)
#                 var = beta / (v * (alpha - 1) + 1e-6)
#                 unc = torch.sqrt(var + 1e-6)
#                 preds.append(pred_time.cpu().numpy())
#                 uncs.append(unc.cpu().numpy())
#         return np.concatenate(preds), np.concatenate(uncs)
    
#     def calibrate_cftnet(self):
#         """事后校准，得到缩放因子"""
#         print("\n--- CFT-Net 事后校准 ---")
#         pred_val, unc_val = self.predict_cftnet(self.Xc_val, self.Xi_val, self.Xa_val)
#         picp_val_raw, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val, 0.8)
#         print(f"验证集原始PICP: {picp_val_raw:.1f}%")
#         self.calibration_scale = post_hoc_calibration(self.y_val_orig, pred_val, unc_val, target_coverage=0.8)
#         print(f"学习到的缩放因子: {self.calibration_scale:.3f}")
#         return self.calibration_scale
    
#     def evaluate_cftnet(self):
#         """评估CFT-Net，收集指标"""
#         pred_test, unc_test_raw = self.predict_cftnet(self.Xc_test, self.Xi_test, self.Xa_test)
#         unc_test_cal = unc_test_raw * self.calibration_scale
#         errors_test = np.abs(self.y_test_orig - pred_test)
        
#         # 指标
#         mae = mean_absolute_error(self.y_test_orig, pred_test)
#         rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_test))
#         smape = calculate_smape(self.y_test_orig, pred_test)
#         corr, _ = spearmanr(unc_test_cal, errors_test)
#         corr = 0.0 if np.isnan(corr) else corr
#         picp, mpiw = calculate_picp_mpiw(self.y_test_orig, pred_test, unc_test_cal, 0.8)
#         ece = calculate_ece_quantile(errors_test, unc_test_cal)
        
#         # 推理时间
#         infer_time = self.measure_inference_time_cftnet()
        
#         self.results['CFT-Net'] = {
#             'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'Corr': corr,
#             'PICP_80': picp, 'MPIW_80': mpiw, 'ECE': ece,
#             'Inference_ms': infer_time * 1000,
#             'predictions': pred_test,
#             'uncertainties': unc_test_cal
#         }
#         print(f"CFT-Net 测试指标: sMAPE={smape:.2f}%, Corr={corr:.3f}, PICP={picp:.1f}%, 推理时间={infer_time*1000:.3f}ms")
#         return self.results['CFT-Net']
    
#     def measure_inference_time_cftnet(self):
#         """测量CFT-Net推理时间（秒/样本）"""
#         batch_size = 1024
#         n = len(self.Xc_test)
#         times = []
#         with torch.no_grad():
#             for i in range(0, n, batch_size):
#                 cx = torch.FloatTensor(self.Xc_test[i:i+batch_size]).to(device)
#                 ix = torch.FloatTensor(self.Xi_test[i:i+batch_size]).to(device)
#                 ax = torch.LongTensor(self.Xa_test[i:i+batch_size]).to(device)
#                 if device.type == 'cuda':
#                     torch.cuda.synchronize()
#                     start = time.perf_counter()
#                 else:
#                     start = time.perf_counter()
#                 _ = self.cftnet(cx, ix, ax)
#                 if device.type == 'cuda':
#                     torch.cuda.synchronize()
#                 times.append(time.perf_counter() - start)
#         total_time = np.sum(times)
#         return total_time / n

#     # def measure_inference_time_cftnet(self):
#     #     """在CPU上测量CFT-Net推理时间（秒/样本）"""
#     #     # 将模型移至CPU
#     #     self.cftnet.cpu()
#     #     # 数据转为CPU Tensor
#     #     Xc_cpu = torch.FloatTensor(self.Xc_test)
#     #     Xi_cpu = torch.FloatTensor(self.Xi_test)
#     #     Xa_cpu = torch.LongTensor(self.Xa_test)
    
#     #     batch_size = 1024
#     #     n = len(self.Xc_test)
#     #     times = []
    
#     #     with torch.no_grad():
#     #         for i in range(0, n, batch_size):
#     #             cx = Xc_cpu[i:i+batch_size]
#     #             ix = Xi_cpu[i:i+batch_size]
#     #             ax = Xa_cpu[i:i+batch_size]
            
#     #             start = time.perf_counter()
#     #             _ = self.cftnet(cx, ix, ax)
#     #             times.append(time.perf_counter() - start)
    
#     #     # 将模型移回GPU（后续计算仍可在GPU上进行）
#     #     self.cftnet.to(device)
#     #     total_time = np.sum(times)
#     #     return total_time / n
#     def train_baselines(self):
#         """训练并评估基线模型"""
#         models = {
#             'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.seed, n_jobs=-1),
#             'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.seed, n_jobs=-1),
#             'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=self.seed, n_jobs=-1, verbose=-1)
#         }
#         print("\n训练基线模型...")
#         for name, model in models.items():
#             print(f"  {name}...")
#             # 训练
#             start = time.perf_counter()
#             model.fit(self.X_train_comb, self.y_train_log)
#             train_time = time.perf_counter() - start
#             # 预测
#             pred_log = model.predict(self.X_test_comb)
#             pred_orig = np.expm1(pred_log)
#             # 指标
#             mae = mean_absolute_error(self.y_test_orig, pred_orig)
#             rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_orig))
#             smape = calculate_smape(self.y_test_orig, pred_orig)
#             # 推理时间
#             infer_time = self.measure_inference_time_sklearn(model, self.X_test_comb)
#             self.results[name] = {
#                 'MAE': mae, 'RMSE': rmse, 'sMAPE': smape,
#                 'Corr': None, 'PICP_80': None, 'MPIW_80': None, 'ECE': None,
#                 'Inference_ms': infer_time * 1000,
#                 'predictions': pred_orig
#             }
#             print(f"    sMAPE={smape:.2f}%, 推理时间={infer_time*1000:.3f}ms")
    
#     def measure_inference_time_sklearn(self, model, X):
#         """测量sklearn风格模型的推理时间"""
#         batch_size = 1024
#         n = len(X)
#         times = []
#         for i in range(0, n, batch_size):
#             X_batch = X[i:i+batch_size]
#             start = time.perf_counter()
#             _ = model.predict(X_batch)
#             times.append(time.perf_counter()- start)
#         total_time = np.sum(times)
#         return total_time / n
    

#     def generate_radar_chart(self):
#         """绘制雷达图：四个维度：精度(sMAPE), 风险感知(Corr), 可靠性(PICP), 轻量化(推理时间)"""
#         # 提取数据
#         models = list(self.results.keys())
#         smapes = [self.results[m]['sMAPE'] for m in models]
#         corrs = [self.results[m]['Corr'] if self.results[m]['Corr'] is not None else 0 for m in models]
#         picps = [self.results[m]['PICP_80'] if self.results[m]['PICP_80'] is not None else 0 for m in models]
#         inf_times = [self.results[m]['Inference_ms'] for m in models]
        
#         # 归一化（0~1，越高越好）
#         # sMAPE: 越小越好 -> 取倒数或1-归一化，这里用 1 - smape/50 (假设最大50%)
#         smape_max = 50  # 可根据数据调整
#         smape_norm = [max(0, 1 - s/smape_max) for s in smapes]
#         # Corr: 越高越好，直接归一化到0~1（假设最大1）
#         corr_norm = [c if c is not None else 0 for c in corrs]
#         # PICP: 越高越好，直接除以100
#         picp_norm = [p/100 for p in picps]
#         # 推理时间: 越小越好，使用倒数并归一化，假设最慢为1ms（根据数据调整）
#         inf_max = max(inf_times) if max(inf_times) > 0 else 1
#         inf_norm = [1 - t/inf_max for t in inf_times]
        
#         # 雷达图数据
#         categories = ['精度 (sMAPE↓)', '风险感知 (Corr↑)', '可靠性 (PICP↑)', '轻量化 (推理↓)']
#         N = len(categories)
#         angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#         angles += angles[:1]  # 闭合
        
#         fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
#         colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
#         for i, model in enumerate(models):
#             values = [smape_norm[i], corr_norm[i], picp_norm[i], inf_norm[i]]
#             values += values[:1]
#             ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
#             ax.fill(angles, values, alpha=0.15, color=colors[i])
        
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(categories, fontsize=12)
#         ax.set_ylim(0, 1)
#         ax.set_title('模型综合能力雷达图', fontsize=14, fontweight='bold', pad=20)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
#         plt.tight_layout()
#         plt.savefig('evaluation_results/radar_chart.png', dpi=300, bbox_inches='tight')
#         plt.show()
#         print("雷达图已保存至 evaluation_results/radar_chart.png")
    
#     def generate_comparison_table(self):
#         """生成对比表格（CSV）"""
#         rows = []
#         for model, metrics in self.results.items():
#             row = {
#                 'Model': model,
#                 'sMAPE (%)': f"{metrics['sMAPE']:.2f}",
#                 'MAE (s)': f"{metrics['MAE']:.2f}",
#                 'RMSE (s)': f"{metrics['RMSE']:.2f}",
#                 'Corr': f"{metrics['Corr']:.3f}" if metrics['Corr'] is not None else '-',
#                 'PICP-80 (%)': f"{metrics['PICP_80']:.1f}" if metrics['PICP_80'] is not None else '-',
#                 'MPIW (s)': f"{metrics['MPIW_80']:.2f}" if metrics['MPIW_80'] is not None else '-',
#                 'ECE': f"{metrics['ECE']:.2f}" if metrics['ECE'] is not None else '-',
#                 'Inference (ms)': f"{metrics['Inference_ms']:.3f}"
#             }
#             rows.append(row)
#         df = pd.DataFrame(rows)
#         df.to_csv('evaluation_results/comparison_table.csv', index=False)
#         print("\n对比表格已保存至 evaluation_results/comparison_table.csv")
#         print("\n" + df.to_string(index=False))
#         return df
    
#     def plot_calibration_curve(self):
#         """绘制校准曲线（CFT-Net）"""
#         if 'CFT-Net' not in self.results:
#             return
#         preds = self.results['CFT-Net']['predictions']
#         uncs = self.results['CFT-Net']['uncertainties']
#         errors = np.abs(self.y_test_orig - preds)
        
#         n_bins = 10
#         quantiles = np.linspace(0, 100, n_bins + 1)
#         bin_edges = np.percentile(uncs, quantiles)
#         bin_edges[-1] += 1e-8
#         bin_centers = []
#         avg_errors = []
#         for i in range(n_bins):
#             in_bin = (uncs >= bin_edges[i]) & (uncs < bin_edges[i+1])
#             if i == n_bins - 1:
#                 in_bin = (uncs >= bin_edges[i]) & (uncs <= bin_edges[i+1])
#             if in_bin.sum() > 0:
#                 bin_centers.append(np.mean(uncs[in_bin]))
#                 avg_errors.append(np.mean(errors[in_bin]))
#             else:
#                 bin_centers.append(np.nan)
#                 avg_errors.append(np.nan)
        
#         plt.figure(figsize=(8, 6))
#         plt.plot(bin_centers, avg_errors, 'o-', label='实际误差')
#         plt.plot(bin_centers, bin_centers, 'r--', label='完美校准')
#         plt.xlabel('平均不确定性 (s)')
#         plt.ylabel('平均绝对误差 (s)')
#         plt.title('CFT-Net 校准曲线')
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.savefig('evaluation_results/calibration_curve.png', dpi=300)
#         plt.show()
#         print("校准曲线已保存")
    
#     def plot_prediction_intervals(self):
#         """绘制前100个样本的预测区间"""
#         if 'CFT-Net' not in self.results:
#             return
#         preds = self.results['CFT-Net']['predictions'][:100]
#         uncs = self.results['CFT-Net']['uncertainties'][:100]
#         y_true = self.y_test_orig[:100]
        
#         z = 1.28
#         lower = preds - z * uncs
#         upper = preds + z * uncs
        
#         plt.figure(figsize=(14, 6))
#         x = np.arange(len(preds))
#         plt.fill_between(x, lower, upper, alpha=0.3, color='blue', label='80%预测区间')
#         plt.plot(x, preds, 'b-', linewidth=1.5, label='预测值')
#         plt.scatter(x, y_true, c='black', s=20, zorder=5, label='真实值')
#         covered = (y_true >= lower) & (y_true <= upper)
#         plt.scatter(x[~covered], y_true[~covered], c='red', s=50, marker='x', label='未覆盖')
#         plt.xlabel('样本索引')
#         plt.ylabel('传输时间 (s)')
#         plt.title('CFT-Net 预测区间可视化')
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.savefig('evaluation_results/prediction_intervals.png', dpi=300)
#         plt.show()
#         print("预测区间图已保存")
    
#     def run_full_evaluation(self):
#         """执行完整评估流程"""
#         self.calibrate_cftnet()
#         self.evaluate_cftnet()
#         self.train_baselines()
#         self.generate_comparison_table()
#         self.generate_radar_chart()
#         self.plot_calibration_curve()
#         self.plot_prediction_intervals()
#         print("\n✅ 所有评估完成！结果保存在 evaluation_results/ 目录")

# # ==============================================================================
# # 5. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     # 指定训练好的CFT-Net模型路径（请根据实际修改）
#     MODEL_PATH = "cts_fixed_0217_1727_seed42.pth"  # 您的最佳模型文件名
    
#     if not os.path.exists(MODEL_PATH):
#         print(f"错误：找不到模型文件 {MODEL_PATH}")
#         exit(1)
    
#     evaluator = ModelEvaluator(MODEL_PATH, seed=SEED)
#     evaluator.run_full_evaluation()



"""
CFT-Net 完整对比评测脚本 (最终视觉增强版)
修改内容：
1. 雷达图改为 2x2 分图显示 (Separated Radar Charts)
2. 雷达图顶点增加真实数值标注
3. 保持数据加载逻辑的一致性 (RMSE 修复版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr, norm
from scipy.optimize import brentq
from collections import Counter
import warnings
import platform

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 基础配置
# ==============================================================================
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300 

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

os.makedirs("evaluation_results", exist_ok=True)

# ==============================================================================
# 1. 模型定义 (保持不变)
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

class CompactCFTNet(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
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
        v = F.softplus(out[:, 1]) + 0.5
        alpha = F.softplus(out[:, 2]) + 1.5
        beta = F.softplus(out[:, 3]) + 1.0
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 2. 评估工具函数
# ==============================================================================
def calculate_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100

def calculate_picp_mpiw(y_true, y_pred, unc, confidence=0.8):
    z = norm.ppf((1 + confidence) / 2)
    lower, upper = y_pred - z * unc, y_pred + z * unc
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    return picp, mpiw

def calculate_ece_quantile(errors, uncertainties, n_bins=10):
    if len(errors) == 0: return 0.0
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(uncertainties, quantiles)
    bin_edges[-1] += 1e-8
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1]) if i == n_bins - 1 else (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if in_bin.sum() > 0:
            avg_unc = uncertainties[in_bin].mean()
            avg_err = errors[in_bin].mean()
            ece += np.abs(avg_err - avg_unc) * (in_bin.sum() / len(errors))
    return ece

def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8):
    def get_picp(s):
        lower, upper = y_pred - s * unc_raw, y_pred + s * unc_raw
        return np.mean((y_true >= lower) & (y_true <= upper))
    try:
        return brentq(lambda s: get_picp(s) - target_coverage, 0.1, 100.0)
    except:
        return 50.0

# ==============================================================================
# 3. 数据加载 (严格保持原始逻辑)
# ==============================================================================
def load_preprocessing_objects():
    with open('preprocessing_objects.pkl', 'rb') as f:
        return pickle.load(f)

def load_data():
    df_exp = pd.read_excel("cts_data.xlsx")
    df_feat = pd.read_csv("image_features_database.csv")
    
    rename_map = {
        "image": "image_name", "method": "algo_name", 
        "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", 
        "mem_limit": "mem_limit_mb"
    }
    df_exp = df_exp.rename(columns=rename_map)
    if 'total_time' not in df_exp.columns: 
        cols = [c for c in df_exp.columns if 'total_tim' in c]
        if cols: df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
            
    df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
    df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    
    cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
    cols_i = [c for c in ['total_size_mb', 'avg_layer_entropy', 'entropy_std', 
                          'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio'] if c in df.columns]
    
    Xc = df[cols_c].values
    Xi = df[cols_i].values
    y = np.log1p(df['total_time'].values)
    algo_names = df['algo_name'].values
    return Xc, Xi, algo_names, y, cols_c, cols_i, df['total_time'].values

# ==============================================================================
# 4. 评估主类
# ==============================================================================
class ModelEvaluator:
    def __init__(self, model_path, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.prep = load_preprocessing_objects()
        
        Xc, Xi, algos, y_log, self.cols_c, self.cols_i, y_orig = load_data()
        N = len(y_log)
        idx = np.random.RandomState(seed).permutation(N)
        
        n_tr = int(N * 0.7)
        n_val = int(N * 0.15)
        self.tr_idx, self.val_idx, self.te_idx = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]
        
        self.Xc_test = self.prep['scaler_c'].transform(Xc[self.te_idx])
        self.Xi_test = self.prep['scaler_i'].transform(Xi[self.te_idx])
        
        def safe_transform(labels):
            known = set(self.prep['enc'].classes_)
            default = self.prep['default_algo_idx']
            return np.array([self.prep['enc'].transform([l])[0] if l in known else default for l in labels])
            
        self.Xa_test = safe_transform(algos[self.te_idx])
        self.y_test_orig = y_orig[self.te_idx]
        
        self.Xc_train = self.prep['scaler_c'].transform(Xc[self.tr_idx])
        self.Xi_train = self.prep['scaler_i'].transform(Xi[self.tr_idx])
        self.Xa_train = self.prep['enc'].transform(algos[self.tr_idx])
        self.y_train_log = y_log[self.tr_idx]
        
        self.X_train_comb = np.hstack([self.Xc_train, self.Xi_train, self.Xa_train.reshape(-1,1)])
        self.X_test_comb = np.hstack([self.Xc_test, self.Xi_test, self.Xa_test.reshape(-1,1)])
        
        checkpoint = torch.load(model_path, map_location=device)
        embed_dim = checkpoint.get('config', {}).get('embed_dim', 32)
        
        self.cftnet = CompactCFTNet(len(self.cols_c), len(self.cols_i), len(self.prep['enc'].classes_), embed_dim=embed_dim).to(device)
        self.cftnet.load_state_dict(checkpoint['model_state_dict'])
        self.cftnet.eval()
        
        self.Xc_val = self.prep['scaler_c'].transform(Xc[self.val_idx])
        self.Xi_val = self.prep['scaler_i'].transform(Xi[self.val_idx])
        self.Xa_val = safe_transform(algos[self.val_idx])
        self.y_val_orig = y_orig[self.val_idx]
        
        self.results = {}

    def predict_cftnet(self, Xc, Xi, Xa):
        batch_size = 1024
        preds, uncs = [], []
        with torch.no_grad():
            for i in range(0, len(Xc), batch_size):
                cx = torch.FloatTensor(Xc[i:i+batch_size]).to(device)
                ix = torch.FloatTensor(Xi[i:i+batch_size]).to(device)
                ax = torch.LongTensor(Xa[i:i+batch_size]).to(device)
                out = self.cftnet(cx, ix, ax)
                gamma, v, alpha, beta = out[:,0], out[:,1], out[:,2], out[:,3]
                preds.append(torch.expm1(gamma).cpu().numpy())
                uncs.append(torch.sqrt(beta / (v * (alpha - 1) + 1e-6)).cpu().numpy())
        return np.concatenate(preds), np.concatenate(uncs)

    def measure_inference_cpu(self):
        model_cpu = self.cftnet.cpu()
        cx = torch.FloatTensor(self.Xc_test[0:1])
        ix = torch.FloatTensor(self.Xi_test[0:1])
        ax = torch.LongTensor(self.Xa_test[0:1])
        with torch.no_grad():
            for _ in range(50): _ = model_cpu(cx, ix, ax)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(1000): _ = model_cpu(cx, ix, ax)
        self.cftnet.to(device)
        return (time.perf_counter() - start) 

    def run_full_evaluation(self):
        print("评估 CFT-Net...")
        p_val, u_val = self.predict_cftnet(self.Xc_val, self.Xi_val, self.Xa_val)
        scale = post_hoc_calibration(self.y_val_orig, p_val, u_val)
        
        p_test, u_test_raw = self.predict_cftnet(self.Xc_test, self.Xi_test, self.Xa_test)
        u_test = u_test_raw * scale
        
        metrics = self._calc_metrics(self.y_test_orig, p_test, u_test)
        metrics['Inference_ms'] = self.measure_inference_cpu()
        self.results['CFT-Net'] = {**metrics, 'preds': p_test, 'uncs': u_test}
        
        print("训练基线模型...")
        baselines = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=SEED),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=SEED),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=SEED, verbose=-1)
        }
        
        for name, model in baselines.items():
            model.fit(self.X_train_comb, self.y_train_log)
            p_test = np.expm1(model.predict(self.X_test_comb))
            
            x_sample = self.X_test_comb[0:1]
            start = time.perf_counter()
            for _ in range(1000): model.predict(x_sample)
            lat = (time.perf_counter() - start)
            
            metrics = self._calc_metrics(self.y_test_orig, p_test)
            metrics['Inference_ms'] = lat
            self.results[name] = {**metrics, 'preds': p_test}
            
        self.save_csv()
        self.plot_radar_chart()          # 2x2 独立雷达图
        self.plot_calibration_curve()
        self.plot_prediction_intervals()
        self.plot_predicted_vs_actual()  # 预测vs真实散点图
        print("\n✅ 所有评估完成！结果保存在 evaluation_results/ 目录")

    def _calc_metrics(self, y, p, u=None):
        e = np.abs(y - p)
        res = {
            'MAE': mean_absolute_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'sMAPE': calculate_smape(y, p),
            'Corr': spearmanr(u, e)[0] if u is not None else None,
            'PICP': calculate_picp_mpiw(y, p, u, 0.8)[0] if u is not None else None,
            'MPIW': calculate_picp_mpiw(y, p, u, 0.8)[1] if u is not None else None,
            'ECE': calculate_ece_quantile(e, u) if u is not None else None
        }
        return res

    def save_csv(self):
        rows = []
        for m, res in self.results.items():
            row = {'Model': m}
            for k, v in res.items():
                if k not in ['preds', 'uncs']:
                    row[k] = f"{v:.4f}" if v is not None else "N/A"
            rows.append(row)
        pd.DataFrame(rows).to_csv('evaluation_results/final_metrics.csv', index=False)

    def plot_predicted_vs_actual(self):
        print("绘制预测对比散点图...")
        models = list(self.results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        y_true = self.y_test_orig
        max_val = np.percentile(y_true, 98)
        
        for idx, model_name in enumerate(models):
            if idx >= 4: break 
            ax = axes[idx]
            preds = self.results[model_name]['preds']
            ax.scatter(y_true, preds, alpha=0.3, s=10, color='#1f77b4', label='样本点')
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测')
            rmse = self.results[model_name]['RMSE']
            r2 = r2_score(y_true, preds)
            ax.text(0.05, 0.95, f'RMSE = {rmse:.2f}\n$R^2$ = {r2:.3f}', 
                    transform=ax.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("真实传输时间 (s)")
            ax.set_ylabel("预测传输时间 (s)")
            ax.set_xlim(-5, max_val)
            ax.set_ylim(-5, max_val)
            ax.grid(True, alpha=0.2)
            if idx == 0: ax.legend(loc='lower right') 
        plt.tight_layout()
        plt.savefig('evaluation_results/pred_vs_actual.png', dpi=300)
        plt.close()

    # ==========================================================================
    # 🌟 核心修改：分图雷达图 (Separated Radar Charts)
    # ==========================================================================
    def plot_radar_chart(self):
        print("绘制分图雷达图...")
        categories = ['精度\n(sMAPE)', '风险感知\n(Corr)', '可靠性\n(PICP)', '轻量化\n(Time)']
        models = list(self.results.keys())
        
        # 归一化逻辑
        data = []
        raw_values = [] # 存储原始值用于标注
        
        for m in models:
            res = self.results[m]
            smape = res['sMAPE']
            corr = res['Corr'] if res['Corr'] is not None else 0
            picp = res['PICP'] if res['PICP'] is not None else 0
            time = res['Inference_ms']
            
            # 记录原始值 (用于 Label)
            raw = [
                f"{smape:.1f}%", 
                f"{corr:.2f}" if corr>0 else "N/A", 
                f"{picp:.1f}%" if picp>0 else "N/A", 
                f"{time:.3f}ms"
            ]
            raw_values.append(raw)
            
            # 计算得分 (用于画图形状, 0.1~1.0)
            s_score = max(0.1, 1 - smape/15.0) # 越准越好
            c_score = max(0.1, corr)           # 越大越好
            p_score = max(0.1, picp / 100.0)   # 越大越好
            t_score = max(0.1, 1 - time/0.06)  # 越快越好 (基线设为0.06ms)
            
            data.append([s_score, c_score, p_score, t_score])
            
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # 创建 2x2 Subplots, 且都是 Polar
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        
        for i, (ax, model_name) in enumerate(zip(axes, models)):
            if i >= 4: break
            values = data[i]
            labels = raw_values[i]
            
            val_plot = values + values[:1]
            
            # 1. 画线和填充
            ax.plot(angles, val_plot, linewidth=2, linestyle='-', color=colors[i])
            ax.fill(angles, val_plot, alpha=0.3, color=colors[i]) # 增加一点透明度让它好看
            
            # 2. 顶点数值标注 (Label)
            for j, (angle, v, text) in enumerate(zip(angles[:-1], values, labels)):
                # 稍微向外推一点，防止压线
                ax.text(angle, v + 0.15, text, ha='center', va='center', 
                        fontsize=10, fontweight='bold', color=colors[i])

            # 3. 设置样式
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1.3) # 留出空间给Label
            ax.set_yticks([])   # 隐藏同心圆刻度
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_title(model_name, y=1.1, fontsize=13, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.3) # 调整间距
        plt.savefig('evaluation_results/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_calibration_curve(self):
        if 'CFT-Net' not in self.results: return
        preds, uncs = self.results['CFT-Net']['preds'], self.results['CFT-Net']['uncs']
        errors = np.abs(self.y_test_orig - preds)
        n_bins = 10
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(uncs, quantiles)
        bin_edges[-1] += 1e-8
        centers, avg_errs = [], []
        for i in range(n_bins):
            mask = (uncs >= bin_edges[i]) & (uncs <= bin_edges[i+1]) if i == n_bins-1 else (uncs >= bin_edges[i]) & (uncs < bin_edges[i+1])
            if mask.sum() > 0:
                centers.append(uncs[mask].mean())
                avg_errs.append(errors[mask].mean())
        plt.figure(figsize=(7, 6))
        plt.plot(centers, avg_errs, 'o-', color='#1f77b4', linewidth=2, label='Actual Error')
        plt.plot([0, max(centers)], [0, max(centers)], 'r--', linewidth=2, label='Ideal Calibration')
        plt.xlabel('Predicted Uncertainty (s)')
        plt.ylabel('Mean Absolute Error (s)')
        plt.title('Uncertainty Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('evaluation_results/calibration_curve.png', dpi=300)
        plt.close()

    def plot_prediction_intervals(self):
        if 'CFT-Net' not in self.results: return
        preds = self.results['CFT-Net']['preds'][:100]
        uncs = self.results['CFT-Net']['uncs'][:100]
        y_true = self.y_test_orig[:100]
        lower, upper = preds - 1.28*uncs, preds + 1.28*uncs
        x = np.arange(len(preds))
        plt.figure(figsize=(15, 6))
        plt.fill_between(x, lower, upper, alpha=0.2, color='blue', label='80% CI')
        plt.plot(x, preds, 'b-', linewidth=1.5, alpha=0.8, label='Prediction')
        plt.scatter(x, y_true, c='black', s=15, zorder=5, label='Ground Truth')
        mask = (y_true < lower) | (y_true > upper)
        plt.scatter(x[mask], y_true[mask], c='red', s=40, marker='x', label='Missed')
        plt.xlabel('Sample Index')
        plt.ylabel('Transfer Time (s)')
        plt.title('Prediction Interval Coverage')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_results/prediction_intervals.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    MODEL_PATH = "cts_fixed_0217_1727_seed42.pth" 
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        exit()
    evaluator = ModelEvaluator(MODEL_PATH, seed=SEED)
    evaluator.run_full_evaluation()
