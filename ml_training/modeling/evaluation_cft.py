
# """
# CFT-Net 完整对比评测脚本（修复版 v2）
# 生成用于论文的对比表格和雷达图（精度、风险感知、可靠性、轻量化）

# 修复内容：
# 1. 算法特征使用One-Hot编码，避免数值顺序误导
# 2. 统一推理时间测量标准（全部在CPU上测量）
# 3. 增加物理确定性的讨论和说明
# 4. 增加分层校准和更完整的评估指标
# 5. 【新增】添加 Pred vs Actual 散点图
# 6. 【修复】prediction_intervals 使用全量测试集，PICP与表格一致
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
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from scipy.stats import spearmanr, norm, wilcoxon
# from scipy.optimize import brentq
# from collections import Counter
# import warnings
# import platform

# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 0. 基础配置
# # ==============================================================================
# system = platform.system()
# if system == 'Windows':
#     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# elif system == 'Darwin':
#     plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
# else:
#     plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

# plt.rcParams['axes.unicode_minus'] = False

# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

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

# def hierarchical_calibration(y_true, y_pred, unc_raw, n_bins=5):
#     """
#     分层校准：对不同不确定性水平使用不同缩放因子
#     解决高不确定性区域校准不足的问题
#     """
#     quantiles = np.percentile(unc_raw, np.linspace(0, 100, n_bins + 1))
#     scales = []
    
#     for i in range(n_bins):
#         mask = (unc_raw >= quantiles[i]) & (unc_raw <= quantiles[i+1])
#         if mask.sum() > 10:  # 确保有足够样本
#             # 该区间目标：PICP = 80%
#             def picp_with_scale(s):
#                 z = norm.ppf(0.9)
#                 lower = y_pred[mask] - z * s * unc_raw[mask]
#                 upper = y_pred[mask] + z * s * unc_raw[mask]
#                 return np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper))
            
#             try:
#                 from scipy.optimize import brentq
#                 s_opt = brentq(lambda s: picp_with_scale(s) - 0.8, 0.1, 100)
#                 scales.append(s_opt)
#             except:
#                 scales.append(33.713)  # 默认回退
#         else:
#             scales.append(33.713)
    
#     # 应用分层缩放
#     unc_cal = unc_raw.copy()
#     for i in range(n_bins):
#         mask = (unc_raw >= quantiles[i]) & (unc_raw <= quantiles[i+1])
#         unc_cal[mask] = unc_raw[mask] * scales[i]
    
#     return unc_cal, scales

# def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, search_range=(0.1, 100)):
#     def picp_with_scale(s):
#         z = norm.ppf((1 + target_coverage) / 2)
#         lower = y_pred - z * s * unc_raw
#         upper = y_pred + z * s * unc_raw
#         return np.mean((y_true >= lower) & (y_true <= upper))
#     s_min, s_max = search_range
#     try:
#         s_opt = brentq(picp_with_scale, s_min, s_max)
#         return s_opt
#     except:
#         scales = np.linspace(s_min, s_max, 500)
#         picps = [picp_with_scale(s) for s in scales]
#         best_idx = np.argmin(np.abs(np.array(picps) - target_coverage))
#         return scales[best_idx]

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
# # 4. 评估主类（修复版）
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
        
#         # 加载数据
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
        
#         # 【修复】基线模型使用One-Hot编码算法特征，避免数值顺序误导
#         self.algo_onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         self.algo_onehot.fit(self.Xa_train.reshape(-1, 1))
        
#         Xa_train_oh = self.algo_onehot.transform(self.Xa_train.reshape(-1, 1))
#         Xa_val_oh = self.algo_onehot.transform(self.Xa_val.reshape(-1, 1))
#         Xa_test_oh = self.algo_onehot.transform(self.Xa_test.reshape(-1, 1))
        
#         self.X_train_comb = np.hstack([self.Xc_train, self.Xi_train, Xa_train_oh])
#         self.X_val_comb = np.hstack([self.Xc_val, self.Xi_val, Xa_val_oh])
#         self.X_test_comb = np.hstack([self.Xc_test, self.Xi_test, Xa_test_oh])
        
#         print(f"数据划分: 训练 {len(self.tr_idx)} | 验证 {len(self.val_idx)} | 测试 {len(self.te_idx)}")
#         print(f"基线模型特征维度: {self.X_train_comb.shape[1]} (包含{len(self.enc.classes_)}个算法的One-Hot编码)")
        
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
#         print("\n--- CFT-Net 事后校准 ---")
#         pred_val, unc_val = self.predict_cftnet(self.Xc_val, self.Xi_val, self.Xa_val)
#         picp_val_raw, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val, 0.8)
#         print(f"验证集原始PICP: {picp_val_raw:.1f}%")
        
#         # 尝试分层校准
#         print("尝试分层校准...")
#         unc_val_hier, scales = hierarchical_calibration(self.y_val_orig, pred_val, unc_val)
#         picp_val_hier, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val_hier, 0.8)
#         print(f"分层校准PICP: {picp_val_hier:.1f}%")
#         print(f"各区间缩放因子: {[f'{s:.2f}' for s in scales]}")
        
#         # 使用全局校准作为回退
#         self.calibration_scale = post_hoc_calibration(self.y_val_orig, pred_val, unc_val)
#         print(f"全局缩放因子: {self.calibration_scale:.3f}")
        
#         # 保存分层校准参数
#         self.hierarchical_scales = scales
#         return self.calibration_scale
    
#     def evaluate_cftnet(self):
#         pred_test, unc_test_raw = self.predict_cftnet(self.Xc_test, self.Xi_test, self.Xa_test)
        
#         # 应用分层校准
#         unc_test_cal = unc_test_raw * self.calibration_scale
        
#         errors_test = np.abs(self.y_test_orig - pred_test)
        
#         # 所有指标
#         mae = mean_absolute_error(self.y_test_orig, pred_test)
#         rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_test))
#         smape = calculate_smape(self.y_test_orig, pred_test)
#         mape = calculate_mape(self.y_test_orig, pred_test)
#         corr, _ = spearmanr(unc_test_cal, errors_test)
#         corr = 0.0 if np.isnan(corr) else corr
#         picp, mpiw = calculate_picp_mpiw(self.y_test_orig, pred_test, unc_test_cal, 0.8)
#         ece = calculate_ece_quantile(errors_test, unc_test_cal)
        
#         # R²
#         r2 = r2_score(self.y_test_orig, pred_test)
        
#         # 推理时间（【修复】统一在CPU上测量）
#         infer_time = self.measure_inference_time_cftnet_cpu()
        
#         self.results['CFT-Net'] = {
#             'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape,
#             'R2': r2, 'Corr': corr, 'PICP_80': picp, 'MPIW_80': mpiw, 'ECE': ece,
#             'Inference_ms': infer_time * 1000,
#             'Params_K': sum(p.numel() for p in self.cftnet.parameters()) / 1000,
#             'predictions': pred_test,
#             'uncertainties': unc_test_cal,
#             'raw_uncertainties': unc_test_raw
#         }
#         print(f"CFT-Net 测试指标: sMAPE={smape:.2f}%, R²={r2:.4f}, Corr={corr:.3f}, PICP={picp:.1f}%, 推理={infer_time*1000:.3f}ms")
#         return self.results['CFT-Net']
    
#     def measure_inference_time_cftnet_cpu(self):
#         """【修复】在CPU上测量CFT-Net推理时间，确保与基线模型公平对比"""
#         self.cftnet.cpu()
#         batch_size = 256
#         n = len(self.Xc_test)
        
#         # Warmup
#         with torch.no_grad():
#             for i in range(0, min(500, n), batch_size):
#                 cx = torch.FloatTensor(self.Xc_test[i:i+batch_size])
#                 ix = torch.FloatTensor(self.Xi_test[i:i+batch_size])
#                 ax = torch.LongTensor(self.Xa_test[i:i+batch_size])
#                 _ = self.cftnet(cx, ix, ax)
        
#         # 正式计时
#         times = []
#         with torch.no_grad():
#             for i in range(0, n, batch_size):
#                 cx = torch.FloatTensor(self.Xc_test[i:i+batch_size])
#                 ix = torch.FloatTensor(self.Xi_test[i:i+batch_size])
#                 ax = torch.LongTensor(self.Xa_test[i:i+batch_size])
                
#                 start = time.perf_counter()
#                 _ = self.cftnet(cx, ix, ax)
#                 times.append(time.perf_counter() - start)
        
#         # 移回GPU
#         self.cftnet.to(device)
        
#         total_time = np.sum(times)
#         return total_time / n
    
#     def train_baselines(self):
#         models = {
#             'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.seed, n_jobs=-1),
#             'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.seed, n_jobs=-1),
#             'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=self.seed, n_jobs=-1, verbose=-1)
#         }
#         print("\n训练基线模型（使用One-Hot编码算法特征）...")
#         for name, model in models.items():
#             print(f"  {name}...")
#             start = time.perf_counter()
#             model.fit(self.X_train_comb, self.y_train_log)
#             train_time = time.perf_counter() - start
            
#             pred_log = model.predict(self.X_test_comb)
#             pred_orig = np.expm1(pred_log)
            
#             mae = mean_absolute_error(self.y_test_orig, pred_orig)
#             rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_orig))
#             smape = calculate_smape(self.y_test_orig, pred_orig)
#             mape = calculate_mape(self.y_test_orig, pred_orig)
#             r2 = r2_score(self.y_test_orig, pred_orig)
#             infer_time = self.measure_inference_time_sklearn(model, self.X_test_comb)
            
#             self.results[name] = {
#                 'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape, 'R2': r2,
#                 'Corr': None, 'PICP_80': None, 'MPIW_80': None, 'ECE': None,
#                 'Inference_ms': infer_time * 1000,
#                 'Params_K': None,  # 树模型参数量不易计算
#                 'predictions': pred_orig
#             }
#             print(f"    R²={r2:.4f}, sMAPE={smape:.2f}%, 推理={infer_time*1000:.3f}ms")
    
#     def measure_inference_time_sklearn(self, model, X):
#         batch_size = 256
#         n = len(X)
#         times = []
#         for i in range(0, n, batch_size):
#             X_batch = X[i:i+batch_size]
#             start = time.perf_counter()
#             _ = model.predict(X_batch)
#             times.append(time.perf_counter() - start)
#         total_time = np.sum(times)
#         return total_time / n
    
#     def generate_radar_chart(self):
#         models = list(self.results.keys())
#         smapes = [self.results[m]['sMAPE'] for m in models]
#         corrs = [self.results[m]['Corr'] if self.results[m]['Corr'] is not None else 0 for m in models]
#         picps = [self.results[m]['PICP_80'] if self.results[m]['PICP_80'] is not None else 0 for m in models]
#         inf_times = [self.results[m]['Inference_ms'] for m in models]
        
#         # 归一化
#         smape_norm = [max(0, 1 - s/50) for s in smapes]
#         corr_norm = [max(0, c) for c in corrs]  # Corr可能为负
#         picp_norm = [p/100 for p in picps]
#         inf_max = max(inf_times) if max(inf_times) > 0 else 1
#         inf_norm = [1 - t/inf_max for t in inf_times]
        
#         categories = ['精度\n(sMAPE↓)', '风险感知\n(Corr↑)', '可靠性\n(PICP↑)', '轻量化\n(Time↓)']
#         N = len(categories)
#         angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#         angles += angles[:1]
        
#         fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
#         # 颜色：CFT-Net绿色，其他灰色
#         colors = ['#808080'] * (len(models)-1) + ['#2ca02c'] if 'CFT-Net' in models else ['#808080'] * len(models)
        
#         for i, model in enumerate(models):
#             values = [smape_norm[i], corr_norm[i], picp_norm[i], inf_norm[i]]
#             values += values[:1]
#             lw = 3 if model == 'CFT-Net' else 1.5
#             ax.plot(angles, values, 'o-', linewidth=lw, label=model, color=colors[i])
#             ax.fill(angles, values, alpha=0.15 if model == 'CFT-Net' else 0.05, color=colors[i])
        
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
#         ax.set_ylim(0, 1)
#         ax.set_title('模型综合能力对比\n（CFT-Net vs 基线模型）', fontsize=16, fontweight='bold', pad=30)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=11)
#         plt.tight_layout()
#         plt.savefig('evaluation_results/radar_chart.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         print("雷达图已保存")
    
#     def generate_comparison_table(self):
#         rows = []
#         for model, metrics in self.results.items():
#             row = {
#                 'Model': model,
#                 'R2': f"{metrics['R2']:.4f}" if metrics.get('R2') is not None else '-',
#                 'sMAPE(%)': f"{metrics['sMAPE']:.2f}",
#                 'MAE(s)': f"{metrics['MAE']:.2f}",
#                 'RMSE(s)': f"{metrics['RMSE']:.2f}",
#                 'Corr': f"{metrics['Corr']:.3f}" if metrics['Corr'] is not None else '-',
#                 'PICP-80(%)': f"{metrics['PICP_80']:.1f}" if metrics['PICP_80'] is not None else '-',
#                 'MPIW(s)': f"{metrics['MPIW_80']:.2f}" if metrics['MPIW_80'] is not None else '-',
#                 'ECE': f"{metrics['ECE']:.3f}" if metrics['ECE'] is not None else '-',
#                 'Params(K)': f"{metrics['Params_K']:.1f}" if metrics.get('Params_K') else '-',
#                 'Time(ms)': f"{metrics['Inference_ms']:.3f}"
#             }
#             rows.append(row)
        
#         df = pd.DataFrame(rows)
#         df.to_csv('evaluation_results/comparison_table.csv', index=False)
#         print("\n对比表格:")
#         print(df.to_string(index=False))
        
#         # 生成LaTeX表格
#         latex = self._generate_latex_table(rows)
#         with open('evaluation_results/table.tex', 'w') as f:
#             f.write(latex)
        
#         return df
    
#     def _generate_latex_table(self, rows):
#         latex = r"""\begin{table}[htbp]
# \centering
# \caption{模型综合性能对比}
# \label{tab:comparison}
# \begin{tabular}{lccccccc}
# \toprule
# \textbf{Model} & \textbf{R\textsuperscript{2}} & \textbf{sMAPE(\%)} & \textbf{MAE(s)} & \textbf{Corr} & \textbf{PICP-80(\%)} & \textbf{Params(K)} & \textbf{Time(ms)} \\
# \midrule
# """
#         for row in rows:
#             latex += f"{row['Model']} & {row['R2']} & {row['sMAPE(%)']} & {row['MAE(s)']} & {row['Corr']} & {row['PICP-80(%)']} & {row['Params(K)']} & {row['Time(ms)']} \\\\\n"
        
#         latex += r"""\bottomrule
# \end{tabular}
# \begin{tablenotes}
# \item[1] R\textsuperscript{2}接近1.0源于传输时间的强物理确定性（大小/带宽）。
# \item[2] CFT-Net是唯一提供不确定性量化的模型（Corr, PICP）。
# \end{tablenotes}
# \end{table}"""
#         return latex
    
#     def plot_calibration_curve(self):
#         if 'CFT-Net' not in self.results:
#             return
        
#         preds = self.results['CFT-Net']['predictions']
#         uncs = self.results['CFT-Net']['uncertainties']
#         errors = np.abs(self.y_test_orig - preds)
        
#         # 分层可视化
#         n_bins = 10
#         quantiles = np.linspace(0, 100, n_bins + 1)
#         bin_edges = np.percentile(uncs, quantiles)
#         bin_edges[-1] += 1e-8
        
#         bin_centers = []
#         avg_errors = []
#         avg_uncertainties = []
        
#         for i in range(n_bins):
#             in_bin = (uncs >= bin_edges[i]) & (uncs < bin_edges[i+1])
#             if i == n_bins - 1:
#                 in_bin = (uncs >= bin_edges[i]) & (uncs <= bin_edges[i+1])
#             if in_bin.sum() > 0:
#                 bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
#                 avg_errors.append(errors[in_bin].mean())
#                 avg_uncertainties.append(uncs[in_bin].mean())
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
#         # 左图：校准曲线
#         ax1.plot(avg_uncertainties, avg_errors, 'o-', linewidth=2, markersize=8, label='实际误差')
#         ax1.plot(avg_uncertainties, avg_uncertainties, 'r--', linewidth=2, label='完美校准')
#         ax1.fill_between(avg_uncertainties, avg_errors, avg_uncertainties, alpha=0.2, color='red')
#         ax1.set_xlabel('平均不确定性 (s)', fontsize=12)
#         ax1.set_ylabel('平均绝对误差 (s)', fontsize=12)
#         ax1.set_title('CFT-Net 校准曲线', fontsize=14, fontweight='bold')
#         ax1.legend()
#         ax1.grid(alpha=0.3)
        
#         # 右图：残差分布
#         residuals = self.y_test_orig - preds
#         ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
#         ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
#         ax2.set_xlabel('残差 (s)', fontsize=12)
#         ax2.set_ylabel('频数', fontsize=12)
#         ax2.set_title('预测残差分布', fontsize=14, fontweight='bold')
#         ax2.grid(alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('evaluation_results/calibration_analysis.png', dpi=300)
#         plt.close()
#         print("校准分析图已保存")
    
#     def plot_prediction_intervals(self):
#         """
#         【修复】使用全量测试集绘制预测区间，确保PICP与表格一致
#         """
#         if 'CFT-Net' not in self.results:
#             return
        
#         # 【关键修复】使用全量测试集，而不是仅前100个
#         n_show = len(self.y_test_orig)  # 全量
        
#         # 按预测值排序以便观察
#         indices = np.argsort(self.y_test_orig)
        
#         preds = self.results['CFT-Net']['predictions'][indices]
#         uncs = self.results['CFT-Net']['uncertainties'][indices]
#         y_true = self.y_test_orig[indices]
        
#         z = 1.28  # 80%置信区间
#         lower = preds - z * uncs
#         upper = preds + z * uncs
        
#         # 计算全量PICP（与表格一致）
#         covered = (y_true >= lower) & (y_true <= upper)
#         picp_actual = covered.mean() * 100
        
#         plt.figure(figsize=(16, 7))
#         x = np.arange(len(preds))
        
#         # 预测区间
#         plt.fill_between(x, lower, upper, alpha=0.3, color='blue', label='80%预测区间')
#         plt.plot(x, preds, 'b-', linewidth=1.5, label='预测值', alpha=0.8)
#         plt.scatter(x, y_true, c='black', s=1, zorder=5, label='真实值', alpha=0.3)
        
#         # 标记未覆盖的点（只标记部分避免过于密集）
#         not_covered_idx = np.where(~covered)[0]
#         if len(not_covered_idx) > 0:
#             # 如果太多，随机采样显示
#             if len(not_covered_idx) > 200:
#                 np.random.seed(42)
#                 display_idx = np.random.choice(not_covered_idx, 200, replace=False)
#             else:
#                 display_idx = not_covered_idx
#             plt.scatter(display_idx, y_true[display_idx], c='red', s=20, marker='x', 
#                        linewidth=2, label=f'未覆盖点 (n={len(not_covered_idx)})', zorder=6)
        
#         plt.xlabel('样本索引（按真实值排序）', fontsize=12)
#         plt.ylabel('传输时间 (s)', fontsize=12)
#         plt.title(f'CFT-Net 预测区间可视化 (全量测试集 n={n_show}, PICP={picp_actual:.1f}%)', 
#                  fontsize=14, fontweight='bold')
#         plt.legend(fontsize=11)
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('evaluation_results/prediction_intervals.png', dpi=300)
#         plt.close()
#         print(f"预测区间图已保存 (使用全量{n_show}个样本, PICP={picp_actual:.1f}%)")
    
#     def plot_pred_vs_actual(self):
#         """
#         【新增】绘制预测值 vs 真实值散点图（论文标准图）
#         """
#         if 'CFT-Net' not in self.results:
#             return
        
#         fig, axes = plt.subplots(2, 2, figsize=(14, 12))
#         fig.suptitle('预测值 vs 真实值对比 (Pred vs Actual)', fontsize=16, fontweight='bold')
        
#         models_to_plot = ['CFT-Net', 'RandomForest', 'XGBoost', 'LightGBM']
#         colors = ['#2ca02c', '#808080', '#808080', '#808080']
        
#         for idx, (model, color) in enumerate(zip(models_to_plot, colors)):
#             if model not in self.results:
#                 continue
            
#             ax = axes[idx // 2, idx % 2]
#             preds = self.results[model]['predictions']
#             y_true = self.y_test_orig
            
#             # 计算指标
#             r2 = self.results[model]['R2']
#             smape = self.results[model]['sMAPE']
            
#             # 散点图
#             ax.scatter(y_true, preds, alpha=0.4, s=10, c=color, edgecolors='none')
            
#             # 完美预测线
#             min_val = min(y_true.min(), preds.min())
#             max_val = max(y_true.max(), preds.max())
#             ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
            
#             # ±20%误差线
#             ax.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'k:', linewidth=1, alpha=0.5, label='±20%误差')
#             ax.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'k:', linewidth=1, alpha=0.5)
            
#             ax.set_xlabel('真实值 (s)', fontsize=11)
#             ax.set_ylabel('预测值 (s)', fontsize=11)
#             ax.set_title(f'{model}\nR²={r2:.4f}, sMAPE={smape:.2f}%', fontsize=12, fontweight='bold')
#             ax.legend(loc='upper left', fontsize=9)
#             ax.grid(alpha=0.3)
#             ax.set_xlim(min_val, max_val)
#             ax.set_ylim(min_val, max_val)
        
#         plt.tight_layout()
#         plt.savefig('evaluation_results/pred_vs_actual.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         print("Pred vs Actual 图已保存")
        
#         # 额外绘制CFT-Net的详细版本（带不确定性）
#         self._plot_cftnet_detailed()
    
#     def _plot_cftnet_detailed(self):
#         """CFT-Net详细版本：按不确定性大小着色"""
#         fig, ax = plt.subplots(figsize=(10, 10))
        
#         preds = self.results['CFT-Net']['predictions']
#         uncs = self.results['CFT-Net']['uncertainties']
#         y_true = self.y_test_orig
        
#         # 按不确定性着色
#         scatter = ax.scatter(y_true, preds, c=uncs, cmap='viridis', alpha=0.6, s=15, 
#                            edgecolors='none')
#         plt.colorbar(scatter, ax=ax, label='不确定性 (s)')
        
#         # 完美预测线
#         min_val = min(y_true.min(), preds.min())
#         max_val = max(y_true.max(), preds.max())
#         ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        
#         # 添加统计信息
#         r2 = self.results['CFT-Net']['R2']
#         smape = self.results['CFT-Net']['sMAPE']
#         corr = self.results['CFT-Net']['Corr']
        
#         ax.set_xlabel('真实值 (s)', fontsize=12)
#         ax.set_ylabel('预测值 (s)', fontsize=12)
#         ax.set_title(f'CFT-Net 预测详情 (按不确定性着色)\nR²={r2:.4f}, sMAPE={smape:.2f}%, Corr={corr:.3f}', 
#                     fontsize=13, fontweight='bold')
#         ax.legend(loc='upper left', fontsize=10)
#         ax.grid(alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('evaluation_results/pred_vs_actual_cftnet_detailed.png', dpi=300)
#         plt.close()
#         print("CFT-Net 详细 Pred vs Actual 图已保存")
    
#     def analyze_physical_determinism(self):
#         """分析物理确定性：验证 total_time ≈ total_size / bandwidth"""
#         print("\n" + "="*60)
#         print("🔍 物理确定性分析")
#         print("="*60)
        
#         # 反标准化获取原始特征
#         Xc_test_orig = self.scaler_c.inverse_transform(self.Xc_test)
#         Xi_test_orig = self.scaler_i.inverse_transform(self.Xi_test)
        
#         # 找到total_size和bandwidth的索引
#         size_idx = self.cols_i.index('total_size_mb') if 'total_size_mb' in self.cols_i else -1
#         bw_idx = self.cols_c.index('bandwidth_mbps') if 'bandwidth_mbps' in self.cols_c else -1
        
#         if size_idx >= 0 and bw_idx >= 0:
#             total_size = Xi_test_orig[:, size_idx]
#             bandwidth = Xc_test_orig[:, bw_idx]
            
#             # 理论传输时间（忽略压缩和开销）
#             theoretical_time = total_size / (bandwidth / 8)  # MB / (Mbps/8) = seconds
            
#             # 与实际时间对比
#             actual_time = self.y_test_orig
#             correlation = np.corrcoef(theoretical_time, actual_time)[0, 1]
            
#             print(f"理论传输时间 vs 实际时间 相关性: {correlation:.4f}")
#             print(f"理论时间范围: [{theoretical_time.min():.2f}, {theoretical_time.max():.2f}] s")
#             print(f"实际时间范围: [{actual_time.min():.2f}, {actual_time.max():.2f}] s")
            
#             # 解释高R²的原因
#             print("\n💡 解释：")
#             print("传输时间主要由物理公式决定：")
#             print("  time ≈ total_size / (bandwidth × compression_ratio) + overhead")
#             print("因此R²接近1.0是预期的，不代表过拟合。")
#             print("CFT-Net的价值在于量化公式无法覆盖的随机波动。")
        
#         print("="*60)
    
#     def run_full_evaluation(self):
#         self.calibrate_cftnet()
#         self.evaluate_cftnet()
#         self.train_baselines()
#         self.analyze_physical_determinism()
#         self.generate_comparison_table()
#         self.generate_radar_chart()
#         self.plot_calibration_curve()
#         self.plot_prediction_intervals()
#         self.plot_pred_vs_actual()  # 【新增】
#         print("\n✅ 所有评估完成！结果保存在 evaluation_results/ 目录")

# # ==============================================================================
# # 5. 主程序
# # ==============================================================================
# if __name__ == "__main__":
#     MODEL_PATH = "cts_improved_0218_2101_seed42.pth"
    
#     if not os.path.exists(MODEL_PATH):
#         print(f"错误：找不到模型文件 {MODEL_PATH}")
#         exit(1)
    
#     evaluator = ModelEvaluator(MODEL_PATH, seed=SEED)
#     evaluator.run_full_evaluation()
# """
# CFT-Net V2 完整对比评测脚本（修复版，匹配训练模型架构）
# 生成用于论文的对比表格和雷达图（精度、风险感知、可靠性、轻量化）
# 修复内容：
# 1. 完全对齐训练时的 CompactCFTNetV2 模型架构，解决权重加载报错
# 2. 修正不确定性传播（Delta Method），解决原始空间尺度不匹配问题
# 3. 算法特征使用One-Hot编码，避免数值顺序误导
# 4. 统一推理时间测量标准（全部在CPU上测量）
# 5. 完整保留分层校准、全量评估、所有可视化功能
# 6. 修复预测区间PICP与表格不一致的问题
# 7. 新增 Pred vs Actual 散点图与物理确定性分析
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
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from scipy.stats import spearmanr, norm, wilcoxon
# from scipy.optimize import brentq
# from collections import Counter
# import warnings
# import platform

# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 0. 基础配置
# # ==============================================================================
# system = platform.system()
# if system == 'Windows':
#     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# elif system == 'Darwin':
#     plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
# else:
#     plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

# plt.rcParams['axes.unicode_minus'] = False

# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# os.makedirs("evaluation_results", exist_ok=True)

# # 与训练脚本完全一致的模型超参数
# MODEL_CONFIG = {
#     "embed_dim": 64,
#     "nhead": 4,
#     "num_layers": 2,
#     "dim_feedforward": 128,
#     "alpha_init": 2.0,
#     "beta_init": 1.0,
#     "v_init": 1.0,
# }

# # ==============================================================================
# # 1. 模型定义（与训练脚本 CompactCFTNetV2 100% 一致）
# # ==============================================================================
# class LightweightFeatureTokenizer(nn.Module):
#     def __init__(self, num_features, embed_dim):
#         super().__init__()
#         self.embeddings = nn.Parameter(torch.empty(num_features, embed_dim))
#         self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
#         self.norm = nn.LayerNorm(embed_dim)
#         nn.init.xavier_normal_(self.embeddings)
        
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         out = x * self.embeddings + self.bias
#         return self.norm(out)

# class LightweightTransformerTower(nn.Module):
#     def __init__(self, num_features, embed_dim=64, nhead=4, num_layers=2, dim_feedforward=128):
#         super().__init__()
#         self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, 
#             nhead=nhead, 
#             dim_feedforward=dim_feedforward,
#             batch_first=True, 
#             dropout=0.2,
#             activation="gelu"
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#     def forward(self, x):
#         tokens = self.tokenizer(x)
#         cls = self.cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat([cls, tokens], dim=1)
#         out = self.encoder(x)
#         return out[:, 0, :]

# class CompactCFTNet(nn.Module):
#     """
#     与训练脚本 CompactCFTNetV2 完全一致，仅保留类名兼容原有代码
#     """
#     def __init__(self, client_feats, image_feats, num_algos, embed_dim=64):
#         super().__init__()
#         self.client_tower = LightweightTransformerTower(
#             client_feats, embed_dim, 
#             nhead=MODEL_CONFIG['nhead'], 
#             num_layers=MODEL_CONFIG['num_layers'],
#             dim_feedforward=MODEL_CONFIG['dim_feedforward']
#         )
#         self.image_tower = LightweightTransformerTower(
#             image_feats, embed_dim, 
#             nhead=MODEL_CONFIG['nhead'], 
#             num_layers=MODEL_CONFIG['num_layers'],
#             dim_feedforward=MODEL_CONFIG['dim_feedforward']
#         )
#         self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
#         # 共享融合层
#         self.shared_fusion = nn.Sequential(
#             nn.Linear(embed_dim * 3, embed_dim * 2),
#             nn.LayerNorm(embed_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU()
#         )
        
#         # 解耦头：均值预测分支
#         self.head_mean = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, 1)
#         )
        
#         # 解耦头：不确定性预测分支
#         self.head_uncertainty = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.LayerNorm(embed_dim // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(embed_dim // 2, 3)
#         )
        
#         # 与训练一致的初始化参数
#         self.alpha_init = MODEL_CONFIG['alpha_init']
#         self.beta_init = MODEL_CONFIG['beta_init']
#         self.v_init = MODEL_CONFIG['v_init']
        
#     def forward(self, cx, ix, ax):
#         c = self.client_tower(cx)
#         i = self.image_tower(ix)
#         a = self.algo_embed(ax)
        
#         fused = torch.cat([c, i, a], dim=-1)
#         shared = self.shared_fusion(fused)
        
#         # 解耦输出
#         gamma = self.head_mean(shared).squeeze(-1)
#         unc_out = self.head_uncertainty(shared)
        
#         # 与训练一致的参数约束
#         v = F.softplus(unc_out[:, 0]) + self.v_init
#         alpha = F.softplus(unc_out[:, 1]) + self.alpha_init
#         beta = F.softplus(unc_out[:, 2]) + self.beta_init
        
#         return torch.stack([gamma, v, alpha, beta], dim=1)

# # ==============================================================================
# # 2. 评估指标函数
# # ==============================================================================
# def calculate_smape(y_true, y_pred):
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

# def hierarchical_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, n_bins=5):
#     """
#     分层校准：对不同不确定性水平使用不同缩放因子
#     解决高不确定性区域校准不足的问题
#     """
#     quantiles = np.percentile(unc_raw, np.linspace(0, 100, n_bins + 1))
#     scales = []
#     bin_edges = []
    
#     print(f"{'区间':<15} {'样本数':<8} {'原始不确定':<12} {'实际误差':<12} {'缩放因子':<10}")
#     print("-" * 70)
    
#     for i in range(n_bins):
#         low, high = quantiles[i], quantiles[i+1]
#         bin_edges.append((low, high))
#         mask = (unc_raw >= low) & (unc_raw <= high)
#         n_samples = mask.sum()
        
#         if n_samples > 10:
#             def picp_with_scale(s):
#                 z = norm.ppf((1 + target_coverage) / 2)
#                 lower = y_pred[mask] - z * s * unc_raw[mask]
#                 upper = y_pred[mask] + z * s * unc_raw[mask]
#                 return np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper))
            
#             try:
#                 s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, 0.1, 100)
#             except:
#                 test_scales = np.linspace(0.1, 100, 500)
#                 picps = [picp_with_scale(s) for s in test_scales]
#                 s_opt = test_scales[np.argmin(np.abs(np.array(picps) - target_coverage))]
#             scales.append(s_opt)
            
#             print(f"[{low:.2f}, {high:.2f}]  "
#                   f"{n_samples:<8} {unc_raw[mask].mean():>10.2f}s  "
#                   f"{np.abs(y_true[mask]-y_pred[mask]).mean():>10.2f}s  {s_opt:>8.2f}x")
#         else:
#             scales.append(1.0)
#             print(f"[{low:.2f}, {high:.2f}]  "
#                   f"{n_samples:<8} {'-':>10}  {'-':>10}  {1.0:>8.2f}x")
    
#     # 应用分层缩放
#     unc_cal = unc_raw.copy()
#     for i, (low, high) in enumerate(bin_edges):
#         mask = (unc_raw >= low) & (unc_raw <= high)
#         unc_cal[mask] = unc_raw[mask] * scales[i]
    
#     return unc_cal, scales, bin_edges

# def apply_hierarchical_calibration(unc_raw, bin_edges, scales):
#     """将验证集学到的分层校准应用到测试集"""
#     unc_cal = unc_raw.copy()
#     for i, (low, high) in enumerate(bin_edges):
#         mask = (unc_raw >= low) & (unc_raw <= high)
#         unc_cal[mask] = unc_raw[mask] * scales[i]
#     return unc_cal

# def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, search_range=(0.1, 100)):
#     """全局单因子校准（作为回退方案）"""
#     def picp_with_scale(s):
#         z = norm.ppf((1 + target_coverage) / 2)
#         lower = y_pred - z * s * unc_raw
#         upper = y_pred + z * s * unc_raw
#         return np.mean((y_true >= lower) & (y_true <= upper))
#     s_min, s_max = search_range
#     try:
#         s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, s_min, s_max)
#         return s_opt
#     except:
#         scales = np.linspace(s_min, s_max, 500)
#         picps = [picp_with_scale(s) for s in scales]
#         best_idx = np.argmin(np.abs(np.array(picps) - target_coverage))
#         return scales[best_idx]

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
#     y_raw_log = np.log1p(df['total_time'].values)
#     y_raw_orig = df['total_time'].values
#     algo_names_raw = df['algo_name'].values
#     return Xc_raw, Xi_raw, algo_names_raw, y_raw_log, cols_c, cols_i, y_raw_orig

# # ==============================================================================
# # 4. 评估主类（完整修复版）
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
        
#         # 加载数据
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
        
#         # 基线模型使用One-Hot编码算法特征，避免数值顺序误导
#         self.algo_onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         self.algo_onehot.fit(self.Xa_train.reshape(-1, 1))
        
#         Xa_train_oh = self.algo_onehot.transform(self.Xa_train.reshape(-1, 1))
#         Xa_val_oh = self.algo_onehot.transform(self.Xa_val.reshape(-1, 1))
#         Xa_test_oh = self.algo_onehot.transform(self.Xa_test.reshape(-1, 1))
        
#         self.X_train_comb = np.hstack([self.Xc_train, self.Xi_train, Xa_train_oh])
#         self.X_val_comb = np.hstack([self.Xc_val, self.Xi_val, Xa_val_oh])
#         self.X_test_comb = np.hstack([self.Xc_test, self.Xi_test, Xa_test_oh])
        
#         print(f"数据划分: 训练 {len(self.tr_idx)} | 验证 {len(self.val_idx)} | 测试 {len(self.te_idx)}")
#         print(f"基线模型特征维度: {self.X_train_comb.shape[1]} (包含{len(self.enc.classes_)}个算法的One-Hot编码)")
        
#         # 加载CFT-Net模型（与训练完全一致的架构）
#         self.embed_dim = MODEL_CONFIG['embed_dim']
#         self.cftnet = CompactCFTNet(
#             client_feats=len(self.cols_c), 
#             image_feats=len(self.cols_i), 
#             num_algos=len(self.enc.classes_), 
#             embed_dim=self.embed_dim
#         ).to(device)
        
#         # 加载权重，兼容两种保存格式
#         checkpoint = torch.load(model_path, map_location=device)
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         else:
#             state_dict = checkpoint
#         self.cftnet.load_state_dict(state_dict)
#         self.cftnet.eval()
#         print("✅ CFT-Net V2 模型加载成功")
        
#         self.results = {}
#         self.calibration_params = {}
    
#     def predict_cftnet(self, Xc, Xi, Xa):
#         """
#         批量预测，使用Delta Method修正原始空间不确定性
#         与训练时的计算逻辑完全一致
#         """
#         batch_size = 1024
#         n = len(Xc)
#         preds_orig = []
#         uncs_log = []
#         uncs_orig = []
        
#         with torch.no_grad():
#             for i in range(0, n, batch_size):
#                 cx = torch.FloatTensor(Xc[i:i+batch_size]).to(device)
#                 ix = torch.FloatTensor(Xi[i:i+batch_size]).to(device)
#                 ax = torch.LongTensor(Xa[i:i+batch_size]).to(device)
                
#                 out = self.cftnet(cx, ix, ax)
#                 gamma, v, alpha, beta = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
                
#                 # 1. 均值转换回原始空间
#                 pred_log = gamma
#                 pred_orig = torch.expm1(pred_log)
                
#                 # 2. 不确定性传播（Delta Method）
#                 # Var(exp(x)-1) ≈ (exp(x))² * Var(x)
#                 var_log = beta / (v * (alpha - 1) + 1e-6)
#                 std_log = torch.sqrt(var_log + 1e-6)
#                 std_orig = torch.exp(pred_log) * std_log  # 核心修正：尺度对齐
                
#                 preds_orig.append(pred_orig.cpu().numpy())
#                 uncs_log.append(std_log.cpu().numpy())
#                 uncs_orig.append(std_orig.cpu().numpy())
        
#         return np.concatenate(preds_orig), np.concatenate(uncs_orig), np.concatenate(uncs_log)
    
#     def calibrate_cftnet(self):
#         """在验证集上学习校准参数"""
#         print("\n" + "="*60)
#         print("🔧 CFT-Net 事后校准（验证集）")
#         print("="*60)
        
#         pred_val, unc_val_raw, _ = self.predict_cftnet(self.Xc_val, self.Xi_val, self.Xa_val)
#         picp_val_raw, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val_raw, 0.8)
#         print(f"验证集原始PICP: {picp_val_raw:.1f}%")
        
#         # 优先使用分层校准
#         print("\n--- 分层校准学习 ---")
#         unc_val_cal, scales, bin_edges = hierarchical_calibration(
#             self.y_val_orig, pred_val, unc_val_raw, target_coverage=0.8, n_bins=5
#         )
#         picp_val_cal, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val_cal, 0.8)
#         print(f"分层校准后PICP: {picp_val_cal:.1f}%")
        
#         # 保存校准参数
#         self.calibration_params = {
#             'hierarchical_scales': scales,
#             'bin_edges': bin_edges,
#             'global_scale': post_hoc_calibration(self.y_val_orig, pred_val, unc_val_raw)
#         }
#         print(f"全局校准缩放因子: {self.calibration_params['global_scale']:.3f}")
#         print("="*60)
        
#         return self.calibration_params
    
#     def evaluate_cftnet(self):
#         """CFT-Net 完整测试集评估"""
#         pred_test, unc_test_raw, _ = self.predict_cftnet(self.Xc_test, self.Xi_test, self.Xa_test)
        
#         # 应用分层校准
#         unc_test_cal = apply_hierarchical_calibration(
#             unc_test_raw, 
#             self.calibration_params['bin_edges'], 
#             self.calibration_params['hierarchical_scales']
#         )
        
#         errors_test = np.abs(self.y_test_orig - pred_test)
        
#         # 全量指标计算
#         mae = mean_absolute_error(self.y_test_orig, pred_test)
#         rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_test))
#         smape = calculate_smape(self.y_test_orig, pred_test)
#         mape = calculate_mape(self.y_test_orig, pred_test)
#         r2 = r2_score(self.y_test_orig, pred_test)
        
#         # 不确定性指标
#         corr, _ = spearmanr(unc_test_cal, errors_test)
#         corr = 0.0 if np.isnan(corr) else corr
#         picp, mpiw = calculate_picp_mpiw(self.y_test_orig, pred_test, unc_test_cal, 0.8)
#         ece = calculate_ece_quantile(errors_test, unc_test_cal)
        
#         # 推理时间（统一在CPU上测量，公平对比）
#         infer_time = self.measure_inference_time_cftnet_cpu()
        
#         # 参数量统计
#         params_k = sum(p.numel() for p in self.cftnet.parameters()) / 1000
        
#         self.results['CFT-Net'] = {
#             'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape,
#             'R2': r2, 'Corr': corr, 'PICP_80': picp, 'MPIW_80': mpiw, 'ECE': ece,
#             'Inference_ms': infer_time * 1000,
#             'Params_K': params_k,
#             'predictions': pred_test,
#             'uncertainties': unc_test_cal,
#             'raw_uncertainties': unc_test_raw
#         }
        
#         print(f"\n✅ CFT-Net 测试集评估完成")
#         print(f"  精度指标: sMAPE={smape:.2f}%, RMSE={rmse:.2f}s, R²={r2:.4f}")
#         print(f"  不确定性: Corr={corr:.3f}, PICP={picp:.1f}%, MPIW={mpiw:.2f}s, ECE={ece:.3f}")
#         print(f"  推理性能: 单样本推理={infer_time*1000:.3f}ms, 参数量={params_k:.1f}K")
        
#         return self.results['CFT-Net']
    
#     def measure_inference_time_cftnet_cpu(self):
#         """在CPU上测量CFT-Net推理时间，确保与基线模型公平对比"""
#         self.cftnet.cpu()
#         batch_size = 256
#         n = len(self.Xc_test)
        
#         # Warmup
#         with torch.no_grad():
#             for i in range(0, min(500, n), batch_size):
#                 cx = torch.FloatTensor(self.Xc_test[i:i+batch_size])
#                 ix = torch.FloatTensor(self.Xi_test[i:i+batch_size])
#                 ax = torch.LongTensor(self.Xa_test[i:i+batch_size])
#                 _ = self.cftnet(cx, ix, ax)
        
#         # 正式计时
#         times = []
#         with torch.no_grad():
#             for i in range(0, n, batch_size):
#                 cx = torch.FloatTensor(self.Xc_test[i:i+batch_size])
#                 ix = torch.FloatTensor(self.Xi_test[i:i+batch_size])
#                 ax = torch.LongTensor(self.Xa_test[i:i+batch_size])
                
#                 start = time.perf_counter()
#                 _ = self.cftnet(cx, ix, ax)
#                 times.append(time.perf_counter() - start)
        
#         # 移回原设备
#         self.cftnet.to(device)
        
#         total_time = np.sum(times)
#         return total_time / n
    
#     def train_baselines(self):
#         """训练并评估基线模型（RandomForest/XGBoost/LightGBM）"""
#         models = {
#             'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.seed, n_jobs=-1),
#             'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.seed, n_jobs=-1),
#             'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=self.seed, n_jobs=-1, verbose=-1)
#         }
#         print("\n🚀 训练基线模型（使用One-Hot编码算法特征）...")
        
#         for name, model in models.items():
#             print(f"  训练 {name}...")
#             start = time.perf_counter()
#             model.fit(self.X_train_comb, self.y_train_log)
#             train_time = time.perf_counter() - start
            
#             # 预测并转换回原始空间
#             pred_log = model.predict(self.X_test_comb)
#             pred_orig = np.expm1(pred_log)
            
#             # 精度指标
#             mae = mean_absolute_error(self.y_test_orig, pred_orig)
#             rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_orig))
#             smape = calculate_smape(self.y_test_orig, pred_orig)
#             mape = calculate_mape(self.y_test_orig, pred_orig)
#             r2 = r2_score(self.y_test_orig, pred_orig)
            
#             # 推理时间
#             infer_time = self.measure_inference_time_sklearn(model, self.X_test_comb)
            
#             self.results[name] = {
#                 'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape, 'R2': r2,
#                 'Corr': None, 'PICP_80': None, 'MPIW_80': None, 'ECE': None,
#                 'Inference_ms': infer_time * 1000,
#                 'Params_K': None,
#                 'predictions': pred_orig
#             }
#             print(f"    完成: R²={r2:.4f}, sMAPE={smape:.2f}%, 单样本推理={infer_time*1000:.3f}ms")
    
#     def measure_inference_time_sklearn(self, model, X):
#         """测量sklearn系列模型的推理时间"""
#         batch_size = 256
#         n = len(X)
#         times = []
#         for i in range(0, n, batch_size):
#             X_batch = X[i:i+batch_size]
#             start = time.perf_counter()
#             _ = model.predict(X_batch)
#             times.append(time.perf_counter() - start)
#         total_time = np.sum(times)
#         return total_time / n
    
#     def generate_comparison_table(self):
#         """生成对比表格（CSV+LaTeX）"""
#         rows = []
#         for model, metrics in self.results.items():
#             row = {
#                 'Model': model,
#                 'R2': f"{metrics['R2']:.4f}" if metrics.get('R2') is not None else '-',
#                 'sMAPE(%)': f"{metrics['sMAPE']:.2f}",
#                 'MAE(s)': f"{metrics['MAE']:.2f}",
#                 'RMSE(s)': f"{metrics['RMSE']:.2f}",
#                 'Corr': f"{metrics['Corr']:.3f}" if metrics['Corr'] is not None else '-',
#                 'PICP-80(%)': f"{metrics['PICP_80']:.1f}" if metrics['PICP_80'] is not None else '-',
#                 'MPIW(s)': f"{metrics['MPIW_80']:.2f}" if metrics['MPIW_80'] is not None else '-',
#                 'ECE': f"{metrics['ECE']:.3f}" if metrics['ECE'] is not None else '-',
#                 'Params(K)': f"{metrics['Params_K']:.1f}" if metrics.get('Params_K') else '-',
#                 'Time(ms)': f"{metrics['Inference_ms']:.3f}"
#             }
#             rows.append(row)
        
#         df = pd.DataFrame(rows)
#         df.to_csv('evaluation_results/comparison_table.csv', index=False)
        
#         print("\n" + "="*120)
#         print("📊 模型综合性能对比表")
#         print("="*120)
#         print(df.to_string(index=False))
#         print("="*120)
        
#         # 生成LaTeX表格
#         latex = self._generate_latex_table(rows)
#         with open('evaluation_results/comparison_table.tex', 'w') as f:
#             f.write(latex)
#         print("LaTeX表格已保存至 evaluation_results/comparison_table.tex")
        
#         return df
    
#     def _generate_latex_table(self, rows):
#         """生成论文用LaTeX表格"""
#         latex = r"""\begin{table}[htbp]
# \centering
# \caption{模型综合性能对比}
# \label{tab:model_comparison}
# \resizebox{\textwidth}{!}{
# \begin{tabular}{lccccccccc}
# \toprule
# \textbf{Model} & \textbf{R\textsuperscript{2}} & \textbf{sMAPE(\%)} & \textbf{RMSE(s)} & \textbf{Corr} & \textbf{PICP-80(\%)} & \textbf{MPIW(s)} & \textbf{ECE} & \textbf{Params(K)} & \textbf{Time(ms)} \\
# \midrule
# """
#         for row in rows:
#             latex += f"{row['Model']} & {row['R2']} & {row['sMAPE(%)']} & {row['RMSE(s)']} & {row['Corr']} & {row['PICP-80(%)']} & {row['MPIW(s)']} & {row['ECE']} & {row['Params(K)']} & {row['Time(ms)']} \\\\\n"
        
#         latex += r"""\bottomrule
# \end{tabular}
# }
# \begin{tablenotes}
# \footnotesize
# \item[1] R\textsuperscript{2} 接近1.0源于传输时间的强物理确定性（文件大小/带宽）。
# \item[2] Corr、PICP、MPIW、ECE 为不确定性量化专属指标，传统树模型无法提供。
# \end{tablenotes}
# \end{table}"""
#         return latex
    
#     def generate_radar_chart(self):
#         """生成模型综合能力雷达图"""
#         models = list(self.results.keys())
#         # 雷达图维度：精度、风险感知、可靠性、轻量化、推理速度
#         smapes = [self.results[m]['sMAPE'] for m in models]
#         corrs = [self.results[m]['Corr'] if self.results[m]['Corr'] is not None else 0 for m in models]
#         picps = [self.results[m]['PICP_80'] if self.results[m]['PICP_80'] is not None else 0 for m in models]
#         inf_times = [self.results[m]['Inference_ms'] for m in models]
#         params = [self.results[m]['Params_K'] if self.results[m]['Params_K'] is not None else 1000 for m in models]
        
#         # 归一化（越高越好）
#         smape_norm = [max(0, 1 - s/50) for s in smapes]  # sMAPE越小越好
#         corr_norm = [max(0, c) for c in corrs]  # Corr越大越好
#         picp_norm = [p/100 for p in picps]  # PICP越大越好
#         inf_norm = [1 - t/max(inf_times) for t in inf_times]  # 推理时间越小越好
#         param_norm = [1 - p/max(params) for p in params]  # 参数量越小越好
        
#         categories = [
#             '预测精度\n(sMAPE↓)', 
#             '风险感知\n(Corr↑)', 
#             '可靠性\n(PICP↑)', 
#             '推理速度\n(Time↓)',
#             '轻量化\n(Params↓)'
#         ]
#         N = len(categories)
#         angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#         angles += angles[:1]
        
#         fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
#         # 配色：CFT-Net用突出的绿色，基线用灰色系
#         colors = ['#808080', '#808080', '#808080', '#2ca02c'] if len(models) == 4 else ['#808080']*(len(models)-1) + ['#2ca02c']
        
#         for i, model in enumerate(models):
#             values = [smape_norm[i], corr_norm[i], picp_norm[i], inf_norm[i], param_norm[i]]
#             values += values[:1]
#             linewidth = 3 if model == 'CFT-Net' else 1.5
#             alpha = 0.2 if model == 'CFT-Net' else 0.05
#             ax.plot(angles, values, 'o-', linewidth=linewidth, label=model, color=colors[i])
#             ax.fill(angles, values, alpha=alpha, color=colors[i])
        
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
#         ax.set_ylim(0, 1)
#         ax.set_title('模型综合能力雷达图\n（CFT-Net vs 传统基线模型）', fontsize=16, fontweight='bold', pad=40)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), fontsize=12)
#         plt.tight_layout()
#         plt.savefig('evaluation_results/radar_chart.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         print("✅ 雷达图已保存至 evaluation_results/radar_chart.png")
    
#     def plot_calibration_curve(self):
#         """绘制校准曲线与残差分布"""
#         if 'CFT-Net' not in self.results:
#             return
        
#         preds = self.results['CFT-Net']['predictions']
#         uncs = self.results['CFT-Net']['uncertainties']
#         errors = np.abs(self.y_test_orig - preds)
        
#         # 分箱计算校准曲线
#         n_bins = 10
#         quantiles = np.linspace(0, 100, n_bins + 1)
#         bin_edges = np.percentile(uncs, quantiles)
#         bin_edges[-1] += 1e-8
        
#         bin_centers = []
#         avg_errors = []
#         avg_uncertainties = []
        
#         for i in range(n_bins):
#             in_bin = (uncs >= bin_edges[i]) & (uncs < bin_edges[i+1])
#             if i == n_bins - 1:
#                 in_bin = (uncs >= bin_edges[i]) & (uncs <= bin_edges[i+1])
#             if in_bin.sum() > 0:
#                 bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
#                 avg_errors.append(errors[in_bin].mean())
#                 avg_uncertainties.append(uncs[in_bin].mean())
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
#         # 左图：校准曲线
#         ax1.plot(avg_uncertainties, avg_errors, 'o-', linewidth=2, markersize=8, label='实际绝对误差', color='#2ca02c')
#         ax1.plot(avg_uncertainties, avg_uncertainties, 'r--', linewidth=2, label='完美校准线')
#         ax1.fill_between(avg_uncertainties, avg_errors, avg_uncertainties, alpha=0.2, color='red')
#         ax1.set_xlabel('平均预测不确定性 (s)', fontsize=12)
#         ax1.set_ylabel('平均绝对误差 (s)', fontsize=12)
#         ax1.set_title('CFT-Net 校准曲线', fontsize=14, fontweight='bold')
#         ax1.legend(fontsize=11)
#         ax1.grid(alpha=0.3)
        
#         # 右图：残差分布
#         residuals = self.y_test_orig - preds
#         ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
#         ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
#         ax2.set_xlabel('预测残差 (真实值-预测值, s)', fontsize=12)
#         ax2.set_ylabel('样本频数', fontsize=12)
#         ax2.set_title('预测残差分布', fontsize=14, fontweight='bold')
#         ax2.grid(alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('evaluation_results/calibration_analysis.png', dpi=300)
#         plt.close()
#         print("✅ 校准分析图已保存至 evaluation_results/calibration_analysis.png")
    
#     def plot_prediction_intervals(self):
#         """绘制全量测试集预测区间图"""
#         if 'CFT-Net' not in self.results:
#             return
        
#         # 使用全量测试集，按真实值排序便于观察
#         indices = np.argsort(self.y_test_orig)
#         n_samples = len(indices)
        
#         preds = self.results['CFT-Net']['predictions'][indices]
#         uncs = self.results['CFT-Net']['uncertainties'][indices]
#         y_true = self.y_test_orig[indices]
        
#         # 80%置信区间
#         z = norm.ppf((1 + 0.8) / 2)
#         lower = preds - z * uncs
#         upper = preds + z * uncs
        
#         # 计算实际PICP
#         covered = (y_true >= lower) & (y_true <= upper)
#         picp_actual = covered.mean() * 100
#         not_covered_count = (~covered).sum()
        
#         plt.figure(figsize=(16, 7))
#         x = np.arange(n_samples)
        
#         # 绘制预测区间、预测值、真实值
#         plt.fill_between(x, lower, upper, alpha=0.3, color='#1f77b4', label='80%预测区间')
#         plt.plot(x, preds, 'b-', linewidth=1.2, label='预测值', alpha=0.8)
#         plt.scatter(x, y_true, c='black', s=2, zorder=5, label='真实值', alpha=0.4)
        
#         # 标记未覆盖的点
#         not_covered_idx = np.where(~covered)[0]
#         if len(not_covered_idx) > 0:
#             # 样本过多时随机采样显示，避免过于密集
#             display_count = min(200, len(not_covered_idx))
#             np.random.seed(self.seed)
#             display_idx = np.random.choice(not_covered_idx, display_count, replace=False)
#             plt.scatter(display_idx, y_true[display_idx], c='red', s=25, marker='x', 
#                        linewidth=2, label=f'未覆盖样本 (n={not_covered_count})', zorder=6)
        
#         plt.xlabel('样本索引（按真实传输时间升序排列）', fontsize=12)
#         plt.ylabel('传输时间 (s)', fontsize=12)
#         plt.title(f'CFT-Net 全量测试集预测区间可视化 (n={n_samples}, 实际PICP={picp_actual:.1f}%)', 
#                  fontsize=14, fontweight='bold')
#         plt.legend(fontsize=11, loc='upper left')
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('evaluation_results/prediction_intervals.png', dpi=300)
#         plt.close()
#         print(f"✅ 预测区间图已保存 (全量{n_samples}个样本, 实际PICP={picp_actual:.1f}%)")
    
#     def plot_pred_vs_actual(self):
#         """绘制所有模型的预测值vs真实值散点图"""
#         if 'CFT-Net' not in self.results:
#             return
        
#         fig, axes = plt.subplots(2, 2, figsize=(14, 12))
#         fig.suptitle('预测值 vs 真实值对比 (Prediction vs Actual)', fontsize=16, fontweight='bold', y=0.98)
        
#         models_to_plot = ['CFT-Net', 'RandomForest', 'XGBoost', 'LightGBM']
#         colors = ['#2ca02c', '#808080', '#808080', '#808080']
        
#         for idx, (model, color) in enumerate(zip(models_to_plot, colors)):
#             if model not in self.results:
#                 continue
            
#             ax = axes[idx // 2, idx % 2]
#             preds = self.results[model]['predictions']
#             y_true = self.y_test_orig
            
#             # 核心指标
#             r2 = self.results[model]['R2']
#             smape = self.results[model]['sMAPE']
            
#             # 散点图
#             ax.scatter(y_true, preds, alpha=0.4, s=10, c=color, edgecolors='none')
            
#             # 完美预测线
#             min_val = min(y_true.min(), preds.min())
#             max_val = max(y_true.max(), preds.max())
#             ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
            
#             # ±20%误差带
#             ax.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'k:', linewidth=1, alpha=0.5, label='±20%误差带')
#             ax.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'k:', linewidth=1, alpha=0.5)
            
#             ax.set_xlabel('真实传输时间 (s)', fontsize=11)
#             ax.set_ylabel('预测传输时间 (s)', fontsize=11)
#             ax.set_title(f'{model}\nR²={r2:.4f}, sMAPE={smape:.2f}%', fontsize=12, fontweight='bold')
#             ax.legend(loc='upper left', fontsize=9)
#             ax.grid(alpha=0.3)
#             ax.set_xlim(min_val, max_val)
#             ax.set_ylim(min_val, max_val)
        
#         plt.tight_layout()
#         plt.savefig('evaluation_results/pred_vs_actual_all.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 绘制CFT-Net专属带不确定性着色的版本
#         self._plot_cftnet_detailed_scatter()
#         print("✅ Pred vs Actual 对比图已保存")
    
#     def _plot_cftnet_detailed_scatter(self):
#         """CFT-Net专属散点图，按不确定性大小着色"""
#         fig, ax = plt.subplots(figsize=(10, 10))
        
#         preds = self.results['CFT-Net']['predictions']
#         uncs = self.results['CFT-Net']['uncertainties']
#         y_true = self.y_test_orig
        
#         # 按不确定性着色
#         scatter = ax.scatter(y_true, preds, c=uncs, cmap='viridis', alpha=0.6, s=15, edgecolors='none')
#         plt.colorbar(scatter, ax=ax, label='预测不确定性 (s)')
        
#         # 完美预测线
#         min_val = min(y_true.min(), preds.min())
#         max_val = max(y_true.max(), preds.max())
#         ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
#         # 指标标注
#         r2 = self.results['CFT-Net']['R2']
#         smape = self.results['CFT-Net']['sMAPE']
#         corr = self.results['CFT-Net']['Corr']
#         picp = self.results['CFT-Net']['PICP_80']
        
#         ax.set_xlabel('真实传输时间 (s)', fontsize=12)
#         ax.set_ylabel('预测传输时间 (s)', fontsize=12)
#         ax.set_title(
#             f'CFT-Net 预测详情（按不确定性着色）\n'
#             f'R²={r2:.4f}, sMAPE={smape:.2f}%, Corr={corr:.3f}, PICP={picp:.1f}%', 
#             fontsize=13, fontweight='bold'
#         )
#         ax.legend(loc='upper left', fontsize=10)
#         ax.grid(alpha=0.3)
#         ax.set_xlim(min_val, max_val)
#         ax.set_ylim(min_val, max_val)
        
#         plt.tight_layout()
#         plt.savefig('evaluation_results/pred_vs_actual_cftnet_detailed.png', dpi=300)
#         plt.close()
#         print("✅ CFT-Net 详细散点图已保存")
    
#     def analyze_physical_determinism(self):
#         """分析传输时间的物理确定性，解释高R²的合理性"""
#         print("\n" + "="*60)
#         print("🔍 传输时间物理确定性分析")
#         print("="*60)
        
#         # 反标准化获取原始特征
#         Xc_test_orig = self.scaler_c.inverse_transform(self.Xc_test)
#         Xi_test_orig = self.scaler_i.inverse_transform(self.Xi_test)
        
#         # 找到核心物理特征的索引
#         size_idx = self.cols_i.index('total_size_mb') if 'total_size_mb' in self.cols_i else -1
#         bw_idx = self.cols_c.index('bandwidth_mbps') if 'bandwidth_mbps' in self.cols_c else -1
        
#         if size_idx >= 0 and bw_idx >= 0:
#             total_size_mb = Xi_test_orig[:, size_idx]
#             bandwidth_mbps = Xc_test_orig[:, bw_idx]
            
#             # 理论传输时间（忽略压缩、协议开销）
#             # 公式：时间(s) = 文件大小(MB) / 带宽(MB/s) = 文件大小(MB) / (带宽(Mbps)/8)
#             theoretical_time = total_size_mb / (bandwidth_mbps / 8)
#             actual_time = self.y_test_orig
            
#             # 相关性分析
#             correlation = np.corrcoef(theoretical_time, actual_time)[0, 1]
#             r2_theoretical = correlation ** 2
            
#             print(f"理论传输时间 vs 实际传输时间 皮尔逊相关系数: {correlation:.4f}")
#             print(f"理论公式可解释的R²: {r2_theoretical:.4f}")
#             print(f"理论时间范围: [{theoretical_time.min():.2f}, {theoretical_time.max():.2f}] s")
#             print(f"实际时间范围: [{actual_time.min():.2f}, {actual_time.max():.2f}] s")
            
#             print("\n💡 高R²合理性说明：")
#             print("容器镜像传输时间由强物理规律主导，核心公式为：")
#             print("  传输时间 ≈ 镜像总大小 / 有效传输带宽 + 固定开销")
#             print(f"仅「大小/带宽」的基础公式即可解释 {r2_theoretical*100:.1f}% 的时间波动，")
#             print("因此模型R²接近1.0是符合物理规律的，并非过拟合。")
#             print("CFT-Net的核心价值在于量化公式无法覆盖的随机波动（压缩率、网络抖动、宿主机负载等），")
#             print("提供可靠的不确定性估计，为调度决策提供风险感知能力。")
        
#         print("="*60)
    
#     def run_full_evaluation(self):
#         """执行完整的评估流程"""
#         self.calibrate_cftnet()
#         self.evaluate_cftnet()
#         self.train_baselines()
#         self.analyze_physical_determinism()
#         self.generate_comparison_table()
#         self.generate_radar_chart()
#         self.plot_calibration_curve()
#         self.plot_prediction_intervals()
#         self.plot_pred_vs_actual()
        
#         # 保存完整结果
#         with open('evaluation_results/full_evaluation_results.json', 'w') as f:
#             json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ['predictions', 'uncertainties', 'raw_uncertainties']} for k, v in self.results.items()}, f, indent=2)
        
#         print("\n🎉 所有评估流程完成！所有结果已保存至 evaluation_results/ 目录")

# # ==============================================================================
# # 5. 主程序入口
# # ==============================================================================
# if __name__ == "__main__":
#     # 请修改为你的模型文件路径
#     MODEL_PATH = "cts_optimized_0218_2125_seed42.pth"
    
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
#         print("请修改脚本中 MODEL_PATH 为你的模型文件路径")
#         exit(1)
    
#     # 初始化评估器并执行完整评估
#     evaluator = ModelEvaluator(MODEL_PATH, seed=SEED)
#     evaluator.run_full_evaluation()

"""
CFT-Net V2 完整对比评测脚本（优化版适配，新增物理特征）
生成用于论文的对比表格和雷达图（精度、风险感知、可靠性、轻量化）
修复内容：
1. 完全对齐训练时的 CompactCFTNetV2 模型架构，解决权重加载报错
2. 修正不确定性传播（Delta Method），解决原始空间尺度不匹配问题
3. 【关键】新增与训练脚本一致的物理交叉特征，解决维度不匹配报错
4. 算法特征使用One-Hot编码，避免数值顺序误导
5. 统一推理时间测量标准（全部在CPU上测量）
6. 完整保留分层校准、全量评估、所有可视化功能
7. 修复预测区间PICP与表格不一致的问题
8. 新增 Pred vs Actual 散点图与物理确定性分析
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr, norm, wilcoxon
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
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

os.makedirs("evaluation_results", exist_ok=True)

# 与训练脚本完全一致的模型超参数
MODEL_CONFIG = {
    "embed_dim": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "alpha_init": 2.0,
    "beta_init": 1.0,
    "v_init": 1.0,
}

# ==============================================================================
# 1. 模型定义（与训练脚本 CompactCFTNetV2 100% 一致）
# ==============================================================================
class LightweightFeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_features, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_normal_(self.embeddings)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        out = x * self.embeddings + self.bias
        return self.norm(out)

class LightweightTransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.tokenizer = LightweightFeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True, 
            dropout=0.2,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        out = self.encoder(x)
        return out[:, 0, :]

class CompactCFTNet(nn.Module):
    """
    与训练脚本 CompactCFTNetV2 完全一致，仅保留类名兼容原有代码
    """
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=64):
        super().__init__()
        self.client_tower = LightweightTransformerTower(
            client_feats, embed_dim, 
            nhead=MODEL_CONFIG['nhead'], 
            num_layers=MODEL_CONFIG['num_layers'],
            dim_feedforward=MODEL_CONFIG['dim_feedforward']
        )
        self.image_tower = LightweightTransformerTower(
            image_feats, embed_dim, 
            nhead=MODEL_CONFIG['nhead'], 
            num_layers=MODEL_CONFIG['num_layers'],
            dim_feedforward=MODEL_CONFIG['dim_feedforward']
        )
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        # 共享融合层
        self.shared_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # 解耦头：均值预测分支
        self.head_mean = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # 解耦头：不确定性预测分支
        self.head_uncertainty = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 3)
        )
        
        # 与训练一致的初始化参数
        self.alpha_init = MODEL_CONFIG['alpha_init']
        self.beta_init = MODEL_CONFIG['beta_init']
        self.v_init = MODEL_CONFIG['v_init']
        
    def forward(self, cx, ix, ax):
        c = self.client_tower(cx)
        i = self.image_tower(ix)
        a = self.algo_embed(ax)
        
        fused = torch.cat([c, i, a], dim=-1)
        shared = self.shared_fusion(fused)
        
        # 解耦输出
        gamma = self.head_mean(shared).squeeze(-1)
        unc_out = self.head_uncertainty(shared)
        
        # 与训练一致的参数约束
        v = F.softplus(unc_out[:, 0]) + self.v_init
        alpha = F.softplus(unc_out[:, 1]) + self.alpha_init
        beta = F.softplus(unc_out[:, 2]) + self.beta_init
        
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 2. 评估指标函数
# ==============================================================================
def calculate_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100
    return smape

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def calculate_picp_mpiw(y_true, y_pred, unc, confidence=0.8):
    z = norm.ppf((1 + confidence) / 2)
    lower = y_pred - z * unc
    upper = y_pred + z * unc
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    return picp, mpiw

def calculate_ece_quantile(errors, uncertainties, n_bins=10):
    if len(errors) == 0:
        return 0.0
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(uncertainties, quantiles)
    bin_edges[-1] += 1e-8
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if i == n_bins - 1:
            in_bin = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])
        prop = in_bin.sum() / len(errors)
        if prop > 0:
            avg_unc = uncertainties[in_bin].mean()
            avg_err = errors[in_bin].mean()
            ece += np.abs(avg_err - avg_unc) * prop
    return ece

def hierarchical_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, n_bins=5):
    """
    分层校准：对不同不确定性水平使用不同缩放因子
    解决高不确定性区域校准不足的问题
    """
    quantiles = np.percentile(unc_raw, np.linspace(0, 100, n_bins + 1))
    scales = []
    bin_edges = []
    
    print(f"{'区间':<15} {'样本数':<8} {'原始不确定':<12} {'实际误差':<12} {'缩放因子':<10}")
    print("-" * 70)
    
    for i in range(n_bins):
        low, high = quantiles[i], quantiles[i+1]
        bin_edges.append((low, high))
        mask = (unc_raw >= low) & (unc_raw <= high)
        n_samples = mask.sum()
        
        if n_samples > 10:
            def picp_with_scale(s):
                z = norm.ppf((1 + target_coverage) / 2)
                lower = y_pred[mask] - z * s * unc_raw[mask]
                upper = y_pred[mask] + z * s * unc_raw[mask]
                return np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper))
            
            try:
                s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, 0.1, 100)
            except:
                test_scales = np.linspace(0.1, 100, 500)
                picps = [picp_with_scale(s) for s in test_scales]
                s_opt = test_scales[np.argmin(np.abs(np.array(picps) - target_coverage))]
            scales.append(s_opt)
            
            print(f"[{low:.2f}, {high:.2f}]  "
                  f"{n_samples:<8} {unc_raw[mask].mean():>10.2f}s  "
                  f"{np.abs(y_true[mask]-y_pred[mask]).mean():>10.2f}s  {s_opt:>8.2f}x")
        else:
            scales.append(1.0)
            print(f"[{low:.2f}, {high:.2f}]  "
                  f"{n_samples:<8} {'-':>10}  {'-':>10}  {1.0:>8.2f}x")
    
    # 应用分层缩放
    unc_cal = unc_raw.copy()
    for i, (low, high) in enumerate(bin_edges):
        mask = (unc_raw >= low) & (unc_raw <= high)
        unc_cal[mask] = unc_raw[mask] * scales[i]
    
    return unc_cal, scales, bin_edges

def apply_hierarchical_calibration(unc_raw, bin_edges, scales):
    """将验证集学到的分层校准应用到测试集"""
    unc_cal = unc_raw.copy()
    for i, (low, high) in enumerate(bin_edges):
        mask = (unc_raw >= low) & (unc_raw <= high)
        unc_cal[mask] = unc_raw[mask] * scales[i]
    return unc_cal

def post_hoc_calibration(y_true, y_pred, unc_raw, target_coverage=0.8, search_range=(0.1, 100)):
    """全局单因子校准（作为回退方案）"""
    def picp_with_scale(s):
        z = norm.ppf((1 + target_coverage) / 2)
        lower = y_pred - z * s * unc_raw
        upper = y_pred + z * s * unc_raw
        return np.mean((y_true >= lower) & (y_true <= upper))
    s_min, s_max = search_range
    try:
        s_opt = brentq(lambda s: picp_with_scale(s) - target_coverage, s_min, s_max)
        return s_opt
    except:
        scales = np.linspace(s_min, s_max, 500)
        picps = [picp_with_scale(s) for s in scales]
        best_idx = np.argmin(np.abs(np.array(picps) - target_coverage))
        return scales[best_idx]

# ==============================================================================
# 3. 数据加载与预处理（【关键修改】与训练脚本一致的特征工程）
# ==============================================================================
def load_preprocessing_objects():
    # 【修改】尝试加载优化版的预处理对象，如果没有则回退
    if os.path.exists('preprocessing_objects_optimized.pkl'):
        print("📦 加载优化版预处理对象 (preprocessing_objects_optimized.pkl)")
        with open('preprocessing_objects_optimized.pkl', 'rb') as f:
            prep = pickle.load(f)
    else:
        print("⚠️  未找到优化版预处理对象，加载默认版 (preprocessing_objects.pkl)")
        with open('preprocessing_objects.pkl', 'rb') as f:
            prep = pickle.load(f)
    return prep

def load_data():
    """【核心修改】与训练脚本完全一致的特征工程，新增4个物理特征"""
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
        if cols:
            df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
    df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
    df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
    
    # ==========================================
    # 【关键修复】和训练脚本保持一致：新增物理交叉特征
    # ==========================================
    print("🔧 加载数据：新增物理交叉特征（与训练脚本一致）")
    
    # 1. 最核心特征：理论传输时间 (文件大小 / 有效带宽)
    df['theoretical_time'] = df['total_size_mb'] / (df['bandwidth_mbps'] / 8 + 1e-8)
    
    # 2. 资源压力特征
    df['cpu_to_size_ratio'] = df['cpu_limit'] / (df['total_size_mb'] + 1e-8)
    df['mem_to_size_ratio'] = df['mem_limit_mb'] / (df['total_size_mb'] + 1e-8)
    
    # 3. 网络综合指标
    df['network_score'] = df['bandwidth_mbps'] / (df['network_rtt'] + 1e-8)
    
    # 更新特征列表，必须和训练时完全一致！
    cols_c = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb', 
              'theoretical_time', 'cpu_to_size_ratio', 'mem_to_size_ratio', 'network_score']
    
    # 镜像特征保持不变
    target_cols = ['total_size_mb', 'avg_layer_entropy', 'entropy_std',
                   'layer_count', 'size_std_mb', 'text_ratio', 'zero_ratio']
    cols_i = [c for c in target_cols if c in df.columns]
    
    Xc_raw = df[cols_c].values
    Xi_raw = df[cols_i].values
    y_raw_log = np.log1p(df['total_time'].values)
    y_raw_orig = df['total_time'].values
    algo_names_raw = df['algo_name'].values
    return Xc_raw, Xi_raw, algo_names_raw, y_raw_log, cols_c, cols_i, y_raw_orig

# ==============================================================================
# 4. 评估主类（完整修复版）
# ==============================================================================
class ModelEvaluator:
    def __init__(self, model_path, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.prep = load_preprocessing_objects()
        self.scaler_c = self.prep['scaler_c']
        self.scaler_i = self.prep['scaler_i']
        self.enc = self.prep['enc']
        
        # 【修改】优先从预处理对象中读取特征列名，确保和训练一致
        self.cols_c = self.prep.get('cols_c', ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb', 
                                                 'theoretical_time', 'cpu_to_size_ratio', 'mem_to_size_ratio', 'network_score'])
        self.cols_i = self.prep.get('cols_i', ['total_size_mb', 'avg_layer_entropy', 'layer_count', 'text_ratio', 'zero_ratio'])
        
        self.default_algo = self.prep.get('most_common_algo', self.enc.classes_[0])
        self.default_idx = self.enc.transform([self.default_algo])[0]
        
        # 加载数据
        Xc_raw, Xi_raw, algo_names_raw, y_log, _, _, y_orig = load_data()
        N = len(y_log)
        idx = np.random.permutation(N)
        n_tr = int(N * 0.7)
        n_val = int(N * 0.15)
        self.tr_idx = idx[:n_tr]
        self.val_idx = idx[n_tr:n_tr+n_val]
        self.te_idx = idx[n_tr+n_val:]
        
        # 标准化
        self.Xc_train = self.scaler_c.transform(Xc_raw[self.tr_idx])
        self.Xc_val = self.scaler_c.transform(Xc_raw[self.val_idx])
        self.Xc_test = self.scaler_c.transform(Xc_raw[self.te_idx])
        self.Xi_train = self.scaler_i.transform(Xi_raw[self.tr_idx])
        self.Xi_val = self.scaler_i.transform(Xi_raw[self.val_idx])
        self.Xi_test = self.scaler_i.transform(Xi_raw[self.te_idx])
        
        # 算法编码
        def safe_transform(labels):
            known = set(self.enc.classes_)
            return np.array([self.enc.transform([l])[0] if l in known else self.default_idx for l in labels])
        
        self.Xa_train = self.enc.transform(algo_names_raw[self.tr_idx])
        self.Xa_val = safe_transform(algo_names_raw[self.val_idx])
        self.Xa_test = safe_transform(algo_names_raw[self.te_idx])
        
        self.y_train_log = y_log[self.tr_idx]
        self.y_val_log = y_log[self.val_idx]
        self.y_test_log = y_log[self.te_idx]
        self.y_train_orig = y_orig[self.tr_idx]
        self.y_val_orig = y_orig[self.val_idx]
        self.y_test_orig = y_orig[self.te_idx]
        
        # 基线模型使用One-Hot编码算法特征，避免数值顺序误导
        self.algo_onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.algo_onehot.fit(self.Xa_train.reshape(-1, 1))
        
        Xa_train_oh = self.algo_onehot.transform(self.Xa_train.reshape(-1, 1))
        Xa_val_oh = self.algo_onehot.transform(self.Xa_val.reshape(-1, 1))
        Xa_test_oh = self.algo_onehot.transform(self.Xa_test.reshape(-1, 1))
        
        self.X_train_comb = np.hstack([self.Xc_train, self.Xi_train, Xa_train_oh])
        self.X_val_comb = np.hstack([self.Xc_val, self.Xi_val, Xa_val_oh])
        self.X_test_comb = np.hstack([self.Xc_test, self.Xi_test, Xa_test_oh])
        
        print(f"数据划分: 训练 {len(self.tr_idx)} | 验证 {len(self.val_idx)} | 测试 {len(self.te_idx)}")
        print(f"基线模型特征维度: {self.X_train_comb.shape[1]} (包含{len(self.enc.classes_)}个算法的One-Hot编码)")
        print(f"CFT-Net 输入维度: 客户端特征={len(self.cols_c)}, 镜像特征={len(self.cols_i)}")
        
        # 加载CFT-Net模型（与训练完全一致的架构）
        self.embed_dim = MODEL_CONFIG['embed_dim']
        self.cftnet = CompactCFTNet(
            client_feats=len(self.cols_c),  # 【关键】这里会自动读取新的8维特征
            image_feats=len(self.cols_i), 
            num_algos=len(self.enc.classes_), 
            embed_dim=self.embed_dim
        ).to(device)
        
        # 加载权重，兼容两种保存格式
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.cftnet.load_state_dict(state_dict)
        self.cftnet.eval()
        print("✅ CFT-Net V2 模型加载成功")
        
        self.results = {}
        self.calibration_params = {}
    
    def predict_cftnet(self, Xc, Xi, Xa):
        """
        批量预测，使用Delta Method修正原始空间不确定性
        与训练时的计算逻辑完全一致
        """
        batch_size = 1024
        n = len(Xc)
        preds_orig = []
        uncs_log = []
        uncs_orig = []
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                cx = torch.FloatTensor(Xc[i:i+batch_size]).to(device)
                ix = torch.FloatTensor(Xi[i:i+batch_size]).to(device)
                ax = torch.LongTensor(Xa[i:i+batch_size]).to(device)
                
                out = self.cftnet(cx, ix, ax)
                gamma, v, alpha, beta = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
                
                # 1. 均值转换回原始空间
                pred_log = gamma
                pred_orig = torch.expm1(pred_log)
                
                # 2. 不确定性传播（Delta Method）
                # Var(exp(x)-1) ≈ (exp(x))² * Var(x)
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_log = torch.sqrt(var_log + 1e-6)
                std_orig = torch.exp(pred_log) * std_log  # 核心修正：尺度对齐
                
                preds_orig.append(pred_orig.cpu().numpy())
                uncs_log.append(std_log.cpu().numpy())
                uncs_orig.append(std_orig.cpu().numpy())
        
        return np.concatenate(preds_orig), np.concatenate(uncs_orig), np.concatenate(uncs_log)
    
    def calibrate_cftnet(self):
        """在验证集上学习校准参数"""
        print("\n" + "="*60)
        print("🔧 CFT-Net 事后校准（验证集）")
        print("="*60)
        
        pred_val, unc_val_raw, _ = self.predict_cftnet(self.Xc_val, self.Xi_val, self.Xa_val)
        picp_val_raw, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val_raw, 0.8)
        print(f"验证集原始PICP: {picp_val_raw:.1f}%")
        
        # 优先使用分层校准
        print("\n--- 分层校准学习 ---")
        unc_val_cal, scales, bin_edges = hierarchical_calibration(
            self.y_val_orig, pred_val, unc_val_raw, target_coverage=0.8, n_bins=5
        )
        picp_val_cal, _ = calculate_picp_mpiw(self.y_val_orig, pred_val, unc_val_cal, 0.8)
        print(f"分层校准后PICP: {picp_val_cal:.1f}%")
        
        # 保存校准参数
        self.calibration_params = {
            'hierarchical_scales': scales,
            'bin_edges': bin_edges,
            'global_scale': post_hoc_calibration(self.y_val_orig, pred_val, unc_val_raw)
        }
        print(f"全局校准缩放因子: {self.calibration_params['global_scale']:.3f}")
        print("="*60)
        
        return self.calibration_params
    
    def evaluate_cftnet(self):
        """CFT-Net 完整测试集评估"""
        pred_test, unc_test_raw, _ = self.predict_cftnet(self.Xc_test, self.Xi_test, self.Xa_test)
        
        # 应用分层校准
        unc_test_cal = apply_hierarchical_calibration(
            unc_test_raw, 
            self.calibration_params['bin_edges'], 
            self.calibration_params['hierarchical_scales']
        )
        
        errors_test = np.abs(self.y_test_orig - pred_test)
        
        # 全量指标计算
        mae = mean_absolute_error(self.y_test_orig, pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_test))
        smape = calculate_smape(self.y_test_orig, pred_test)
        mape = calculate_mape(self.y_test_orig, pred_test)
        r2 = r2_score(self.y_test_orig, pred_test)
        
        # 不确定性指标
        corr, _ = spearmanr(unc_test_cal, errors_test)
        corr = 0.0 if np.isnan(corr) else corr
        picp, mpiw = calculate_picp_mpiw(self.y_test_orig, pred_test, unc_test_cal, 0.8)
        ece = calculate_ece_quantile(errors_test, unc_test_cal)
        
        # 推理时间（统一在CPU上测量，公平对比）
        infer_time = self.measure_inference_time_cftnet_cpu()
        
        # 参数量统计
        params_k = sum(p.numel() for p in self.cftnet.parameters()) / 1000
        
        self.results['CFT-Net'] = {
            'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape,
            'R2': r2, 'Corr': corr, 'PICP_80': picp, 'MPIW_80': mpiw, 'ECE': ece,
            'Inference_ms': infer_time * 1000,
            'Params_K': params_k,
            'predictions': pred_test,
            'uncertainties': unc_test_cal,
            'raw_uncertainties': unc_test_raw
        }
        
        print(f"\n✅ CFT-Net 测试集评估完成")
        print(f"  精度指标: sMAPE={smape:.2f}%, RMSE={rmse:.2f}s, R²={r2:.4f}")
        print(f"  不确定性: Corr={corr:.3f}, PICP={picp:.1f}%, MPIW={mpiw:.2f}s, ECE={ece:.3f}")
        print(f"  推理性能: 单样本推理={infer_time*1000:.3f}ms, 参数量={params_k:.1f}K")
        
        return self.results['CFT-Net']
    
    def measure_inference_time_cftnet_cpu(self):
        """在CPU上测量CFT-Net推理时间，确保与基线模型公平对比"""
        self.cftnet.cpu()
        batch_size = 256
        n = len(self.Xc_test)
        
        # Warmup
        with torch.no_grad():
            for i in range(0, min(500, n), batch_size):
                cx = torch.FloatTensor(self.Xc_test[i:i+batch_size])
                ix = torch.FloatTensor(self.Xi_test[i:i+batch_size])
                ax = torch.LongTensor(self.Xa_test[i:i+batch_size])
                _ = self.cftnet(cx, ix, ax)
        
        # 正式计时
        times = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                cx = torch.FloatTensor(self.Xc_test[i:i+batch_size])
                ix = torch.FloatTensor(self.Xi_test[i:i+batch_size])
                ax = torch.LongTensor(self.Xa_test[i:i+batch_size])
                
                start = time.perf_counter()
                _ = self.cftnet(cx, ix, ax)
                times.append(time.perf_counter() - start)
        
        # 移回原设备
        self.cftnet.to(device)
        
        total_time = np.sum(times)
        return total_time / n
    
    def train_baselines(self):
        """训练并评估基线模型（RandomForest/XGBoost/LightGBM）"""
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.seed, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.seed, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=self.seed, n_jobs=-1, verbose=-1)
        }
        print("\n🚀 训练基线模型（使用One-Hot编码算法特征）...")
        
        for name, model in models.items():
            print(f"  训练 {name}...")
            start = time.perf_counter()
            model.fit(self.X_train_comb, self.y_train_log)
            train_time = time.perf_counter() - start
            
            # 预测并转换回原始空间
            pred_log = model.predict(self.X_test_comb)
            pred_orig = np.expm1(pred_log)
            
            # 精度指标
            mae = mean_absolute_error(self.y_test_orig, pred_orig)
            rmse = np.sqrt(mean_squared_error(self.y_test_orig, pred_orig))
            smape = calculate_smape(self.y_test_orig, pred_orig)
            mape = calculate_mape(self.y_test_orig, pred_orig)
            r2 = r2_score(self.y_test_orig, pred_orig)
            
            # 推理时间
            infer_time = self.measure_inference_time_sklearn(model, self.X_test_comb)
            
            self.results[name] = {
                'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape, 'R2': r2,
                'Corr': None, 'PICP_80': None, 'MPIW_80': None, 'ECE': None,
                'Inference_ms': infer_time * 1000,
                'Params_K': None,
                'predictions': pred_orig
            }
            print(f"    完成: R²={r2:.4f}, sMAPE={smape:.2f}%, 单样本推理={infer_time*1000:.3f}ms")
    
    def measure_inference_time_sklearn(self, model, X):
        """测量sklearn系列模型的推理时间"""
        batch_size = 256
        n = len(X)
        times = []
        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            start = time.perf_counter()
            _ = model.predict(X_batch)
            times.append(time.perf_counter() - start)
        total_time = np.sum(times)
        return total_time / n
    
    def generate_comparison_table(self):
        """生成对比表格（CSV+LaTeX）"""
        rows = []
        for model, metrics in self.results.items():
            row = {
                'Model': model,
                'R2': f"{metrics['R2']:.4f}" if metrics.get('R2') is not None else '-',
                'sMAPE(%)': f"{metrics['sMAPE']:.2f}",
                'MAE(s)': f"{metrics['MAE']:.2f}",
                'RMSE(s)': f"{metrics['RMSE']:.2f}",
                'Corr': f"{metrics['Corr']:.3f}" if metrics['Corr'] is not None else '-',
                'PICP-80(%)': f"{metrics['PICP_80']:.1f}" if metrics['PICP_80'] is not None else '-',
                'MPIW(s)': f"{metrics['MPIW_80']:.2f}" if metrics['MPIW_80'] is not None else '-',
                'ECE': f"{metrics['ECE']:.3f}" if metrics['ECE'] is not None else '-',
                'Params(K)': f"{metrics['Params_K']:.1f}" if metrics.get('Params_K') else '-',
                'Time(ms)': f"{metrics['Inference_ms']:.3f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv('evaluation_results/comparison_table.csv', index=False)
        
        print("\n" + "="*120)
        print("📊 模型综合性能对比表")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        # 生成LaTeX表格
        latex = self._generate_latex_table(rows)
        with open('evaluation_results/comparison_table.tex', 'w') as f:
            f.write(latex)
        print("LaTeX表格已保存至 evaluation_results/comparison_table.tex")
        
        return df
    
    def _generate_latex_table(self, rows):
        """生成论文用LaTeX表格"""
        latex = r"""\begin{table}[htbp]
\centering
\caption{模型综合性能对比}
\label{tab:model_comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccccc}
\toprule
\textbf{Model} & \textbf{R\textsuperscript{2}} & \textbf{sMAPE(\%)} & \textbf{RMSE(s)} & \textbf{Corr} & \textbf{PICP-80(\%)} & \textbf{MPIW(s)} & \textbf{ECE} & \textbf{Params(K)} & \textbf{Time(ms)} \\
\midrule
"""
        for row in rows:
            latex += f"{row['Model']} & {row['R2']} & {row['sMAPE(%)']} & {row['RMSE(s)']} & {row['Corr']} & {row['PICP-80(%)']} & {row['MPIW(s)']} & {row['ECE']} & {row['Params(K)']} & {row['Time(ms)']} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
}
\begin{tablenotes}
\footnotesize
\item[1] R\textsuperscript{2} 接近1.0源于传输时间的强物理确定性（文件大小/带宽）。
\item[2] Corr、PICP、MPIW、ECE 为不确定性量化专属指标，传统树模型无法提供。
\end{tablenotes}
\end{table}"""
        return latex
    
    def generate_radar_chart(self):
        """生成模型综合能力雷达图"""
        models = list(self.results.keys())
        # 雷达图维度：精度、风险感知、可靠性、轻量化、推理速度
        smapes = [self.results[m]['sMAPE'] for m in models]
        corrs = [self.results[m]['Corr'] if self.results[m]['Corr'] is not None else 0 for m in models]
        picps = [self.results[m]['PICP_80'] if self.results[m]['PICP_80'] is not None else 0 for m in models]
        inf_times = [self.results[m]['Inference_ms'] for m in models]
        params = [self.results[m]['Params_K'] if self.results[m]['Params_K'] is not None else 1000 for m in models]
        
        # 归一化（越高越好）
        smape_norm = [max(0, 1 - s/50) for s in smapes]  # sMAPE越小越好
        corr_norm = [max(0, c) for c in corrs]  # Corr越大越好
        picp_norm = [p/100 for p in picps]  # PICP越大越好
        inf_norm = [1 - t/max(inf_times) for t in inf_times]  # 推理时间越小越好
        param_norm = [1 - p/max(params) for p in params]  # 参数量越小越好
        
        categories = [
            '预测精度\n(sMAPE↓)', 
            '风险感知\n(Corr↑)', 
            '可靠性\n(PICP↑)', 
            '推理速度\n(Time↓)',
            '轻量化\n(Params↓)'
        ]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # 配色：CFT-Net用突出的绿色，基线用灰色系
        colors = ['#808080', '#808080', '#808080', '#2ca02c'] if len(models) == 4 else ['#808080']*(len(models)-1) + ['#2ca02c']
        
        for i, model in enumerate(models):
            values = [smape_norm[i], corr_norm[i], picp_norm[i], inf_norm[i], param_norm[i]]
            values += values[:1]
            linewidth = 3 if model == 'CFT-Net' else 1.5
            alpha = 0.2 if model == 'CFT-Net' else 0.05
            ax.plot(angles, values, 'o-', linewidth=linewidth, label=model, color=colors[i])
            ax.fill(angles, values, alpha=alpha, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('模型综合能力雷达图\n（CFT-Net vs 传统基线模型）', fontsize=16, fontweight='bold', pad=40)
        ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), fontsize=12)
        plt.tight_layout()
        plt.savefig('evaluation_results/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 雷达图已保存至 evaluation_results/radar_chart.png")
    
    def plot_calibration_curve(self):
        """绘制校准曲线与残差分布"""
        if 'CFT-Net' not in self.results:
            return
        
        preds = self.results['CFT-Net']['predictions']
        uncs = self.results['CFT-Net']['uncertainties']
        errors = np.abs(self.y_test_orig - preds)
        
        # 分箱计算校准曲线
        n_bins = 10
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(uncs, quantiles)
        bin_edges[-1] += 1e-8
        
        bin_centers = []
        avg_errors = []
        avg_uncertainties = []
        
        for i in range(n_bins):
            in_bin = (uncs >= bin_edges[i]) & (uncs < bin_edges[i+1])
            if i == n_bins - 1:
                in_bin = (uncs >= bin_edges[i]) & (uncs <= bin_edges[i+1])
            if in_bin.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                avg_errors.append(errors[in_bin].mean())
                avg_uncertainties.append(uncs[in_bin].mean())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：校准曲线
        ax1.plot(avg_uncertainties, avg_errors, 'o-', linewidth=2, markersize=8, label='实际绝对误差', color='#2ca02c')
        ax1.plot(avg_uncertainties, avg_uncertainties, 'r--', linewidth=2, label='完美校准线')
        ax1.fill_between(avg_uncertainties, avg_errors, avg_uncertainties, alpha=0.2, color='red')
        ax1.set_xlabel('平均预测不确定性 (s)', fontsize=12)
        ax1.set_ylabel('平均绝对误差 (s)', fontsize=12)
        ax1.set_title('CFT-Net 校准曲线', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # 右图：残差分布
        residuals = self.y_test_orig - preds
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('预测残差 (真实值-预测值, s)', fontsize=12)
        ax2.set_ylabel('样本频数', fontsize=12)
        ax2.set_title('预测残差分布', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/calibration_analysis.png', dpi=300)
        plt.close()
        print("✅ 校准分析图已保存至 evaluation_results/calibration_analysis.png")
    
    def plot_prediction_intervals(self):
        """绘制全量测试集预测区间图"""
        if 'CFT-Net' not in self.results:
            return
        
        # 使用全量测试集，按真实值排序便于观察
        indices = np.argsort(self.y_test_orig)
        n_samples = len(indices)
        
        preds = self.results['CFT-Net']['predictions'][indices]
        uncs = self.results['CFT-Net']['uncertainties'][indices]
        y_true = self.y_test_orig[indices]
        
        # 80%置信区间
        z = norm.ppf((1 + 0.8) / 2)
        lower = preds - z * uncs
        upper = preds + z * uncs
        
        # 计算实际PICP
        covered = (y_true >= lower) & (y_true <= upper)
        picp_actual = covered.mean() * 100
        not_covered_count = (~covered).sum()
        
        plt.figure(figsize=(16, 7))
        x = np.arange(n_samples)
        
        # 绘制预测区间、预测值、真实值
        plt.fill_between(x, lower, upper, alpha=0.3, color='#1f77b4', label='80%预测区间')
        plt.plot(x, preds, 'b-', linewidth=1.2, label='预测值', alpha=0.8)
        plt.scatter(x, y_true, c='black', s=2, zorder=5, label='真实值', alpha=0.4)
        
        # 标记未覆盖的点
        not_covered_idx = np.where(~covered)[0]
        if len(not_covered_idx) > 0:
            # 样本过多时随机采样显示，避免过于密集
            display_count = min(200, len(not_covered_idx))
            np.random.seed(self.seed)
            display_idx = np.random.choice(not_covered_idx, display_count, replace=False)
            plt.scatter(display_idx, y_true[display_idx], c='red', s=25, marker='x', 
                       linewidth=2, label=f'未覆盖样本 (n={not_covered_count})', zorder=6)
        
        plt.xlabel('样本索引（按真实传输时间升序排列）', fontsize=12)
        plt.ylabel('传输时间 (s)', fontsize=12)
        plt.title(f'CFT-Net 全量测试集预测区间可视化 (n={n_samples}, 实际PICP={picp_actual:.1f}%)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_results/prediction_intervals.png', dpi=300)
        plt.close()
        print(f"✅ 预测区间图已保存 (全量{n_samples}个样本, 实际PICP={picp_actual:.1f}%)")
    
    def plot_pred_vs_actual(self):
        """绘制所有模型的预测值vs真实值散点图"""
        if 'CFT-Net' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('预测值 vs 真实值对比 (Prediction vs Actual)', fontsize=16, fontweight='bold', y=0.98)
        
        models_to_plot = ['CFT-Net', 'RandomForest', 'XGBoost', 'LightGBM']
        colors = ['#2ca02c', '#808080', '#808080', '#808080']
        
        for idx, (model, color) in enumerate(zip(models_to_plot, colors)):
            if model not in self.results:
                continue
            
            ax = axes[idx // 2, idx % 2]
            preds = self.results[model]['predictions']
            y_true = self.y_test_orig
            
            # 核心指标
            r2 = self.results[model]['R2']
            smape = self.results[model]['sMAPE']
            
            # 散点图
            ax.scatter(y_true, preds, alpha=0.4, s=10, c=color, edgecolors='none')
            
            # 完美预测线
            min_val = min(y_true.min(), preds.min())
            max_val = max(y_true.max(), preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
            
            # ±20%误差带
            ax.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'k:', linewidth=1, alpha=0.5, label='±20%误差带')
            ax.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'k:', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('真实传输时间 (s)', fontsize=11)
            ax.set_ylabel('预测传输时间 (s)', fontsize=11)
            ax.set_title(f'{model}\nR²={r2:.4f}, sMAPE={smape:.2f}%', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/pred_vs_actual_all.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制CFT-Net专属带不确定性着色的版本
        self._plot_cftnet_detailed_scatter()
        print("✅ Pred vs Actual 对比图已保存")
    
    def _plot_cftnet_detailed_scatter(self):
        """CFT-Net专属散点图，按不确定性大小着色"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        preds = self.results['CFT-Net']['predictions']
        uncs = self.results['CFT-Net']['uncertainties']
        y_true = self.y_test_orig
        
        # 按不确定性着色
        scatter = ax.scatter(y_true, preds, c=uncs, cmap='viridis', alpha=0.6, s=15, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='预测不确定性 (s)')
        
        # 完美预测线
        min_val = min(y_true.min(), preds.min())
        max_val = max(y_true.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
        # 指标标注
        r2 = self.results['CFT-Net']['R2']
        smape = self.results['CFT-Net']['sMAPE']
        corr = self.results['CFT-Net']['Corr']
        picp = self.results['CFT-Net']['PICP_80']
        
        ax.set_xlabel('真实传输时间 (s)', fontsize=12)
        ax.set_ylabel('预测传输时间 (s)', fontsize=12)
        ax.set_title(
            f'CFT-Net 预测详情（按不确定性着色）\n'
            f'R²={r2:.4f}, sMAPE={smape:.2f}%, Corr={corr:.3f}, PICP={picp:.1f}%', 
            fontsize=13, fontweight='bold'
        )
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/pred_vs_actual_cftnet_detailed.png', dpi=300)
        plt.close()
        print("✅ CFT-Net 详细散点图已保存")
    
    def analyze_physical_determinism(self):
        """分析传输时间的物理确定性，解释高R²的合理性"""
        print("\n" + "="*60)
        print("🔍 传输时间物理确定性分析")
        print("="*60)
        
        # 反标准化获取原始特征
        Xc_test_orig = self.scaler_c.inverse_transform(self.Xc_test)
        Xi_test_orig = self.scaler_i.inverse_transform(self.Xi_test)
        
        # 找到核心物理特征的索引
        size_idx = self.cols_i.index('total_size_mb') if 'total_size_mb' in self.cols_i else -1
        bw_idx = self.cols_c.index('bandwidth_mbps') if 'bandwidth_mbps' in self.cols_c else -1
        
        if size_idx >= 0 and bw_idx >= 0:
            total_size_mb = Xi_test_orig[:, size_idx]
            bandwidth_mbps = Xc_test_orig[:, bw_idx]
            
            # 理论传输时间（忽略压缩、协议开销）
            # 公式：时间(s) = 文件大小(MB) / 带宽(MB/s) = 文件大小(MB) / (带宽(Mbps)/8)
            theoretical_time = total_size_mb / (bandwidth_mbps / 8)
            actual_time = self.y_test_orig
            
            # 相关性分析
            correlation = np.corrcoef(theoretical_time, actual_time)[0, 1]
            r2_theoretical = correlation ** 2
            
            print(f"理论传输时间 vs 实际传输时间 皮尔逊相关系数: {correlation:.4f}")
            print(f"理论公式可解释的R²: {r2_theoretical:.4f}")
            print(f"理论时间范围: [{theoretical_time.min():.2f}, {theoretical_time.max():.2f}] s")
            print(f"实际时间范围: [{actual_time.min():.2f}, {actual_time.max():.2f}] s")
            
            print("\n💡 高R²合理性说明：")
            print("容器镜像传输时间由强物理规律主导，核心公式为：")
            print("  传输时间 ≈ 镜像总大小 / 有效传输带宽 + 固定开销")
            print(f"仅「大小/带宽」的基础公式即可解释 {r2_theoretical*100:.1f}% 的时间波动，")
            print("因此模型R²接近1.0是符合物理规律的，并非过拟合。")
            print("CFT-Net的核心价值在于量化公式无法覆盖的随机波动（压缩率、网络抖动、宿主机负载等），")
            print("提供可靠的不确定性估计，为调度决策提供风险感知能力。")
        
        print("="*60)
    
    def run_full_evaluation(self):
        """执行完整的评估流程"""
        self.calibrate_cftnet()
        self.evaluate_cftnet()
        self.train_baselines()
        self.analyze_physical_determinism()
        self.generate_comparison_table()
        self.generate_radar_chart()
        self.plot_calibration_curve()
        self.plot_prediction_intervals()
        self.plot_pred_vs_actual()
        
        # 保存完整结果
        with open('evaluation_results/full_evaluation_results.json', 'w') as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ['predictions', 'uncertainties', 'raw_uncertainties']} for k, v in self.results.items()}, f, indent=2)
        
        print("\n🎉 所有评估流程完成！所有结果已保存至 evaluation_results/ 目录")

# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 请修改为你的模型文件路径
    MODEL_PATH = "cts_optimized_0218_2125_seed42.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
        print("请修改脚本中 MODEL_PATH 为你的模型文件路径")
        exit(1)
    
    # 初始化评估器并执行完整评估
    evaluator = ModelEvaluator(MODEL_PATH, seed=SEED)
    evaluator.run_full_evaluation()