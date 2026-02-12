import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import os
import platform

warnings.filterwarnings('ignore')

# --- 字体自动配置 ---
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 模型定义 (保持一致)
# ==============================================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
    def forward(self, x):
        return x.unsqueeze(-1) * self.weights + self.biases

class TransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True, dropout=0.1)
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
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 修正后的评估器
# ==============================================================================
class UncertaintyEvaluatorFixed:
    
    def __init__(self):
        self.model = None
        self.scaler_c = StandardScaler()
        self.scaler_i = StandardScaler()
        self.enc_algo = LabelEncoder()
        # 特征列定义
        self.col_client = ['bandwidth_mbps', 'cpu_limit', 'network_rtt', 'mem_limit_mb']
        self.col_image = ['total_size_mb', 'avg_layer_entropy', 'text_ratio', 'layer_count', 'zero_ratio']
        
    def _find_file(self, filename):
        for path in [filename, os.path.join('..', 'modeling', filename), os.path.join('ml_training', 'modeling', filename)]:
            if os.path.exists(path): return path
        return filename

    def load_resources(self):
        """加载数据和模型"""
        print("正在加载资源...")
        data_path = self._find_file('cts_data.xlsx')
        feat_path = self._find_file('image_features_database.csv')
        model_path = self._find_file('cts_best_model_full_modified.pth')
        
        # 1. 数据处理
        df_exp = pd.read_excel(data_path)
        df_feat = pd.read_csv(feat_path)
        
        rename_map = {"image": "image_name", "method": "algo_name", "network_bw": "bandwidth_mbps", "network_delay": "network_rtt", "mem_limit": "mem_limit_mb"}
        df_exp = df_exp.rename(columns=rename_map)
        if 'total_time' not in df_exp.columns:
            cols = [c for c in df_exp.columns if 'total_tim' in c]
            if cols: df_exp = df_exp.rename(columns={cols[0]: 'total_time'})
        
        df_exp = df_exp[(df_exp['status'] == 'SUCCESS') & (df_exp['total_time'] > 0)]
        if 'mem_limit_mb' not in df_exp.columns: df_exp['mem_limit_mb'] = 1024.0
        
        self.full_df = pd.merge(df_exp, df_feat, on="image_name", how="inner")
        
        # 2. 拟合Scaler
        self.scaler_c.fit(self.full_df[self.col_client].values)
        self.scaler_i.fit(self.full_df[self.col_image].values)
        self.enc_algo.fit(self.full_df['algo_name'].values)
        
        # 3. 加载模型
        self.model = CTSDualTowerModel(
            client_feats=len(self.col_client),
            image_feats=len(self.col_image),
            num_algos=len(self.enc_algo.classes_),
            embed_dim=32
        )
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("✅ 模型与数据加载完成")

    def create_calibrated_ood(self):
        """
        构造更合理的 OOD 数据：
        不是通过数值爆炸，而是通过'逻辑冲突'和'噪声扰动'
        """
        print("构造 OOD 测试集...")
        
        # 1. ID 数据 (In-Distribution): 从真实数据中采样
        id_df = self.full_df.sample(n=1000, random_state=42).copy()
        id_df['is_ood'] = 0
        id_df['ood_type'] = 'ID (Normal)'
        
        # 2. OOD-1: 噪声扰动 (Noisy Features)
        # 给特征加上强高斯噪声，使其脱离原始分布，但不过分
        ood_noise = self.full_df.sample(n=500, random_state=101).copy()
        for col in self.col_client + self.col_image:
            std = ood_noise[col].std()
            # 加上 3倍标准差的噪声 -> 统计学上的异常值
            ood_noise[col] = ood_noise[col] + np.random.normal(0, 3 * std, len(ood_noise))
        ood_noise['is_ood'] = 1
        ood_noise['ood_type'] = 'Noisy Input'
        
        # 3. OOD-2: 逻辑冲突 (Conflicting Features)
        # 例如：极高带宽(10000) 但 RTT 极高(5000ms) -> 物理上矛盾
        ood_conflict = self.full_df.sample(n=500, random_state=102).copy()
        ood_conflict['bandwidth_mbps'] = 10000.0  # 超快网
        ood_conflict['network_rtt'] = 5000.0      # 超高延迟
        ood_conflict['total_size_mb'] = 0.1       # 极小文件
        ood_conflict['avg_layer_entropy'] = 0.01  # 极低熵
        ood_conflict['is_ood'] = 1
        ood_conflict['ood_type'] = 'Logic Conflict'
        
        return pd.concat([id_df, ood_noise, ood_conflict], ignore_index=True)

    def predict(self, df):
        X_c = self.scaler_c.transform(df[self.col_client].values)
        X_i = self.scaler_i.transform(df[self.col_image].values)
        # 处理可能的未知算法标签
        try:
            X_a = self.enc_algo.transform(df['algo_name'].values)
        except:
            # 如果OOD构造出了未知算法，默认用0
            X_a = np.zeros(len(df), dtype=int)
            
        cx = torch.FloatTensor(X_c)
        ix = torch.FloatTensor(X_i)
        ax = torch.LongTensor(X_a)
        
        with torch.no_grad():
            preds = self.model(cx, ix, ax)
            gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            
            # [关键修改] 使用总方差 (Total Variance)
            # Var[y] = Beta * (1 + v) / (v * (Alpha - 1))
            # 包含 Aleatoric (数据噪音) + Epistemic (模型不懂)
            # 这种指标对 OOD 检测更鲁棒
            uncertainty = (beta * (1 + v)) / (v * (alpha - 1))
            
            pred_time = np.expm1(gamma.numpy())
            
        return pred_time, uncertainty.numpy()

    def run_evaluation(self):
        self.load_resources()
        test_df = self.create_calibrated_ood()
        
        preds, unc = self.predict(test_df)
        test_df['uncertainty'] = unc
        test_df['pred_time'] = preds
        
        # 1. 打印统计数据
        id_u = test_df[test_df['is_ood']==0]['uncertainty'].mean()
        ood_u = test_df[test_df['is_ood']==1]['uncertainty'].mean()
        auroc = roc_auc_score(test_df['is_ood'], test_df['uncertainty'])
        
        print("\n" + "="*40)
        print("📊 修正后的不确定性统计")
        print("="*40)
        print(f"ID 样本平均不确定性  : {id_u:.4f}")
        print(f"OOD 样本平均不确定性 : {ood_u:.4f}")
        print(f"OOD 检测 AUROC       : {auroc:.4f}")
        print("="*40)
        
        if ood_u > id_u:
            print("✅ 验证成功：异常样本的不确定性显著高于正常样本！")
        else:
            print("⚠️ 警告：OOD不确定性仍未超过ID，可能是模型对Log空间方差的理解问题。")

        # 2. 绘制图表
        self.plot_results(test_df)
        
        # 保存数据
        stats = {
            'id_uncertainty': float(id_u),
            'ood_uncertainty': float(ood_u),
            'ood_auroc': float(auroc)
        }
        with open('chapter3_3_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def plot_results(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # (a) 分布直方图 (KDE)
        # 截断极值以便绘图好看 (取95分位数)
        limit = np.percentile(df['uncertainty'], 95)
        plot_data = df[df['uncertainty'] < limit]
        
        sns.kdeplot(data=plot_data[plot_data['is_ood']==0], x='uncertainty', fill=True, ax=axes[0], color='blue', label='ID (正常)')
        sns.kdeplot(data=plot_data[plot_data['is_ood']==1], x='uncertainty', fill=True, ax=axes[0], color='red', label='OOD (异常)')
        axes[0].set_title('(a) 认知不确定性分布密度估计', fontsize=14)
        axes[0].set_xlabel('总预测不确定性 (Total Variance)', fontsize=12)
        axes[0].legend()
        
        # (b) 误差相关性 (只看 ID 数据，证明模型知道自己哪里不准)
        id_df = df[df['is_ood']==0].copy()
        id_df['abs_error'] = np.abs(id_df['total_time'] - id_df['pred_time'])
        # 分箱计算
        id_df['unc_bin'] = pd.qcut(id_df['uncertainty'], q=10, duplicates='drop')
        bin_stats = id_df.groupby('unc_bin').agg({'abs_error': 'mean', 'uncertainty': 'mean'}).reset_index()
        
        sns.regplot(data=bin_stats, x='uncertainty', y='abs_error', ax=axes[1], 
                    scatter_kws={'s':100, 'alpha':0.7}, line_kws={'color':'red'})
        axes[1].set_title('(b) 预测误差与不确定性的相关性 (AUSE验证)', fontsize=14)
        axes[1].set_xlabel('平均不确定性 (Bin)', fontsize=12)
        axes[1].set_ylabel('平均绝对误差 (MAE)', fontsize=12)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('figure_3_5_uncertainty_analysis_fixed.png', dpi=300)
        print("✅ 图表已生成: figure_3_5_uncertainty_analysis_fixed.png")

if __name__ == "__main__":
    evaluator = UncertaintyEvaluatorFixed()
    evaluator.run_evaluation()