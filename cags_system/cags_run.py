import torch
import torch.nn as nn
import torch.nn.functional as F

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
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.2
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
        # 输出层升级为 4 个神经元 (Gamma, v, Alpha, Beta)
        self.head = nn.Linear(64, 4) 

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        hidden = self.hidden(combined)
        out = self.head(hidden)
        
        # 施加数学约束 (Softplus)
        gamma = out[:, 0]
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys

# === 1. 引入 CAGS 调度器核心类 ===
from cags_scheduler import CAGSStrategyLayer, CAGSTacticalLayer, CAGSCorrectionLayer

# ==============================================================================
# === 2. 粘贴模型定义 (AI部分) - 必须与 train.py 和 experiment_graph_ai.py 100% 一致 ===
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
        # [修改点 1] 输出层升级为 4 个神经元 (Gamma, v, Alpha, Beta)
        self.head = nn.Linear(64, 4) 

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        hidden = self.hidden(combined)
        out = self.head(hidden)
        
        # [修改点 2] 施加数学约束 (Softplus)
        gamma = out[:, 0]
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 🚀 主程序：AI 驱动的 CAGS 仿真 (含不确定性演示)
# ==============================================================================
def run_cags_simulation():
    print("="*60)
    print("🚀 启动 CAGS 自适应流水线传输系统 (Uncertainty-Aware AI Mode)")
    print("="*60)

    # ---------------------------------------------------------
    # Step 1: 加载训练好的大脑 (CFT-Net)
    # ---------------------------------------------------------
    device = torch.device("cpu") 
    
    # ---------------------------------------------------------
    # Step 1: 加载训练好的大脑 (CFT-Net)
    # ---------------------------------------------------------
    device = torch.device("cpu") 
    
    # 更新并扩展可能的模型路径
    possible_paths = [
        "cts_best_model_full.pth",
        "ml_training/modeling/cts_best_model_full.pth",      # 新增：项目根目录下
        "../ml_training/modeling/cts_best_model_full.pth",   # 原有
        "../../ml_training/modeling/cts_best_model_full.pth", # 新增：更深一层
        "../cags_system/ml_training/modeling/cts_best_model_full.pth" # 常见IDE结构
    ]
    
    # 使用 next() 和生成器表达式简化路径查找
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)

    # 统一的模型初始化
    model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(device)
    ai_uncertainty = 0.5  # 默认不确定性

    if model_path:
        print(f"📥 正在加载 AI 模型: {model_path} ...")
        try:
            state_dict = torch.load(model_path, map_location=device)
            # 维度检查
            if state_dict['head.weight'].shape[0] == 4:
                model.load_state_dict(state_dict)
                model.eval()
                print("✅ 模型加载成功！EDL 不确定性推断已就绪。")
            else:
                print("⚠️ 检测到旧模型权重。切换到模拟模式。")
                ai_uncertainty = 0.8
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}, 使用随机权重演示...")
            ai_uncertainty = 0.8
    else:
        print("⚠️ [演示模式] 未找到模型文件，使用随机权重演示...")

    # ---------------------------------------------------------
    # Step 2: 构造输入场景
    # ---------------------------------------------------------
    print("\n🌍 [环境感知] 正在采集上下文特征...")
    # 构造：带宽低 (5Mbps) + 文件大 (1.5GB) -> 预期风险较高
    client_vec = torch.FloatTensor([[5.0, 0.8, 200.0, 1024.0]])
    image_vec = torch.FloatTensor([[1500.0, 0.8, 0.1, 5.0, 0.1]])
    algo_vec = torch.LongTensor([0])

    # ---------------------------------------------------------
    # Step 3: AI 推理 (计算不确定性)
    # ---------------------------------------------------------
    with torch.no_grad():
        preds = model(client_vec, image_vec, algo_vec)
        gamma, v, alpha, beta = preds[0]
        
        # [修改点 3] 计算不确定性
        # Uncertainty = Beta / (v * (Alpha - 1))
        uncertainty_val = beta / (v * (alpha - 1))
        predicted_time_s = torch.expm1(gamma).item()
        
        # 更新 AI 不确定性
        ai_uncertainty = min(1.0, max(0.0, uncertainty_val.item() / 10.0)) # 归一化

    predicted_risk_prob = 0.05 if predicted_time_s > 60 else 0.01
    predicted_bw = 5.0 
    
    print(f"🤖 [AI 推理结果]")
    print(f"   👉 预测耗时: {predicted_time_s:.2f} 秒")
    print(f"   👉 模型不确定性 (Uncertainty): {ai_uncertainty:.4f}") # 打印出来给老师看

    # ---------------------------------------------------------
    # Step 4: 战略层决策 (传入不确定性)
    # ---------------------------------------------------------
    strategy = CAGSStrategyLayer()
    
    # [修改点 4] 传入 model_uncertainty
    # 即使 risk_prob 很低，如果 uncertainty 很高，也会触发风险放大
    best_config, cost = strategy.optimize(predicted_bw, predicted_risk_prob, 0.8, model_uncertainty=ai_uncertainty)
    chunk_size, concurrency = best_config

    print(f"\n💡 [战略层] 基于 AI 预测 (含不确定性加权) 的效用决策")
    print(f"   👉 最优切片: {chunk_size/1024} KB")
    print(f"   👉 最优并发: {concurrency} 线程")
    
    if ai_uncertainty > 0.5:
        print(f"   👉 决策理由: 模型不确定性较高 ({ai_uncertainty:.2f})，系统自动启用了【风险放大机制】，倾向于保守配置。")
    elif predicted_risk_prob > 0.02:
        print(f"   👉 决策理由: AI 预测耗时过长，判定为高风险环境，强制选择稳健粒度。")
    else:
        print(f"   👉 决策理由: 模型置信度高且预测风险低，启用激进配置以提升吞吐。")

    # ---------------------------------------------------------
    # Step 5: 战术执行
    # ---------------------------------------------------------
    tactical = CAGSTacticalLayer()
    correction = CAGSCorrectionLayer(initial_chunk_size=chunk_size)
    
    print("\n🔄 [战术层 & 修正层] 启动自适应传输流水线...")
    print("-" * 60)

    for i in range(10):
        # 模拟：中间发生网络抖动
        is_jitter = (i == 3 or i == 4 or i == 5)
        status = 'TIMEOUT' if is_jitter else 'SUCCESS'
        
        # A. 修正层 (AIMD)
        current_size = correction.feedback(status, rtt_ms=200) 
        
        # B. 战术层 (乱序模拟)
        actual_id = i
        if i == 1: actual_id = 2
        if i == 2: actual_id = 1
        
        if status == 'SUCCESS':
            tactical.on_download_complete(actual_id, current_size/1024)
        
        time.sleep(0.1)

    print("-" * 60)
    print("✅ 仿真结束！系统展示了从 [不确定性感知] 到 [AIMD自愈] 的完整闭环。")

if __name__ == "__main__":
    run_cags_simulation()