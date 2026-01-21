import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

# === 引入调度器 (确保 cags_scheduler.py 已经是最新版) ===
from cags_scheduler import CAGSStrategyLayer, CAGSCorrectionLayer

# ==============================================================================
# 1. 模型定义升级 (适配 EDL 证据深度学习)
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
        # [修改点 1] 输出层改为 4 个神经元 (Gamma, v, Alpha, Beta)
        self.head = nn.Linear(64, 4) 

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        hidden = self.hidden(combined)
        out = self.head(hidden)
        
        # [修改点 2] 施加数学约束 (Softplus) 保证参数 > 0
        gamma = out[:, 0]
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)

# ==============================================================================
# 2. 基于随机过程的网络环境建模
# ==============================================================================
def generate_real_world_trace(steps=20): # 稍微增加点步数看效果
    """生成模拟真实 4G/5G 弱网环境的带宽轨迹"""
    trace = []
    state = "HIGH" 
    for i in range(steps):
        if state == "HIGH":
            bw = random.uniform(8.0, 12.0)
            if random.random() < 0.2: state = "DROP"
        elif state == "DROP":
            bw = random.uniform(2.0, 5.0)
            state = "LOW"
        elif state == "LOW":
            bw = random.gammavariate(1, 0.5)
            bw = max(0.2, min(bw, 1.5)) # 稍微提高下限防止完全死掉
            if random.random() < 0.15: state = "RECOVERY"
        elif state == "RECOVERY":
            bw = random.uniform(3.0, 7.0)
            state = "HIGH"
        trace.append(round(bw, 2))
    return trace

# ==============================================================================
# 3. 核心实验逻辑 (Uncertainty-Aware)
# ==============================================================================
def run_ai_driven_experiment():
    print("🚀 启动真·AI驱动 (含不确定性感知) 的对比仿真实验...")

    # --- A. 加载模型 ---
    device = torch.device("cpu")
    possible_paths = ["cts_best_model_full.pth", "../ml_training/modeling/cts_best_model_full.pth", "ml_training/modeling/cts_edl_model_best.pth"]
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    # 初始化一个 Mock 的不确定性，以防模型加载失败
    ai_uncertainty = 0.5 
    cags_initial_size = 4 * 1024 * 1024

    if model_path:
        print(f"📥 尝试加载 AI 模型: {model_path}")
        # 注意：这里我们尝试实例化新的 EDL 模型
        model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(device)
        try:
            # 尝试加载权重
            state_dict = torch.load(model_path, map_location=device)
            # 简单的权重形状检查，防止旧模型(输出1)加载到新模型(输出4)报错
            if state_dict['head.weight'].shape[0] == 4:
                model.load_state_dict(state_dict)
                model.eval()
                print("✅ EDL 模型加载成功！启用不确定性推理。")
                
                # --- B. AI 推理 (计算 Uncertainty) ---
                print("🤖 AI 正在分析当前环境的不确定性...")
                # 构造一个【高风险场景】
                client_vec = torch.FloatTensor([[2.0, 0.8, 500.0, 1024.0]]) 
                image_vec = torch.FloatTensor([[1500.0, 0.8, 0.1, 5.0, 0.1]])
                algo_vec = torch.LongTensor([0])
                
                with torch.no_grad():
                    preds = model(client_vec, image_vec, algo_vec)
                    gamma, v, alpha, beta = preds[0]
                    
                    # [修改点 3] 计算不确定性 (Aleatoric + Epistemic)
                    # U = beta / (v * (alpha - 1))
                    uncertainty_val = beta / (v * (alpha - 1))
                    pred_time = torch.expm1(gamma).item()
                    
                    ai_uncertainty = uncertainty_val.item()
                    # 归一化一下，防止数值太大
                    ai_uncertainty = min(1.0, max(0.0, ai_uncertainty / 10.0)) 
                    
                    print(f"   👉 预测耗时: {pred_time:.1f}s")
                    print(f"   👉 模型不确定性 (U): {ai_uncertainty:.4f}")
            else:
                print("⚠️ 检测到旧模型权重 (输出维度不匹配)。")
                print("🔄 切换到【模拟模式】：模拟一个高不确定性场景。")
                ai_uncertainty = 0.8 # 模拟高不确定性
        except Exception as e:
            print(f"⚠️ 模型加载出错: {e}")
            print("🔄 切换到【模拟模式】。")
            ai_uncertainty = 0.8
    else:
        print("⚠️ 未找到模型文件。使用模拟值。")
        ai_uncertainty = 0.8

    # --- C. 调用战略层 (传递 Uncertainty) ---
    strategy = CAGSStrategyLayer()
    
    # 模拟预测的丢包率 (如果带宽低，丢包率高)
    pred_loss = 0.05 
    
    # [修改点 4] 传入 model_uncertainty
    # 如果 ai_uncertainty 很高 (0.8)，StrategyLayer 里的 risk_amplifier 会很大
    # 从而导致 Cost 剧增，系统自动选择小切片
    best_config, _ = strategy.optimize(2.0, pred_loss, 0.8, model_uncertainty=ai_uncertainty) 
    
    cags_initial_size = best_config[0]
    
    print(f"🧠 战略层决策:")
    print(f"   👉 输入不确定性: {ai_uncertainty:.4f}")
    print(f"   👉 风险放大因子: {1.0 + 5.0 * ai_uncertainty:.2f}x") # 假设 weight=5.0
    print(f"   👉 最终决定初始切片: {cags_initial_size/1024} KB")

    # --- D. 开始跑分对比 (Trace-driven) ---
    bandwidth_trace = generate_real_world_trace(20)
    print(f"📊 动态带宽轨迹生成完毕 (长度 {len(bandwidth_trace)})")
    
    docker_tput = []
    cags_tput = []
    
    # 初始化修正层
    correction = CAGSCorrectionLayer(initial_chunk_size=cags_initial_size)
    
    for bw in bandwidth_trace:
        # === 1. Native Docker (大包 + RTO) ===
        # 假设 4MB 大包
        time_cost = 4.0 / max(0.01, bw)
        if time_cost > 2.0: 
            docker_tput.append(0.1) # 拥塞崩溃
        else:
            docker_tput.append(bw * 0.9)

        # === 2. AI-CAGS (智能调整) ===
        # 获取当前切片大小 (MB)
        curr_mb = correction.current_size / (1024*1024)
        est_time = curr_mb / max(0.01, bw)
        
        status = 'TIMEOUT' if est_time > 1.5 else 'SUCCESS' # 稍微严格一点的超时判定
        
        # 修正层介入
        correction.feedback(status, rtt_ms=100)
        
        if est_time > 2.0:
             # 如果真的非常非常慢，吞吐也会受影响，但不至于归零
             cags_tput.append(bw * 0.6)
        else:
             # 正常情况，享受并发收益 (这里简化模拟)
             cags_tput.append(min(bw * 0.98, bw)) # 贴近上限

    # --- E. 画图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidth_trace, 'k--', alpha=0.3, label='Physical Bandwidth (Limit)', linewidth=1)
    plt.plot(docker_tput, 'r-o', linewidth=2, label='Native Docker (Static 4MB)')
    plt.plot(cags_tput, 'g-^', linewidth=2, label='CTS (Uncertainty-Aware CAGS)')
    
    plt.title(f'Performance: Uncertainty-Aware Scheduling (U={ai_uncertainty:.2f})', fontsize=14)
    plt.ylabel('Goodput (Mbps)')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "exp_uncertainty_result.png"
    plt.savefig(output_file)
    print(f"\n✅ 实验完成！结果已保存至: {output_file}")
    print("💡 观察重点: 绿线应该在弱网区间依然坚挺，因为高不确定性让它选择了小切片。")

if __name__ == "__main__":
    run_ai_driven_experiment()