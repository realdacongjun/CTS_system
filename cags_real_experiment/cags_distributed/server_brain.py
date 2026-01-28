from flask import Flask, request, jsonify
import torch
import numpy as np
import sys
import os

# 导入现有的模型和调度器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cts_model import CTSDualTowerModel
from cags_scheduler import CAGSStrategyLayer

app = Flask(__name__)

# ==========================================
# 0. 全局配置
# ==========================================
# ⚠️ 替换为你的真实公网 IP
MY_PUBLIC_IP = "47.121.137.243" 
DEFAULT_DOWNLOAD_URL = f"http://{MY_PUBLIC_IP}/real_test.bin"

# ==========================================
# 1. 特征标准化器
# ==========================================
class SimpleScaler:
    def transform(self, val, mean, std):
        return (val - mean) / (std + 1e-6)

# 这里的均值和方差是基于训练数据估算的
CLIENT_STATS = {
    'bandwidth_mbps': (20.0, 30.0), 
    'cpu_load': (0.5, 0.3),          
    'network_rtt': (50.0, 80.0),      
    'memory_gb': (8.0, 4.0)          
}
IMAGE_STATS = {
    'total_size_mb': (200.0, 150.0), 
    'avg_layer_entropy': (6.5, 1.0),
    'text_ratio': (0.1, 0.1),
    'layer_count': (10.0, 5.0),
    'zero_ratio': (0.05, 0.05)
}

# ==========================================
# 2. 全局模型加载
# ==========================================
print("🔄 正在加载AI模型...")
device = torch.device("cpu")
# 注意：确保参数与 train.py 一致 (client=4, image=5)
model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(device)

try:
    # 自动寻找模型路径
    possible_paths = [
        "cts_best_model_full.pth",
        "../ml_training/modeling/cts_best_model_full.pth",
        "../../ml_training/modeling/cts_best_model_full.pth"
    ]
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if model_path:
        print(f"📥 加载权重文件: {model_path}")
        # 使用安全方式加载模型
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ AI模型加载成功！")
    else:
        raise FileNotFoundError("未找到 .pth 模型文件")
        
except Exception as e:
    print(f"❌ AI模型加载失败: {e}")
    print("⚠️  [演示模式] 使用随机初始化模型继续运行...")

def calculate_uncertainty(beta, v, alpha):
    """计算不确定性 U"""
    return beta / (v * (alpha - 1) + 1e-6) 

@app.route('/negotiate', methods=['POST'])
def negotiate_strategy():
    """
    核心接口：接收客户端信息 -> AI 推理 -> 战略决策 -> 返回策略
    """
    try:
        data = request.json
        client_info = data.get('client_info', {})
        image_info = data.get('image_info', {})
        server_info = data.get('server_info', {})
        
        # 获取基础 URL (例如 http://47.121.xx.xx/generalized_mixed.tar)
        base_url = server_info.get('download_url', DEFAULT_DOWNLOAD_URL)
        
        # --- A. 特征预处理 (保持不变) ---
        scaler = SimpleScaler()
        raw_bw = float(client_info.get('bandwidth_mbps', 10.0))
        raw_cpu = float(client_info.get('cpu_load', 0.5))
        raw_rtt = float(client_info.get('rtt_ms', 50.0))
        raw_mem = float(client_info.get('memory_gb', 4.0))
        
        # 标准化
        norm_bw = scaler.transform(raw_bw, *CLIENT_STATS['bandwidth_mbps'])
        norm_cpu = scaler.transform(raw_cpu, *CLIENT_STATS['cpu_load'])
        norm_rtt = scaler.transform(raw_rtt, *CLIENT_STATS['network_rtt'])
        norm_mem = scaler.transform(raw_mem, *CLIENT_STATS['memory_gb'])
        
        client_vec = torch.FloatTensor([[norm_bw, norm_cpu, norm_rtt, norm_mem]]).to(device)
        
        # Image 特征
        raw_size = float(image_info.get('size_mb', 100.0))
        norm_size = scaler.transform(raw_size, *IMAGE_STATS['total_size_mb'])
        image_vec = torch.FloatTensor([[norm_size, 0.5, 0.1, 5.0, 0.05]]).to(device)
        algo_vec = torch.LongTensor([0]).to(device)

        # --- B. AI 推理 (保持不变) ---
        with torch.no_grad():
            preds = model(client_vec, image_vec, algo_vec)
            gamma, v, alpha, beta = preds[0]
            
            uncertainty_val = calculate_uncertainty(beta, v, alpha)
            ai_uncertainty = min(1.0, max(0.0, uncertainty_val.item() / 10.0))
            predicted_time_s = torch.expm1(gamma).item()
        
        print(f"[AI Brain] 预测耗时: {predicted_time_s:.2f}s | 不确定性 U: {ai_uncertainty:.4f}")
        
        # --- C. 战略层决策 (Chunk & Concurrency) ---
        strategy = CAGSStrategyLayer()
        
        # 估算丢包率风险 (RTT越高，丢包风险越大)
        predicted_risk_prob = 0.05 if predicted_time_s > 60 else 0.01
        
        best_config, cost = strategy.optimize(
            predicted_bw_mbps=raw_bw, 
            predicted_loss_rate=predicted_risk_prob, 
            client_cpu_load=raw_cpu, 
            model_uncertainty=ai_uncertainty
        )
        chunk_size, concurrency = best_config
        
        print(f"[Strategy] 决策: 分片 {chunk_size/1024:.0f}KB | 并发 {concurrency}")
        
        # --- D. 压缩算法排序 & URL 构造 (⭐⭐⭐ 核心修改区 ⭐⭐⭐) ---
        
        # 1. 获取算法排序
        c_profile = {'bandwidth_mbps': raw_bw, 'cpu_score': 2000, 'decompression_speed': 200}
        i_profile = {'total_size_mb': raw_size, 'avg_layer_entropy': 0.65}
        
        if hasattr(strategy, 'predict_compression_times'):
            sorted_algorithms = strategy.predict_compression_times(c_profile, i_profile)
        else:
            sorted_algorithms = [('gzip-6', 0.0)] 

        # 2. 拿到 AI 认为最好的算法
        top_algo_name = sorted_algorithms[0][0]  # 例如 'brotli-6' 或 'lz4-fast'
        
        # 3. 映射算法名到文件后缀 (这步至关重要！)
        suffix_map = {
            'brotli': '.br',
            'zstd': '.zst',
            'lz4': '.lz4',
            'gzip': '.gz'
        }
        
        # 模糊匹配：只要名字里包含 'brotli' 就用 .br
        final_suffix = '.gz' # 默认
        for key, sfx in suffix_map.items():
            if key in top_algo_name:
                final_suffix = sfx
                break
        
        # 4. 构造最终下载链接
        # 假设 base_url 是 ".../file.tar"，我们要变成 ".../file.tar.br"
        # 先去掉可能存在的旧后缀，再加新后缀 (或者直接追加，取决于你生成文件的方式)
        # 简单起见，我们假设客户端传来的 url 是不带压缩后缀的基础名
        final_target_url = f"{base_url}{final_suffix}"
        
        print(f"[Server Decision] Algo:{top_algo_name} -> Suffix:{final_suffix} -> Threads:{concurrency}")

        # --- E. 返回响应 ---
        response_data = {
            'target_url': final_target_url,  # <--- 这里返回的是带后缀的 URL
            'strategy': {
                'initial_chunk_size': int(chunk_size),
                'concurrency': int(concurrency)
            },
            'meta_info': {
                'predicted_time_s': predicted_time_s,
                'uncertainty': ai_uncertainty,
                'cost': float(cost),
                'top_algorithms': sorted_algorithms[:3],
                'selected_algo': top_algo_name
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ [Server] 决策异常: {e}")
        import traceback
        traceback.print_exc()
        # 返回保底策略
        return jsonify({
            'target_url': DEFAULT_DOWNLOAD_URL,
            'strategy': {'initial_chunk_size': 1024*1024, 'concurrency': 2},
            'meta_info': {'error': str(e), 'uncertainty': 1.0}
        })

if __name__ == '__main__':
    # 监听 0.0.0.0 才能被公网访问
    app.run(host='0.0.0.0', port=5000, debug=False)