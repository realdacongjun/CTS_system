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
# 1. 特征标准化器
# ==========================================
class SimpleScaler:
    def transform(self, val, mean, std):
        return (val - mean) / (std + 1e-6)

# 这里的均值和方差是基于训练数据估算的
CLIENT_STATS = {
    'bandwidth_mbps': (20.0, 30.0),    # 平均带宽 20, 波动 30
    'cpu_load': (0.5, 0.3),           # CPU 负载
    'network_rtt': (50.0, 80.0),      # RTT
    'memory_gb': (8.0, 4.0)           # 内存 (GB)
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
model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(device)

try:
    # 模型路径可能需要根据实际情况调整
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_training', 'modeling', 'cts_best_model_full.pth')
    # 使用安全方式加载模型
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ AI模型加载成功！")
except Exception as e:
    print(f"❌ AI模型加载失败: {e}")
    print("⚠️  使用随机初始化模型进行演示")
    # 创建随机模型继续运行
    pass

def calculate_uncertainty(beta, v, alpha):
    """
    计算不确定性 U = beta / (v * (alpha - 1))
    """
    return beta / (v * (alpha - 1) + 1e-6)  # 防止除零

@app.route('/negotiate', methods=['POST'])
def negotiate_strategy():
    """
    接收客户端环境信息，返回AI决策的下载策略
    """
    try:
        # 获取客户端上报的环境信息
        data = request.json
        client_info = data.get('client_info', {})
        image_info = data.get('image_info', {})
        server_info = data.get('server_info', {})
        
        print(f"[Server] 接收到客户端请求: {client_info}")
        
        # 构建标准化输入特征
        scaler = SimpleScaler()
        
        # Client特征: [Bandwidth, CPU_Limit, RTT, Memory_Limit]
        raw_bw = client_info.get('bandwidth_mbps', 10.0)
        raw_cpu = client_info.get('cpu_load', 0.5)
        raw_rtt = client_info.get('rtt_ms', 50.0)
        raw_mem = client_info.get('memory_gb', 4.0)
        
        norm_bw = scaler.transform(raw_bw, CLIENT_STATS['bandwidth_mbps'][0], CLIENT_STATS['bandwidth_mbps'][1])
        norm_cpu = scaler.transform(raw_cpu, CLIENT_STATS['cpu_load'][0], CLIENT_STATS['cpu_load'][1])
        norm_rtt = scaler.transform(raw_rtt, CLIENT_STATS['network_rtt'][0], CLIENT_STATS['network_rtt'][1])
        norm_mem = scaler.transform(raw_mem, CLIENT_STATS['memory_gb'][0], CLIENT_STATS['memory_gb'][1])
        
        client_vec = torch.FloatTensor([[
            norm_bw, 
            norm_cpu, 
            norm_rtt, 
            norm_mem
        ]]).to(device)
        
        # Image特征: [Total_Size, Entropy, Text, Layer, Zero]
        raw_size = image_info.get('size_mb', 100.0)
        norm_size = scaler.transform(raw_size, IMAGE_STATS['total_size_mb'][0], IMAGE_STATS['total_size_mb'][1])
        
        image_vec = torch.FloatTensor([[
            norm_size, 
            0.0,  # 其他特征简化处理
            0.0, 
            0.0, 
            0.0
        ]]).to(device)
        
        algo_vec = torch.LongTensor([0]).to(device)

        # AI推理
        with torch.no_grad():
            preds = model(client_vec, image_vec, algo_vec)
            gamma, v, alpha, beta = preds[0]
            
            # 计算不确定性
            uncertainty_val = calculate_uncertainty(beta, v, alpha)
            ai_uncertainty = min(1.0, max(0.0, uncertainty_val.item() / 10.0))
            
            # 反标准化预测时间
            predicted_time_s = torch.expm1(gamma).item()
        
        print(f"[Server] AI推理结果: 预测耗时 {predicted_time_s:.2f}s, 不确定性 {ai_uncertainty:.4f}")
        
        # 战略层决策
        strategy = CAGSStrategyLayer()
        predicted_risk_prob = 0.05 if predicted_time_s > 60 else 0.01
        best_config, cost = strategy.optimize(
            raw_bw,  # 使用原始带宽值
            predicted_risk_prob, 
            raw_cpu,  # 使用原始CPU负载
            model_uncertainty=ai_uncertainty
        )
        chunk_size, concurrency = best_config
        
        print(f"[Server] 战略层决策: 块大小 {chunk_size/(1024*1024):.2f}MB, 并发数 {concurrency}")
        
        # 返回决策结果
        response_data = {
            'target_url': server_info.get('download_url', 'http://192.168.1.100:80/download'),  # IP占位符，需根据实际环境修改
            'strategy': {
                'initial_chunk_size': int(chunk_size),
                'concurrency': int(concurrency)
            },
            'meta_info': {
                'predicted_time_s': predicted_time_s,
                'uncertainty': ai_uncertainty,
                'cost': float(cost)
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"[Server] 决策过程出错: {e}")
        # 返回默认策略
        return jsonify({
            'target_url': 'http://192.168.1.100:80/download',  # IP占位符，需根据实际环境修改
            'strategy': {
                'initial_chunk_size': 1024*1024,  # 1MB
                'concurrency': 2
            },
            'meta_info': {
                'predicted_time_s': 0,
                'uncertainty': 0.1,
                'cost': 1.0,
                'error': str(e)
            }
        })

if __name__ == '__main__':
    # 注意：IP地址需要根据实际部署环境修改
    # 示例中使用 192.168.1.100 作为服务端IP，请根据实际环境修改
    app.run(host='0.0.0.0', port=5000, debug=False)