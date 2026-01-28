from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# 加载脚本 B 生成的智慧成果
with open("pareto_rules.json", "r") as f:
    POLICY_MAP = json.load(f)

# 风险势垒阈值 (基于实验数据确定的临界点)
LOSS_BARRIER_THRESHOLD = 5.0 # 丢包率超过 5% 视为“高危区域”

@app.route('/negotiate', methods=['POST'])
def negotiate():
    data = request.json
    client = data.get('client_info', {})
    image = data.get('image_info', {})
    
    # 1. 实时感知 (Telemetry)
    bw = client.get('bandwidth_mbps', 10)
    loss = client.get('loss_rate', 0)
    rtt = client.get('rtt_ms', 50)
    file_id = image.get('file_id', 'Med_Mixed') # 也可以根据 AI 预测的大小映射

    # 2. 场景映射 (Scenario Mapping)
    if loss > 4 or bw < 5: 
        scene = "Weak"
    elif bw > 50 and loss < 1: 
        scene = "Cloud"
    else: 
        scene = "Edge"

    # 3. 基础策略查表 (Pareto Lookup)
    base_strategy = POLICY_MAP.get(scene, {}).get(file_id, {"threads": 2, "chunk_mb": -1})
    
    final_threads = base_strategy['threads']
    final_chunk = base_strategy['chunk_mb']
    
    # 4. 核心创新：风险势垒触发 (Risk Barrier Activation)
    # 不管帕累托选了多少，只要网络触发高危红线，强制降为 1 线程确保“生存”
    barrier_active = False
    if loss >= LOSS_BARRIER_THRESHOLD:
        final_threads = 1
        final_chunk = -1 # 回退到最稳的自动分片模式
        barrier_active = True

    # 5. 返回决策 (Decision Response)
    return jsonify({
        "status": "success",
        "decision": {
            "threads": final_threads,
            "chunk_size_mb": final_chunk,
            "compression_algo": "zstd", # 这里本该由 AI 脚本 B 决定，我们先假设
            "meta": {
                "scenario": scene,
                "barrier_triggered": barrier_active,
                "theory": "Pareto Knee Point"
            }
        }
    })

if __name__ == "__main__":
    # 部署在 2核2G 的服务端机器上
    app.run(host='0.0.0.0', port=5000)