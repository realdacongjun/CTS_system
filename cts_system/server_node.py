"""
CTS Server Node (服务端)
运行位置：云服务器
职责：接收请求 -> 分析镜像 -> AI决策 -> 下发策略
"""
from flask import Flask, request, jsonify
from cts_core import ImageAnalyzer, CompressionCachePool
from brain import DecisionBrain
import time

app = Flask(__name__)

# 1. 初始化核心组件 (只初始化一次，常驻内存)
print("🚀 [Server] 正在初始化 CTS 决策引擎...")
analyzer = ImageAnalyzer()
brain = DecisionBrain()  # 加载 PyTorch 模型
cache = CompressionCachePool()
print("✅ [Server] 服务已就绪，监听 8000 端口...")


# server_node.py 的补充部分

@app.route('/ping', methods=['GET'])
def handle_ping():
    """配合 ClientProbe 测 RTT"""
    return "pong"

@app.route('/speedtest', methods=['GET'])
def handle_speedtest():
    """配合 ClientProbe 测带宽"""
    # 生成 1MB 的随机垃圾数据
    size_mb = 1 
    return os.urandom(size_mb * 1024 * 1024)
@app.route('/pull_request', methods=['POST'])
def handle_pull():
    """
    处理客户端的拉取请求
    """
    start_time = time.time()
    data = request.json
    
    image_name = data.get('image_name')
    client_profile = data.get('client_profile') # 接收客户端传来的体检报告
    
    print(f"\n📡 [收到请求] 客户端想拉取: {image_name}")
    print(f"📝 [客户端画像] 带宽: {client_profile['bandwidth_mbps']}Mbps, CPU分: {client_profile['cpu_score']}")

    # Step 1: 分析镜像 (右塔)
    # (在真实系统中，这里应该查数据库，而不是现场分析)
    img_feats = analyzer.analyze(image_name)
    
    # Step 2: AI 决策 (Brain)
    best_algo, reason = brain.make_decision(client_profile, img_feats)
    
    # Step 3: 检查缓存 (Execution)
    cache_key = f"{image_name}_{best_algo}"
    hit_cache = cache.get(cache_key) is not None
    
    # 模拟处理耗时
    process_time = (time.time() - start_time) * 1000
    
    print(f"🧠 [决策结果] 推荐算法: {best_algo} (依据: {reason})")
    print(f"⏱️ [处理耗时] {process_time:.2f}ms")

    # 返回指令给客户端
    return jsonify({
        "status": "success",
        "strategy": best_algo,
        "reason": reason,
        "cache_hit": hit_cache,
        "download_url": f"http://cts-repo/{image_name}/layer?algo={best_algo}" # 模拟链接
    })

if __name__ == '__main__':
    # 监听 0.0.0.0 代表允许外部访问
    app.run(host='0.0.0.0', port=8000)