"""
CTS Client Node (需求端)
运行位置：用户电脑 / 边缘设备
职责：自检硬件 -> 发送画像 -> 获取最优策略 -> 执行拉取
"""
import requests
import time
from cts_core import ClientProbe

# 配置服务端的 IP 地址 (本地测试用 localhost，远程用公网IP)
SERVER_URL = "http://39.106.147.155:8000"

def smart_pull(image_name):
    print(f"\n{'='*40}")
    print(f"🚀 [Client] 准备拉取镜像: {image_name}")
    
    # 1. 自身体检 (运行 ClientProbe)
    print("🏥 [Client] 正在进行环境感知 (CPU/网络)...")
    probe = ClientProbe()
    my_profile = probe.probe() # 这一步是真实的！
    
    print(f" -> 测得带宽: {my_profile['bandwidth_mbps']} Mbps")
    print(f" -> 测得算力: {my_profile['cpu_score']} 分")

    # 2. 发送请求给大脑 (携带画像)
    payload = {
        "image_name": image_name,
        "client_profile": my_profile
    }
    
    try:
        print("📨 [Client] 向决策中心发送请求...")
        t0 = time.time()
        response = requests.post(SERVER_URL, json=payload)
        rtt = (time.time() - t0) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ [Client] 收到决策指令 (RTT: {rtt:.1f}ms):")
            print(f" -> 🎯 推荐算法: [{result['strategy']}]")
            print(f" -> 💡 决策理由: {result['reason']}")
            
            # 3. (模拟) 根据指令开始下载
            print(f" -> ⬇️ 开始使用 {result['strategy']} 协议下载数据流...")
            # real_download(result['download_url'])
        else:
            print(f"❌ 服务端错误: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 连接失败: {e} (请确认 server_node.py 是否已启动)")

if __name__ == "__main__":
    # 模拟用户行为
    smart_pull("redis:latest")
    
    # 可以模拟休息一会儿再拉另一个
    # time.sleep(2)
    # smart_pull("mysql:latest")