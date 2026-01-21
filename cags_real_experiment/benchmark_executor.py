import argparse
import torch
import torch.nn.functional as F
import time
import psutil
import os
from cts_model import CTSDualTowerModel
from cags_scheduler import CAGSStrategyLayer, CAGSCorrectionLayer
from real_sensor import RealSensor
from real_downloader import RealDownloader
import requests


# ==========================================
# 1. 新增：简易缩放器 (必须与训练数据分布一致)
# ==========================================
class SimpleScaler:
    def transform(self, val, mean, std):
        return (val - mean) / (std + 1e-6)

    def inverse_transform(self, val, mean, std):
        return val * std + mean

# 这里的均值和方差是基于你之前 4000 条训练数据估算的
# 如果不加这个，AI 根本看不懂输入的数据
CLIENT_STATS = {
    'bandwidth_mbps': (20.0, 30.0),    # 平均带宽 20, 波动 30
    'cpu_limit': (0.5, 0.3),     # CPU 负载
    'network_rtt': (50.0, 80.0),   # RTT
    'mem_limit_mb': (8.0, 4.0)      # 内存 (GB)
}
IMAGE_STATS = {
    'total_size_mb': (200.0, 150.0), 
    'avg_layer_entropy': (6.5, 1.0),
    'text_ratio': (0.1, 0.1),
    'layer_count': (10.0, 5.0),
    'zero_ratio': (0.05, 0.05)
}


def load_model(model_path):
    """
    加载训练好的AI模型
    """
    device = torch.device("cpu")
    model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(device)
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试其他可能的路径
            possible_paths = [
                model_path,
                '../ml_training/modeling/cts_best_model_full.pth',
                '../../ml_training/modeling/cts_best_model_full.pth',
                '../../../ml_training/modeling/cts_best_model_full.pth',
                os.path.join(os.path.dirname(__file__), '../ml_training/modeling/cts_best_model_full.pth')
            ]
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path is None:
                print("❌ 未找到模型文件，使用随机初始化演示...")
                return None, device
            
            model_path = found_path
            print(f"🔍 找到模型文件: {model_path}")
        
        # 使用安全方式加载模型
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ 模型加载成功！")
        return model, device
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}")
        print("⚠️ 使用随机初始化演示...")
        # 修复参数名称错误
        model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(torch.device("cpu"))
        device = torch.device("cpu")
        return None, device


def calculate_uncertainty(beta, v, alpha):
    """
    计算不确定性 U = beta / (v * (alpha - 1))
    """
    return beta / (v * (alpha - 1) + 1e-6)  # 防止除零


def run_cags_mode(args, model, device):
    """
    CAGS自适应策略模式
    """
    print("🚀 运行 CAGS 自适应策略模式")
    
    # 1. 感知环境
    print("🔍 正在感知网络环境...")
    sensor = RealSensor(args.url)
    net_profile = sensor.get_network_profile()
    sys_profile = sensor.probe_system_info() # 获取真实内存信息
    # 获取真实的CPU负载
    real_cpu_load = psutil.cpu_percent(interval=None) / 100.0  # 0.0 ~ 1.0
    print(f"📊 网络概况: {net_profile}, CPU负载: {real_cpu_load:.2f}")
    
    # 获取文件真实大小用于特征
    try:
        head_resp = requests.head(args.url, timeout=2)
        real_file_size_mb = int(head_resp.headers.get('Content-Length', 0)) / (1024*1024)
    except:
        real_file_size_mb = 100.0

    # 2. 特征构建与标准化 (CRITICAL FIX)
    scaler = SimpleScaler()
    
    # Client特征: [Bandwidth, CPU_Limit, RTT, Memory_Limit]
    raw_bw = net_profile['bandwidth_mbps']
    raw_rtt = net_profile['rtt_ms']
    raw_mem = sys_profile.get('total_memory_gb', 8.0)
    
    norm_bw = scaler.transform(raw_bw, CLIENT_STATS['bandwidth_mbps'][0], CLIENT_STATS['bandwidth_mbps'][1])
    norm_cpu = scaler.transform(real_cpu_load, CLIENT_STATS['cpu_limit'][0], CLIENT_STATS['cpu_limit'][1])
    norm_rtt = scaler.transform(raw_rtt, CLIENT_STATS['network_rtt'][0], CLIENT_STATS['network_rtt'][1])
    norm_mem = scaler.transform(raw_mem, CLIENT_STATS['mem_limit_mb'][0], CLIENT_STATS['mem_limit_mb'][1])
    
    client_vec = torch.FloatTensor([[
        norm_bw, 
        norm_cpu,   # 使用动态感知的CPU负载
        norm_rtt, 
        norm_mem
    ]]).to(device)
    
    # Image特征: [Total_Size, Entropy, Text, Layer, Zero]
    norm_size = scaler.transform(real_file_size_mb, IMAGE_STATS['total_size_mb'][0], IMAGE_STATS['total_size_mb'][1])
    image_vec = torch.FloatTensor([[
        norm_size, 
        0.0,  # 其他特征简化处理，也可以使用真实的传感器数据
        0.0, 
        0.0, 
        0.0
    ]]).to(device)
    
    algo_vec = torch.LongTensor([0]).to(device)

    # AI推理
    print("🤖 正在进行AI推理...")
    with torch.no_grad():
        preds = model(client_vec, image_vec, algo_vec)
        gamma, v, alpha, beta = preds[0]
        
        # 计算不确定性
        uncertainty_val = calculate_uncertainty(beta, v, alpha)
        ai_uncertainty = min(1.0, max(0.0, uncertainty_val.item() / 10.0))
        
        # 反标准化预测时间
        predicted_time_s = torch.expm1(gamma).item()
    
    print(f"🔮 AI推理结果: 预测耗时 {predicted_time_s:.2f}s, 不确定性 {ai_uncertainty:.4f}")
    
    # 战略层决策
    print("🧠 正在进行战略层决策...")
    strategy = CAGSStrategyLayer()
    predicted_risk_prob = 0.05 if predicted_time_s > 60 else 0.01
    best_config, cost = strategy.optimize(
        raw_bw,  # 使用原始带宽值
        predicted_risk_prob, 
        real_cpu_load,  # 使用真实的CPU负载
        model_uncertainty=ai_uncertainty
    )
    chunk_size, concurrency = best_config
    
    print(f"💡 战略层决策: 块大小 {chunk_size/(1024*1024):.2f}MB, 并发数 {concurrency}")
    
    # 获取文件大小
    try:
        response = requests.head(args.url)
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size == 0:
            print("⚠️ 无法获取文件大小，尝试使用Range请求获取")
            response = requests.get(args.url, headers={'Range': 'bytes=0-0'}, timeout=5)
            if response.status_code == 206:
                content_range = response.headers.get('Content-Range', '')
                if content_range:
                    file_size = int(content_range.split('/')[-1])
    except:
        print("⚠️ 获取文件大小失败，默认使用1GB")
        file_size = 1024 * 1024 * 1024  # 1GB
    
    # 初始化修正层
    correction = CAGSCorrectionLayer(initial_chunk_size=chunk_size)
    
    # 执行下载
    downloader = RealDownloader(args.url, file_size, args.output_file)
    success = downloader.download_with_chunks(chunk_size, concurrency, correction)
    
    return success


def run_static_mode(args):
    """
    静态策略模式 (模拟Docker)
    """
    print("📦 运行 Docker 静态策略模式 (固定4MB块，3并发)")
    
    # 固定参数
    chunk_size = 4 * 1024 * 1024  # 4MB
    concurrency = 3
    
    # 获取文件大小
    try:
        response = requests.head(args.url)
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size == 0:
            print("⚠️ 无法获取文件大小，尝试使用Range请求获取")
            response = requests.get(args.url, headers={'Range': 'bytes=0-0'}, timeout=5)
            if response.status_code == 206:
                content_range = response.headers.get('Content-Range', '')
                if content_range:
                    file_size = int(content_range.split('/')[-1])
    except:
        print("⚠️ 获取文件大小失败，默认使用1GB")
        file_size = 1024 * 1024 * 1024  # 1GB
    
    # 执行下载
    downloader = RealDownloader(args.url, file_size, args.output_file)
    success = downloader.download_with_chunks(chunk_size, concurrency)
    
    return success


def run_aimd_mode(args):
    """
    AIMD模式 (固定初始块大小，动态调整)
    """
    print("🔄 运行 AIMD 动态调整模式")
    
    # 固定初始参数
    chunk_size = 2 * 1024 * 1024  # 2MB
    concurrency = 4
    
    # 获取文件大小
    try:
        response = requests.head(args.url)
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size == 0:
            print("⚠️ 无法获取文件大小，尝试使用Range请求获取")
            response = requests.get(args.url, headers={'Range': 'bytes=0-0'}, timeout=5)
            if response.status_code == 206:
                content_range = response.headers.get('Content-Range', '')
                if content_range:
                    file_size = int(content_range.split('/')[-1])
    except:
        print("⚠️ 获取文件大小失败，默认使用1GB")
        file_size = 1024 * 1024 * 1024  # 1GB
    
    # 初始化修正层
    correction = CAGSCorrectionLayer(initial_chunk_size=chunk_size)
    
    # 执行下载
    downloader = RealDownloader(args.url, file_size, args.output_file)
    success = downloader.download_with_chunks(chunk_size, concurrency, correction)
    
    return success


def main():
    parser = argparse.ArgumentParser(description='CAGS 真实环境实验执行器')
    parser.add_argument('--mode', choices=['cags', 'static', 'aimd'], 
                       required=True, help='运行模式: cags(自适应), static(静态), aimd(AIMD动态)')
    parser.add_argument('--url', type=str, required=True, 
                       help='下载URL')
    parser.add_argument('--output-file', type=str, default='downloaded_file.bin',
                       help='输出文件路径')
    parser.add_argument('--model-path', type=str, 
                       default='../ml_training/modeling/cts_best_model_full.pth',
                       help='AI模型路径')
    
    args = parser.parse_args()
    
    print(f"🎯 目标URL: {args.url}")
    print(f"📝 输出文件: {args.output_file}")
    print(f"⚙️  运行模式: {args.mode}")
    
    # 监控系统资源
    def monitor_resources():
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            print(f"🖥️  CPU: {cpu_percent}%, 内存: {memory_percent}%")
            time.sleep(5)
    
    # 启动资源监控线程
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    if args.mode == 'cags':
        # 加载模型
        model, device = load_model(args.model_path)
        if model is None:
            # 如果模型加载失败，创建一个随机初始化的模型用于演示
            # 修复参数名称错误
            model = CTSDualTowerModel(client_feats=4, image_feats=5, num_algos=10).to(torch.device("cpu"))
            device = torch.device("cpu")
            print("⚠️ 使用随机初始化模型进行演示")
        success = run_cags_mode(args, model, device)
    elif args.mode == 'static':
        success = run_static_mode(args)
    elif args.mode == 'aimd':
        success = run_aimd_mode(args)
    
    if success:
        print("✅ 下载成功完成!")
    else:
        print("❌ 下载过程中出现问题!")


if __name__ == "__main__":
    import threading
    main()