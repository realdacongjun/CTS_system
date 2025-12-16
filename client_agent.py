"""
客户端代理模块
用于模拟客户端接收并响应探针请求
"""

from flask import Flask, request, jsonify
import time
import gzip
import zlib
import psutil
import cpuinfo
import random

app = Flask(__name__)

@app.route('/probe', methods=['POST'])
def handle_probe():
    probe_type = request.args.get('type')
    
    if probe_type == 'B_net':
        # 网络探测 - 模拟网络性能
        data_size = len(request.data)
        # 模拟网络延迟和带宽限制
        simulated_delay = random.uniform(0.05, 0.2)  # 50-200ms延迟
        time.sleep(simulated_delay)
        
        # 模拟带宽计算 (假设100Mbps带宽)
        bandwidth_mbps = random.uniform(30, 70)  # 30-70 Mbps
        rtt = simulated_delay * 1000  # 转换为毫秒
        loss_rate = random.uniform(0, 0.05)  # 0-5%丢包率
        
        return jsonify({
            'bandwidth': round(bandwidth_mbps, 2),
            'rtt': round(rtt, 2),
            'loss_rate': round(loss_rate, 4)
        })
    
    elif probe_type == 'B_cpu':
        # 解压探测 - 模拟CPU解压性能
        try:
            # 获取请求数据
            if request.is_json:
                data = request.get_json()
                algorithm = data.get('algorithm', 'gzip')
                compressed_data = data.get('data', '').encode() if isinstance(data.get('data'), str) else data.get('data', b'')
            else:
                algorithm = 'gzip'
                compressed_data = request.data
            
            # 模拟解压操作时间
            data_size_mb = len(compressed_data) / (1024 * 1024)
            
            # 根据算法和数据大小模拟解压时间
            if algorithm == 'gzip':
                # 模拟gzip解压速度 50-200 MB/s
                speed = random.uniform(50, 200)
            elif algorithm == 'zlib':
                # 模拟zlib解压速度 80-250 MB/s
                speed = random.uniform(80, 250)
            else:
                speed = 100
            
            # 模拟解压时间
            duration = data_size_mb / speed if speed > 0 else 0.1
            time.sleep(min(duration, 1.0))  # 限制最大等待时间
            
            return jsonify({
                'throughput': round(speed, 2),
                'algorithm': algorithm,
                'duration': round(duration, 4),
                'data_size_mb': round(data_size_mb, 2)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif probe_type == 'B_io':
        # IO探测 - 模拟磁盘IO性能
        data_size = len(request.data)
        data_size_mb = data_size / (1024 * 1024)
        
        # 模拟IO写入时间 (假设磁盘写入速度 50-300 MB/s)
        io_speed = random.uniform(50, 300)
        duration = data_size_mb / io_speed if io_speed > 0 else 0.1
        time.sleep(min(duration, 1.0))  # 限制最大等待时间
        
        return jsonify({
            'io_write_throughput': round(io_speed, 2),
            'data_size_mb': round(data_size_mb, 2),
            'duration': round(duration, 4)
        })
    
    else:
        return jsonify({'error': f'Unknown probe type: {probe_type}'}), 400

def get_client_capabilities():
    """
    获取客户端能力信息
    """
    try:
        # 获取CPU信息
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get('brand_raw', 'Unknown')
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        # 获取内存信息
        mem_info = psutil.virtual_memory()
        total_memory_gb = round(mem_info.total / (1024**3), 2)
        
        # 获取磁盘信息
        disk_info = psutil.disk_usage('/')
        free_disk_gb = round(disk_info.free / (1024**3), 2)
        
        return {
            'cpu': {
                'name': cpu_name,
                'cores': cpu_cores,
                'threads': cpu_threads
            },
            'memory': {
                'total_gb': total_memory_gb
            },
            'disk': {
                'free_gb': free_disk_gb
            }
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/capabilities', methods=['GET'])
def client_capabilities():
    """
    返回客户端能力信息
    """
    return jsonify(get_client_capabilities())

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"客户端代理启动中，监听端口: {port}")
    print("支持的端点:")
    print("  POST /probe?type=B_net    - 网络探测")
    print("  POST /probe?type=B_cpu    - CPU解压探测")
    print("  POST /probe?type=B_io     - IO探测")
    print("  GET  /capabilities        - 客户端能力信息")
    print(f"\n使用方法: python main.py http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)

"""
客户端能力注册工具
用于一次性执行探针并注册节点能力，替代需要持续运行的服务模式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.client_probe import ClientProbe
from modules.capability_registry import CapabilityRegistry


def register_node_capability():
    """
    执行一次性节点能力探针并注册
    """
    print("开始执行节点能力探针...")
    print("=" * 40)
    
    # 执行本地探测
    client_probe = ClientProbe()
    capability_data = client_probe.get_client_profile()
    
    # 生成节点ID（在实际应用中，这应该基于更稳定的硬件特征）
    registry = CapabilityRegistry()
    client_info = {
        "ip": "localhost",  # 在实际应用中应该获取真实IP
        "user_agent": "Client_Register_Tool/1.0"
    }
    node_id = registry.generate_node_id(client_info)
    
    # 注册能力
    success = registry.register_capability(node_id, capability_data)
    
    if success:
        print("✓ 节点能力注册成功!")
        print(f"  节点ID: {node_id}")
        print(f"  CPU评分: {capability_data.get('cpu_score')}")
        print(f"  网络带宽: {capability_data.get('bandwidth_mbps')} Mbps")
        print(f"  解压速度: {capability_data.get('decompression_speed')}")
        print("\n请保存此节点ID，后续请求中使用以获得优化服务:")
        print(f"  python main.py node:{node_id}")
    else:
        print("✗ 节点能力注册失败!")


def show_registered_nodes():
    """
    显示已注册的节点
    """
    registry = CapabilityRegistry()
    capabilities = registry.capabilities
    
    if not capabilities:
        print("暂无已注册节点")
        return
    
    print("已注册节点列表:")
    print("=" * 40)
    for node_id, capability in capabilities.items():
        print(f"节点ID: {node_id}")
        print(f"  CPU评分: {capability.cpu_score}")
        print(f"  网络带宽: {capability.bandwidth_mbps} Mbps")
        print(f"  最后更新: {capability.last_updated}")
        print(f"  置信度: {capability.confidence}")
        print("-" * 30)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            show_registered_nodes()
        else:
            print("用法:")
            print("  python client_agent.py          # 注册当前节点能力")
            print("  python client_agent.py list     # 显示已注册节点")
    else:
        register_node_capability()


if __name__ == "__main__":
    main()
