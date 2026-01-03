"""
探针服务端 - 模拟接收端处理探针请求
"""

from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

@app.route('/probe', methods=['POST'])
def handle_probe():
    """处理探针请求"""
    probe_type = request.args.get('type', 'unknown')
    
    # 模拟处理时间
    start_time = time.time()
    
    # 获取请求数据大小
    data = request.data
    data_size = len(data)
    
    # 模拟不同类型探针的处理
    if probe_type == 'B_net':
        # 网络探测 - 模拟网络传输
        time.sleep(random.uniform(0.01, 0.1))
        processing_time = time.time() - start_time
        
        # 根据数据大小和处理时间计算带宽 (Mbps)
        bandwidth = (data_size * 8) / (processing_time * 1024 * 1024)
        
        result = {
            'probe_type': 'B_net',
            'bandwidth': round(bandwidth, 2),
            'rtt': round(processing_time * 1000, 2),  # ms
            'loss_rate': random.uniform(0, 0.05)
        }
        
    elif probe_type == 'B_cpu':
        # CPU探测 - 模拟解压缩
        # 进行一些计算密集型操作来模拟解压
        dummy_calc = sum(i * i for i in range(int(1e6)))
        processing_time = time.time() - start_time
        
        # 模拟解压吞吐量 (MB/s)
        decompress_throughput = random.uniform(20, 200)
        
        result = {
            'probe_type': 'B_cpu',
            'decompress_throughput': round(decompress_throughput, 2),
            'processing_time': round(processing_time, 4)
        }
        
    elif probe_type == 'B_io':
        # IO探测 - 模拟文件写入
        time.sleep(random.uniform(0.001, 0.05))
        processing_time = time.time() - start_time
        
        # 模拟IO写入吞吐量 (MB/s)
        io_throughput = random.uniform(50, 500)
        
        result = {
            'probe_type': 'B_io',
            'io_write_throughput': round(io_throughput, 2),
            'processing_time': round(processing_time, 4)
        }
        
    else:
        result = {
            'probe_type': probe_type,
            'error': 'Unknown probe type'
        }
    
    print(f"处理探针类型: {probe_type}, 数据大小: {data_size} 字节")
    return jsonify(result)

@app.route('/')
def index():
    return '<h1>探针接收端服务</h1><p>服务正在运行，等待探针请求...</p>'

if __name__ == '__main__':
    print("探针服务端启动中...")
    print("请访问 http://localhost:5000 查看服务状态")
    print("探针端点: http://localhost:5000/probe?type=B_net|B_cpu|B_io")
    app.run(host='0.0.0.0', port=5000, debug=True)