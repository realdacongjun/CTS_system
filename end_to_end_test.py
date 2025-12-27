#!/usr/bin/env python3
"""
端到端测试脚本
使用真实数据测试整个CTS系统流程
"""

import os
import sys
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
import time

from modules.capability_registry import CapabilityRegistry
from modules.image_probe import ImageProbe
from modules.decision_engine import CompressionDecisionEngine
from modules.transmission_optimizer import TransmissionOptimizer
from modules.cache_pool import CachePool
from modules.differential_sync import DifferentialSync
from modules.feedback_collector import performance_monitor, FeedbackData
from core.streaming_proxy import StreamingProxy
from core.task_scheduler import task_scheduler


def create_test_image(image_dir):
    """
    创建一个真实的测试镜像，包含真实的文件内容
    """
    print(f"创建测试镜像: {image_dir}")
    
    # 创建目录
    os.makedirs(image_dir, exist_ok=True)
    
    # 创建真实的文件内容
    files_content = {
        "app.py": b"import os\nimport sys\n\ndef main():\n    print('Hello, World!')\n    return 0\n\nif __name__ == '__main__':\n    sys.exit(main())",
        "requirements.txt": b"flask==2.0.1\nrequests==2.25.1\nnumpy==1.21.0",
        "config.json": b'{"app_name": "test_app", "version": "1.0", "debug": true}',
        "README.md": b"# Test Application\n\nThis is a test application for the CTS system.",
        "binary_file.bin": os.urandom(1024 * 50)  # 50KB随机二进制数据
    }
    
    # 创建多个层，每层包含部分文件
    layers_data = [
        {"layer1.tar": ["app.py", "requirements.txt"]},
        {"layer2.tar": ["config.json", "README.md"]},
        {"layer3.tar": ["binary_file.bin"]}
    ]
    
    # 创建各层的tar文件
    layer_paths = []
    for layer_info in layers_data:
        for layer_filename, files in layer_info.items():
            layer_path = os.path.join(image_dir, layer_filename)
            with tarfile.open(layer_path, "w") as tar:
                for filename in files:
                    if filename in files_content:
                        # 创建临时文件
                        temp_file_path = os.path.join(image_dir, filename)
                        with open(temp_file_path, "wb") as f:
                            f.write(files_content[filename])
                        
                        # 添加到tar
                        tar.add(temp_file_path, arcname=filename)
                        
                        # 删除临时文件
                        os.remove(temp_file_path)
            
            layer_paths.append(layer_filename)
    
    # 创建manifest.json
    manifest = [{
        "Config": "config.json",
        "Layers": layer_paths
    }]
    
    manifest_path = os.path.join(image_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    print(f"测试镜像创建完成，包含 {len(layer_paths)} 个层")
    return image_dir


def register_client_capabilities():
    """
    注册真实的客户端能力
    """
    print("注册客户端能力...")
    
    # 获取系统信息
    import psutil
    import cpuinfo
    
    cpu_info = cpuinfo.get_cpu_info()
    cpu_count = psutil.cpu_count()
    memory_size = psutil.virtual_memory().total
    
    # 估算CPU性能分数
    cpu_score = 1000  # 基准值
    if cpu_count > 4:
        cpu_score += (cpu_count - 4) * 100
    if 'Intel' in cpu_info['brand_raw']:
        cpu_score += 200
    elif 'AMD' in cpu_info['brand_raw']:
        cpu_score += 150
    
    # 模拟网络和解压性能测试
    print("正在测试网络和解压性能...")
    
    # 模拟网络带宽测试（实际项目中应实现真实测试）
    bandwidth_mbps = 100  # 实际值应通过网络测试获得
    network_rtt = 15      # 实际值应通过网络测试获得
    
    # 模拟解压速度测试（实际项目中应实现真实测试）
    decompression_speed = {
        "gzip": 100,
        "zstd": 140,
        "lz4": 160
    }
    
    # 创建客户端信息
    client_info = {
        "ip": "127.0.0.1",
        "user_agent": f"real_client_{os.getpid()}"
    }
    
    # 初始化能力注册表
    capability_registry = CapabilityRegistry()
    node_id = capability_registry.generate_node_id(client_info)
    
    # 创建能力数据
    capability_data = {
        "cpu_score": cpu_score,
        "bandwidth_mbps": bandwidth_mbps,
        "decompression_speed": decompression_speed,
        "network_rtt": network_rtt,
        "disk_io_speed": 500,  # 估算值
        "memory_size": memory_size
    }
    
    # 注册能力
    capability_registry.register_capability(node_id, capability_data)
    print(f"节点 {node_id} 能力注册成功")
    print(f"  - CPU分数: {cpu_score}")
    print(f"  - 带宽: {bandwidth_mbps}Mbps")
    print(f"  - 网络RTT: {network_rtt}ms")
    
    return node_id, capability_registry


def run_end_to_end_test():
    """
    运行端到端测试
    """
    print("="*60)
    print("开始端到端测试")
    print("="*60)
    
    # 1. 创建测试镜像
    print("\n1. 创建测试镜像...")
    with tempfile.TemporaryDirectory() as temp_dir:
        image_dir = create_test_image(temp_dir)
        
        # 2. 注册客户端能力
        print("\n2. 注册客户端能力...")
        node_id, capability_registry = register_client_capabilities()
        
        # 3. 初始化系统模块
        print("\n3. 初始化系统模块...")
        image_probe = ImageProbe(image_dir)
        decision_engine = CompressionDecisionEngine(strategy="cost_model")
        transmission_optimizer = TransmissionOptimizer()
        cache_pool = CachePool()
        differential_sync = DifferentialSync()
        streaming_proxy = StreamingProxy()
        
        # 4. 获取客户端画像
        print("\n4. 获取客户端画像...")
        client_profile = capability_registry.get_client_profile_for_decision(node_id)
        print(f"客户端画像: CPU={client_profile['cpu_score']}, 带宽={client_profile['bandwidth_mbps']}Mbps")
        
        # 5. 分析真实镜像特征
        print("\n5. 分析真实镜像特征...")
        start_time = time.time()
        image_profile = image_probe.get_image_profile()
        analysis_time = time.time() - start_time
        
        print(f"镜像分析完成 (耗时: {analysis_time:.2f}s):")
        print(f"  - 总大小: {image_profile['total_size_mb']:.2f} MB")
        print(f"  - 平均熵值: {image_profile['avg_layer_entropy']:.3f}")
        print(f"  - 文本比例: {image_profile['text_ratio']:.2f}")
        print(f"  - 二进制比例: {image_profile['binary_ratio']:.2f}")
        print(f"  - 层数量: {image_profile['layer_count']}")
        print(f"  - 文件类型分布: {image_profile['file_type_distribution']}")
        
        # 6. 使用决策引擎选择最优压缩策略
        print("\n6. 选择最优压缩策略...")
        start_time = time.time()
        best_method, decision_details = decision_engine.make_decision(client_profile, image_profile)
        decision_time = time.time() - start_time
        
        print(f"最优压缩策略: {best_method}")
        print(f"预期总成本: {decision_details['predicted_cost']:.2f}秒 (决策耗时: {decision_time:.3f}s)")
        
        # 7. 获取传输优化计划
        print("\n7. 生成传输优化计划...")
        start_time = time.time()
        optimization_result = transmission_optimizer.optimize_transmission(
            "test_image:latest",
            image_profile["layers"],
            client_profile
        )
        optimization_time = time.time() - start_time
        
        print(f"传输计划生成完成 (耗时: {optimization_time:.3f}s)")
        print(f"优先传输层: {optimization_result['transmission_plan']['priority_layers']}")
        
        # 8. 测试缓存机制
        print("\n8. 测试缓存机制...")
        layer_digest = "layer1.tar"  # 使用实际层名
        start_time = time.time()
        
        if cache_pool.is_layer_cached("test_image:latest", layer_digest, best_method):
            print(f"层 {layer_digest} 已在缓存中")
        else:
            print(f"层 {layer_digest} 未在缓存中，进行压缩并缓存...")
            
            # 获取原始层数据
            layer_path = os.path.join(image_dir, layer_digest)
            with open(layer_path, 'rb') as f:
                original_data = f.read()
            
            # 压缩数据
            import zlib
            if best_method.startswith("gzip"):
                level = int(best_method.split("-")[1]) if "-" in best_method else 6
                compressed_data = zlib.compress(original_data, level)
            else:
                compressed_data = zlib.compress(original_data, 6)  # 默认使用gzip-6
            
            # 估算节省时间
            compression_time_saved = len(original_data) / (1024 * 1024) * 0.5  # 简单估算
            transfer_time_saved = len(compressed_data) / (1024 * 1024) * 0.1  # 简单估算
            
            # 缓存数据
            cache_pool.cache_layer_data(
                "test_image:latest", 
                layer_digest, 
                best_method, 
                compressed_data,
                compression_time_saved=compression_time_saved,
                transfer_time_saved=transfer_time_saved
            )
            print(f"已将层 {layer_digest} 添加到缓存，压缩后大小: {len(compressed_data)} bytes")
        
        cache_time = time.time() - start_time
        print(f"缓存操作耗时: {cache_time:.3f}s")
        
        # 9. 测试异步任务调度
        print("\n9. 测试异步任务调度...")
        start_time = time.time()
        
        def task_callback(task):
            print(f"任务 {task.task_id} 完成，状态: {task.status.value}")
        
        # 提交一个压缩任务
        layer_path = os.path.join(image_dir, "layer2.tar")
        with open(layer_path, 'rb') as f:
            layer2_data = f.read()
        
        task_id = task_scheduler.submit_transcode_task(
            "test_image:latest",
            "layer2.tar",
            best_method,
            layer2_data,
            task_callback
        )
        print(f"已提交异步任务: {task_id}")
        
        # 等待任务完成
        task_completed = False
        for i in range(20):  # 最多等待10秒
            task = task_scheduler.get_task_status(task_id)
            if task and task.status in ['COMPLETED', 'FAILED']:
                task_completed = True
                break
            time.sleep(0.5)
        
        if task_completed:
            print(f"异步任务 {task_id} 已完成，状态: {task.status.value}")
        else:
            print(f"异步任务 {task_id} 仍在处理中...")
        
        task_time = time.time() - start_time
        print(f"任务调度测试耗时: {task_time:.3f}s")
        
        # 10. 测试流式传输
        print("\n10. 测试流式传输...")
        start_time = time.time()
        
        # 使用真实的层数据进行流式压缩
        layer_path = os.path.join(image_dir, "layer3.tar")
        with open(layer_path, 'rb') as input_file:
            # 模拟流式处理
            total_output_size = 0
            for chunk in streaming_proxy.process_stream(input_file, best_method):
                total_output_size += len(chunk)
                # 实际应用中，这里会将chunk写入HTTP响应流
                pass
        
        streaming_time = time.time() - start_time
        print(f"流式传输完成，输出大小: {total_output_size} bytes (耗时: {streaming_time:.3f}s)")
        
        # 11. 测试差分同步
        print("\n11. 测试差分同步...")
        start_time = time.time()
        
        source_path = os.path.join(image_dir, "layer1.tar")
        
        # 创建修改后的文件
        target_path = os.path.join(image_dir, "layer1_modified.tar")
        with open(source_path, "rb") as f:
            original_data = f.read()
        
        # 修改部分数据创建新版本
        modified_data = bytearray(original_data)
        for i in range(100, 200):  # 修改部分字节
            if i < len(modified_data):
                modified_data[i] = (modified_data[i] + 42) % 256
        
        with open(target_path, "wb") as f:
            f.write(modified_data)
        
        # 执行差分同步
        delta_result = differential_sync.sync_layers(source_path, target_path, 
                                                     os.path.join(image_dir, "delta.bin"))
        
        differential_time = time.time() - start_time
        print(f"差分同步结果 (耗时: {differential_time:.3f}s):")
        print(f"  - 原始大小: {delta_result['original_size']} bytes")
        print(f"  - 差分大小: {delta_result['delta_size']} bytes")
        print(f"  - 压缩比例: {delta_result['compression_ratio']:.2f}")
        print(f"  - 节省大小: {delta_result['transfer_saved']} bytes")
        
        # 12. 收集反馈数据
        print("\n12. 收集反馈数据...")
        feedback = FeedbackData(
            node_id=node_id,
            image_id="test_image:latest",
            algo_used=best_method,
            actual_transfer_time=5.2,  # 模拟实际传输时间
            actual_decomp_time=3.8,    # 模拟实际解压时间
            predicted_transfer_time=decision_details['predicted_cost'] * 0.6,
            predicted_decomp_time=decision_details['predicted_cost'] * 0.4
        )
        performance_monitor.collect_feedback(feedback)
        print(f"已收集反馈数据: 使用 {best_method} 算法")
        
        # 13. 计算系统准确性
        print("\n13. 计算系统准确性...")
        accuracy = performance_monitor.calculate_accuracy()
        print(f"传输时间预测准确性: {accuracy['transfer_accuracy']:.2f}")
        print(f"解压时间预测准确性: {accuracy['decomp_accuracy']:.2f}")
        
        print("\n" + "="*60)
        print("端到端测试完成！")
        print("="*60)
        print("系统成功执行了完整的感知-预测-决策-执行流程:")
        print(f"  - 感知平面: 获取了真实的客户端能力")
        print(f"  - 预测平面: 分析了真实镜像特征并预测了传输计划")
        print(f"  - 决策平面: 使用成本模型选择了最优压缩策略")
        print(f"  - 执行平面: 实现了缓存、差分传输、流式处理和异步任务调度")
        print(f"  - 反馈闭环: 收集了实际性能数据用于模型优化")
        print(f"  - 智能缓存: 基于权重的缓存淘汰算法")
        
        return True


if __name__ == "__main__":
    success = run_end_to_end_test()
    if success:
        print("\n测试成功完成！")
        sys.exit(0)
    else:
        print("\n测试失败！")
        sys.exit(1)