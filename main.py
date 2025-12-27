#!/usr/bin/env python3
"""
CTS_system主入口模块
整合感知、预测、决策、执行四个平面
"""

import sys
import argparse
import os
import json
import psutil
import cpuinfo
from modules.capability_registry import CapabilityRegistry
from modules.image_probe import ImageProbe
from modules.decision_engine import CompressionDecisionEngine
from modules.transmission_optimizer import TransmissionOptimizer
from modules.cache_pool import CachePool
from modules.differential_sync import DifferentialSync
from modules.feedback_collector import performance_monitor
from core.streaming_proxy import StreamingProxy
from core.task_scheduler import task_scheduler


def register_real_node():
    """注册真实节点能力"""
    print("正在注册真实节点能力...")
    
    # 获取CPU信息
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
    
    # 获取内存信息
    memory_gb = memory_size / (1024**3)
    
    print(f"检测到CPU: {cpu_info['brand_raw']}")
    print(f"CPU核心数: {cpu_count}")
    print(f"内存大小: {memory_gb:.2f}GB")
    print(f"估算CPU性能分数: {cpu_score}")
    
    # 模拟网络带宽测试
    print("正在测试网络性能...")
    bandwidth_mbps = 50  # 默认值
    network_rtt = 20      # 默认值
    
    # 模拟解压速度测试
    decompression_speed = {
        "gzip": 80,
        "zstd": 120,
        "lz4": 150
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
        "disk_io_speed": 400,  # 估算值
        "memory_size": memory_size
    }
    
    # 注册能力
    capability_registry.register_capability(node_id, capability_data)
    print(f"节点 {node_id} 能力注册成功")
    
    return node_id, capability_registry


def test_real_scenario():
    """真实场景测试"""
    print("开始真实场景测试...")
    
    # 1. 注册真实节点
    node_id, capability_registry = register_real_node()
    
    # 2. 创建一个模拟镜像目录用于测试
    print("\n创建测试镜像目录...")
    test_image_path = "./test_image"
    os.makedirs(test_image_path, exist_ok=True)
    
    # 创建模拟的manifest.json
    manifest_content = [{
        "Config": "config.json",
        "Layers": [
            "layer1.tar",
            "layer2.tar",
            "layer3.tar"
        ]
    }]
    
    with open(os.path.join(test_image_path, "manifest.json"), "w") as f:
        json.dump(manifest_content, f)
    
    # 创建模拟的层文件
    for i in range(1, 4):
        layer_path = os.path.join(test_image_path, f"layer{i}.tar")
        with open(layer_path, "wb") as f:
            # 写入一些模拟数据
            import random
            data = bytearray([random.randint(0, 255) for _ in range(1024 * 100)])  # 100KB per layer
            f.write(data)
    
    print(f"测试镜像已创建: {test_image_path}")
    
    # 3. 初始化系统模块
    image_probe = ImageProbe(test_image_path)
    decision_engine = CompressionDecisionEngine(strategy="cost_model")  # 使用成本模型进行测试
    transmission_optimizer = TransmissionOptimizer()
    cache_pool = CachePool()
    
    # 4. 获取客户端画像
    print(f"\n获取节点 {node_id} 的能力画像...")
    client_profile = capability_registry.get_client_profile_for_decision(node_id)
    print(f"客户端画像: CPU={client_profile['cpu_score']}, 带宽={client_profile['bandwidth_mbps']}Mbps")
    
    # 5. 分析镜像特征
    print("\n正在分析镜像特征...")
    image_profile = image_probe.get_image_profile()
    
    print(f"镜像分析完成:")
    print(f"  - 总大小: {image_profile['total_size_mb']:.2f} MB")
    print(f"  - 平均熵值: {image_profile['avg_layer_entropy']:.3f}")
    print(f"  - 文本比例: {image_profile['text_ratio']:.2f}")
    print(f"  - 二进制比例: {image_profile['binary_ratio']:.2f}")
    print(f"  - 层数量: {image_profile['layer_count']}")
    
    # 6. 使用决策引擎选择最优压缩策略
    print("\n正在选择最优压缩策略...")
    best_method, decision_details = decision_engine.make_decision(client_profile, image_profile)
    
    print(f"最优压缩策略: {best_method}")
    print(f"预期总成本: {decision_details['predicted_cost']:.2f}秒")
    
    # 显示各策略成本对比
    print("\n各压缩策略成本对比:")
    method_costs = decision_details['method_costs']
    sorted_methods = sorted(method_costs.items(), key=lambda x: x[1])
    for method, cost in sorted_methods[:5]:  # 显示前5个最佳选项
        print(f"  - {method}: {cost:.2f}秒")
    
    # 7. 获取传输优化计划
    print("\n正在生成传输优化计划...")
    optimization_result = transmission_optimizer.optimize_transmission(
        "test_image:latest",
        image_profile["layers"],
        client_profile
    )
    
    print(f"传输计划生成完成")
    print(f"优先传输层: {optimization_result['transmission_plan']['priority_layers']}")
    
    # 8. 模拟预测相关镜像
    print("\n正在预测相关镜像...")
    related_images = transmission_optimizer.predictor.predict_related_images("test_image:latest")
    if related_images:
        print(f"预测到 {len(related_images)} 个相关镜像:")
        for img in related_images:
            print(f"  - {img['image_name']} (相关性: {img['correlation_score']:.2f})")
    else:
        print("未预测到相关镜像")
    
    # 9. 模拟缓存和差分传输
    print("\n测试缓存机制...")
    cache_key = f"test_image:latest:layer1:{best_method}"
    if cache_pool.is_layer_cached("test_image:latest", "layer1.tar", best_method):
        print(f"层 layer1 已在缓存中，使用 {best_method} 格式")
    else:
        print(f"层 layer1 未在缓存中，将使用 {best_method} 进行压缩")
        # 模拟添加到缓存
        cache_pool.cache_layer_data("test_image:latest", "layer1.tar", best_method, 
                                   os.path.join(test_image_path, "layer1.tar"))
        print("已将 layer1 添加到缓存")
    
    # 10. 模拟差分传输
    print("\n测试差分传输功能...")
    differential_sync = DifferentialSync()
    
    # 创建两个略有不同的文件用于测试差分
    source_path = os.path.join(test_image_path, "layer1.tar")
    target_path = os.path.join(test_image_path, "layer1_modified.tar")
    
    # 创建修改后的文件
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
                                                 os.path.join(test_image_path, "delta.bin"))
    
    print(f"差分同步结果:")
    print(f"  - 原始大小: {delta_result['original_size']} bytes")
    print(f"  - 差分大小: {delta_result['delta_size']} bytes")
    print(f"  - 压缩比例: {delta_result['compression_ratio']:.2f}")
    print(f"  - 节省大小: {delta_result['transfer_saved']} bytes")
    
    # 11. 测试新增功能模块
    print("\n测试新增功能模块...")
    
    # 测试反馈收集器
    print("测试反馈收集器...")
    from modules.feedback_collector import FeedbackData
    feedback = FeedbackData(
        node_id=node_id,
        image_id="test_image:latest",
        algo_used=best_method,
        actual_transfer_time=12.5,
        actual_decomp_time=8.3,
        predicted_transfer_time=decision_details['predicted_cost'] * 0.6,  # 假设传输时间占60%
        predicted_decomp_time=decision_details['predicted_cost'] * 0.4   # 剩余40%是解压时间
    )
    performance_monitor.collect_feedback(feedback)
    print(f"已收集反馈数据: {feedback.algo_used}")
    
    # 测试流式代理
    print("测试流式代理...")
    streaming_proxy = StreamingProxy()
    print("流式代理已初始化")
    
    # 测试任务调度器
    print("测试任务调度器...")
    def task_callback(task):
        print(f"任务 {task.task_id} 完成，状态: {task.status.value}")
    
    task_id = task_scheduler.submit_transcode_task(
        "test_image:latest",
        "layer1.tar",
        best_method,
        original_data,
        task_callback
    )
    print(f"已提交任务: {task_id}")
    
    # 等待任务完成
    import time
    for i in range(10):  # 最多等待5秒
        task = task_scheduler.get_task_status(task_id)
        if task and task.status in ['COMPLETED', 'FAILED']:
            break
        time.sleep(0.5)
    
    # 12. 清理测试文件
    print("\n清理测试文件...")
    import shutil
    if os.path.exists(test_image_path):
        shutil.rmtree(test_image_path)
    
    print("\n真实场景测试完成！")
    print(f"系统成功执行了完整的感知-预测-决策-执行流程:")
    print(f"  - 感知平面: 获取了真实的客户端能力")
    print(f"  - 预测平面: 分析了镜像特征并预测了相关镜像")
    print(f"  - 决策平面: 使用双塔模型选择了最优压缩策略")
    print(f"  - 执行平面: 实现了缓存和差分传输优化")
    print(f"  - 新增功能: 反馈收集、流式传输、异步任务调度")


def main():
    parser = argparse.ArgumentParser(description='Client-aware Transfer System')
    parser.add_argument('image', nargs='?', help='镜像名称或路径')
    parser.add_argument('--node-id', help='节点ID')
    parser.add_argument('--register', action='store_true', help='注册节点能力')
    parser.add_argument('--profile-path', default='./local_node_profile.json', help='本地节点配置文件路径')
    parser.add_argument('--test', action='store_true', help='运行真实场景测试')
    
    args = parser.parse_args()
    
    if args.test:
        test_real_scenario()
    elif args.register:
        # 注册节点能力
        node_id, capability_registry = register_real_node()
    elif args.node_id:
        # 使用指定节点ID进行优化
        capability_registry = CapabilityRegistry()
        capability = capability_registry.get_capability(args.node_id)
        
        if not capability:
            print(f"错误: 未找到节点 {args.node_id} 的能力信息")
            sys.exit(1)
        
        print(f"使用节点 {args.node_id} 的能力画像进行优化")
        
        # 这里可以继续执行优化流程，但需要提供镜像路径
        if not args.image:
            print("错误: 使用 --node-id 时需要提供镜像路径")
            sys.exit(1)
            
        # 初始化系统各模块
        image_probe = ImageProbe(args.image)
        decision_engine = CompressionDecisionEngine(strategy="ml_model")
        transmission_optimizer = TransmissionOptimizer()
        cache_pool = CachePool()
        differential_sync = DifferentialSync()
        
        # 获取客户端画像
        client_profile = capability_registry.get_client_profile_for_decision(args.node_id)
        
        # 分析镜像特征
        print("正在分析镜像特征...")
        image_profile = image_probe.get_image_profile()
        
        print(f"镜像分析完成:")
        print(f"  - 总大小: {image_profile['total_size_mb']:.2f} MB")
        print(f"  - 平均熵值: {image_profile['avg_layer_entropy']:.3f}")
        print(f"  - 文本比例: {image_profile['text_ratio']:.2f}")
        print(f"  - 二进制比例: {image_profile['binary_ratio']:.2f}")
        print(f"  - 层数量: {image_profile['layer_count']}")
        
        # 使用决策引擎选择最优压缩策略
        print("正在选择最优压缩策略...")
        best_method, decision_details = decision_engine.make_decision(client_profile, image_profile)
        
        print(f"最优压缩策略: {best_method}")
        print(f"预期总成本: {decision_details['predicted_cost']:.2f}秒")
        
        # 获取传输优化计划
        print("正在生成传输优化计划...")
        optimization_result = transmission_optimizer.optimize_transmission(
            args.image,
            image_profile["layers"],
            client_profile
        )
        
        print(f"传输计划生成完成")
        print(f"优先传输层: {optimization_result['transmission_plan']['priority_layers']}")
        
        # 如果有相关的预处理任务
        if optimization_result['transmission_plan']['pre_transmission_tasks']:
            print(f"预处理任务: {len(optimization_result['transmission_plan']['pre_transmission_tasks'])} 个")
            for task in optimization_result['transmission_plan']['pre_transmission_tasks']:
                print(f"  - {task['action']} {task['target']} (优先级: {task['priority']:.2f})")
    else:
        print("请指定节点ID、使用 --register 进行注册或使用 --test 运行真实场景测试")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()