#!/usr/bin/env python3
"""
智能镜像优化系统主入口
整合所有模块，提供完整的镜像优化服务
"""

import argparse
import os
import sys

from modules.client_probe import ClientProbe
from modules.image_probe import ImageProbe
from modules.decision_engine import CompressionDecisionEngine
from modules.cache_pool import CompressionCachePool
from modules.transmission_optimizer import TransmissionOptimizer
from modules.docker_analyzer import DockerAnalyzer
from modules.network_profiler import NetworkProfiler
from modules.capability_registry import CapabilityRegistry


def run_image_optimization(image_name, node_id=None):
    """
    运行完整的镜像优化流程
    
    Args:
        image_name: 镜像名称
        node_id: 节点ID（用于查找预注册的能力画像）
    """
    print(f"开始优化镜像: {image_name}")
    print("=" * 50)
    
    # 1. 客户端探针
    print("1. 获取客户端能力...")
    if node_id:
        # 使用预注册的能力画像
        print(f"   使用预注册节点能力，节点ID: {node_id}")
        capability_registry = CapabilityRegistry()
        client_profile = capability_registry.get_client_profile_for_decision(node_id)
        print(f"   客户端配置（预注册）: {client_profile}")
    else:
        # 使用本地探测方式
        print("   使用本地探测获取客户端能力")
        client_probe = ClientProbe()
        client_profile = client_probe.get_client_profile()
        print(f"   客户端配置（本地探测）: {client_profile}")
    
    # 2. 镜像探针（真实分析）
    print("\n2. 执行镜像探针...")
    try:
        docker_analyzer = DockerAnalyzer()
        image_profile = docker_analyzer.get_image_profile(image_name)
        print(f"   镜像特征: {image_profile}")
    except Exception as e:
        print(f"   真实镜像分析失败: {e}")
        print("   使用模拟数据进行演示...")
        # 使用模拟数据
        image_profile = {
            "layer_count": 6,
            "total_size_mb": 742,
            "avg_layer_entropy": 0.78,
            "text_ratio": 0.18,
            "binary_ratio": 0.82,
            "compressed_file_ratio": 0.12,
            "avg_file_size_kb": 42.3,
            "median_file_size_kb": 8.1,
            "small_file_ratio": 0.67,
            "big_file_ratio": 0.15,
            "predict_popularity": 0.54
        }
        print(f"   模拟镜像特征: {image_profile}")
    
    # 3. 压缩决策
    print("\n3. 执行压缩决策...")
    decision_engine = CompressionDecisionEngine()
    decision = decision_engine.make_decision(client_profile, image_profile)
    print(f"   压缩决策: {decision}")
    
    # 4. 检查缓存
    print("\n4. 检查缓存...")
    cache_pool = CompressionCachePool()
    selected_compression = decision["selected_compression"]
    
    if cache_pool.is_cached(image_name, selected_compression):
        print(f"   缓存命中: {image_name} ({selected_compression})")
        cached_data_path = cache_pool.get_cached_data(image_name, selected_compression)
        print(f"   缓存路径: {cached_data_path}")
    else:
        print(f"   缓存未命中，需要生成: {image_name} ({selected_compression})")
        # 这里应该执行实际的压缩操作，然后缓存结果
        # 示例中我们模拟这个过程
        cache_pool.cache_data(image_name, selected_compression, "temp/image_extract")
        print(f"   数据已缓存")
    
    # 5. 传输优化
    print("\n5. 执行传输优化...")
    optimizer = TransmissionOptimizer()
    
    # 获取真实的镜像数据大小
    image_work_dir = os.path.join("image", image_name.replace("/", "_").replace(":", "_"))
    export_path = os.path.join(image_work_dir, "image.tar")
    image_data = b""
    try:
        # 尝试读取真实的镜像数据
        if os.path.exists(export_path):
            # 获取文件大小作为数据大小
            image_size = os.path.getsize(export_path)
            image_data = b"x" * image_size  # 模拟对应大小的数据
        else:
            # 回退到使用镜像分析得到的大小
            image_size_mb = image_profile.get("total_size_mb", 100)
            image_data = b"x" * int(image_size_mb * 1024 * 1024)  # 转换为字节
    except Exception as e:
        print(f"   读取镜像数据时出错: {e}")
        # 回退到使用镜像分析得到的大小
        image_size_mb = image_profile.get("total_size_mb", 100)
        image_data = b"x" * int(image_size_mb * 1024 * 1024)  # 转换为字节
    
    # 获取缓存数据用于差分计算
    cached_data = None
    if cache_pool.is_cached(image_name, selected_compression):
        cached_data = image_data  # 简化处理
    
    optimization_result = optimizer.optimize_transmission(
        image_name, client_profile, image_data, cached_data)
    print(f"   传输优化结果: {optimization_result}")
    
    # 6. 输出最终结果
    print("\n" + "=" * 50)
    print("优化完成，最终结果:")
    print(f"  选定压缩算法: {decision['selected_compression']}")
    print(f"  预期压缩后大小: {decision['expected_compressed_size']} MB")
    print(f"  预估总耗时: {decision['estimated_total_cost']} 秒")
    print(f"  传输策略: {optimization_result['transmission_method']}")
    print(f"  传输大小: {optimization_result['estimated_transmission_size']:.2f} MB")


def register_node_capability():
    """
    注册节点能力（一次性探针执行模式）
    """
    print("执行一次性节点能力探针...")
    
    # 执行本地探测
    client_probe = ClientProbe()
    capability_data = client_probe.get_client_profile()
    
    # 生成节点ID（在实际应用中，这应该基于更稳定的硬件特征）
    registry = CapabilityRegistry()
    client_info = {
        "ip": "localhost",  # 在实际应用中应该获取真实IP
        "user_agent": "CLI_Register_Tool/1.0"
    }
    node_id = registry.generate_node_id(client_info)
    
    # 注册能力
    registry.register_capability(node_id, capability_data)
    
    print(f"节点能力注册成功!")
    print(f"节点ID: {node_id}")
    print(f"能力数据: {capability_data}")
    print("\n后续请求中请使用此节点ID以获得优化服务")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能镜像优化系统')
    parser.add_argument('image', nargs='?', default='nginx:latest', help='要优化的镜像名称')
    parser.add_argument('--node-id', help='预注册的节点ID')
    parser.add_argument('--register', action='store_true', help='执行一次性节点能力探针并注册')
    
    args = parser.parse_args()
    
    if args.register:
        # 注册模式
        register_node_capability()
    else:
        # 图像优化模式
        run_image_optimization(args.image, args.node_id)


if __name__ == "__main__":
    main()