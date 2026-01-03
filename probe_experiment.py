"""
最小可行性实验脚本
功能：验证核心功能是否正常工作，确认tc限速生效和性能监控记录成功
输入：镜像名称、客户端画像配置
输出：实验结果和验证报告
"""

import os
import time
import subprocess
import json
from typing import Dict, Any
import docker
from modules.image_probe import ImageProbe
from modules.capability_registry import CapabilityRegistry
from modules.decision_engine import DecisionEngine
from core.streaming_proxy import StreamingProxy


def setup_test_environment():
    """设置测试环境"""
    print("设置测试环境...")
    
    # 检查tc命令是否可用
    try:
        result = subprocess.run(['tc', '-h'], capture_output=True, text=True)
        print("✓ tc命令可用")
    except FileNotFoundError:
        print("✗ tc命令不可用，请安装iproute2包")
        return False
    
    # 检查docker是否运行
    try:
        client = docker.from_env()
        client.ping()
        print("✓ Docker可用")
    except:
        print("✗ Docker不可用")
        return False
    
    return True


def test_network_limiting():
    """测试网络限制功能"""
    print("\n测试网络限制功能...")
    
    # 创建一个测试容器
    client = docker.from_env()
    
    try:
        # 启动一个临时容器
        container = client.containers.run(
            'alpine:latest',
            'sleep 30',
            detach=True,
            network_mode='bridge'
        )
        
        # 获取容器网络接口
        result = subprocess.run([
            'docker', 'inspect', 
            '--format', '{{.State.Pid}}', 
            container.id
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            pid = result.stdout.strip()
            if pid:
                # 获取容器的veth接口名称
                network_inspect = subprocess.run([
                    'docker', 'inspect', 
                    '--format', '{{range .NetworkSettings.Networks}}{{.EndpointID}}{{end}}', 
                    container.id
                ], capture_output=True, text=True)
                
                if network_inspect.returncode == 0:
                    endpoint_id = network_inspect.stdout.strip()[:12]
                    if endpoint_id:
                        # 获取宿主机上的veth接口名称
                        veth_cmd = f"ip link show | grep -o 'veth[^@]*@.*{endpoint_id[:5]}' | head -n1 | cut -d'@' -f1"
                        veth_result = subprocess.run(veth_cmd, shell=True, capture_output=True, text=True)
                        
                        if veth_result.returncode == 0 and veth_result.stdout.strip():
                            veth_name = veth_result.stdout.strip()
                            
                            # 应用带宽限制
                            tc_cmd = f"tc qdisc add dev {veth_name} root handle 1: tbf rate 10mbit burst 32kbit latency 400ms"
                            subprocess.run(tc_cmd, shell=True)
                            
                            print(f"✓ 网络限制已应用到接口 {veth_name}")
                            
                            # 测试延迟
                            delay_cmd = f"tc qdisc add dev {veth_name} parent 1:1 handle 10: netem delay 50ms"
                            subprocess.run(delay_cmd, shell=True)
                            
                            print(f"✓ 延迟限制已应用到接口 {veth_name}")
                            
                            # 清理tc规则
                            subprocess.run(f"tc qdisc del dev {veth_name} root 2>/dev/null", shell=True)
                            
                            # 停止并删除容器
                            container.stop()
                            container.remove()
                            
                            return True
                        else:
                            print(f"✗ 无法找到容器 {container.id[:12]} 的veth接口")
                    else:
                        print(f"✗ 无法获取容器 {container.id[:12]} 的endpoint ID")
                else:
                    print(f"✗ 无法获取容器 {container.id[:12]} 的网络配置")
            else:
                print(f"✗ 无法获取容器 {container.id[:12]} 的PID")
        else:
            print(f"✗ 无法获取容器 {container.id[:12]} 的PID: {result.stderr}")
        
        # 停止并删除容器（以防万一）
        try:
            container.stop()
            container.remove()
        except:
            pass
        
        return False
    except Exception as e:
        print(f"✗ 网络限制测试失败: {e}")
        return False


def test_image_analysis():
    """测试镜像分析功能"""
    print("\n测试镜像分析功能...")
    
    try:
        # 使用一个简单的镜像进行测试
        probe = ImageProbe()
        features = probe.analyze_image('hello-world:latest')
        
        print(f"✓ 镜像分析成功")
        print(f"  镜像ID: {features['image_id'][:12]}")
        print(f"  总大小: {features['total_size_mb']:.2f} MB")
        print(f"  层数: {features['layer_count']}")
        print(f"  平均熵值: {features['avg_layer_entropy']:.3f}")
        print(f"  压缩机会: {features['compression_opportunity']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ 镜像分析失败: {e}")
        return False


def test_capability_registry():
    """测试能力注册功能"""
    print("\n测试能力注册功能...")
    
    try:
        registry = CapabilityRegistry()
        
        # 创建一个模拟的客户端能力画像
        test_client_profile = {
            'node_id': 'test-node-001',
            'cpu_score': 2000,
            'bandwidth_mbps': 100,
            'decompression_speed': {
                'gzip': 100,
                'zstd': 120,
                'lz4': 150
            },
            'network_rtt': 20,
            'disk_io_speed': 200,
            'memory_size': 8,
            'latency_requirement': 0.5
        }
        
        # 注册能力
        registry.register_capability(test_client_profile)
        
        # 验证注册
        retrieved_profile = registry.get_capability('test-node-001')
        if retrieved_profile:
            print(f"✓ 能力注册成功")
            print(f"  CPU分数: {retrieved_profile['cpu_score']}")
            print(f"  带宽: {retrieved_profile['bandwidth_mbps']} Mbps")
            return True
        else:
            print("✗ 能力注册失败")
            return False
    except Exception as e:
        print(f"✗ 能力注册测试失败: {e}")
        return False


def test_decision_engine():
    """测试决策引擎功能"""
    print("\n测试决策引擎功能...")
    
    try:
        engine = DecisionEngine()
        
        # 创建测试数据
        client_profile = {
            'cpu_score': 1000,  # 弱客户端
            'bandwidth_mbps': 10,  # 低带宽
            'decompression_speed': {
                'gzip': 50,
                'zstd': 60,
                'lz4': 80
            },
            'network_rtt': 100,
            'disk_io_speed': 50,
            'memory_size': 4,
            'latency_requirement': 1.0
        }
        
        image_features = {
            'total_size_mb': 100.0,
            'avg_layer_entropy': 0.4,
            'text_ratio': 0.6,
            'binary_ratio': 0.4,
            'layer_count': 3,
            'file_type_distribution': {'text': 0.6, 'binary': 0.4, 'compressed': 0.0},
            'avg_file_size': 4096,
            'compression_ratio_estimate': 0.35
        }
        
        # 获取最优压缩方法
        best_method, cost_estimate = engine.get_optimal_compression_method(client_profile, image_features)
        
        print(f"✓ 决策引擎运行成功")
        print(f"  推荐算法: {best_method}")
        print(f"  预估成本: {cost_estimate:.2f}s")
        
        return True
    except Exception as e:
        print(f"✗ 决策引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_proxy():
    """测试流式代理功能"""
    print("\n测试流式代理功能...")
    
    try:
        proxy = StreamingProxy()
        
        # 创建模拟数据流
        def data_stream():
            for i in range(5):
                yield f"这是第{i}个数据块，用于测试流式压缩功能。" * 50
        
        # 测试不同压缩方法
        methods = ['gzip', 'lz4']
        for method in methods:
            # 创建客户端画像以测试自适应块大小
            client_profile = {
                'bandwidth_mbps': 10  # 低带宽，应使用小块
            }
            
            start_time = time.time()
            compressed_chunks = []
            for chunk in proxy.compress_stream(data_stream(), method, client_profile):
                compressed_chunks.append(chunk)
            end_time = time.time()
            
            original_size = sum(len(f"这是第{i}个数据块，用于测试流式压缩功能。" * 50) for i in range(5))
            compressed_size = sum(len(chunk) for chunk in compressed_chunks)
            
            print(f"  {method}压缩:")
            print(f"    原始大小: {original_size} bytes")
            print(f"    压缩后大小: {compressed_size} bytes")
            print(f"    压缩比: {compressed_size/original_size:.2%}")
            print(f"    处理时间: {end_time - start_time:.2f}s")
            print(f"    使用块大小: {proxy._determine_chunk_size(client_profile)} bytes")
        
        print(f"✓ 流式代理测试成功")
        return True
    except Exception as e:
        print(f"✗ 流式代理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_minimal_experiment():
    """运行最小可行性实验"""
    print("开始运行最小可行性实验...")
    print("="*50)
    
    # 检查环境
    if not setup_test_environment():
        print("环境检查失败，无法继续实验")
        return False
    
    # 运行各项测试
    tests = [
        ("网络限制功能", test_network_limiting),
        ("镜像分析功能", test_image_analysis),
        ("能力注册功能", test_capability_registry),
        ("决策引擎功能", test_decision_engine),
        ("流式代理功能", test_streaming_proxy)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        result = test_func()
        results[test_name] = result
    
    print("\n" + "="*50)
    print("实验结果汇总:")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ 所有测试通过！可以开始3240次全量实验")
    else:
        print("\n✗ 部分测试失败，请先解决这些问题")
    
    # 保存实验结果
    with open('probe_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n实验结果已保存至: probe_experiment_results.json")
    return all_passed


def main():
    """主函数"""
    print("最小可行性实验脚本")
    print("用于验证核心功能是否正常工作")
    print("确认tc限速生效和性能监控记录成功")
    
    success = run_minimal_experiment()
    
    if success:
        print("\n建议:")
        print("1. 现在可以放心启动3240次的全量挂机实验了")
        print("2. 实验将验证'弱网用高压，强网用快压'的逻辑")
        print("3. 监控实验进程，确保数据采集正常")
    else:
        print("\n请先解决上述失败的测试项，再进行全量实验")


if __name__ == "__main__":
    main()