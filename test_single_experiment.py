"""
单个实验测试脚本
用于在Linux环境下测试单个实验流程，完全按照orchestrator的逻辑，但规模更小
"""
import sys
import os
import json
import time
import docker
import subprocess
import shutil
from pathlib import Path
from ml_training.exp_orchestrator import ExperimentOrchestrator
from ml_training.config import get_client_capabilities, get_image_profiles, get_compression_config

def test_single_experiment():
    """测试单个实验流程，完全按照orchestrator的逻辑"""
    print("开始测试单个实验流程...")
    print("注意：此脚本需要在Linux环境下运行，以支持完整的网络仿真功能")
    
    # 检查是否在Linux环境下运行
    if sys.platform != 'linux':
        print("警告：此脚本设计用于Linux环境，当前运行环境为：", sys.platform)
        print("在非Linux环境下，网络仿真功能可能不可用")
    
    # 创建orchestrator实例，不启用云模式以使用完整功能
    orchestrator = ExperimentOrchestrator(cloud_mode=False)
    
    # 定义测试参数 - 规模更小的实验
    client_types = ["C1", "C2", "C3"]  # 只测试3种客户端类型而不是全部6种
    image_name = "node:latest"  # 单个镜像
    algorithm = "zstd-3"  # 单个压缩算法
    
    print(f"测试参数: 镜像={image_name}, 算法={algorithm}")
    print(f"客户端类型: {client_types}")
    
    results = []
    
    for client_type in client_types:
        print(f"\n开始测试客户端类型: {client_type}")
        try:
            # 执行单个实验，完全按照orchestrator的逻辑
            result = orchestrator.run_experiment(client_type, image_name, algorithm)
            results.append(result)
            
            print(f"客户端 {client_type} 实验完成:")
            print(f"  状态: {result.get('status', 'unknown')}")
            if result.get('status') == 'ABNORMAL':
                print(f"  错误: {result.get('error', 'No error message')}")
                print(f"  原始输出: {result.get('raw_output', 'N/A')}")
                print(f"  客户端类型: {result.get('client_type', 'N/A')}")
            else:
                print(f"  解压时间: {result.get('decompression_time', 'N/A')}s")
                print(f"  压缩大小: {result.get('compressed_size_bytes', 'N/A')} bytes")
                print(f"  解压大小: {result.get('uncompressed_size_bytes', 'N/A')} bytes")
                print(f"  CPU用户时间: {result.get('cpu_user_time_delta', 'N/A')}s")
                print(f"  峰值内存: {result.get('memory_rss_bytes_peak', 'N/A')} bytes")
                print(f"  实际带宽: {result.get('actual_bandwidth', 'N/A')} Mbps")
                print(f"  带宽标准差: {result.get('bandwidth_std', 'N/A')}")
                print(f"  平均CPU使用率: {result.get('avg_cpu_usage', 'N/A')}%")
                print(f"  峰值内存: {result.get('peak_memory', 'N/A')} MB")
                print(f"  是否噪声数据: {result.get('is_noisy_data', 'N/A')}")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"子进程执行失败: {str(e)}\n命令: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}\n输出: {e.output if e.output else 'N/A'}"
            print(f"客户端 {client_type} 实验失败: {error_msg}")
            results.append({
                "status": "ABNORMAL",
                "error": error_msg,
                "client_type": client_type,
                "image_name": image_name,
                "method": algorithm
            })
        except docker.errors.APIError as e:
            error_msg = f"Docker API错误: {str(e)}"
            print(f"客户端 {client_type} 实验失败: {error_msg}")
            import traceback
            traceback.print_exc()
            results.append({
                "status": "ABNORMAL",
                "error": error_msg,
                "client_type": client_type,
                "image_name": image_name,
                "method": algorithm
            })
        except FileNotFoundError as e:
            error_msg = f"文件或命令未找到: {str(e)}\n这通常意味着缺少必要的系统工具，如tc、pumba等"
            print(f"客户端 {client_type} 实验失败: {error_msg}")
            import traceback
            traceback.print_exc()
            results.append({
                "status": "ABNORMAL",
                "error": error_msg,
                "client_type": client_type,
                "image_name": image_name,
                "method": algorithm
            })
        except PermissionError as e:
            error_msg = f"权限不足: {str(e)}\n这通常意味着没有sudo权限或Docker权限"
            print(f"客户端 {client_type} 实验失败: {error_msg}")
            import traceback
            traceback.print_exc()
            results.append({
                "status": "ABNORMAL",
                "error": error_msg,
                "client_type": client_type,
                "image_name": image_name,
                "method": algorithm
            })
        except Exception as e:
            error_msg = f"客户端 {client_type} 实验失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            results.append({
                "status": "ABNORMAL",
                "error": error_msg,
                "client_type": client_type,
                "image_name": image_name,
                "method": algorithm
            })
        
        # 添加一些延迟以避免资源竞争
        time.sleep(2)
    
    # 输出测试结果摘要
    print(f"\n测试完成! 总共执行了 {len(results)} 个实验")
    
    success_count = sum(1 for r in results if r.get('status') == 'SUCCESS')
    abnormal_count = sum(1 for r in results if r.get('status') == 'ABNORMAL')
    
    print(f"成功: {success_count}, 异常: {abnormal_count}")
    
    # 详细分析结果
    for i, result in enumerate(results):
        client_type = client_types[i]
        print(f"\n{client_type} 详细结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    # 保存测试结果
    with open('single_experiment_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("测试结果已保存到 single_experiment_test_results.json")
    
    return results

def check_environment():
    """检查环境是否满足实验要求"""
    print("检查环境依赖...")
    
    # 检查Docker是否可用
    try:
        subprocess.run(["docker", "version"], check=True, capture_output=True)
        print("✅ Docker 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker 不可用，请安装Docker并启动服务")
        return False
    
    # 检查tc命令
    if shutil.which("tc"):
        print("✅ tc 命令可用")
    else:
        print("⚠️ tc 命令不可用，将自动切换到云模式")
    
    # 检查sudo权限
    try:
        subprocess.run(["sudo", "-n", "true"], check=True, capture_output=True)
        print("✅ 有sudo权限")
    except subprocess.CalledProcessError:
        print("⚠️ 无sudo权限，某些功能可能受限")
    
    # 检查Pumba（云模式必需）
    if shutil.which("pumba"):
        print("✅ Pumba 可用")
    else:
        print("⚠️ Pumba 不可用，云模式将不可用")
    
    return True

if __name__ == "__main__":
    # 确保在项目根目录运行
    current_dir = os.getcwd()
    if not os.path.exists("ml_training") or not os.path.exists("client_agent.py"):
        print("错误: 请在 CTS_system 项目根目录下运行此脚本")
        sys.exit(1)
    
    # 检查环境
    if not check_environment():
        print("环境检查失败，无法继续执行测试")
        sys.exit(1)
    
    print("开始执行单个实验测试...")
    test_results = test_single_experiment()
    
    print("\n测试完成!")
"""
单个实验测试脚本
用于在Windows环境下测试单个实验流程
"""
import sys
import os
import json
import time
import docker
from ml_training.exp_orchestrator import ExperimentOrchestrator
from ml_training.config import get_client_capabilities, get_image_profiles, get_compression_config

def get_base_algorithm(algorithm):
    """从复合算法名获取基础算法名，例如 'zstd-3' -> 'zstd'"""
    if '-' in algorithm:
        base_algo = algorithm.split('-')[0]
        # 确保基础算法名是client_agent支持的
        if base_algo in ['gzip', 'zstd', 'lz4', 'uncompressed']:
            return base_algo
        else:
            # 如果基础算法不在支持列表中，返回默认值
            return 'gzip'
    return algorithm

def test_single_experiment():
    """测试单个实验流程"""
    print("开始测试单个实验流程...")
    
    # 启用云模式以兼容Windows环境
    orchestrator = ExperimentOrchestrator(cloud_mode=True)
    
    # 定义测试参数
    client_types = ["C1", "C2", "C3", "C4", "C5", "C6"]  # 6种客户端类型
    image_name = "node:latest"  # 使用完整镜像名
    algorithm = "zstd-3"  # 单个压缩算法
    
    print(f"测试参数: 镜像={image_name}, 算法={algorithm}")
    print(f"客户端类型: {client_types}")
    print(f"基础算法名: {get_base_algorithm(algorithm)}")
    
    results = []
    
    for client_type in client_types:
        print(f"\n开始测试客户端类型: {client_type}")
        try:
            # 执行单个实验
            result = orchestrator.run_experiment(client_type, image_name, algorithm)
            results.append(result)
            
            print(f"客户端 {client_type} 实验完成:")
            print(f"  状态: {result.get('status', 'unknown')}")
            if result.get('status') == 'ABNORMAL':
                print(f"  错误: {result.get('error', 'No error message')}")
            else:
                print(f"  解压时间: {result.get('decompression_time', 'N/A')}s")
                print(f"  实际带宽: {result.get('actual_bandwidth', 'N/A')} Mbps")
                
        except Exception as e:
            print(f"客户端 {client_type} 实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "status": "ABNORMAL",
                "error": str(e),
                "client_type": client_type,
                "image_name": image_name,
                "method": algorithm
            })
        
        # 添加一些延迟以避免资源竞争
        time.sleep(2)
    
    # 输出测试结果摘要
    print(f"\n测试完成! 总共执行了 {len(results)} 个实验")
    
    success_count = sum(1 for r in results if r.get('status') == 'SUCCESS')
    abnormal_count = sum(1 for r in results if r.get('status') == 'ABNORMAL')
    
    print(f"成功: {success_count}, 异常: {abnormal_count}")
    
    # 保存测试结果
    with open('single_experiment_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("测试结果已保存到 single_experiment_test_results.json")
    
    return results

if __name__ == "__main__":
    # 确保在项目根目录运行
    current_dir = os.getcwd()
    if not os.path.exists("ml_training") or not os.path.exists("client_agent.py"):
        print("错误: 请在 CTS_system 项目根目录下运行此脚本")
        sys.exit(1)
    
    print("开始执行单个实验测试...")
    test_results = test_single_experiment()
    
    print("\n测试完成!")