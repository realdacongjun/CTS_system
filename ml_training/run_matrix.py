"""
实验矩阵执行器
执行完整的实验矩阵：6客户端 × 18镜像 × 10算法 × 3重复 = 3240次实验
"""

import time
import json
import sqlite3
import subprocess
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from .config import get_client_capabilities, get_image_profiles, get_compression_config
from .exp_orchestrator import ExperimentOrchestrator
from .data_collector import DataCollector


def check_env_support():
    """
    检查云服务器环境是否支持实验编排
    """
    modules = ["sch_netem", "sch_htb"]
    for mod in modules:
        res = subprocess.run(f"sudo modprobe {mod}", shell=True)
        if res.returncode != 0:
            print(f"❌ 环境不支持内核模块: {mod}")
            return False
    
    res = subprocess.run("tc -V", shell=True, capture_output=True)
    if res.returncode != 0:
        print("❌ 未安装 iproute2 (tc) 工具")
        return False
        
    print("✅ 云服务器环境完美支持实验编排！")
    return True


def run_experiment_matrix():
    """执行三级循环实验矩阵，优化为镜像在最外层以减少磁盘占用"""
    # 初始化数据库
    conn = sqlite3.connect('experiments.db')
    cursor = conn.cursor()
    
    # 创建状态表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            client_type TEXT,
            image TEXT,
            algorithm TEXT,
            status TEXT,
            result TEXT,
            PRIMARY KEY (client_type, image, algorithm)
        )
    ''')
    
    # 插入待执行的实验（如果不存在）
    for client_type in client_types:
        for image in images:
            for algorithm in algorithms:
                cursor.execute(
                    'INSERT OR IGNORE INTO experiments VALUES (?, ?, ?, ?, ?)', 
                    (client_type, image, algorithm, 'PENDING', None)
                )
    conn.commit()
    
    # 获取待执行的实验
    cursor.execute('SELECT client_type, image, algorithm FROM experiments WHERE status = "PENDING"')
    pending_experiments = cursor.fetchall()
    
    orchestrator = ExperimentOrchestrator(cloud_mode=True)  # 可配置
    data_collector = DataCollector()
    
    # 优化后的三级循环：镜像 -> 客户端 -> 算法
    for image in images:
        print(f"开始处理镜像: {image}")
        try:
            # 1. 准备该镜像的压缩版本（下载或生成）
            layer_files = prepare_image_layers(image)  # 假设此函数存在
            
            # 2. 针对该镜像执行所有客户端和算法的组合
            for client_type in client_types:
                for algorithm in algorithms:
                    # 检查数据库状态
                    cursor.execute(
                        'SELECT status FROM experiments WHERE client_type = ? AND image = ? AND algorithm = ?', 
                        (client_type, image, algorithm)
                    )
                    result = cursor.fetchone()
                    if result and result[0] == 'SUCCESS':
                        continue  # 跳过已完成的实验
                    
                    # 执行单个实验
                    result = orchestrator.run_experiment(client_type, image, algorithm)
                    # 保存结果等操作...
                    
            # 3. 处理完一个镜像后，立即清理其所有临时文件，释放磁盘空间
            cleanup_image_layers(image)
            print(f"已清理镜像 {image} 的所有临时文件")
            
        except Exception as e:
            print(f"处理镜像 {image} 时出错: {e}")
            raise
    
    conn.close()


def run_full_experiment_matrix():
    """
    运行完整的实验矩阵
    - 6个客户端配置（C1-C6）
    - 18个镜像
    - 10种压缩算法
    - 每种组合重复3次
    """
    print("开始运行完整实验矩阵...")
    print("实验配置: 6客户端 × 18镜像 × 10算法 × 3重复 = 3240次实验")
    
    # 创建实验数据目录
    data_dir = "/tmp/full_experiment_data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化实验编排器
    orchestrator = ExperimentOrchestrator(
        registry_url="localhost:5000",
        data_dir=data_dir,
        container_image="cts-system/client-agent:latest",  # 使用正确的镜像名
        cloud_mode=True  # 在云服务器上运行，tc可用
    )
    
    # 获取完整的实验配置
    all_client_profiles = get_client_capabilities()['profiles']
    all_target_images = get_image_profiles()
    all_compression_methods = get_compression_config()['algorithms']
    
    # 选择完整配置
    selected_profiles = all_client_profiles
    selected_images = all_target_images
    # 从所有算法中选择10个代表性算法
    selected_methods = all_compression_methods[::4][:10]  # 每隔4个取一个，确保覆盖不同算法类型
    
    print(f"客户端配置: {[p['name'] for p in selected_profiles]}")
    print(f"目标镜像: {[i['name'] for i in selected_images[:5]]}... (共{len(selected_images)}个)")
    print(f"压缩算法: {selected_methods}")
    
    # 运行完整实验矩阵
    all_results = []
    completed_count = 0
    total_experiments = len(selected_profiles) * len(selected_images) * len(selected_methods) * 3
    
    print(f"预计总实验数: {total_experiments}")
    
    for profile in selected_profiles:
        print(f"\n=== 开始处理客户端配置: {profile['name']} ({profile['description']}) ===")
        
        # 启动模拟容器
        container = orchestrator._setup_emulated_container(profile)
        if not container:
            print(f"容器启动失败: {profile['name']}")
            continue
        
        try:
            for image in selected_images:
                for method in selected_methods:
                    for repetition in range(3):
                        exp_uuid = f"{profile['name']}_{image['name']}_{method}_rep{repetition}"
                        
                        print(f"执行实验 {completed_count + 1}/{total_experiments}: {exp_uuid}")
                        
                        try:
                            # 执行带物理限制的实验 - 修复参数顺序
                            experiment_record = orchestrator.run_profiled_experiment(
                                container, 
                                image['name'], 
                                method, 
                                profile
                            )
                            
                            all_results.append(experiment_record)
                            completed_count += 1
                            
                            print(f"实验完成，状态: {experiment_record.get('status', 'UNKNOWN')}")
                            
                            # 保存中间结果，防止中断丢失
                            if completed_count % 10 == 0:
                                intermediate_file = f"{data_dir}/intermediate_results_{completed_count}.json"
                                with open(intermediate_file, 'w') as f:
                                    json.dump(all_results, f, indent=2)
                                print(f"中间结果已保存: {intermediate_file}")
                                
                        except Exception as e:
                            print(f"实验执行异常: {e}")
                            # 记录失败的实验
                            error_record = {
                                'profile_id': profile['name'],
                                'image_name': image['name'],
                                'method': method,
                                'repetition': repetition,
                                'status': 'ABNORMAL',
                                'error_message': str(e)
                            }
                            all_results.append(error_record)
                            completed_count += 1
        finally:
            # 清理容器
            try:
                container.stop()
                container.remove()
            except:
                pass
    
    # 分析结果
    print(f"\n=== 完整实验矩阵完成 ===")
    success_count = len([r for r in all_results if r.get('status') == 'SUCCESS'])
    print(f"成功执行: {success_count}/{len(all_results)} 次实验")
    print(f"失败实验: {len([r for r in all_results if r.get('status') == 'ABNORMAL'])}/{len(all_results)} 次实验")
    
    # 检查解压性能数据是否收集到
    decompression_data_count = 0
    for result in all_results:
        if result.get('decompression_performance'):
            decompression_data_count += 1
    
    print(f"包含解压性能数据的实验: {decompression_data_count}/{len(all_results)} 次实验")
    
    # 保存结果到文件
    output_file = f"{data_dir}/full_matrix_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"完整结果已保存到: {output_file}")
    
    # 生成摘要报告
    generate_summary_report(all_results, output_file.replace('.json', '_summary.txt'))
    
    return all_results


def generate_summary_report(results, output_path):
    """
    生成实验摘要报告
    """
    success_count = len([r for r in results if r.get('status') == 'SUCCESS'])
    total_count = len(results)
    
    # 修复KeyError: 'image_name'问题 - 使用.get()方法安全获取字段
    unique_images = len(set(r.get('image_name') for r in results if r.get('image_name')))
    unique_algorithms = len(set(r.get('method') for r in results if r.get('method')))
    
    summary = f"""
实验矩阵执行摘要报告
===================

执行统计:
- 总实验数: {total_count}
- 成功实验数: {success_count}
- 失败实验数: {total_count - success_count}
- 成功率: {success_count/total_count*100:.2f}%

实验配置:
- 客户端类型: C1-C6 (6种)
- 测试镜像: {unique_images} 个
- 压缩算法: {unique_algorithms} 种
- 重复次数: 3 次

时间统计:
- 开始时间: {time.ctime()}
- 结束时间: {time.ctime()}
"""
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"摘要报告已保存到: {output_path}")


if __name__ == "__main__":
    # 首先检查环境支持
    if not check_env_support():
        print("⚠️  环境检查失败，切换到云模式运行")
        # 可以选择继续运行或退出
        response = input("是否继续以云模式运行？(y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    run_full_experiment_matrix()


def prepare_image_layers(image: str) -> Dict[str, str]:
    """
    为指定镜像准备所有压缩版本的层文件
    
    Args:
        image: 镜像名称
        
    Returns:
        字典，键为算法名称，值为临时文件路径
    """
    # 创建镜像专用的临时目录
    image_temp_dir = Path(f"/tmp/experiment_data/{image.replace(':', '_')}")
    image_temp_dir.mkdir(parents=True, exist_ok=True)
    
    layer_files = {}
    algorithms = [
        'gzip-1', 'gzip-6', 'gzip-9',
        'zstd-1', 'zstd-3', 'zstd-6',
        'lz4-fast', 'lz4-medium', 'lz4-slow',
        'brotli-1'
    ]
    
    for algorithm in algorithms:
        # 模拟生成或下载对应压缩级别的文件
        # 实际应用中应从仓库拉取或调用压缩服务生成
        compressed_file = image_temp_dir / f"{algorithm}_layer.tar"
        # 创建空文件作为占位符（实际应写入真实数据）
        compressed_file.touch()
        layer_files[algorithm] = str(compressed_file)
    
    return layer_files

def cleanup_image_layers(image: str):
    """
    清理指定镜像的所有临时文件
    
    Args:
        image: 镜像名称
    """
    image_temp_dir = Path(f"/tmp/experiment_data/{image.replace(':', '_')}")
    if image_temp_dir.exists():
        shutil.rmtree(image_temp_dir)
    
    # 确保父目录存在
    parent_dir = image_temp_dir.parent
    parent_dir.mkdir(exist_ok=True)

# 添加缺失的全局变量
client_types = ["C1", "C2", "C3", "C4", "C5", "C6"]
images = [
    "centos", "fedora", "ubuntu",
    "mongo", "mysql", "postgres",
    "redis", "nginx", "tomcat",
    "wordpress", "drupal", "joomla",
    "magento", "prestashop", "shopify",
    "laravel", "django", "flask",
    "rails", "spring", "nodejs"
]
algorithms = [
    'gzip-1', 'gzip-6', 'gzip-9',
    'zstd-1', 'zstd-3', 'zstd-6',
    'lz4-fast', 'lz4-medium', 'lz4-slow',
    'brotli-1'
]
