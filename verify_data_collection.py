"""
验证数据收集功能脚本
功能：验证ml_training模块是否能正确收集实验数据
输入：无
输出：验证报告
"""

import os
import json
from pathlib import Path
import tempfile
from ml_training.config import get_client_capabilities, get_image_profiles, get_compression_config
from ml_training.exp_orchestrator import ExperimentOrchestrator


def verify_data_collection():
    """
    验证数据收集功能
    """
    print("开始验证数据收集功能...")
    
    # 创建临时实验数据目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"使用临时目录: {temp_dir}")
        
        # 初始化实验编排器
        orchestrator = ExperimentOrchestrator(
            registry_url="localhost:5000",
            data_dir=temp_dir,
            container_image="hello-world:latest",  # 使用hello-world镜像进行测试
            cloud_mode=True  # 使用云模式，跳过tc命令检查
        )
        
        # 获取测试配置
        all_client_profiles = get_client_capabilities()['profiles']
        all_target_images = get_image_profiles()
        all_compression_methods = get_compression_config()['algorithms']
        
        # 选择最小的测试子集
        test_profile = all_client_profiles[0]  # 选择第一个客户端配置
        test_image = all_target_images[0]      # 选择第一个镜像
        test_method = all_compression_methods[0]  # 选择第一个压缩算法
        
        print(f"测试配置:")
        print(f"  - 客户端: {test_profile['name']} ({test_profile['description']})")
        print(f"  - 镜像: {test_image['name']}")
        print(f"  - 算法: {test_method}")
        
        # 尝试启动容器
        try:
            print("\n正在启动测试容器...")
            container = orchestrator._setup_emulated_container(test_profile)
            print(f"容器启动成功: {container.id[:12]}")
            
            # 执行一个简单的实验
            print(f"\n正在执行测试实验...")
            result = orchestrator._execute_experiment(container, test_image['name'], test_method)
            
            # 模拟实验时间
            import time
            start_time = time.time() - 5
            end_time = time.time()
            decompression_time = result.get('decompression_performance', {}).get('decompression_time', 0)
            
            # 收集实验指标
            experiment_record = orchestrator.collect_experiment_metrics(
                test_profile['name'], 
                test_image['name'], 
                test_method, 
                0,
                start_time,
                end_time,
                decompression_time,
                result.get('error')
            )
            
            print(f"实验执行完成，状态: {experiment_record['status']}")
            
            # 销毁容器
            container.stop()
            container.remove()
            print("测试容器已销毁")
            
            # 验证数据收集
            print("\n=== 验证数据收集结果 ===")
            
            # 检查是否生成了JSON文件
            json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            print(f"生成的JSON文件数量: {len(json_files)}")
            
            if json_files:
                print(f"JSON文件列表: {json_files}")
                
                # 读取并检查第一个JSON文件
                first_json = json_files[0]
                with open(os.path.join(temp_dir, first_json), 'r') as f:
                    data = json.load(f)
                
                print(f"\nJSON文件内容示例:")
                print(f"  - UUID: {data.get('uuid', 'N/A')}")
                print(f"  - 客户端ID: {data.get('profile_id', 'N/A')}")
                print(f"  - 镜像名称: {data.get('image_name', 'N/A')}")
                print(f"  - 压缩方法: {data.get('method', 'N/A')}")
                print(f"  - 总耗时: {data.get('cost_total', 'N/A')}秒")
                print(f"  - 解压时间: {data.get('decompression_time', 'N/A')}秒")
                print(f"  - 解压性能数据: {data.get('decompression_performance', 'N/A')}")
                print(f"  - 宿主机CPU负载: {data.get('host_cpu_load', 'N/A')}")
                print(f"  - 宿主机内存使用: {data.get('host_memory_usage', 'N/A')}%")
                print(f"  - 实验状态: {data.get('status', 'N/A')}")
                
                # 检查关键字段是否存在
                required_fields = [
                    'uuid', 'profile_id', 'image_name', 'method', 
                    'cost_total', 'decompression_time', 'decompression_performance',
                    'host_cpu_load', 'host_memory_usage', 'status'
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"\n缺少字段: {missing_fields}")
                else:
                    print(f"\n所有必需字段都已找到")
                
                # 检查解压性能数据
                perf_data = data.get('decompression_performance', {})
                if perf_data:
                    print(f"\n解压性能数据验证:")
                    print(f"  - 解压时间: {perf_data.get('decompression_time', 'N/A')}")
                    print(f"  - CPU使用率: {perf_data.get('cpu_usage', 'N/A')}")
                    print(f"  - 内存使用: {perf_data.get('memory_usage', 'N/A')}")
                    print(f"  - 磁盘I/O: {perf_data.get('disk_io', 'N/A')}")
                    print(f"  - 压缩方法: {perf_data.get('method', 'N/A')}")
                    
                    # 检查解压性能数据的完整性
                    perf_required_fields = ['decompression_time', 'cpu_usage', 'memory_usage', 'disk_io', 'method']
                    missing_perf_fields = [field for field in perf_required_fields if field not in perf_data]
                    if not missing_perf_fields:
                        print("  - 解压性能数据完整")
                    else:
                        print(f"  - 解压性能数据缺少字段: {missing_perf_fields}")
                else:
                    print(f"\n未找到解压性能数据")
            
            # 检查数据库记录
            db_path = os.path.join(temp_dir, "experiment_manifest.db")
            if os.path.exists(db_path):
                print(f"\n数据库文件存在: {db_path}")
                
                # 尝试连接数据库并检查记录
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM experiments")
                count = cursor.fetchone()[0]
                print(f"数据库中实验记录数量: {count}")
                
                if count > 0:
                    cursor.execute("SELECT * FROM experiments LIMIT 1")
                    record = cursor.fetchone()
                    col_names = [description[0] for description in cursor.description]
                    print(f"数据库列名: {col_names}")
                
                conn.close()
            else:
                print(f"\n数据库文件不存在: {db_path}")
            
            print(f"\n=== 数据收集验证完成 ===")
            print(f"验证结果: {'通过' if len(json_files) > 0 else '失败'}")
            
            return len(json_files) > 0
            
        except Exception as e:
            print(f"验证过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = verify_data_collection()
    if success:
        print("\n数据收集功能验证成功！")
    else:
        print("\n数据收集功能验证失败！")