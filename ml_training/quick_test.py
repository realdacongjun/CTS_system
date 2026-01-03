"""
增强版快速验证脚本 (Data Quality Ready)
功能：执行小型实验矩阵，并验证收集到的实测数据质量是否满足双塔模型训练要求
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_training.config import get_client_capabilities, get_image_profiles, get_compression_config
from ml_training.exp_orchestrator import ExperimentOrchestrator

def run_quick_test():
    print("🚀 开始运行增强版验证实验...")
    
    # 1. 初始化设置
    data_dir = "/tmp/quick_test_data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # 建议：如果是本地测试设为 False，如果在云端测试设为 True
    IS_CLOUD = False 
    
    orchestrator = ExperimentOrchestrator(
        registry_url="localhost:5000",
        data_dir=data_dir,
        container_image="cts_client:latest",
        cloud_mode=IS_CLOUD
    )
    
    # 2. 选取实验子集 (1x2x2)
    all_client_profiles = get_client_capabilities()['profiles']
    selected_profiles = [p for p in all_client_profiles if p['name'] == 'C1'][:1]
    
    all_target_images = get_image_profiles()
    # 选取一个超小镜像和一个中型镜像，观察数据差异
    selected_images = [i for i in all_target_images if i['name'] in ['hello-world', 'alpine']][:2]
    
    selected_methods = ['gzip-1', 'zstd-1']
    
    all_results = []
    
    # 3. 执行实验循环 - 需要启动容器并运行实验
    for profile in selected_profiles:
        print(f"\n[Profile: {profile['name']}] 设定带宽目标: {profile.get('bw_rate', profile.get('bandwidth_mbps', 'N/A'))} Mbps")
        
        # 启动容器
        container = orchestrator._setup_emulated_container(profile)
        
        try:
            for image in selected_images:
                for method in selected_methods:
                    print(f"正在测试: Image={image['name']} | Method={method}...")
                    
                    try:
                        # 调用修改后的 orchestrator (应包含实时监控逻辑)
                        record = orchestrator.run_profiled_experiment(
                            container,
                            f"{orchestrator.registry_url}/{image['name']}",
                            method,
                            profile
                        )
                        
                        # 补充镜像物理特征 (为了给双塔模型提供右塔输入)
                        record['static_image_size_mb'] = image.get('size_mb', 0)
                        record['static_layer_count'] = image.get('layer_count', 0)
                        
                        all_results.append(record)
                        
                        # 实时反馈实测质量
                        if record['status'] == 'SUCCESS':
                            actual_bw = record.get('actual_bandwidth', 0)
                            bw_std = record.get('bandwidth_std', 0)
                            
                            print(f" ✅ 成功 | 实测带宽: {actual_bw:.2f} Mbps (波动: {bw_std:.2f})")
                            if record.get('is_noisy_data'):
                                print(f" ⚠️  警告: 此条数据波动过大，将被模型标记为噪声")
                        else:
                            print(f" ❌ 失败 | 原因: {record.get('error')}")

                    except Exception as e:
                        print(f" 💥 脚本崩溃: {e}")
                        import traceback
                        traceback.print_exc()
        finally:
            # 清理容器
            try:
                container.stop()
                container.remove()
            except:
                pass

    # 4. 汇总与模型准备度分析
    print("\n" + "="*30)
    print("📊 实验结果分析")
    print("="*30)
    
    successes = [r for r in all_results if r['status'] == 'SUCCESS']
    noisy = [r for r in successes if r.get('is_noisy_data', False)]
    
    print(f"1. 总样本数: {len(all_results)}")
    print(f"2. 有效样本 (用于训练): {len(successes) - len(noisy)}")
    print(f"3. 噪声样本 (云端干扰): {len(noisy)}")
    
    # 打印一条样本预览，检查字段是否完整
    if successes:
        print("\n[一条可用于训练的样本预览]:")
        sample = successes[0]
        # 挑选模型关心的字段
        training_features = {
            "X_Client": [sample.get('actual_bandwidth'), sample.get('avg_cpu_usage')],
            "X_Image": [sample.get('static_image_size_mb'), sample.get('static_layer_count')],
            "Y_Label": sample.get('decompression_time', 0)
        }
        print(json.dumps(training_features, indent=4))

    # 5. 保存
    output_file = os.path.join(data_dir, "model_training_ready_data.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n数据已保存至: {output_file}")

if __name__ == "__main__":
    run_quick_test()