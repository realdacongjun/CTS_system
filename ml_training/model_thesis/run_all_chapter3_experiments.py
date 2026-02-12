import os
import subprocess
import sys
import time
from pathlib import Path

def run_experiment(script_name, description):
    """运行单个实验脚本"""
    print(f"\n{'='*60}")
    print(f"开始执行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ {description} 执行成功!")
            print(f"执行时间: {end_time - start_time:.2f} 秒")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print(f"✗ {description} 执行失败!")
            print(f"错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ 执行 {script_name} 时发生异常: {str(e)}")
        return False
    
    return True

def main():
    """主执行函数"""
    print("CTS系统第三章实验执行器")
    print("="*60)
    print("本脚本将依次执行第三章所需的四个核心实验")
    print("="*60)
    
    # 定义实验脚本列表
    experiments = [
        ("chapter3_1_heterogeneity_analysis.py", "3.1 异构性特征分析实验"),
        ("chapter3_2_prediction_accuracy.py", "3.2 性能预测准确性评估"),
        ("chapter3_3_uncertainty_quantification.py", "3.3 不确定性量化实验"),
        ("chapter3_4_ablation_study.py", "3.4 消融实验研究")
    ]
    
    # 确保在正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"工作目录: {script_dir}")
    
    # 检查依赖
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"警告: 缺少以下Python包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return
    
    # 执行所有实验
    successful_experiments = []
    failed_experiments = []
    
    for script_name, description in experiments:
        if os.path.exists(script_name):
            success = run_experiment(script_name, description)
            if success:
                successful_experiments.append(description)
            else:
                failed_experiments.append(description)
        else:
            print(f"✗ 找不到脚本文件: {script_name}")
            failed_experiments.append(description)
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("实验执行总结")
    print(f"{'='*60}")
    print(f"成功执行: {len(successful_experiments)} 个实验")
    print(f"执行失败: {len(failed_experiments)} 个实验")
    
    if successful_experiments:
        print("\n成功实验列表:")
        for exp in successful_experiments:
            print(f"  ✓ {exp}")
    
    if failed_experiments:
        print("\n失败实验列表:")
        for exp in failed_experiments:
            print(f"  ✗ {exp}")
    
    # 列出生成的文件
    print(f"\n生成的主要文件:")
    generated_files = [
        "compression_heterogeneity_results.csv",
        "environment_heterogeneity_results.csv", 
        "figure_3_1_layer_compression_comparison.png",
        "figure_3_2_environment_performance.png",
        "figure_3_3_compression_tradeoff.png",
        "chapter3_statistics.json",
        
        "synthetic_performance_data.csv",
        "table_3_1_model_comparison.csv",
        "figure_3_4_prediction_accuracy.png",
        "chapter3_2_statistics.json",
        
        "id_training_data.csv",
        "ood_test_data.csv",
        "figure_3_5_uncertainty_analysis.png",
        "chapter3_3_statistics.json",
        "ause_results.json",
        
        "ablation_experiment_data.csv",
        "table_3_2_ablation_comparison.csv",
        "figure_3_6_component_contribution.png",
        "chapter3_4_statistics.json"
    ]
    
    existing_files = [f for f in generated_files if os.path.exists(f)]
    missing_files = [f for f in generated_files if not os.path.exists(f)]
    
    if existing_files:
        print("已生成的文件:")
        for file in existing_files:
            file_size = os.path.getsize(file) if os.path.exists(file) else 0
            print(f"  ✓ {file} ({file_size:,} 字节)")
    
    if missing_files:
        print("缺失的文件:")
        for file in missing_files:
            print(f"  ✗ {file}")
    
    print(f"\n{'='*60}")
    print("第三章实验执行完成!")
    print("请检查生成的图表和数据文件，用于论文写作。")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()