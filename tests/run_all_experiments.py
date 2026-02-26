#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTS闭环系统完整评估实验主入口
============================
优化版：兼容新的 model_wrapper 和 exp1-exp5
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path

# 添加项目根目录到Python路径 (更健壮的路径处理)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from src.environment import EnvironmentController
from src.executor import DockerExecutor
from src.model_wrapper import CFTNetWrapper
# 注意：这里不再导入 CAGSEngine，因为实验内部会直接初始化 CAGSDecisionEngine

# 导入实验模块
import experiments.exp1_end_to_end as exp1
import experiments.exp2_ablation as exp2
import experiments.exp3_robustness as exp3
import experiments.exp4_lightweight as exp4
import experiments.exp5_stability as exp5

def setup_environment(global_config: Dict):
    """环境初始化和检查"""
    print("="*80)
    print("🔍 CTS闭环系统实验环境检查")
    print("="*80)
    
    # 检查root权限（Linux/macOS）
    if os.name != 'nt':  # 非Windows系统
        if os.geteuid() != 0:
            print("⚠️  警告: 建议以root权限运行以获得完整网络控制能力")
            print("   某些网络配置功能可能受限")
    
    # 检查必要目录
    required_dirs = [
        PROJECT_ROOT / "configs", 
        PROJECT_ROOT / "data", 
        PROJECT_ROOT / "src", 
        PROJECT_ROOT / "experiments"
    ]
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"❌ 缺少必要目录: {dir_path}")
            return False
        print(f"✅ 目录检查通过: {dir_path}")
    
    # 检查配置文件
    config_files = [
        PROJECT_ROOT / "configs" / "global_config.yaml", 
        PROJECT_ROOT / "configs" / "model_config.yaml"
    ]
    for config_file in config_files:
        if not config_file.exists():
            print(f"❌ 缺少配置文件: {config_file}")
            return False
        print(f"✅ 配置文件存在: {config_file}")
    
    # 检查模型文件 (从配置中读取路径)
    model_cfg_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(model_cfg_path, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)
    
    model_files = [
        PROJECT_ROOT / model_cfg['model']['model_path'],
        PROJECT_ROOT / model_cfg['model']['preprocess_path']
    ]
    for model_file in model_files:
        if not model_file.exists():
            print(f"❌ 缺少模型文件: {model_file}")
            print("   请确保模型文件路径在 model_config.yaml 中配置正确")
            return False
        print(f"✅ 模型文件存在: {model_file}")
    
    # 创建输出目录
    os.makedirs(global_config['global']['result_save_dir'], exist_ok=True)
    os.makedirs(global_config['global']['log_dir'], exist_ok=True)
    
    return True

def load_global_config():
    """加载全局配置"""
    try:
        config_path = PROJECT_ROOT / "configs" / "global_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        raise

def setup_logging(global_config):
    """设置日志系统"""
    log_dir = Path(global_config['global']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "experiment_main.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("📝 日志系统初始化完成")
    logger.info(f"📄 日志文件: {log_file}")

def initialize_components(global_config):
    """【优化】初始化实验组件"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔧 初始化实验组件...")
        
        # 1. 初始化环境控制器
        env_controller = EnvironmentController(str(PROJECT_ROOT / "configs" / "global_config.yaml"))
        logger.info("✅ 环境控制器初始化完成")
        
        # 2. 初始化Docker执行器
        # 注意：如果 global_config 没有 registry，使用默认值
        registry_addr = global_config.get('registry', {}).get('address', 'docker.io')
        docker_executor = DockerExecutor(registry_addr)
        logger.info("✅ Docker执行器初始化完成")
        
        # 3. 初始化模型包装器 (核心！)
        model_wrapper = CFTNetWrapper(str(PROJECT_ROOT / "configs" / "model_config.yaml"))
        logger.info("✅ CFT-Net模型包装器初始化完成")
        
        # 注意：不再单独初始化 CAGSEngine
        # 因为每个实验内部会根据 model_wrapper 初始化 CAGSDecisionEngine
        
        return env_controller, docker_executor, model_wrapper
        
    except Exception as e:
        logger.error(f"❌ 组件初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """主执行函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           CTS闭环系统完整评估实验平台 (优化版)               ║
║                                                              ║
║  验证目标: CFT-Net性能预测 + CAGS自适应决策 双创新点组合     ║
║  实验数量: 5个核心实验                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. 加载配置 (先加载，用于环境检查)
    try:
        global_config = load_global_config()
    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        sys.exit(1)
    
    # 2. 环境检查
    if not setup_environment(global_config):
        print("\n❌ 环境检查失败，请检查上述错误信息")
        sys.exit(1)
    
    # 3. 设置日志
    setup_logging(global_config)
    logger = logging.getLogger(__name__)
    
    # 4. 初始化组件
    try:
        env_controller, docker_executor, model_wrapper = initialize_components(global_config)
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        sys.exit(1)
    
    # 存储所有实验结果
    all_results = {}
    
    try:
        # 执行所有实验
        print("\n" + "="*80)
        print("🚀 开始执行CTS闭环系统完整评估实验")
        print("="*80)
        
        # -----------------------------------------------------------
        # 实验一：端到端性能基准
        # -----------------------------------------------------------
        print("\n" + "="*80)
        print("📊 实验一：端到端性能基准测试")
        print("="*80)
        exp1_results = exp1.run(
            env_controller, 
            docker_executor, 
            model_wrapper,  # 【新增】传入 model_wrapper
            global_config
        )
        exp1.print_summary(exp1_results)
        all_results['exp1'] = exp1_results
        
        # -----------------------------------------------------------
        # 实验二：协同增益消融
        # -----------------------------------------------------------
        print("\n" + "="*80)
        print("🔬 实验二：协同增益消融实验")
        print("="*80)
        exp2_results = exp2.run(
            env_controller, 
            docker_executor, 
            model_wrapper,  # 【新增】传入 model_wrapper
            global_config
        )
        exp2.print_summary(exp2_results)
        all_results['exp2'] = exp2_results
        
        # -----------------------------------------------------------
        # 实验三：动态鲁棒性
        # -----------------------------------------------------------
        print("\n" + "="*80)
        print("🛡️  实验三：动态鲁棒性测试")
        print("="*80)
        exp3_results = exp3.run(
            env_controller, 
            docker_executor, 
            model_wrapper,  # 传入 model_wrapper
            global_config
        )
        exp3.print_summary(exp3_results)
        all_results['exp3'] = exp3_results
        
        # -----------------------------------------------------------
        # 实验四：轻量化部署
        # -----------------------------------------------------------
        print("\n" + "="*80)
        print("⚙️  实验四：轻量化部署验证")
        print("="*80)
        exp4_results = exp4.run(
            global_config,
            model_wrapper  # 【新增】传入 model_wrapper (可选，用于优化)
        )
        exp4.print_summary(exp4_results)
        all_results['exp4'] = exp4_results
        
        # -----------------------------------------------------------
        # 实验五：长期稳定性
        # -----------------------------------------------------------
        print("\n" + "="*80)
        print("🔁 实验五：长期稳定性测试")
        print("="*80)
        exp5_results = exp5.run(
            env_controller, 
            docker_executor, 
            model_wrapper,  # 【新增】传入 model_wrapper
            global_config
        )
        exp5.print_summary(exp5_results)
        all_results['exp5'] = exp5_results
        
        # -----------------------------------------------------------
        # 保存所有原始结果
        # -----------------------------------------------------------
        result_file = Path(global_config['global']['result_save_dir']) / "all_raw_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 所有原始实验数据已保存到: {result_file}")
        
        # -----------------------------------------------------------
        # 打印最终总结
        # -----------------------------------------------------------
        print("\n" + "="*80)
        print("🎉 CTS闭环系统完整评估实验完成！")
        print("="*80)
        
        print(f"\n📁 实验结果位置:")
        print(f"   原始数据: {global_config['global']['result_save_dir']}")
        print(f"   日志文件: {global_config['global']['log_dir']}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  实验被用户中断")
        print("\n⚠️  实验被用户中断，正在清理...")
    except Exception as e:
        logger.error(f"\n❌ 实验执行过程中发生错误: {e}", exc_info=True)
        print(f"\n❌ 实验失败: {e}")
        raise
    finally:
        # 清理环境
        try:
            env_controller.cleanup()
            logger.info("✅ 环境已清理")
        except:
            pass
        
        print("\n✅ CTS实验平台执行完毕")

if __name__ == "__main__":
    main()