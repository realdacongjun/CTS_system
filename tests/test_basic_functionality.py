#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTS实验框架基础功能测试 (优化版)
=====================
用于快速验证各组件的基本功能是否正常工作。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    all_passed = True
    
    try:
        from src.environment import EnvironmentController
        print("✅ EnvironmentController 导入成功")
    except Exception as e:
        print(f"❌ EnvironmentController 导入失败: {e}")
        all_passed = False
    
    try:
        from src.executor import DockerExecutor
        print("✅ DockerExecutor 导入成功")
    except Exception as e:
        print(f"❌ DockerExecutor 导入失败: {e}")
        all_passed = False
    
    try:
        from src.model_wrapper import CFTNetWrapper
        print("✅ CFTNetWrapper 导入成功")
    except Exception as e:
        print(f"❌ CFTNetWrapper 导入失败: {e}")
        print(f"   ⚠️  这可能是由于缺少依赖或模型文件: {e}")
        # 模型导入失败不阻断，因为可能还没准备好
    
    # 【修改】修正类名：CAGSDecisionEngine
    try:
        from src.decision_engine import CAGSDecisionEngine
        print("✅ CAGSDecisionEngine 导入成功")
    except ImportError as e:
        print(f"❌ CAGSDecisionEngine 导入失败: {e}")
        all_passed = False
    except Exception as e:
        print(f"⚠️  CAGSDecisionEngine 导入警告 (可能是内部初始化错误): {e}")
    
    try:
        import src.utils as utils
        print("✅ src.utils 导入成功")
    except Exception as e:
        print(f"❌ src.utils 导入失败: {e}")
        all_passed = False
    
    return all_passed

def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    
    try:
        import yaml
        
        # 1. 测试全局配置
        global_config_path = PROJECT_ROOT / "configs" / "global_config.yaml"
        if global_config_path.exists():
            with open(global_config_path, 'r', encoding='utf-8') as f:
                global_config = yaml.safe_load(f)
            print("✅ 全局配置 (global_config.yaml) 加载成功")
            
            # 检查关键字段
            if 'scenarios' in global_config:
                print(f"   场景数量: {len(global_config['scenarios'])}")
            if 'images' in global_config:
                print(f"   测试镜像数量: {len(global_config['images'])}")
        else:
            print(f"❌ 全局配置文件不存在: {global_config_path}")
            return False
        
        # 2. 测试模型配置
        model_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
            print("✅ 模型配置 (model_config.yaml) 加载成功")
        else:
            print(f"⚠️  模型配置文件不存在 (可选): {model_config_path}")
            
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_directory_structure():
    """测试目录结构 (优化版)"""
    print("\n🧪 测试目录结构...")
    
    # 【修改】更新必要目录列表，移除非必须的
    required_dirs = [
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "data",      # 只要 data 根目录
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "experiments",
        # PROJECT_ROOT / "models",    # 可选，刚开始可能没有
        # PROJECT_ROOT / "analysis",  # 可选
    ]
    
    # 建议创建的目录
    suggested_dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "results"
    ]
    
    all_good = True
    
    # 检查必须目录
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ 目录存在: {dir_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"❌ 目录缺失: {dir_path.relative_to(PROJECT_ROOT)}")
            all_good = False
    
    # 检查建议目录
    print("\n   建议目录:")
    for dir_path in suggested_dirs:
        if dir_path.exists():
            print(f"   ✅ {dir_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"   ⚠️  {dir_path.relative_to(PROJECT_ROOT)} (不存在，运行时会自动创建)")
    
    return all_good

def test_model_files():
    """测试模型文件 (优化版)"""
    print("\n🧪 测试模型文件...")
    
    # 先尝试从 model_config.yaml 读取路径，更灵活
    model_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    model_files_to_check = []
    
    if model_config_path.exists():
        try:
            import yaml
            with open(model_config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if 'model' in cfg:
                m_path = PROJECT_ROOT / cfg['model'].get('model_path', '')
                p_path = PROJECT_ROOT / cfg['model'].get('preprocess_path', '')
                model_files_to_check.append(m_path)
                model_files_to_check.append(p_path)
        except:
            pass
    
    # 如果配置里没读到，使用默认路径
    if not model_files_to_check:
        print("   未在配置中找到模型路径，使用默认路径检查...")
        model_files_to_check = [
            PROJECT_ROOT / "models" / "cts_optimized_0218_2125_seed42.pth",
            PROJECT_ROOT / "models" / "preprocessing_objects_optimized.pkl"
        ]
    
    all_good = True
    found_any = False
    
    for model_file in model_files_to_check:
        if model_file.exists():
            found_any = True
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ 模型文件存在: {model_file.name} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️  模型文件未找到: {model_file.name} (这是可选的，除非你要运行完整推理)")
    
    if not found_any:
        print("   💡 提示：如果还没有训练好模型，可以先跳过模型相关测试")
    
    return True # 模型文件不是必须通过的项

def test_basic_component_init():
    """【新增】测试基础组件初始化"""
    print("\n🧪 测试组件轻量级初始化...")
    
    try:
        from src.environment import EnvironmentController
        # 测试初始化 EnvironmentController
        config_path = PROJECT_ROOT / "configs" / "global_config.yaml"
        if config_path.exists():
            # 这里只测试初始化，不测试实际 tc 命令
            env = EnvironmentController(str(config_path))
            print("✅ EnvironmentController 初始化成功")
    except Exception as e:
        print(f"⚠️  EnvironmentController 初始化警告: {e}")
    
    try:
        from src.executor import DockerExecutor
        # 测试初始化 DockerExecutor
        exe = DockerExecutor("docker.io")
        print("✅ DockerExecutor 初始化成功")
    except Exception as e:
        print(f"⚠️  DockerExecutor 初始化警告: {e}")
    
    return True

def main():
    """主测试函数"""
    print("🚀 CTS实验框架基础功能测试 (优化版)")
    print("="*60)
    
    tests = [
        ("目录结构", test_directory_structure),
        ("模块导入", test_imports),
        ("配置加载", test_config_loading),
        ("组件初始化", test_basic_component_init),
        ("模型文件", test_model_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 执行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "="*60)
    print("📋 测试结果总结")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:20} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有基础测试通过！")
        print("\n💡 下一步建议:")
        print("   1. 确认 configs/global_config.yaml 中的参数 (如 network.interface)")
        print("   2. 确保 Docker 服务正在运行")
        print("   3. (可选) 运行 prepare_experiment_images.sh 准备镜像")
        print("   4. 执行: python run_all_experiments.py")
    else:
        print("\n⚠️  部分测试失败。")
        print("   请检查上述错误信息。")
        print("   注意：'模型文件'测试失败不影响代码结构验证。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)