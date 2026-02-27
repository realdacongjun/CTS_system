#!/usr/bin/env python3
"""
新架构测试脚本 - 验证重构后的CTS系统功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_module_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    modules_to_test = [
        ('src.model_wrapper', 'CFTNetWrapper'),
        ('src.decision_engine', 'CAGSDecisionEngine'),
        ('baseline_downloader', 'pull_with_docker'),
        ('adaptive_downloader', 'cts_download')
    ]
    
    success_count = 0
    for module_name, class_or_func in modules_to_test:
        try:
            if '.' in module_name:
                # 处理子模块导入
                parts = module_name.split('.')
                module = __import__(module_name, fromlist=[class_or_func])
                getattr(module, class_or_func)
            else:
                # 处理同级模块导入
                module = __import__(module_name)
                getattr(module, class_or_func)
            print(f"✅ {module_name}.{class_or_func} 导入成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_or_func} 导入失败: {e}")
    
    return success_count == len(modules_to_test)

def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    
    try:
        import yaml
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ 配置文件加载成功")
            print(f"   实验轮次: {config.get('experiment', {}).get('repeat_times', 'N/A')}")
            print(f"   测试镜像数: {len(config.get('images', []))}")
            return True
        else:
            print("❌ 配置文件不存在")
            return False
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n🧪 测试目录结构...")
    
    required_paths = [
        "src",
        "models",
        "data",
        "results",
        "configs"
    ]
    
    success_count = 0
    for path in required_paths:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            print(f"✅ 目录存在: {path}")
            success_count += 1
        else:
            print(f"⚠️  目录不存在: {path} (将自动创建)")
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ 已创建: {path}")
                success_count += 1
            except Exception as e:
                print(f"   ❌ 创建失败: {e}")
    
    return success_count >= len(required_paths) - 1  # 允许一个目录创建失败

def test_model_files():
    """测试模型文件存在性"""
    print("\n🧪 测试模型文件...")
    
    model_files = [
        "models/cts_optimized.pth",
        "models/preprocessing_objects.pkl"
    ]
    
    found_count = 0
    for file_path in model_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ 模型文件存在: {file_path} ({size_mb:.2f} MB)")
            found_count += 1
        else:
            print(f"⚠️  模型文件不存在: {file_path}")
    
    if found_count == 0:
        print("💡 提示: 请确保模型文件已放置在正确位置")
    
    return True  # 模型文件不是必须的

def main():
    """主测试函数"""
    print("🚀 CTS新架构功能测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_module_imports),
        ("配置加载", test_config_loading),
        ("目录结构", test_directory_structure),
        ("模型文件", test_model_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 执行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 50)
    print("📋 测试结果总结")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:15} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        print("\n💡 下一步建议:")
        print("   1. 运行: python prepare_images.py")
        print("   2. 启动代理服务器: python proxy_server.py")
        print("   3. 运行实验: python experiment_runner.py")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)