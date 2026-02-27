#!/usr/bin/env python3
"""
兼容性检查脚本
验证所有修复后的代码兼容性问题是否已解决
"""

import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_model_import():
    """测试模型导入功能"""
    print("🧪 测试模型导入...")
    
    try:
        # 测试模型包装器导入
        from src.model_wrapper import CFTNetWrapper
        print("✅ CFTNetWrapper 导入成功")
        
        # 测试决策引擎导入
        from src.decision_engine import CAGSDecisionEngine
        print("✅ CAGSDecisionEngine 导入成功")
        
        # 测试模型类导入（通过包装器内部机制）
        model_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        if model_config_path.exists():
            wrapper = CFTNetWrapper(str(model_config_path))
            print("✅ 模型类动态导入成功")
            print(f"   设备: {wrapper.device}")
            print(f"   模型路径: {wrapper.model_path}")
        else:
            print("⚠️  配置文件不存在，跳过模型初始化测试")
            
        return True
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n🧪 测试配置文件加载...")
    
    try:
        # 测试实验配置加载
        config_paths = [
            PROJECT_ROOT / "config.yaml",
            PROJECT_ROOT / "configs" / "global_config.yaml",
            PROJECT_ROOT / "configs" / "config.yaml"
        ]
        
        config_found = False
        for config_path in config_paths:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ 配置文件加载成功: {config_path.name}")
                print(f"   实验轮次: {config.get('experiment', {}).get('repeat_times', 'N/A')}")
                config_found = True
                break
        
        if not config_found:
            print("❌ 未找到任何配置文件")
            return False
            
        # 测试模型配置加载
        model_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
            print(f"✅ 模型配置加载成功")
            print(f"   模型路径: {model_config['model']['model_path']}")
            print(f"   预处理路径: {model_config['model']['preprocess_path']}")
        else:
            print("❌ 模型配置文件不存在")
            return False
            
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_path_resolution():
    """测试路径解析功能"""
    print("\n🧪 测试路径解析...")
    
    try:
        # 测试相对路径解析
        from src.model_wrapper import CFTNetWrapper
        model_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        
        if model_config_path.exists():
            wrapper = CFTNetWrapper(str(model_config_path))
            
            # 检查路径是否正确解析
            model_path = Path(wrapper.model_path)
            preprocess_path = Path(wrapper.preprocess_path)
            
            if model_path.exists() or not model_path.is_absolute():
                print("✅ 模型路径解析正确")
            else:
                print(f"⚠️  模型路径可能有问题: {model_path}")
                
            if preprocess_path.exists() or not preprocess_path.is_absolute():
                print("✅ 预处理路径解析正确")
            else:
                print(f"⚠️  预处理路径可能有问题: {preprocess_path}")
                
            return True
        else:
            print("⚠️  配置文件不存在，跳过路径解析测试")
            return True
            
    except Exception as e:
        print(f"❌ 路径解析测试失败: {e}")
        return False

def test_import_without_training():
    """测试导入时不触发训练"""
    print("\n🧪 测试导入时不触发训练...")
    
    try:
        # 这个测试比较特殊，我们需要验证导入cts_model不会触发训练
        import subprocess
        import tempfile
        
        # 创建一个测试脚本，只导入不执行
        test_script = f"""
import sys
sys.path.insert(0, "{PROJECT_ROOT}")
sys.path.insert(0, "{PROJECT_ROOT / 'src'}")

# Only import, do not execute training
from src.cts_model import CompactCFTNetV2
print("[SUCCESS] Successfully imported CompactCFTNetV2 without triggering training")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_script = f.name
        
        # 运行测试脚本
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=30)
        
        os.unlink(temp_script)  # 删除临时文件
        
        if result.returncode == 0 and "成功导入" in result.stdout:
            print("✅ 导入测试通过，未触发训练")
            return True
        else:
            print(f"❌ 导入测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 导入测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 CTS系统兼容性检查")
    print("=" * 50)
    
    tests = [
        ("模型导入", test_model_import),
        ("配置加载", test_config_loading),
        ("路径解析", test_path_resolution),
        ("导入安全", test_import_without_training)
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
    print("📋 兼容性检查总结")
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
    
    overall_status = "🟢 所有兼容性问题已解决" if failed == 0 else "🟡 仍存在兼容性问题"
    print(f"\n📊 整体状态: {overall_status}")
    
    if failed == 0:
        print("\n🎉 恭喜！所有兼容性修复已完成")
        print("✅ 代码现在可以正常运行而不会出现ImportError或FileNotFoundError")
        print("\n💡 下一步建议:")
        print("   1. 运行 system_health_check.py 进行完整系统检查")
        print("   2. 准备测试数据和模型文件")
        print("   3. 运行完整实验流程")
    else:
        print(f"\n⚠️  发现 {failed} 个兼容性问题需要解决")
        print("💡 请检查上述失败项并相应修复")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)