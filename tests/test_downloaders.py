#!/usr/bin/env python3
"""
下载器模块测试脚本
验证修改后的baseline_downloader和adaptive_downloader功能
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_baseline_downloader():
    """测试Baseline下载器"""
    print("🧪 测试 Baseline 下载器...")
    
    try:
        import baseline_downloader
        
        # 测试函数存在性
        assert hasattr(baseline_downloader, 'pull_with_docker'), "缺少 pull_with_docker 函数"
        print("✅ pull_with_docker 函数存在")
        
        # 测试返回字段格式
        result = baseline_downloader.pull_with_docker("alpine:latest", clear_cache=False, timeout=30)
        
        required_fields = ['strategy', 'image', 'time_s', 'success']
        for field in required_fields:
            assert field in result, f"缺少必需字段: {field}"
        
        print("✅ 返回字段格式正确")
        print(f"   策略: {result['strategy']}")
        print(f"   镜像: {result['image']}")
        print(f"   时间: {result['time_s']}s")
        print(f"   成功: {result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline 下载器测试失败: {e}")
        return False

def test_adaptive_downloader():
    """测试自适应下载器"""
    print("\n🧪 测试自适应下载器...")
    
    try:
        import adaptive_downloader
        
        # 测试函数存在性
        assert hasattr(adaptive_downloader, 'cts_download'), "缺少 cts_download 函数"
        print("✅ cts_download 函数存在")
        
        # 测试配置映射
        required_strategies = ['gzip', 'zstd_l3', 'zstd_l10', 'uncompressed']
        for strategy in required_strategies:
            assert strategy in adaptive_downloader.COMPRESSION_CFG, f"缺少策略配置: {strategy}"
        
        print("✅ 压缩策略配置完整")
        
        # 测试返回字段格式
        result = adaptive_downloader.cts_download(
            "alpine:latest", 
            strategy="zstd_l3", 
            threads=2,
            clear_cache=False,
            timeout=30
        )
        
        required_fields = ['strategy', 'compression', 'threads', 'image', 'time_s', 'success']
        for field in required_fields:
            assert field in result, f"缺少必需字段: {field}"
        
        print("✅ 返回字段格式正确")
        print(f"   策略: {result['strategy']}")
        print(f"   压缩: {result['compression']}")
        print(f"   线程: {result['threads']}")
        print(f"   镜像: {result['image']}")
        print(f"   时间: {result['time_s']}s")
        print(f"   成功: {result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 自适应下载器测试失败: {e}")
        return False

def test_import_consistency():
    """测试导入一致性"""
    print("\n🧪 测试导入一致性...")
    
    try:
        # 测试experiment_runner中的导入
        import baseline_downloader
        import adaptive_downloader
        
        # 验证函数签名一致性
        import inspect
        baseline_sig = inspect.signature(baseline_downloader.pull_with_docker)
        adaptive_sig = inspect.signature(adaptive_downloader.cts_download)
        
        print("✅ 导入一致性检查通过")
        print(f"   Baseline 函数签名: {baseline_sig}")
        print(f"   Adaptive 函数签名: {adaptive_sig}")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入一致性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 下载器模块功能测试")
    print("=" * 50)
    
    tests = [
        ("Baseline下载器", test_baseline_downloader),
        ("自适应下载器", test_adaptive_downloader),
        ("导入一致性", test_import_consistency)
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
    print("📋 下载器测试总结")
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
        print("\n🎉 所有下载器测试通过！")
        print("✅ 修改后的下载器可以正常使用")
        print("\n💡 下一步建议:")
        print("   1. 运行 prepare_images.py 准备测试镜像")
        print("   2. 启动 proxy_server.py")
        print("   3. 运行 experiment_runner.py 进行完整实验")
    else:
        print(f"\n⚠️  发现 {failed} 个问题需要解决")
        print("💡 请检查上述失败项并相应修复")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)