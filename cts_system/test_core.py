import os
import shutil
import time
import json
from cts_core import (
    DifferentialSync, 
    CompressionCachePool, 
    CapabilityRegistry,
    ClientProbe,
    ImageAnalyzer
)

def test_differential_sync():
    print("\n=== 1. 测试差分同步引擎 (手术刀测试) ===")
    diff = DifferentialSync(block_size=1024) # 这里的块设置小一点方便测试
    
    # 场景：模拟 Docker 层稍微变了一点点
    # 文件A: 100KB 全是 'A'
    # 文件B: 100KB 全是 'A' + 中间插了几个 'B'
    data_source = b"A" * (100 * 1024)
    # 在中间修改数据，模拟版本更新
    data_target = data_source[:50000] + b"BBBBBBBBBB" + data_source[50010:]
    
    os.makedirs("temp_test", exist_ok=True)
    with open("temp_test/source.layer", "wb") as f: f.write(data_source)
    with open("temp_test/target.layer", "wb") as f: f.write(data_target)
    
    # 执行同步
    start = time.time()
    result = diff.sync_layers(
        "temp_test/source.layer", 
        "temp_test/target.layer", 
        "temp_test/delta.bin"
    )
    cost = (time.time() - start) * 1000
    
    print(f"原始文件大小: {result['original_size']} Bytes")
    print(f"差分包大小:   {result['delta_size']} Bytes")
    print(f"体积节省比例: {result['savings']*100:.2f}%")
    
    # 验证逻辑：源文件和目标文件几乎一样，delta包应该非常小
    # 如果节省比例超过 90%，说明差分算法生效了
    if result['savings'] > 0.9:
        print("✅ 测试通过：成功识别重复数据，极大减少了传输量！")
    else:
        print("❌ 测试失败：差分包过大，去重失效。")

def test_cache_eviction():
    print("\n=== 2. 测试智能缓存淘汰 (库管测试) ===")
    # 初始化一个小容量缓存池，只允许存 3 个
    pool = CompressionCachePool(root="temp_test/cache")
    pool.capacity = 3 
    
    # 场景：我们需要存入 A, B, C, D 四个数据，必然有一个要被踢走
    # 淘汰算法权重：热度 * 0.3 + 收益 * 0.7
    
    print("存入 A (低收益, 访问1次)...")
    pool.put("Key_A", b"Data_A", predicted_gain=0.1) # 分数低
    
    print("存入 B (高收益, 访问1次)...")
    pool.put("Key_B", b"Data_B", predicted_gain=0.9) # 分数高
    
    print("存入 C (中收益, 访问多)...")
    pool.put("Key_C", b"Data_C", predicted_gain=0.5) 
    pool.get("Key_C") # 多访问一次，增加热度
    pool.get("Key_C") 
    
    # 此时池子满了: [A, B, C]
    
    print("存入 D (触发淘汰)...")
    pool.put("Key_D", b"Data_D", predicted_gain=0.6)
    
    # 验证：谁被删了？
    # A 分数最低 (热度1, 收益0.1)，理论上 A 应该消失
    
    path_a = pool.get("Key_A")
    path_b = pool.get("Key_B")
    
    if path_a is None and path_b is not None:
        print("✅ 测试通过：价值最低的 A 被淘汰，高价值的 B 被保留！")
    else:
        print(f"❌ 测试失败：淘汰逻辑不符合预期 (A存在: {path_a is not None})")

def test_probe():
    print("\n=== 3. 测试客户端探针 (体检测试) ===")
    probe = ClientProbe()
    
    # 运行真实探测
    profile = probe.probe()
    
    print(f"节点ID: {profile['node_id']}")
    print(f"CPU评分: {profile['cpu_score']} (越高越快)")
    print(f"内存大小: {profile['memory_size']} MB")
    
    if profile['cpu_score'] > 0 and profile['memory_size'] > 0:
        print("✅ 测试通过：成功读取到底层硬件信息！")
    else:
        print("❌ 测试失败：未能获取硬件信息。")

def cleanup():
    # 清理垃圾文件
    if os.path.exists("temp_test"):
        shutil.rmtree("temp_test")
    print("\n=== 测试结束，清理现场 ===")

if __name__ == "__main__":
    try:
        test_differential_sync()
        test_cache_eviction()
        test_probe()
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()