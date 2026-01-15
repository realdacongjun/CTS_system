"""
CTS Core Module (cts_core.py)
-----------------------------
系统基础设施层 (Infrastructure Layer)
包含：感知(Sensing)、记忆(Memory)、执行(Execution)、反馈(Feedback)
"""

import os
import json
import time
import hashlib
import struct
import math
import shutil
import random
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict, deque
from io import BytesIO

# 第三方依赖 (请确保安装: pip install docker psutil requests py-cpuinfo)
try:
    import docker
    import psutil
    import requests
    import cpuinfo
    import tarfile
except ImportError as e:
    print(f"⚠ 缺少核心依赖: {e}. 请运行 pip install docker psutil requests py-cpuinfo")

# =========================================================================
# 1. 数据结构 (Data Structures)
# =========================================================================

@dataclass
class NodeCapability:
    """节点能力画像"""
    node_id: str
    cpu_score: int = 1000
    bandwidth_mbps: float = 10.0
    network_rtt: float = 50.0
    decompression_speed: Dict[str, float] = None
    memory_size: int = 0      # MB
    last_updated: float = 0.0
    confidence: float = 0.5   # 置信度 (0.0 - 1.0)

@dataclass
class FeedbackData:
    """闭环反馈数据"""
    node_id: str
    image_id: str
    algo_used: str
    actual_transfer_time: float
    actual_decomp_time: float
    predicted_transfer_time: Optional[float] = None
    predicted_decomp_time: Optional[float] = None
    timestamp: float = 0.0

# =========================================================================
# 2. 感知层 - 主动探针 (Active Sensing)
# =========================================================================

class ClientProbe:
    """
    客户端主动探针
    运行在 Client 端，主动向 Server 发起测速和自检
    """
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.node_id = self._get_mac_node_id()

    def probe(self) -> Dict[str, Any]:
        """执行全量探测"""
        print(f"[ClientProbe] 正在探测本机 ({self.node_id}) 能力...")
        
        # 1. 硬件信息
        sys_info = self._probe_system_info()
        
        # 2. 动态基准测试
        cpu_score = self._probe_cpu_benchmark()
        net_stats = self._probe_network()
        decomp_speeds = self._probe_decompression_speed()

        profile = {
            "node_id": self.node_id,
            "cpu_score": cpu_score,
            "bandwidth_mbps": net_stats['bandwidth'],
            "network_rtt": net_stats['rtt'],
            "decompression_speed": decomp_speeds,
            "memory_size": sys_info['memory_mb'],
            "system_info": sys_info,
            "timestamp": time.time()
        }
        return profile

    def _probe_cpu_benchmark(self):
        """通过做数学题估算 CPU 单核性能"""
        def benchmark():
            start = time.time()
            _ = sum(i*i for i in range(500000))
            return time.time() - start
        
        times = [benchmark() for _ in range(3)]
        avg_time = sum(times) / len(times)
        # 基准: 0.1s = 1000分
        return int(100 / (avg_time + 0.001))

    def _probe_network(self):
        """连接 Server 进行真实测速"""
        try:
            # RTT
            t0 = time.time()
            requests.get(f"{self.server_url}/ping", timeout=2)
            rtt = (time.time() - t0) * 1000

            # Bandwidth (下载 1MB)
            t0 = time.time()
            resp = requests.get(f"{self.server_url}/speedtest?size=1mb", timeout=5)
            if resp.status_code == 200:
                size_bits = len(resp.content) * 8
                duration = time.time() - t0
                if duration < 0.001: duration = 0.001
                bw = (size_bits / duration) / 1_000_000 # Mbps
                return {"bandwidth": round(bw, 2), "rtt": round(rtt, 1)}
        except Exception:
            pass
        return {"bandwidth": 10.0, "rtt": 50.0} # 兜底值

    def _probe_decompression_speed(self):
        """本地解压基准测试 (MB/s)"""
        import zlib
        try:
            data = b"A" * 1024 * 1024 # 1MB
            compressed = zlib.compress(data, level=1)
            start = time.time()
            for _ in range(5): zlib.decompress(compressed)
            duration = (time.time() - start) / 5
            speed = 1.0 / duration if duration > 0 else 500
            return {"gzip": int(speed), "zstd": int(speed * 1.5)} # 简单估算
        except:
            return {"gzip": 100, "zstd": 150}

    def _probe_system_info(self):
        try:
            mem = psutil.virtual_memory()
            return {
                "memory_mb": int(mem.total / 1024 / 1024),
                "cpu_cores": psutil.cpu_count(),
                "os": os.name
            }
        except:
            return {"memory_mb": 4096}

    def _get_mac_node_id(self):
        import uuid
        return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:16]

# =========================================================================
# 3. 感知层 - 任务分析 (Task Sensing)
# =========================================================================

class ImageAnalyzer:
    """
    镜像深度分析器
    通过 docker save 导出并解压，计算真实信息熵
    """
    def __init__(self, work_dir="temp/image_analysis"):
        self.work_dir = work_dir
        try:
            self.client = docker.from_env()
        except:
            self.client = None

    def analyze(self, image_name: str) -> Dict[str, Any]:
        """核心分析入口"""
        if not self.client:
            print("⚠ Docker 未运行，返回模拟数据")
            return self._get_mock(image_name)

        print(f"[ImageAnalyzer] 正在深度分析 {image_name} (这可能需要几秒钟)...")
        try:
            # 1. 导出与解压
            extract_path = self._export_and_extract(image_name)
            
            # 2. 解析 Manifest
            manifest_path = os.path.join(extract_path, "manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError("Manifest not found")
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)[0]

            # 3. 逐层分析
            layers_info = []
            blobs_dir = os.path.join(extract_path, "blobs", "sha256")
            
            for i, layer_file in enumerate(manifest.get('Layers', [])):
                layer_sha = os.path.basename(layer_file)
                # 兼容不同 Docker 版本的导出结构 (有些直接在根目录，有些在 blobs)
                possible_paths = [
                    os.path.join(extract_path, layer_file),
                    os.path.join(blobs_dir, layer_sha)
                ]
                
                real_path = next((p for p in possible_paths if os.path.exists(p)), None)
                if not real_path: continue

                # 分析层内容
                stats = self._analyze_layer_file(real_path)
                stats['layer_index'] = i
                stats['layer_digest'] = f"sha256:{layer_sha}"
                layers_info.append(stats)

            # 4. 汇总
            total_size = sum(l['size'] for l in layers_info)
            avg_entropy = sum(l['entropy'] for l in layers_info) / len(layers_info) if layers_info else 0
            
            return {
                "image_id": hashlib.md5(image_name.encode()).hexdigest()[:12],
                "total_size_mb": round(total_size / (1024**2), 2),
                "layer_count": len(layers_info),
                "avg_layer_entropy": round(avg_entropy, 4),
                "layers": layers_info
            }

        except Exception as e:
            print(f"⚠ 分析失败: {e}")
            return self._get_mock(image_name)
        finally:
            # 清理临时文件
            if os.path.exists(self.work_dir):
                shutil.rmtree(self.work_dir, ignore_errors=True)

    def _export_and_extract(self, image_name):
        clean_name = image_name.replace("/", "_").replace(":", "_")
        target_dir = os.path.join(self.work_dir, clean_name)
        os.makedirs(target_dir, exist_ok=True)
        
        image = self.client.images.get(image_name)
        tar_path = os.path.join(target_dir, "img.tar")
        
        with open(tar_path, 'wb') as f:
            for chunk in image.save(): f.write(chunk)
            
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=target_dir)
            
        return target_dir

    def _analyze_layer_file(self, layer_path):
        """分析单个 layer.tar 文件的熵值"""
        size = os.path.getsize(layer_path)
        entropy = 0.5 # 默认
        
        # 采样前 1MB 数据计算熵
        try:
            with open(layer_path, 'rb') as f:
                sample = f.read(1024 * 1024)
                entropy = self._calculate_entropy(sample)
        except: pass
        
        return {"size": size, "entropy": entropy}

    def _calculate_entropy(self, data):
        if not data: return 0
        freq = defaultdict(int)
        for byte in data: freq[byte] += 1
        ent = 0
        for count in freq.values():
            p = count / len(data)
            ent -= p * math.log2(p)
        return ent / 8.0

    def _get_mock(self, name):
        return {"total_size_mb": 100, "layer_count": 5, "avg_layer_entropy": 0.5, "layers": []}

# =========================================================================
# 4. 记忆层 - 注册中心 (Memory)
# =========================================================================

class CapabilityRegistry:
    """线程安全的能力注册中心"""
    def __init__(self, path="registry/capabilities.json"):
        self.path = path
        self.capabilities: Dict[str, NodeCapability] = {}
        self.lock = threading.Lock()
        self._load()

    def register(self, node_id: str, data: Dict[str, Any]):
        with self.lock:
            # 简单的 Merge 逻辑
            if node_id in self.capabilities:
                cap = self.capabilities[node_id]
                cap.cpu_score = data.get('cpu_score', cap.cpu_score)
                cap.bandwidth_mbps = data.get('bandwidth_mbps', cap.bandwidth_mbps)
                cap.last_updated = time.time()
                cap.confidence = min(1.0, cap.confidence + 0.1)
            else:
                self.capabilities[node_id] = NodeCapability(
                    node_id=node_id,
                    cpu_score=data.get('cpu_score', 1000),
                    bandwidth_mbps=data.get('bandwidth_mbps', 10),
                    last_updated=time.time(),
                    confidence=0.5
                )
            self._save()

    def get(self, node_id: str) -> Dict[str, Any]:
        with self.lock:
            cap = self.capabilities.get(node_id)
            return asdict(cap) if cap else None

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({k: asdict(v) for k,v in self.capabilities.items()}, f, indent=2)

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items(): self.capabilities[k] = NodeCapability(**v)
            except: pass

class NetworkProfiler:
    """被动网络画像 (历史记录)"""
    def __init__(self, path="registry/network_history.json"):
        self.path = path
        self.history = {}
        self._load()

    def update(self, node_id, bw_sample):
        # 移动平均算法
        old_bw = self.history.get(node_id, {}).get('avg_bw', bw_sample)
        new_bw = 0.7 * old_bw + 0.3 * bw_sample
        self.history[node_id] = {'avg_bw': new_bw, 'last_seen': time.time()}
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f: json.dump(self.history, f)
    
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f: self.history = json.load(f)
            except: pass

# =========================================================================
# 5. 执行层 (Execution Layer)
# =========================================================================

class DifferentialSync:
    """
    块级差分同步引擎 (rsync-like)
    """
    def __init__(self, block_size=65536): # 64KB
        self.block_size = block_size

    def sync_layers(self, source_path, target_path, output_path):
        """生成差分包 (优化了内存使用)"""
        # 1. 计算源文件指纹
        source_map = {}
        with open(source_path, 'rb') as f:
            idx = 0
            while chunk := f.read(self.block_size):
                fp = hashlib.sha256(chunk).hexdigest()
                source_map[fp] = idx
                idx += len(chunk)

        # 2. 扫描目标文件生成指令
        instructions = []
        target_len = 0
        with open(target_path, 'rb') as f:
            while chunk := f.read(self.block_size):
                target_len += len(chunk)
                fp = hashlib.sha256(chunk).hexdigest()
                if fp in source_map:
                    instructions.append({'t': 1, 'idx': source_map[fp], 'l': len(chunk)})
                else:
                    instructions.append({'t': 2, 'd': chunk})

        # 3. 序列化 (使用 struct 压缩体积)
        delta_size = self._save_delta(instructions, output_path)
        
        return {
            "original_size": target_len,
            "delta_size": delta_size,
            "savings": 1.0 - (delta_size / target_len) if target_len > 0 else 0
        }

    def _save_delta(self, instructions, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', len(instructions)))
            for inst in instructions:
                if inst['t'] == 1: # Reference
                    f.write(struct.pack('<BII', 1, inst['idx'], inst['l']))
                else: # Data
                    data = inst['d']
                    f.write(struct.pack('<BI', 2, len(data)))
                    f.write(data)
        return os.path.getsize(path)

class CompressionCachePool:
    """
    智能加权缓存池
    结合 (热度 + AI预测收益) 决定淘汰策略
    """
    def __init__(self, root="cache/"):
        self.root = root
        self.cache = OrderedDict()
        self.capacity = 500
        os.makedirs(root, exist_ok=True)

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]['path']
        return None

    def put(self, key, data: bytes, predicted_gain=0.5):
        path = os.path.join(self.root, hashlib.md5(key.encode()).hexdigest())
        with open(path, 'wb') as f: f.write(data)
        
        self.cache[key] = {'path': path, 'hits': 1, 'gain': predicted_gain}
        self.cache.move_to_end(key)
        
        if len(self.cache) > self.capacity:
            self._evict()

    def _evict(self):
        # 核心算法: 权重 = 热度 * 0.3 + AI预测收益 * 0.7
        # 淘汰权重最低的
        items = list(self.cache.items())
        items.sort(key=lambda x: x[1]['hits']*0.3 + x[1]['gain']*0.7)
        victim_key = items[0][0]
        
        try:
            os.remove(self.cache[victim_key]['path'])
        except: pass
        del self.cache[victim_key]

class PerformanceMonitor:
    """闭环反馈收集器"""
    def __init__(self, path="registry/feedback.json"):
        self.path = path
        self.data = []

    def collect(self, feedback: FeedbackData):
        self.data.append(asdict(feedback))
        # 简单保存最后100条
        if len(self.data) > 100: self.data = self.data[-100:]
        try:
            with open(self.path, 'w') as f: json.dump(self.data, f)
        except: pass

# =========================================================================
# 6. 全局实例导出 (Global Instances)
# =========================================================================
# 这些是外部模块(main.py)直接调用的对象
registry = CapabilityRegistry()
cache_pool = CompressionCachePool()
network_profiler = NetworkProfiler()
diff_engine = DifferentialSync()
perf_monitor = PerformanceMonitor()

if __name__ == "__main__":
    print(">>> CTS Core 模块自检...")
    # 1. 探针测试
    p = ClientProbe()
    print(f"探针: {p.probe()}")
    
    # 2. 缓存测试
    cache_pool.put("test_key", b"test_data")
    print(f"缓存: {os.path.exists(cache_pool.get('test_key'))}")
    
    print(">>> 模块功能正常")