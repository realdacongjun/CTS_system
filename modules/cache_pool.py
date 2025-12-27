"""
压缩缓存池（Compression Cache Pool）
目的：缓存已压缩的镜像层，避免重复压缩
"""


import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from collections import OrderedDict

"""
压缩缓存池（Compression Cache Pool）
目的：存储同一镜像的多种压缩格式，提高响应速度

目录结构示例：
cache/
  ubuntu:latest/
    gzip-6/
    zstd-3/
    zstd-5/
    lz4-fast/
"""


class LRUCache:
    """LRU缓存实现，增加热度分值和预测权重"""
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str):
        if key not in self.cache:
            return None
        # 更新使用时间戳
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            # 更新热度分值
            current_value = self.cache[key]
            current_value['data'] = value['data']
            current_value['last_access'] = time.time()
            current_value['hotness_score'] = current_value.get('hotness_score', 0) + 1
        else:
            # 新增缓存项，初始热度为1
            self.cache[key] = {
                'data': value['data'],
                'last_access': time.time(),
                'hotness_score': 1,
                'access_count': 1,
                'predicted_value': value.get('predicted_value', 0.5),  # 预测价值分值
                'compression_time_saved': value.get('compression_time_saved', 0),  # 预估节省的压缩时间
                'transfer_time_saved': value.get('transfer_time_saved', 0)  # 预估节省的传输时间
            }
        
        self.cache.move_to_end(key)
        
        if len(self.cache) > self.capacity:
            # 根据热度、预测价值和节省时间的综合权重排序，移除最不重要的项
            # 权重计算：cache_weight = (hotness_score + predicted_value + time_saved_factor) / 3
            def calculate_cache_weight(item):
                hotness = item[1]['hotness_score']
                predicted = item[1]['predicted_value']
                time_saved_factor = (item[1]['compression_time_saved'] + item[1]['transfer_time_saved']) / 2
                return (hotness + predicted + time_saved_factor) / 3
            
            sorted_items = sorted(self.cache.items(), key=calculate_cache_weight)
            lru_key = sorted_items[0][0]
            del self.cache[lru_key]


class CompressionCachePool:
    """压缩缓存池"""
    
    def __init__(self, cache_root: str = "cache/"):
        self.cache_root = cache_root
        self.lru_cache = LRUCache(1000)  # LRU缓存，容量1000
        os.makedirs(self.cache_root, exist_ok=True)
    
    def cache_layer_data(self, image_name: str, layer_digest: str, compression_method: str, layer_data: bytes, 
                        predicted_value: float = 0.5, compression_time_saved: float = 0.0, transfer_time_saved: float = 0.0):
        """
        缓存压缩后的层数据
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            layer_data: 层数据
            predicted_value: 预测价值分值（来自访问预测器）
            compression_time_saved: 预估节省的压缩时间
            transfer_time_saved: 预估节省的传输时间
        """
        # 清理镜像名称以适合作为文件名
        clean_image_name = image_name.replace("/", "_").replace(":", "_")
        
        # 创建缓存目录
        layer_cache_dir = os.path.join(
            self.cache_root,
            clean_image_name,
            "layers",
            layer_digest,
            compression_method
        )
        os.makedirs(layer_cache_dir, exist_ok=True)
        
        # 保存压缩数据
        cache_path = os.path.join(layer_cache_dir, "data")
        with open(cache_path, "wb") as f:
            f.write(layer_data)
        
        # 更新LRU缓存
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        self.lru_cache.put(cache_key, {
            'data': layer_data,
            'predicted_value': predicted_value,
            'compression_time_saved': compression_time_saved,
            'transfer_time_saved': transfer_time_saved
        })
    
    def get_cached_layer_data(self, image_name: str, layer_digest: str, compression_method: str) -> Optional[bytes]:
        """
        获取缓存的压缩层数据
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            
        Returns:
            层数据或None
        """
        # 检查LRU缓存
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        cached_item = self.lru_cache.get(cache_key)
        if cached_item:
            # 更新热度分值
            cached_item['hotness_score'] = cached_item.get('hotness_score', 0) + 1
            cached_item['access_count'] = cached_item.get('access_count', 0) + 1
            cached_item['last_access'] = time.time()
            return cached_item['data']
        
        # 从文件系统加载
        clean_image_name = image_name.replace("/", "_").replace(":", "_")
        cache_path = os.path.join(
            self.cache_root,
            clean_image_name,
            "layers",
            layer_digest,
            compression_method,
            "data"
        )
        
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = f.read()
            
            # 添加到LRU缓存
            self.lru_cache.put(cache_key, {
                'data': data,
                'predicted_value': 0.5  # 默认预测价值
            })
            
            return data
        
        return None
    
    def is_layer_cached(self, image_name: str, layer_digest: str, compression_method: str) -> bool:
        """
        检查层是否已缓存
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            
        Returns:
            是否已缓存
        """
        return self.get_cached_layer_data(image_name, layer_digest, compression_method) is not None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        total_size = 0
        total_layers = 0
        
        for root, dirs, files in os.walk(self.cache_root):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
            total_layers += len(dirs)
        
        return {
            "total_size_bytes": total_size,
            "total_layers": total_layers,
            "cached_items_count": len(self.lru_cache.cache),
            "cache_path": self.cache_root
        }
    
    def update_hotness_score(self, image_name: str, layer_digest: str, compression_method: str):
        """
        手动更新缓存项的热度分值
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
        """
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        cached_item = self.lru_cache.get(cache_key)
        if cached_item:
            cached_item['hotness_score'] = cached_item.get('hotness_score', 0) + 1
            cached_item['access_count'] = cached_item.get('access_count', 0) + 1
            cached_item['last_access'] = time.time()
    
    def get_hotness_score(self, image_name: str, layer_digest: str, compression_method: str) -> int:
        """
        获取缓存项的热度分值
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            
        Returns:
            热度分值
        """
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        cached_item = self.lru_cache.get(cache_key)
        if cached_item:
            return cached_item.get('hotness_score', 0)
        return 0
    
    def get_predicted_value(self, image_name: str, layer_digest: str, compression_method: str) -> float:
        """
        获取缓存项的预测价值分值
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            
        Returns:
            预测价值分值
        """
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        cached_item = self.lru_cache.get(cache_key)
        if cached_item:
            return cached_item.get('predicted_value', 0.5)
        return 0.5
    
    def set_predicted_value(self, image_name: str, layer_digest: str, compression_method: str, predicted_value: float):
        """
        设置缓存项的预测价值分值
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            predicted_value: 预测价值分值
        """
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        cached_item = self.lru_cache.get(cache_key)
        if cached_item:
            cached_item['predicted_value'] = predicted_value
    
    def get_cache_weight(self, image_name: str, layer_digest: str, compression_method: str) -> float:
        """
        获取缓存项的综合权重
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            
        Returns:
            综合权重值
        """
        cache_key = f"{image_name}:{layer_digest}:{compression_method}"
        cached_item = self.lru_cache.get(cache_key)
        if cached_item:
            hotness = cached_item.get('hotness_score', 0)
            predicted = cached_item.get('predicted_value', 0.5)
            time_saved_factor = (cached_item.get('compression_time_saved', 0) + 
                                cached_item.get('transfer_time_saved', 0)) / 2
            return (hotness + predicted + time_saved_factor) / 3
        return 0.0


# 简化别名
CachePool = CompressionCachePool