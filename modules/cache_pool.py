"""
多格式缓存池（Compression Cache Pool）
目的：存储同一镜像的多种压缩格式，提高响应速度

目录结构示例：
cache/
  ubuntu:latest/
    gzip-6/
    zstd-3/
    zstd-5/
    lz4-fast/
"""


import os
import shutil
from collections import OrderedDict


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, capacity=100):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # 移动到末尾表示最近使用
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            # 更新现有键值
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # 删除最久未使用的项
            self.cache.popitem(last=False)
        self.cache[key] = value


class CompressionCachePool:
    """压缩缓存池"""
    
    def __init__(self, cache_root="cache"):
        self.cache_root = cache_root
        self.lru_cache = LRUCache(1000)  # 内存中的LRU缓存
        # 确保缓存根目录存在
        if not os.path.exists(cache_root):
            os.makedirs(cache_root)
    
    def get_cache_path(self, image_name, compression_method):
        """
        获取镜像特定压缩格式的缓存路径
        
        Args:
            image_name: 镜像名称
            compression_method: 压缩方法
            
        Returns:
            path: 缓存路径
        """
        # 清理镜像名称中的特殊字符
        clean_image_name = image_name.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_root, clean_image_name, compression_method)
    
    def is_cached(self, image_name, compression_method):
        """
        检查特定压缩格式是否已缓存
        
        Args:
            image_name: 镜像名称
            compression_method: 压缩方法
            
        Returns:
            bool: 是否已缓存
        """
        cache_key = f"{image_name}:{compression_method}"
        # 先检查内存LRU缓存
        if self.lru_cache.get(cache_key) is not None:
            return True
            
        # 检查磁盘缓存
        cache_path = self.get_cache_path(image_name, compression_method)
        exists = os.path.exists(cache_path)
        
        if exists:
            # 添加到内存LRU缓存
            self.lru_cache.put(cache_key, True)
            
        return exists
    
    def get_cached_data(self, image_name, compression_method):
        """
        获取缓存的数据
        
        Args:
            image_name: 镜像名称
            compression_method: 压缩方法
            
        Returns:
            data: 缓存数据路径，如果不存在返回None
        """
        if not self.is_cached(image_name, compression_method):
            return None
            
        cache_path = self.get_cache_path(image_name, compression_method)
        if os.path.exists(cache_path):
            # 更新LRU缓存
            cache_key = f"{image_name}:{compression_method}"
            self.lru_cache.put(cache_key, True)
            return cache_path
            
        return None
    
    def cache_data(self, image_name, compression_method, data_path):
        """
        缓存数据
        
        Args:
            image_name: 镜像名称
            compression_method: 压缩方法
            data_path: 数据路径
        """
        cache_path = self.get_cache_path(image_name, compression_method)
        
        # 确保目录存在
        cache_dir = os.path.dirname(cache_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 复制数据到缓存位置
        if os.path.exists(data_path):
            if os.path.isdir(data_path):
                shutil.copytree(data_path, cache_path)
            else:
                shutil.copy2(data_path, cache_path)
            
            # 更新LRU缓存
            cache_key = f"{image_name}:{compression_method}"
            self.lru_cache.put(cache_key, True)
    
    def invalidate_cache(self, image_name=None, compression_method=None):
        """
        使缓存失效
        
        Args:
            image_name: 镜像名称，如果为None则清除所有
            compression_method: 压缩方法，如果为None则清除该镜像的所有格式
        """
        if image_name is None:
            # 清除所有缓存
            if os.path.exists(self.cache_root):
                shutil.rmtree(self.cache_root)
                os.makedirs(self.cache_root)
            self.lru_cache = LRUCache(1000)
            return
        
        if compression_method is None:
            # 清除特定镜像的所有缓存
            clean_image_name = image_name.replace("/", "_").replace(":", "_")
            image_cache_dir = os.path.join(self.cache_root, clean_image_name)
            if os.path.exists(image_cache_dir):
                shutil.rmtree(image_cache_dir)
        else:
            # 清除特定镜存
            cache_path = self.get_cache_path(image_name, compression_method)
            if os.path.exists(cache_path):
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                else:
                    os.remove(cache_path)
        
        # 清理LRU缓存中相关的条目
        keys_to_remove = []
        for key in self.lru_cache.cache.keys():
            if key.startswith(f"{image_name}:"):
                if compression_method is None or key.endswith(f":{compression_method}"):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.lru_cache.cache.pop(key, None)


def main():
    """测试缓存池"""
    cache_pool = CompressionCachePool()
    
    # 测试缓存功能
    image_name = "ubuntu:latest"
    compression_method = "zstd-5"
    
    print(f"检查 {image_name} 的 {compression_method} 是否已缓存: {cache_pool.is_cached(image_name, compression_method)}")
    
    # 模拟缓存数据
    print("缓存数据...")
    cache_pool.cache_data(image_name, compression_method, "temp/image_extract")
    
    print(f"再次检查是否已缓存: {cache_pool.is_cached(image_name, compression_method)}")
    
    cached_path = cache_pool.get_cached_data(image_name, compression_method)
    print(f"获取缓存数据路径: {cached_path}")


if __name__ == "__main__":
    main()