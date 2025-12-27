"""
镜像端探针（Image Probe）
目的：给镜像做"体检"，找出其压缩偏好，供决策器使用

基本步骤：
1. 解析镜像结构
2. 遍历每个layer（tar流分析，不落盘）
3. 计算关键压缩偏好指标
4. 生成镜像画像JSON
"""

import json
import os
import tarfile
import math
from collections import defaultdict
from typing import Dict, Any, List


class ImageProbe:
    """镜像探针类"""
    
    def __init__(self, image_path):
        """
        初始化镜像探针
        
        Args:
            image_path: 镜像目录路径（包含manifest.json等文件）
        """
        self.image_path = image_path
        self.manifest = None
        self.layers = []
    
    def parse_image_structure(self):
        """
        解析镜像结构
        读取manifest.json获取layer列表
        """
        manifest_path = os.path.join(self.image_path, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
                # 获取layers信息
                if isinstance(self.manifest, list) and len(self.manifest) > 0:
                    self.layers = self.manifest[0].get("Layers", [])
                else:
                    self.layers = self.manifest.get("Layers", [])
        
        print(f"发现 {len(self.layers)} 个镜像层")
        return self.layers
    
    def analyze_layers(self):
        """
        遍历每个layer进行分析
        """
        layer_profiles = []
        
        for layer_path in self.layers:
            full_layer_path = os.path.join(self.image_path, layer_path)
            if os.path.exists(full_layer_path):
                layer_profile = self._analyze_single_layer(full_layer_path)
                layer_profiles.append(layer_profile)
        
        return layer_profiles
    
    def _analyze_single_layer(self, layer_path):
        """
        分析单个layer
        
        Args:
            layer_path: layer文件路径
            
        Returns:
            layer_profile: 包含该层分析结果的字典
        """
        layer_profile = {
            "layer_path": layer_path,
            "size": os.path.getsize(layer_path),
            "avg_entropy": 0,
            "file_type_distribution": {},
            "text_ratio": 0,
            "binary_ratio": 0,
            "compressed_ratio": 0
        }
        
        try:
            # 尝试作为tar文件处理
            with tarfile.open(layer_path, 'r') as tar:
                entropy_sum = 0
                file_count = 0
                file_type_counts = defaultdict(int)
                total_size = 0
                text_size = 0
                binary_size = 0
                
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            content = f.read()
                            if content:
                                # 计算文件熵
                                entropy = self.calculate_entropy(content)
                                entropy_sum += entropy
                                file_count += 1
                                
                                # 分析文件类型
                                file_type = self._detect_file_type(content, member.name)
                                file_type_counts[file_type] += 1
                                
                                # 统计文本/二进制大小
                                if file_type == "text":
                                    text_size += len(content)
                                else:
                                    binary_size += len(content)
                                
                                total_size += len(content)
                
                if file_count > 0:
                    layer_profile["avg_entropy"] = entropy_sum / file_count
                layer_profile["file_type_distribution"] = dict(file_type_counts)
                
                if total_size > 0:
                    layer_profile["text_ratio"] = text_size / total_size
                    layer_profile["binary_ratio"] = binary_size / total_size
        except tarfile.ReadError:
            # 如果不是tar文件，直接分析整个文件
            with open(layer_path, 'rb') as f:
                content = f.read()
                if content:
                    # 计算整个文件的熵
                    entropy = self.calculate_entropy(content)
                    layer_profile["avg_entropy"] = entropy
                    
                    # 检测文件类型
                    file_type = self._detect_file_type(content, layer_path)
                    layer_profile["file_type_distribution"] = {file_type: 1}
                    
                    # 设置文本/二进制比例
                    if file_type == "text":
                        layer_profile["text_ratio"] = 1.0
                        layer_profile["binary_ratio"] = 0.0
                    else:
                        layer_profile["text_ratio"] = 0.0
                        layer_profile["binary_ratio"] = 1.0
        
        return layer_profile
    
    def calculate_entropy(self, data):
        """
        计算数据的熵值（压缩友好度）
        
        Args:
            data: 字节数据
            
        Returns:
            entropy: 熵值 (0-1)
        """
        if not data:
            return 0
            
        # 统计每个字节值的出现频率
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        # 计算熵值
        entropy = 0
        data_len = len(data)
        for count in freq.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # 归一化到0-1范围 (理论上最大熵是8位字节的最大熵log2(256)=8)
        return entropy / 8.0 if entropy > 0 else 0
    
    def _detect_file_type(self, content, filename):
        """
        检测文件类型
        
        Args:
            content: 文件内容
            filename: 文件名
            
        Returns:
            file_type: 文件类型
        """
        # 根据文件扩展名判断
        if filename.endswith(('.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.js', '.py', '.java', '.cpp', '.c', '.h')):
            return "text"
        
        # 检查是否为文本内容（基于可打印字符比例）
        try:
            text_content = content.decode('utf-8', errors='ignore')
            printable_chars = sum(1 for c in text_content if c.isprintable() or c.isspace())
            printable_ratio = printable_chars / len(text_content) if text_content else 0
            
            if printable_ratio > 0.9:
                return "text"
            else:
                return "binary"
        except:
            return "binary"
    
    def get_image_profile(self):
        """
        获取完整的镜像配置文件
        """
        # 解析镜像结构
        self.parse_image_structure()
        
        # 分析各层
        layer_profiles = self.analyze_layers()
        
        # 汇总镜像整体特征
        total_size = sum(layer["size"] for layer in layer_profiles)
        avg_entropy = sum(layer["avg_entropy"] for layer in layer_profiles) / len(layer_profiles) if layer_profiles else 0
        total_text_size = sum(
            layer["size"] * layer["text_ratio"] for layer in layer_profiles
        )
        total_binary_size = sum(
            layer["size"] * layer["binary_ratio"] for layer in layer_profiles
        )
        text_ratio = total_text_size / total_size if total_size > 0 else 0
        binary_ratio = total_binary_size / total_size if total_size > 0 else 0
        
        # 合并所有文件类型分布
        file_type_dist = defaultdict(int)
        for layer in layer_profiles:
            for file_type, count in layer["file_type_distribution"].items():
                file_type_dist[file_type] += count
        
        total_files = sum(file_type_dist.values())
        file_type_distribution = {
            file_type: count / total_files if total_files > 0 else 0
            for file_type, count in file_type_dist.items()
        }
        
        image_profile = {
            "total_size_mb": total_size / (1024 * 1024),
            "avg_layer_entropy": avg_entropy,
            "text_ratio": text_ratio,
            "binary_ratio": binary_ratio,
            "file_type_distribution": file_type_distribution,
            "layer_count": len(layer_profiles),
            "layers": layer_profiles
        }
        
        return image_profile