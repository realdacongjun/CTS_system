"""
镜像特征分析模块
功能：分析Docker镜像的特征，包括层结构、文件类型分布、熵值等
输入：镜像ID或镜像名称
输出：镜像特征字典
"""

import docker
import json
import os
from typing import Dict, Any, List
from collections import Counter
import hashlib
import gzip
import tarfile
from io import BytesIO


class ImageProbe:
    """镜像探针"""
    
    def __init__(self):
        """初始化镜像探针"""
        self.client = docker.from_env()
    
    def analyze_image(self, image_name: str) -> Dict[str, Any]:
        """
        分析镜像特征
        
        Args:
            image_name: 镜像名称
            
        Returns:
            镜像特征字典
        """
        # 拉取镜像（如果本地不存在）
        try:
            image = self.client.images.get(image_name)
        except docker.errors.ImageNotFound:
            print(f"镜像 {image_name} 不存在，正在拉取...")
            self.client.images.pull(image_name)
            image = self.client.images.get(image_name)
        
        # 获取镜像信息
        image_info = image.attrs
        
        # 分析镜像层
        layers_info = self._analyze_layers(image_info)
        
        # 计算镜像整体特征
        features = {
            'image_id': image.id,
            'total_size_mb': image.attrs.get('Size', 0) / (1024 * 1024),
            'layer_count': len(layers_info),
            'layers_info': layers_info,
            'avg_layer_entropy': self._calculate_avg_entropy(layers_info),
            'file_type_distribution': self._calculate_file_type_distribution(layers_info),
            'layer_fingerprints': self._calculate_layer_fingerprints(layers_info),
            'compression_opportunity': self._calculate_compression_opportunity(layers_info),
            'avg_file_size': self._calculate_avg_file_size(layers_info),
            'compression_ratio_estimate': self._calculate_compression_ratio_estimate(layers_info)
        }
        
        return features
    
    def _analyze_layers(self, image_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析镜像各层
        
        Args:
            image_info: 镜像信息
            
        Returns:
            各层信息列表
        """
        layers = []
        
        # 获取层ID列表
        layers_ids = image_info.get('RootFS', {}).get('Layers', [])
        
        for layer_id in layers_ids:
            layer_info = self._analyze_single_layer(layer_id)
            layers.append(layer_info)
        
        return layers
    
    def _analyze_single_layer(self, layer_id: str) -> Dict[str, Any]:
        """
        分析单个镜像层
        
        Args:
            layer_id: 层ID
            
        Returns:
            层信息字典
        """
        # 导出层内容
        try:
            layer_content = self.client.api.get_image(layer_id)
            layer_data = layer_content.data
            
            # 计算层大小
            layer_size = len(layer_data)
            
            # 提取文件信息
            file_info = self._extract_file_info_from_tar(layer_data)
            
            # 计算熵值
            entropy = self._calculate_entropy(layer_data)
            
            # 分析文件类型
            file_types = self._analyze_file_types(file_info)
            
            # 计算层指纹
            layer_fingerprint = self._calculate_layer_fingerprint(layer_data)
            
            return {
                'layer_id': layer_id,
                'size': layer_size,
                'entropy': entropy,
                'file_info': file_info,
                'file_types': file_types,
                'fingerprint': layer_fingerprint
            }
        except Exception as e:
            print(f"分析层 {layer_id} 时出错: {e}")
            return {
                'layer_id': layer_id,
                'size': 0,
                'entropy': 0,
                'file_info': [],
                'file_types': {},
                'fingerprint': ''
            }
    
    def _extract_file_info_from_tar(self, tar_data: bytes) -> List[Dict[str, Any]]:
        """
        从tar数据中提取文件信息
        
        Args:
            tar_data: tar格式的层数据
            
        Returns:
            文件信息列表
        """
        file_info_list = []
        
        try:
            # 使用BytesIO创建文件对象
            tar_file = BytesIO(tar_data)
            
            # 打开tar文件
            with tarfile.open(fileobj=tar_file, mode='r') as tar:
                for member in tar.getmembers():
                    file_info = {
                        'name': member.name,
                        'size': member.size,
                        'type': member.type,
                        'mtime': member.mtime,
                        'is_dir': member.isdir(),
                        'is_file': member.isfile(),
                        'is_link': member.islnk(),
                        'is_symlink': member.issym()
                    }
                    file_info_list.append(file_info)
        except Exception as e:
            print(f"提取tar文件信息时出错: {e}")
        
        return file_info_list
    
    def _calculate_entropy(self, data: bytes) -> float:
        """
        计算数据的熵值
        
        Args:
            data: 二进制数据
            
        Returns:
            熵值（0-1之间）
        """
        if not data:
            return 0
        
        # 计算字节频率
        byte_counts = Counter(data)
        total_bytes = len(data)
        
        # 计算熵值
        entropy = 0
        for count in byte_counts.values():
            probability = count / total_bytes
            entropy -= probability * (probability.bit_length() - 1)
        
        # 归一化到0-1范围
        max_entropy = 8  # 8 bits per byte
        return entropy / max_entropy
    
    def _analyze_file_types(self, file_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析文件类型分布
        
        Args:
            file_info_list: 文件信息列表
            
        Returns:
            文件类型分布字典
        """
        text_extensions = {'.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.js', '.py', '.java', '.cpp', '.c', '.h', '.sh', '.md', '.conf', '.ini', '.cfg', '.log'}
        binary_extensions = {'.exe', '.bin', '.so', '.dll', '.dylib', '.o', '.obj', '.lib', '.a', '.jar', '.war', '.class', '.dat', '.bin'}
        compressed_extensions = {'.gz', '.zip', '.tar', '.bz2', '.xz', '.rar', '.7z', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz'}
        
        text_count = 0
        binary_count = 0
        compressed_count = 0
        total_files = len(file_info_list)
        
        for file_info in file_info_list:
            if not file_info['is_file']:
                continue
            
            filename = file_info['name'].lower()
            ext = os.path.splitext(filename)[1]
            
            if ext in text_extensions:
                text_count += 1
            elif ext in binary_extensions:
                binary_count += 1
            elif ext in compressed_extensions or any(compressed_ext in filename for compressed_ext in ['.tar.gz', '.tar.bz2', '.tar.xz']):
                compressed_count += 1
        
        return {
            'text': text_count / total_files if total_files > 0 else 0,
            'binary': binary_count / total_files if total_files > 0 else 0,
            'compressed': compressed_count / total_files if total_files > 0 else 0
        }
    
    def _calculate_avg_entropy(self, layers_info: List[Dict[str, Any]]) -> float:
        """
        计算平均层熵值
        
        Args:
            layers_info: 各层信息列表
            
        Returns:
            平均熵值
        """
        if not layers_info:
            return 0
        
        total_entropy = sum(layer['entropy'] for layer in layers_info)
        return total_entropy / len(layers_info)
    
    def _calculate_file_type_distribution(self, layers_info: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算文件类型分布
        
        Args:
            layers_info: 各层信息列表
            
        Returns:
            文件类型分布字典
        """
        if not layers_info:
            return {'text': 0, 'binary': 0, 'compressed': 0}
        
        total_text = sum(layer['file_types']['text'] for layer in layers_info)
        total_binary = sum(layer['file_types']['binary'] for layer in layers_info)
        total_compressed = sum(layer['file_types']['compressed'] for layer in layers_info)
        
        layer_count = len(layers_info)
        
        return {
            'text': total_text / layer_count,
            'binary': total_binary / layer_count,
            'compressed': total_compressed / layer_count
        }
    
    def _calculate_layer_fingerprints(self, layers_info: List[Dict[str, Any]]) -> List[str]:
        """
        计算层指纹列表
        
        Args:
            layers_info: 各层信息列表
            
        Returns:
            层指纹列表
        """
        return [layer['fingerprint'] for layer in layers_info]
    
    def _calculate_layer_fingerprint(self, layer_data: bytes) -> str:
        """
        计算层指纹
        
        Args:
            layer_data: 层数据
            
        Returns:
            层指纹（SHA256哈希）
        """
        return hashlib.sha256(layer_data).hexdigest()[:16]
    
    def _calculate_compression_opportunity(self, layers_info: List[Dict[str, Any]]) -> float:
        """
        计算压缩机会
        
        Args:
            layers_info: 各层信息列表
            
        Returns:
            压缩机会评估值
        """
        if not layers_info:
            return 0
        
        # 压缩机会 = 高熵值层的比例 + 已压缩文件的比例
        high_entropy_count = sum(1 for layer in layers_info if layer['entropy'] > 0.8)
        high_entropy_ratio = high_entropy_count / len(layers_info)
        
        # 从文件类型分布中获取已压缩文件比例
        avg_compressed_ratio = sum(layer['file_types']['compressed'] for layer in layers_info) / len(layers_info)
        
        # 综合评估压缩机会
        # 如果高熵值层多，说明已压缩内容多，压缩机会小
        # 如果已压缩文件比例高，说明压缩机会小
        compression_opportunity = 1 - (high_entropy_ratio * 0.7 + avg_compressed_ratio * 0.3)
        
        return max(0, min(1, compression_opportunity))
    
    def _calculate_avg_file_size(self, layers_info: List[Dict[str, Any]]) -> float:
        """
        计算平均文件大小
        
        Args:
            layers_info: 各层信息列表
            
        Returns:
            平均文件大小（字节）
        """
        total_files = 0
        total_size = 0
        
        for layer in layers_info:
            for file_info in layer['file_info']:
                if file_info['is_file']:
                    total_size += file_info['size']
                    total_files += 1
        
        return total_size / total_files if total_files > 0 else 0
    
    def _calculate_compression_ratio_estimate(self, layers_info: List[Dict[str, Any]]) -> float:
        """
        估算压缩比
        
        Args:
            layers_info: 各层信息列表
            
        Returns:
            估算压缩比
        """
        if not layers_info:
            return 0.5  # 默认压缩比
        
        # 基于熵值估算压缩比
        avg_entropy = self._calculate_avg_entropy(layers_info)
        
        # 熵值越低，压缩比越高
        # 使用简单的线性模型：压缩比 = 1 - (熵值 * 0.7)
        estimated_compression_ratio = 1 - (avg_entropy * 0.7)
        
        # 确保压缩比在合理范围内
        return max(0.1, min(0.9, estimated_compression_ratio))


def main():
    """主函数，用于测试镜像分析功能"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python image_probe.py <镜像名称>")
        return
    
    image_name = sys.argv[1]
    
    print(f"开始分析镜像: {image_name}")
    
    probe = ImageProbe()
    features = probe.analyze_image(image_name)
    
    print(f"镜像特征分析结果:")
    print(f"  镜像ID: {features['image_id']}")
    print(f"  总大小: {features['total_size_mb']:.2f} MB")
    print(f"  层数: {features['layer_count']}")
    print(f"  平均熵值: {features['avg_layer_entropy']:.3f}")
    print(f"  文件类型分布: {features['file_type_distribution']}")
    print(f"  压缩机会: {features['compression_opportunity']:.3f}")
    print(f"  平均文件大小: {features['avg_file_size']:.2f} bytes")
    print(f"  估算压缩比: {features['compression_ratio_estimate']:.3f}")
    
    print(f"\n各层详细信息:")
    for i, layer in enumerate(features['layers_info']):
        print(f"  层 {i+1}:")
        print(f"    ID: {layer['layer_id'][:12]}...")
        print(f"    大小: {layer['size'] / (1024*1024):.2f} MB")
        print(f"    熵值: {layer['entropy']:.3f}")
        print(f"    文件类型: {layer['file_types']}")
        print(f"    指纹: {layer['fingerprint']}")


if __name__ == "__main__":
    main()