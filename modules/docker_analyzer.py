"""
Docker镜像分析器
用于分析Docker镜像的特征，为压缩决策提供依据
"""

import docker
import tarfile
import os
import json
import hashlib
import math
from collections import defaultdict
import random
from typing import List, Dict, Any


class DockerAnalyzer:
    """Docker镜像分析器"""
# 构造函数
    def __init__(self, work_dir: str = "temp/image"):
        """
        初始化Docker镜像分析器
        
        Args:
            work_dir: 工作目录
        """
        self.work_dir = work_dir
        self.client = None
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Docker客户端初始化失败: {e}")
    
    def pull_image(self, image_name: str):
        """
        拉取Docker镜像
        
        Args:
            image_name: 镜像名称
        """
        if not self.client:
            raise Exception("Docker客户端未初始化")
        
        try:
            print(f"正在拉取镜像: {image_name}")
            self.client.images.pull(image_name)
            print(f"镜像拉取完成: {image_name}")
        except Exception as e:
            print(f"镜像拉取失败: {e}")
            raise
    
    def export_image(self, image_name: str, export_path: str):
        """
        导出Docker镜像为tar文件
        
        Args:
            image_name: 镜像名称
            export_path: 导出路径
        """
        if not self.client:
            raise Exception("Docker客户端未初始化")
        
        try:
            print(f"正在导出镜像: {image_name}")
            image = self.client.images.get(image_name)
            
            with open(export_path, 'wb') as f:
                for chunk in image.save():
                    f.write(chunk)
            
            print(f"镜像导出完成: {export_path}")
        except Exception as e:
            print(f"镜像导出失败: {e}")
            raise
    
    def extract_image_tar(self, tar_path, extract_path):
        """
        解压镜像tar文件
        
        Args:
            tar_path: tar文件路径
            extract_path: 解压路径
        """
        print(f"正在解压镜像tar文件: {tar_path}")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_path)
            print(f"镜像解压完成: {extract_path}")
        except Exception as e:
            print(f"镜像解压失败: {e}")
            raise
    
    def parse_manifest(self, extract_path):
        """
        解析镜像manifest文件
        
        Args:
            extract_path: 解压路径
            
        Returns:
            manifest: manifest数据
        """
        manifest_path = os.path.join(extract_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise Exception(f"找不到manifest文件: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        return manifest[0] if isinstance(manifest, list) else manifest
    
    def analyze_layers(self, extract_path: str, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析镜像的所有层
        
        Args:
            extract_path: 解压路径
            manifest: manifest数据
            
        Returns:
            layers_info: 层信息列表
        """
        layers_info = []
        layers = manifest.get("Layers", [])
        
        print(f"发现 {len(layers)} 个镜像层")
        
        # OCI格式的层文件在blobs/sha256目录下
        blobs_path = os.path.join(extract_path, "blobs", "sha256")
        
        for i, layer_path in enumerate(layers):
            print(f"正在分析第 {i+1} 层: {layer_path}")
            
            # 对于OCI格式，layer_path是相对路径，我们需要找到实际文件
            layer_filename = os.path.basename(layer_path)
            layer_full_path = os.path.join(blobs_path, layer_filename)
            
            if not os.path.exists(layer_full_path):
                print(f"警告: 层文件不存在: {layer_full_path}")
                continue
            
            # 计算层摘要
            layer_digest = self._calculate_digest(layer_full_path)
            
            # 分析层内容
            layer_content_info = self._analyze_layer_content(blobs_path, layer_filename)
            
            layer_info = {
                "layer_path": layer_path,
                "layer_size": os.path.getsize(layer_full_path),
                "layer_digest": layer_digest,
                "layer_index": i
            }
            
            layer_info.update(layer_content_info)
            layers_info.append(layer_info)
        
        return layers_info
    
    def _analyze_layer_content(self, base_path: str, layer_filename: str) -> Dict[str, Any]:
        """
        分析层内容
        
        Args:
            base_path: 基础路径 (blobs/sha256目录)
            layer_filename: 层文件名
            
        Returns:
            layer_info: 层信息
        """
        layer_info = {
            "file_count": 0,
            "total_size": 0,
            "text_files": 0,
            "binary_files": 0,
            "compressed_files": 0,
            "avg_entropy": 0
        }
        
        layer_full_path = os.path.join(base_path, layer_filename)
        if not os.path.exists(layer_full_path):
            return layer_info
        
        try:
            entropy_sum = 0
            sample_count = 0
            file_count = 0
            
            print(f"  打开层文件: {layer_filename}")
            with tarfile.open(layer_full_path, 'r') as tar:
                members = tar.getmembers()
                print(f"  层中成员数量: {len(members)}")
                
                for member in members:
                    if not member.isfile():
                        continue
                    
                    file_count += 1
                    layer_info["total_size"] += member.size
                    
                    # 判断文件类型
                    filename = member.name
                    if self._is_text_file(filename):
                        layer_info["text_files"] += 1
                    elif self._is_compressed_file(filename):
                        layer_info["compressed_files"] += 1
                    else:
                        layer_info["binary_files"] += 1
                    
                    # 采样计算熵值（仅对较小的文件）
                    # 不管文件大小都尝试采样
                    if file_count <= 20:  # 限制采样数量
                        try:
                            f = tar.extractfile(member)
                            if f:
                                # 读取文件内容用于熵值计算
                                # 对大文件只读取前1KB，小文件全部读取
                                if member.size > 1024:
                                    data = f.read(1024)  # 读取前1KB
                                else:
                                    data = f.read()  # 读取全部内容
                                
                                if data and len(data) > 0:
                                    entropy = self._calculate_entropy(data)
                                    entropy_sum += entropy
                                    sample_count += 1
                                    print(f"    分析文件 {filename}, 大小: {member.size}, 熵值: {entropy}")
                                f.close()
                        except Exception as e:
                            print(f"    读取文件 {member.name} 时出错: {e}")
                            # 忽略单个文件读取错误
            
            layer_info["file_count"] = file_count
            print(f"  总文件数: {file_count}, 采样数: {sample_count}")
            
            layer_info["avg_entropy"] = entropy_sum / sample_count if sample_count > 0 else 0.0
            
            # 如果没有任何文件被分析，可能是权限或其他问题
            if file_count == 0:
                print(f"警告: 未能分析层 {layer_filename} 中的任何文件")
            
        except Exception as e:
            print(f"层内容分析失败: {e}")
            import traceback
            traceback.print_exc()
            layer_info.update({
                "file_count": 0,
                "total_size": 0,
                "text_files": 0,
                "binary_files": 0,
                "compressed_files": 0,
                "avg_entropy": 0.0
            })
        
        return layer_info
    
    def _calculate_digest(self, file_path):
        """
        计算文件SHA256摘要
        
        Args:
            file_path: 文件路径
            
        Returns:
            digest: SHA256摘要
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _is_text_file(self, filename):
        """判断是否为文本文件"""
        text_extensions = {'.txt', '.json', '.xml', '.yaml', '.yml', '.conf', '.cfg', '.ini', '.md', '.log'}
        _, ext = os.path.splitext(filename.lower())
        return ext in text_extensions
    
    def _is_compressed_file(self, filename):
        """判断是否为已压缩文件"""
        compressed_extensions = {'.gz', '.bz2', '.xz', '.zip', '.rar', '.7z', '.tar', '.tgz', '.tbz2'}
        _, ext = os.path.splitext(filename.lower())
        return ext in compressed_extensions
    
    def _calculate_entropy(self, data):
        """
        计算数据熵值
        
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
            if count > 0:  # 确保count大于0
                probability = count / data_len
                if probability > 0:
                    entropy -= probability * math.log2(probability)
        
        # 归一化到0-1范围
        return entropy / 8.0 if entropy > 0 else 0
    
    def get_image_profile(self, image_name):
        """
        获取完整的镜像分析结果
        
        Args:
            image_name: 镜像名称
            
        Returns:
            profile: 镜像分析结果
        """
        if not self.client:
            raise Exception("Docker客户端未初始化")
        
        # 使用项目目录下的image文件夹
        image_work_dir = os.path.join(self.work_dir, image_name.replace("/", "_").replace(":", "_"))
        if not os.path.exists(image_work_dir):
            os.makedirs(image_work_dir)
        
        try:
            # 1. 拉取镜像
            self.pull_image(image_name)
            
            # 2. 导出镜像
            export_path = os.path.join(image_work_dir, "image.tar")
            self.export_image(image_name, export_path)
            
            # 3. 解压镜像
            extract_path = os.path.join(image_work_dir, "extracted")
            os.makedirs(extract_path, exist_ok=True)
            self.extract_image_tar(export_path, extract_path)
            
            # 4. 解析manifest
            manifest = self.parse_manifest(extract_path)
            
            # 5. 分析层
            layers_info = self.analyze_layers(extract_path, manifest)
            
            # 6. 生成整体分析结果
            total_size = sum(layer["layer_size"] for layer in layers_info)
            avg_entropy = sum(layer["avg_entropy"] for layer in layers_info) / len(layers_info) if layers_info else 0.0
            
            # 计算文件类型分布
            total_files = sum(layer["file_count"] for layer in layers_info)
            text_files = sum(layer["text_files"] for layer in layers_info)
            binary_files = sum(layer["binary_files"] for layer in layers_info)
            compressed_files = sum(layer["compressed_files"] for layer in layers_info)
            
            text_ratio = text_files / total_files if total_files > 0 else 0
            binary_ratio = binary_files / total_files if total_files > 0 else 0
            compressed_ratio = compressed_files / total_files if total_files > 0 else 0
            
            profile = {
                "layer_count": len(layers_info),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "avg_layer_entropy": round(avg_entropy, 4),
                "text_ratio": round(text_ratio, 4),
                "binary_ratio": round(binary_ratio, 4),
                "compressed_file_ratio": round(compressed_ratio, 4),
                "layers": layers_info
            }
            
            return profile
        finally:
            # 清理临时文件（可选）
            # 如果需要保留文件用于调试，可以注释掉以下代码
            # import shutil
            # shutil.rmtree(image_work_dir, ignore_errors=True)
            pass


def main():
    """测试Docker镜像分析器"""
    analyzer = DockerAnalyzer()
    
    # 测试分析nginx镜像
    try:
        image_name = "nginx:latest"
        print(f"开始分析镜像: {image_name}")
        
        profile = analyzer.get_image_profile(image_name)
        
        print("\n镜像分析结果:")
        print(f"  层数量: {profile['layer_count']}")
        print(f"  总大小: {profile['total_size_mb']} MB")
        print(f"  平均熵值: {profile['avg_layer_entropy']}")
        print(f"  文本文件比例: {profile['text_ratio']}")
        print(f"  二进制文件比例: {profile['binary_ratio']}")
        print(f"  已压缩文件比例: {profile['compressed_file_ratio']}")
        
        print("\n各层详情:")
        for layer in profile['layers']:
            print(f"  层 {layer['layer_index']}:")
            print(f"    大小: {round(layer['layer_size'] / 1024, 2)} KB")
            print(f"    摘要: {layer['layer_digest'][:12]}...")
            print(f"    文件数: {layer['file_count']}")
            print(f"    平均熵值: {layer['avg_entropy']}")
            
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()