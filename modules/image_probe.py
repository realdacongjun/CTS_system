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
import math
from collections import defaultdict


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
        # TODO: 实现layer分析逻辑
        # 流式解压tar包，分析文件类型和可压缩属性
        pass
    
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
    
    def analyze_file_types(self):
        """
        分析文件类型分布
        统计文本/二进制/已压缩文件比例
        """
        # TODO: 实现文件类型分析逻辑
        # 根据文件扩展名或内容特征判断文件类型
        pass
    
    def get_image_profile(self):
        """
        获取完整的镜像配置文件
        """
        # 解析镜像结构
        self.parse_image_structure()
        
        # TODO: 实际分析各层数据
        # 目前返回模拟数据
        
        return {
            "layer_count": len(self.layers),
            "total_size_mb": 742,  # 模拟数据
            "avg_layer_entropy": 0.78,
            "text_ratio": 0.18,
            "binary_ratio": 0.82,
            "compressed_file_ratio": 0.12,
            "avg_file_size_kb": 42.3,
            "median_file_size_kb": 8.1,
            "small_file_ratio": 0.67,
            "big_file_ratio": 0.15,
            "predict_popularity": 0.54
        }


def main():
    """测试镜像探针"""
    # 假设镜像数据在temp/image_extract目录下
    probe = ImageProbe("temp/image_extract")
    profile = probe.get_image_profile()
    print("镜像探针结果:")
    for key, value in profile.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()