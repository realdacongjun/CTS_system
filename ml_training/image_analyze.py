import os
import json
import math
import shutil
import tarfile
import docker
import time
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ==============================================================================
# 1. 配置区域：你的目标镜像列表
# ==============================================================================
TARGET_IMAGES = [
    'quay.io/centos/centos:stream9', 
    'fedora:latest', 
    'ubuntu:latest',
    'mongo:latest', 
    'mysql:latest', 
    'postgres:latest',
    'rust:latest', 
    'ruby:latest', 
    'python:latest',
    'nginx:latest', 
    'httpd:latest', 
    'rabbitmq:latest', 
    'wordpress:latest', 
    'nextcloud:latest',
    'gradle:latest', 
    'node:latest'
]

# 临时工作目录 (用完会删除)
TEMP_DIR = "temp_feature_extraction"
# 采样大小 (读取每层的前 10MB 进行分析，足以代表整体)
SAMPLE_SIZE_BYTES = 10 * 1024 * 1024 

class FeatureExtractor:
    def __init__(self):
        try:
            self.client = docker.from_env()
            print("✅ Docker Client 连接成功")
        except Exception as e:
            print(f"❌ Docker 连接失败，请确保 docker 服务已启动: {e}")
            exit(1)
        
        # 确保环境干净
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

    def process_all(self, image_list):
        results = []
        print(f"🚀 开始分析 {len(image_list)} 个镜像的物理特征...")
        
        # 使用 tqdm 显示进度条
        for img in tqdm(image_list, desc="Processing Images"):
            try:
                # 1. 确保镜像存在
                self._ensure_image(img)
                # 2. 提取特征
                features = self._analyze_image(img)
                if features:
                    results.append(features)
            except Exception as e:
                print(f"\n❌ 处理 {img} 失败: {e}")
        
        # 3. 保存结果
        self._save_results(results)

    def _ensure_image(self, image_name):
        """如果本地没有，先拉取"""
        try:
            self.client.images.get(image_name)
        except docker.errors.ImageNotFound:
            print(f"\n⬇️ 正在拉取 {image_name} (这可能需要一点时间)...")
            self.client.images.pull(image_name)

    def _analyze_image(self, image_name):
        """核心分析逻辑"""
        clean_name = image_name.replace("/", "_").replace(":", "_")
        extract_path = os.path.join(TEMP_DIR, clean_name)
        os.makedirs(extract_path, exist_ok=True)
        
        try:
            # A. 导出镜像 (docker save)
            img_obj = self.client.images.get(image_name)
            tar_stream = img_obj.save()
            tar_file = os.path.join(extract_path, "image.tar")
            
            with open(tar_file, 'wb') as f:
                for chunk in tar_stream:
                    f.write(chunk)
            
            # B. 解压
            with tarfile.open(tar_file) as tar:
                tar.extractall(path=extract_path)
            os.remove(tar_file) # 省空间

            # C. 解析 Manifest 找 Layers
            manifest_path = os.path.join(extract_path, "manifest.json")
            if not os.path.exists(manifest_path):
                return None
            
            with open(manifest_path) as f:
                manifest = json.load(f)[0]
            
            # D. 逐层分析
            layer_stats = []
            blobs_dir = os.path.join(extract_path, "blobs", "sha256")
            
            for layer_file in manifest.get('Layers', []):
                # 兼容不同的存储路径结构
                layer_path = self._find_layer(extract_path, blobs_dir, layer_file)
                if layer_path:
                    stat = self._compute_file_features(layer_path)
                    layer_stats.append(stat)

            # E. 聚合特征 (Weighted Average)
            return self._aggregate_features(image_name, layer_stats)

        finally:
            # 清理，防止磁盘爆满 (Rust 解压出来很大)
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)

    def _find_layer(self, root, blobs, filename):
        p1 = os.path.join(root, filename)
        if os.path.exists(p1): return p1
        p2 = os.path.join(blobs, os.path.basename(filename))
        if os.path.exists(p2): return p2
        return None

    def _compute_file_features(self, filepath):
        """计算单个文件的物理属性"""
        size = os.path.getsize(filepath)
        if size == 0:
            return {'size': 0, 'entropy': 0, 'text_ratio': 0, 'zero_ratio': 0}

        # 采样读取
        read_len = min(size, SAMPLE_SIZE_BYTES)
        with open(filepath, 'rb') as f:
            data = f.read(read_len)
        
        # 1. 熵 (Entropy)
        entropy = 0
        if data:
            counts = Counter(data)
            total = len(data)
            for count in counts.values():
                p = count / total
                entropy -= p * math.log2(p)
            entropy /= 8.0 # 归一化 0-1

        # 2. 文本率 (Text Ratio)
        text_chars = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
        text_ratio = text_chars / len(data) if data else 0

        # 3. 稀疏率 (Zero Ratio) - 很多二进制文件包含大量空洞
        zero_count = data.count(b'\x00')
        zero_ratio = zero_count / len(data) if data else 0

        return {
            'size': size,
            'entropy': entropy,
            'text_ratio': text_ratio,
            'zero_ratio': zero_ratio
        }

    def _aggregate_features(self, image_name, stats):
        """加权聚合：大层的特征决定整体特征"""
        total_size = sum(s['size'] for s in stats)
        if total_size == 0: return None

        # 加权平均
        avg_entropy = sum(s['entropy'] * s['size'] for s in stats) / total_size
        avg_text = sum(s['text_ratio'] * s['size'] for s in stats) / total_size
        avg_zero = sum(s['zero_ratio'] * s['size'] for s in stats) / total_size

        return {
            "image_name": image_name,
            "total_size_mb": round(total_size / (1024**2), 2),
            "layer_count": len(stats),
            "avg_layer_entropy": round(avg_entropy, 4),
            "text_ratio": round(avg_text, 4),
            "zero_ratio": round(avg_zero, 4)  # 稀疏度
        }

    def _save_results(self, results):
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("📊 特征分析结果预览:")
        print("="*50)
        print(df.to_string())
        
        # 保存为 CSV，方便后续 Dataset 类读取
        csv_path = "image_features_database.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 结果已保存至: {os.path.abspath(csv_path)}")
        print("💡 下一步：在训练代码中加载此文件，使用 'pd.merge' 将其合并到 experiments 记录中。")

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process_all(TARGET_IMAGES)