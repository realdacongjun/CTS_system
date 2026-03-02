#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
镜像准备脚本 - 对齐你的具体配置
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "preprocessed_images"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# ==========================================
# 🔧 直接使用你提供的压缩方法配置
# ==========================================
COMPRESSION_METHODS = {
    'gzip-1':     ["gzip", "-1", "-c"],
    'gzip-6':     ["gzip", "-6", "-c"],
    'gzip-9':     ["gzip", "-9", "-c"],
    
    'zstd-1':     ["zstd", "-1", "--force", "-c"],
    'zstd-3':     ["zstd", "-3", "--force", "-c"],
    'zstd-6':     ["zstd", "-6", "--force", "-c"],
    
    'lz4-fast':   ["lz4", "-1", "-c"],
    'lz4-medium': ["lz4", "-3", "-c"],
    'lz4-slow':   ["lz4", "-9", "-c"],
    
    'brotli-1':   ["brotli", "-1", "-c"]
}

# ==========================================
# 🔧 直接使用你提供的镜像列表
# ==========================================
TARGET_IMAGES = [
    "ubuntu:latest",
    "nginx:latest",
    "mysql:latest"
]

def pull_images(image_list: List[str]) -> Tuple[List[str], List[str]]:
    success, failed = [], []
    for image in image_list:
        logger.info(f"[1/4] 正在拉取镜像: {image}")
        try:
            # 加上 quiet 减少日志刷屏
            result = subprocess.run(
                ["docker", "pull", "-q", image],
                capture_output=True, text=True, timeout=1200 # mysql 比较大，给够时间
            )
            if result.returncode == 0:
                logger.info(f"✅ 拉取成功: {image}")
                success.append(image)
            else:
                logger.error(f"❌ 拉取失败: {image}")
                failed.append(image)
        except Exception as e:
            logger.error(f"❌ 拉取异常: {e}")
            failed.append(image)
    return success, failed

def save_and_compress_image(image_name: str) -> bool:
    safe_name = image_name.replace(":", "_").replace("/", "_")
    tar_path = CACHE_DIR / f"{safe_name}.tar"
    
    # Step 1: Docker Save
    if not tar_path.exists():
        logger.info(f"[2/4] 导出镜像: docker save {image_name} -> {tar_path.name}")
        try:
            with open(tar_path, 'wb') as f:
                subprocess.run(
                    ["docker", "save", image_name],
                    stdout=f, check=True, timeout=1200
                )
        except Exception as e:
            logger.error(f"❌ Save 失败: {e}")
            return False

    # Step 2: 遍历压缩 (使用你的字典)
    all_good = True
    for algo_name, cmd_base in COMPRESSION_METHODS.items():
        comp_path = CACHE_DIR / f"{safe_name}.tar.{algo_name}"
        if comp_path.exists():
            size_mb = comp_path.stat().st_size / 1024 / 1024
            logger.info(f"[3/4] 跳过 [{algo_name}] (已存在, {size_mb:.2f}MB)")
            continue
            
        logger.info(f"[3/4] 正在压缩 [{algo_name}]...")
        try:
            # 确保命令以 -c 结尾 (输出到 stdout)
            cmd = cmd_base if cmd_base[-1] == '-c' else cmd_base + ['-c']
            
            with open(tar_path, 'rb') as f_in:
                with open(comp_path, 'wb') as f_out:
                    subprocess.run(cmd, stdin=f_in, stdout=f_out, check=True, timeout=1200)
            
            size_mb = comp_path.stat().st_size / 1024 / 1024
            logger.info(f"    ✅ 完成: {size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"    ❌ 压缩失败 [{algo_name}]: {e}")
            all_good = False

    return all_good

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("🚀 CTS 镜像准备 (对齐你的配置)")
    logger.info(f"📂 缓存目录: {CACHE_DIR}")
    logger.info(f"🗜️  压缩算法: {list(COMPRESSION_METHODS.keys())}")
    logger.info(f"🖼️  目标镜像: {TARGET_IMAGES}")
    logger.info("="*60)

    # 1. 拉取
    pulled_ok, pulled_fail = pull_images(TARGET_IMAGES)
    
    if not pulled_ok:
        logger.error("❌ 没有成功拉取任何镜像，退出")
        return

    # 2. 预处理
    logger.info(f"\n开始预处理 {len(pulled_ok)} 个镜像 (这可能需要很长时间)...")
    for img in pulled_ok:
        logger.info(f"\n----- 处理镜像: {img} -----")
        save_and_compress_image(img)

    logger.info("\n" + "="*60)
    logger.info("✅ 准备工作完成！")
    logger.info("="*60)

if __name__ == "__main__":
    main()
