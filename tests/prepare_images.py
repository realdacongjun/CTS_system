#!/usr/bin/env python3
"""
镜像准备脚本 - 预先拉取测试所需的容器镜像
"""

import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)

def pull_images(image_list: List[str]) -> dict:
    """
    批量拉取镜像
    
    Args:
        image_list: 镜像名称列表
    
    Returns:
        成功和失败的镜像统计
    """
    results = {
        'success': [],
        'failed': []
    }
    
    for image in image_list:
        logger.info(f"正在拉取镜像: {image}")
        try:
            cmd = ["docker", "pull", image]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ 成功拉取: {image}")
                results['success'].append(image)
            else:
                logger.error(f"❌ 拉取失败: {image} - {result.stderr}")
                results['failed'].append((image, result.stderr))
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 拉取超时: {image}")
            results['failed'].append((image, "Timeout"))
        except Exception as e:
            logger.error(f"❌ 拉取异常: {image} - {e}")
            results['failed'].append((image, str(e)))
    
    return results

def cleanup_unused_images() -> bool:
    """清理未使用的镜像"""
    try:
        cmd = ["docker", "image", "prune", "-f"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ 清理完成")
            return True
        else:
            logger.error(f"❌ 清理失败: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ 清理异常: {e}")
        return False

def main():
    """主函数"""
    # 测试镜像列表
    test_images = [
        "ubuntu:latest",
        "nginx:latest",
        "mysql:latest"
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 开始准备测试镜像...")
    
    results = pull_images(test_images)
    
    logger.info("\n" + "="*50)
    logger.info("📊 准备结果统计")
    logger.info("="*50)
    logger.info(f"成功: {len(results['success'])} 个")
    logger.info(f"失败: {len(results['failed'])} 个")
    
    if results['failed']:
        logger.info("\n失败详情:")
        for image, error in results['failed']:
            logger.info(f"  - {image}: {error}")
    
    # 询问是否清理
    if input("\n是否清理未使用的镜像? (y/N): ").lower() == 'y':
        cleanup_unused_images()

if __name__ == "__main__":
    main()