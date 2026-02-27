#!/usr/bin/env python3
"""
Baseline 下载器 - 使用固定策略进行容器镜像下载
"""

import time
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def pull_with_docker(
    image_name: str, 
    clear_cache: bool = True, 
    timeout: int = 300
) -> Dict[str, Any]:
    """
    使用 Docker 原生拉取作为基线
    
    Args:
        image_name: 镜像名称 (如 "alpine:latest")
        clear_cache: 是否先清理缓存
        timeout: 超时时间(秒)
    
    Returns:
        包含时间、成功率等信息的字典
    """
    logger.info(f"[Baseline] 开始拉取镜像: {image_name}")
    
    # 清理缓存
    if clear_cache:
        try:
            subprocess.run(
                ["docker", "image", "prune", "-af"],
                check=True,
                capture_output=True,
                timeout=60
            )
        except Exception as e:
            logger.debug(f"缓存清理跳过: {e}")
    
    start_time = time.perf_counter()
    
    try:
        # 执行 Docker pull 命令
        cmd = ["docker", "pull", image_name]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed_time = time.perf_counter() - start_time
        
        if result.returncode == 0:
            logger.info(f"[Baseline] 拉取成功: {elapsed_time:.2f}s")
            return {
                'strategy': 'Baseline',  # 大写B，配合 experiment_runner
                'image': image_name,
                'time_s': round(elapsed_time, 4),
                'success': True
            }
        else:
            logger.error(f"[Baseline] 拉取失败: {result.stderr[:200]}")
            return {
                'strategy': 'Baseline',
                'image': image_name,
                'time_s': round(elapsed_time, 4),
                'success': False,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"[Baseline] 拉取超时 ({timeout}s)")
        return {
            'strategy': 'Baseline',
            'image': image_name,
            'time_s': timeout,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"[Baseline] 拉取异常: {e}")
        return {
            'strategy': 'Baseline',
            'image': image_name,
            'time_s': elapsed_time,
            'success': False,
            'error': str(e)
        }

def cleanup_image(image_name: str) -> bool:
    """清理已拉取的镜像"""
    try:
        cmd = ["docker", "rmi", image_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"清理镜像失败: {e}")
        return False

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    result = pull_with_docker("alpine:latest")
    print(result)