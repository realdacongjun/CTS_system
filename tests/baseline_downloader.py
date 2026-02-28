#!/usr/bin/env python3
"""
Baseline 下载器 - 使用 Docker 原生拉取（gzip+单线程）作为基线
核心：完全对齐 Docker 默认行为，无任何自定义优化
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
    使用 Docker 原生拉取作为基线（gzip+单线程，Docker 默认行为）
    
    Args:
        image_name: 镜像名称 (如 "alpine:latest")
        clear_cache: 是否先清理目标镜像缓存（而非全量清理）
        timeout: 超时时间(秒)
    
    Returns:
        包含时间、成功率、算法/线程等信息的字典（对齐CTS返回格式）
    """
    logger.info(f"[Baseline] 开始拉取镜像: {image_name} (Docker原生gzip+单线程)")
    
    # 关键修正1：精准清理目标镜像缓存（而非全量prune）
    if clear_cache:
        try:
            # 先检查镜像是否存在
            check_cmd = ["docker", "images", "-q", image_name]
            img_id = subprocess.run(
                check_cmd, capture_output=True, text=True, timeout=60
            ).stdout.strip()
            
            if img_id:  # 仅删除目标镜像，避免误删其他镜像
                subprocess.run(
                    ["docker", "rmi", "-f", img_id],
                    check=True,
                    capture_output=True,
                    timeout=60
                )
                logger.info(f"[Baseline] 清理目标镜像缓存: {image_name} (ID: {img_id[:12]})")
            else:
                logger.debug(f"[Baseline] 目标镜像未缓存，无需清理: {image_name}")
        except Exception as e:
            logger.warning(f"[Baseline] 缓存清理失败（不影响拉取）: {e}")
    
    start_time = time.perf_counter()
    
    try:
        # 执行 Docker pull 命令（纯原生，无任何自定义参数）
        cmd = ["docker", "pull", image_name]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed_time = time.perf_counter() - start_time
        
        if result.returncode == 0:
            logger.info(f"[Baseline] 拉取成功: {image_name} | 耗时: {elapsed_time:.2f}s")
            return {
                'strategy': 'Baseline',       # 保持原有字段，兼容experiment_runner
                'algorithm': 'gzip',          # 新增：明确标注Docker默认算法
                'threads': 1,                 # 新增：明确标注Docker默认单线程
                'image': image_name,
                'time_s': round(elapsed_time, 4),
                'success': True,
                'error': None                 # 统一返回error字段，避免KeyError
            }
        else:
            error_msg = result.stderr[:200]
            logger.error(f"[Baseline] 拉取失败: {image_name} | 错误: {error_msg}")
            return {
                'strategy': 'Baseline',
                'algorithm': 'gzip',
                'threads': 1,
                'image': image_name,
                'time_s': round(elapsed_time, 4),
                'success': False,
                'error': error_msg
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"[Baseline] 拉取超时: {image_name} | 超时时间: {timeout}s")
        return {
            'strategy': 'Baseline',
            'algorithm': 'gzip',
            'threads': 1,
            'image': image_name,
            'time_s': timeout,  # 超时后耗时记为超时时间
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        error_msg = str(e)[:200]
        logger.error(f"[Baseline] 拉取异常: {image_name} | 异常: {error_msg}")
        return {
            'strategy': 'Baseline',
            'algorithm': 'gzip',
            'threads': 1,
            'image': image_name,
            'time_s': round(elapsed_time, 4),
            'success': False,
            'error': error_msg
        }

def cleanup_image(image_name: str) -> bool:
    """清理指定镜像（精准清理，而非全量）"""
    try:
        cmd = ["docker", "rmi", "-f", image_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"[Baseline] 清理镜像成功: {image_name}")
            return True
        else:
            logger.warning(f"[Baseline] 清理镜像失败: {image_name} | 错误: {result.stderr[:100]}")
            return False
    except Exception as e:
        logger.warning(f"[Baseline] 清理镜像异常: {image_name} | 异常: {e}")
        return False

if __name__ == "__main__":
    # 测试代码（验证原生gzip基线）
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    result = pull_with_docker("alpine:latest", timeout=60)
    print("\n[Baseline 测试结果]")
    for k, v in result.items():
        print(f"{k}: {v}")