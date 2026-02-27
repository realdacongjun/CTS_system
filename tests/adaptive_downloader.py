#!/usr/bin/env python3
"""
自适应下载器 - 【方案二完整版】
流程: 请求策略API -> 并行下载 -> 解压 -> Docker Load
"""

import os
import time
import requests
import subprocess
import logging
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 全局配置
PROXY_HOST = "localhost"
PROXY_PORT = 8000
API_BASE_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
TEMP_DIR = Path("./.cts_temp")
TEMP_DIR.mkdir(exist_ok=True)

# 解压命令映射 (仅用于本地 fallback，主要策略由 API 决定)
DECOMPRESS_CMD = {
    '.tar.gz': ['gzip', '-d'],
    '.tar.zst': ['zstd', '-d', '--force'],
    '.tar': None
}

def get_strategy_from_proxy(
    image_name: str, 
    env_state: Dict[str, Any], 
    image_features: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    【核心新增】调用代理服务器获取下载策略
    """
    url = f"{API_BASE_URL}/api/v1/strategy"
    payload = {
        "image_name": image_name,
        "env_state": env_state,
        "image_features": image_features
    }
    
    try:
        logger.info(f"[CTS] 请求策略: {image_name}")
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        
        strategy = result.get('strategy', {})
        metrics = result.get('metrics', {})
        
        logger.info(f"[CTS] 代理决策: Algo={strategy.get('algo')}, Threads={strategy.get('threads')}")
        if metrics:
            logger.info(f"[CTS] 预测指标: PredTime={metrics.get('pred_time_s', 0):.2f}s, Unc={metrics.get('epistemic_unc', 0):.3f}")
        
        return strategy
        
    except requests.exceptions.ConnectionError:
        logger.error(f"[CTS] 无法连接到代理服务器: {API_BASE_URL}")
        return None
    except Exception as e:
        logger.error(f"[CTS] 策略请求失败: {e}")
        return None

def _download_single_chunk(url: str, start: int, end: int, part_path: Path):
    """下载单个分片"""
    headers = {"Range": f"bytes={start}-{end}"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30)
        r.raise_for_status()
        with open(part_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"分片下载失败: {e}")
        return False

def _parallel_download_file(url: str, save_path: Path, threads: int = 4):
    """并行下载文件"""
    try:
        # 1. 获取文件大小
        head = requests.head(url, timeout=10)
        head.raise_for_status()
        file_size = int(head.headers.get("Content-Length", 0))
    except Exception as e:
        logger.error(f"无法连接代理服务器或获取文件大小: {e}")
        return False

    if file_size == 0:
        logger.error("文件大小为0")
        return False

    # 2. 计算分片
    chunk_size = file_size // threads
    parts = []
    futures = []

    logger.info(f"开始下载: ({file_size / 1024 / 1024:.2f} MB, {threads} 线程)")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(threads):
            start = i * chunk_size
            end = file_size - 1 if i == threads - 1 else (i + 1) * chunk_size - 1
            part_path = TEMP_DIR / f".part_{i}"
            parts.append(part_path)
            futures.append(executor.submit(_download_single_chunk, url, start, end, part_path))
        
        for future in as_completed(futures):
            if not future.result():
                for p in parts:
                    if p.exists(): p.unlink()
                return False

    # 3. 合并文件
    logger.info("合并文件...")
    try:
        with open(save_path, 'wb') as outfile:
            for part_path in parts:
                if part_path.exists():
                    with open(part_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    part_path.unlink()
    except Exception as e:
        logger.error(f"文件合并失败: {e}")
        return False

    return True

def cts_download(
    image_name: str, 
    strategy: Optional[str] = None, # 【修改】现在可以为 None，由 API 决定
    threads: Optional[int] = None,   # 【修改】现在可以为 None，由 API 决定
    env_state: Optional[Dict[str, Any]] = None,
    image_features: Optional[Dict[str, Any]] = None,
    clear_cache: bool = True,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    使用 CTS 策略进行下载 (完整版)
    """
    # 0. 初始化默认值
    final_strategy_name = strategy if strategy else "api_guided"
    final_threads = threads if threads else 4
    download_url = None
    
    logger.info(f"[CTS] 开始处理: {image_name}")
    
    # 0.1 清理 Docker 缓存
    if clear_cache:
        try:
            subprocess.run(["docker", "image", "prune", "-af"], check=True, capture_output=True, timeout=60)
        except:
            pass

    start_time = time.perf_counter()
    
    # ==========================================
    # 1. 【核心】请求 API 获取策略
    # ==========================================
    if env_state is not None and image_features is not None:
        api_strategy = get_strategy_from_proxy(image_name, env_state, image_features)
        if api_strategy:
            download_url = api_strategy.get('download_url')
            final_strategy_name = api_strategy.get('algo', final_strategy_name)
            final_threads = api_strategy.get('threads', final_threads)
    
    # ==========================================
    # 2. Fallback: 如果 API 没返回 URL，手动构造
    # ==========================================
    safe_name = image_name.replace(":", "_").replace("/", "_")
    compressed_file = None
    
    if not download_url:
        logger.warning("[CTS] API 未返回 URL，使用本地 fallback 逻辑")
        # 简单的 fallback 映射
        if final_strategy_name == 'gzip':
            subdir, ext = 'gzip', 'tar.gz'
        elif 'zstd' in final_strategy_name:
            subdir, ext = 'zstd_l3', 'tar.zst'
        else:
            subdir, ext = 'raw', 'tar'
            
        download_url = f"{API_BASE_URL}/data/{subdir}/{safe_name}.{ext}"
        compressed_file = TEMP_DIR / f"{safe_name}.{ext}"
    else:
        # 从 URL 提取文件名
        filename = download_url.split('/')[-1]
        compressed_file = TEMP_DIR / filename

    tar_file = TEMP_DIR / f"{safe_name}.tar"
    
    try:
        # 3. 下载
        dl_success = _parallel_download_file(download_url, compressed_file, final_threads)
        if not dl_success:
            raise RuntimeError(f"下载失败: {download_url}")

        # 4. 解压 (根据后缀自动判断)
        load_path = compressed_file
        file_suffix = "".join(compressed_file.suffixes) # .tar.gz 或 .tar.zst
        
        decomp_cmd = None
        if '.gz' in file_suffix:
            decomp_cmd = ['gzip', '-d']
        elif '.zst' in file_suffix:
            decomp_cmd = ['zstd', '-d', '--force']
        
        if decomp_cmd:
            logger.info("解压文件...")
            with open(tar_file, 'wb') as f_out:
                subprocess.run(
                    decomp_cmd + [str(compressed_file)],
                    check=True,
                    stdout=f_out,
                    timeout=timeout
                )
            load_path = tar_file
            compressed_file.unlink()

        # 5. Docker Load
        logger.info("加载镜像到 Docker...")
        subprocess.run(
            ["docker", "load", "-i", str(load_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # 6. 清理
        if load_path.exists() and load_path != compressed_file:
            load_path.unlink()
        if compressed_file.exists():
            compressed_file.unlink()

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"[CTS] 成功: {elapsed_time:.2f}s")
        
        return {
            'strategy': 'CTS',
            'compression': final_strategy_name,
            'threads': final_threads,
            'image': image_name,
            'time_s': round(elapsed_time, 4),
            'success': True
        }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"[CTS] 超时 ({timeout}s)")
        return {
            'strategy': 'CTS',
            'compression': final_strategy_name,
            'threads': final_threads,
            'image': image_name,
            'time_s': elapsed_time,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"[CTS] 异常: {e}", exc_info=False)
        for f in [compressed_file, tar_file]:
            if f and f.exists(): f.unlink()
        return {
            'strategy': 'CTS',
            'compression': final_strategy_name,
            'threads': final_threads,
            'image': image_name,
            'time_s': elapsed_time,
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("测试 CTS 下载器 (请确保 proxy_server 运行中)...")
    
    # 测试参数
    test_env = {
        'bandwidth_mbps': 100.0, 'network_rtt': 20.0, 'cpu_limit': 8.0, 
        'mem_limit_mb': 16384.0, 'theoretical_time': 10.0
    }
    test_feat = {'total_size_mb': 100.0}
    
    # result = cts_download(
    #     "nginx:latest", 
    #     env_state=test_env, 
    #     image_features=test_feat
    # )
    # print(result)