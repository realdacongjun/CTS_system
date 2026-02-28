#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTS 自适应下载器 - 物理真实版
核心逻辑：
1. 调用 CAGS 引擎决策
2. 实际执行 docker save -> 压缩 -> 解压 -> docker load
3. 记录真实耗时
"""
import os
import sys
import time
import subprocess
import logging
import psutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# 路径配置
CTS_ROOT = Path(__file__).parent.parent
SRC_DIR = CTS_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

# 全局缓存
_ENGINE_CACHE: Optional[Tuple] = None

def _get_cts_engine():
    global _ENGINE_CACHE
    if _ENGINE_CACHE:
        return _ENGINE_CACHE
    
    try:
        from model_wrapper import CFTNetWrapper
        from cags_decision import CAGSDecisionEngine
        
        logger.info("🚀 [CTS] 正在加载模型...")
        model_cfg = CTS_ROOT / "configs" / "model_config.yaml"
        
        wrapper = CFTNetWrapper(str(model_cfg))
        engine = CAGSDecisionEngine(
            model=wrapper.model,
            scaler_c=wrapper.scaler_c,
            scaler_i=wrapper.scaler_i,
            enc=wrapper.enc,
            cols_c=wrapper.cols_c,
            cols_i=wrapper.cols_i,
            device=wrapper.device
        )
        _ENGINE_CACHE = (wrapper, engine)
        return wrapper, engine
    except Exception as e:
        logger.warning(f"⚠️  模型加载失败，将使用默认策略: {e}")
        return None, None

def _get_image_features(image_name: str) -> Dict:
    csv_path = CTS_ROOT / "data" / "image_features_database.csv"
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            base = image_name.split(':')[0]
            match = df[df['image_name'].str.contains(base, na=False, case=False)]
            if not match.empty:
                return match.iloc[0].to_dict()
    except:
        pass
    return {'total_size_mb': 200.0}

def _collect_env_state() -> Dict:
    """采集真实环境状态"""
    try:
        cpu_count = psutil.cpu_count(logical=True)
        mem_total = psutil.virtual_memory().total / 1024 / 1024
        return {
            'bandwidth_mbps': 100.0,  # 可由外部传入覆盖
            'network_rtt': 10.0,
            'cpu_limit': float(cpu_count),
            'mem_limit_mb': float(mem_total)
        }
    except:
        return {'bandwidth_mbps': 100.0, 'cpu_limit': 4.0, 'mem_limit_mb': 8192.0}

def _build_compress_command(algo_name: str, threads: int, input_path: str, output_path: str) -> list:
    """
    核心映射：将模型决策的 algo_name 映射为实际的 Shell 命令
    要求容器内已安装：gzip, lz4, zstd, brotli
    """
    # 通用格式：command [args] > output
    # 注意：为了性能对比，我们使用管道流，避免双重IO
    
    algo_map = {
        # Gzip 系列
        "gzip": ["pigz", "-p", str(max(1, threads)), "-c"],  # pigz 是并行 gzip
        "gzip_fast": ["pigz", "-p", str(max(1, threads)), "-1", "-c"],
        
        # LZ4 系列 (极快)
        "lz4": ["lz4", "-c", "-z"],
        "lz4_high": ["lz4", "-c", "-z", "-9"],
        
        # ZSTD 系列 (平衡)
        "zstd_l1": ["zstd", "-T", str(max(1, threads)), "-1", "-c"],
        "zstd_l3": ["zstd", "-T", str(max(1, threads)), "-3", "-c"],
        "zstd_l5": ["zstd", "-T", str(max(1, threads)), "-5", "-c"],
        "zstd_l9": ["zstd", "-T", str(max(1, threads)), "-9", "-c"],
        "zstd_l12": ["zstd", "-T", str(max(1, threads)), "-12", "-c"],
        
        # Brotli 系列 (高压缩)
        "brotli_l4": ["brotli", "-q", "4", "-c"]
    }
    
    # 默认回退
    return algo_map.get(algo_name, ["zstd", "-T", str(threads), "-3", "-c"])

def _build_decompress_command(algo_name: str) -> list:
    """构建解压命令"""
    algo_map = {
        "gzip": ["pigz", "-d", "-c"],
        "gzip_fast": ["pigz", "-d", "-c"],
        "lz4": ["lz4", "-d", "-c"],
        "lz4_high": ["lz4", "-d", "-c"],
        "brotli_l4": ["brotli", "-d", "-c"],
    }
    # zstd 可以自动识别，其他默认用 zstd -d
    if "zstd" in algo_name:
        return ["zstd", "-d", "-c"]
    return algo_map.get(algo_name, ["zstd", "-d", "-c"])

def _run_cts_pipeline(
    image_name: str, 
    algo_name: str, 
    threads: int
) -> Tuple[bool, float, str]:
    """
    执行真实的 CTS 流水线：
    Save -> Compress -> Decompress -> Load
    """
    temp_dir = Path("/tmp/cts")
    temp_dir.mkdir(exist_ok=True)
    tar_path = temp_dir / f"img_{os.getpid()}.tar"
    
    total_start = time.perf_counter()
    error_msg = ""
    success = False

    try:
        # 1. Docker Save (获取原始镜像数据流)
        logger.info(f"[CTS] Step 1/4: docker save {image_name}...")
        t_save = time.perf_counter()
        with open(tar_path, 'wb') as f:
            subprocess.run(
                ["docker", "save", image_name],
                stdout=f, check=True, timeout=600
            )
        save_time = time.perf_counter() - t_save
        logger.info(f"[CTS] Save done: {save_time:.2f}s")

        # 2. 压缩 (应用决策)
        logger.info(f"[CTS] Step 2/4: Compressing with {algo_name} @ {threads} threads...")
        t_comp = time.perf_counter()
        
        # 构建命令
        comp_cmd = _build_compress_command(algo_name, threads, str(tar_path), "")
        comp_path = tar_path.with_suffix(f".tar.{algo_name.split('_')[0]}")
        
        with open(tar_path, 'rb') as f_in:
            with open(comp_path, 'wb') as f_out:
                p1 = subprocess.Popen(comp_cmd, stdin=f_in, stdout=f_out, stderr=subprocess.PIPE)
                _, err = p1.communicate(timeout=600)
        
        if p1.returncode != 0:
            logger.warning(f"Compress warning: {err.decode()[:200]}")
            
        comp_time = time.perf_counter() - t_comp
        logger.info(f"[CTS] Compress done: {comp_time:.2f}s")

        # 3. 解压 (模拟接收端)
        logger.info(f"[CTS] Step 3/4: Decompressing...")
        t_decomp = time.perf_counter()
        
        decomp_cmd = _build_decompress_command(algo_name)
        recovered_tar = tar_path.with_suffix(".recovered.tar")
        
        with open(comp_path, 'rb') as f_in:
            with open(recovered_tar, 'wb') as f_out:
                p2 = subprocess.Popen(decomp_cmd, stdin=f_in, stdout=f_out, stderr=subprocess.PIPE)
                _, err2 = p2.communicate(timeout=600)
        
        decomp_time = time.perf_counter() - t_decomp
        logger.info(f"[CTS] Decompress done: {decomp_time:.2f}s")

        # 4. Docker Load
        logger.info(f"[CTS] Step 4/4: docker load...")
        t_load = time.perf_counter()
        
        with open(recovered_tar, 'rb') as f:
            subprocess.run(
                ["docker", "load"],
                stdin=f, check=True, stdout=subprocess.DEVNULL, timeout=600
            )
        
        load_time = time.perf_counter() - t_load
        logger.info(f"[CTS] Load done: {load_time:.2f}s")

        total_time = time.perf_counter() - total_start
        success = True
        
        # 清理临时文件
        try:
            tar_path.unlink(missing_ok=True)
            comp_path.unlink(missing_ok=True)
            recovered_tar.unlink(missing_ok=True)
        except:
            pass

        return success, total_time, ""

    except subprocess.TimeoutExpired:
        return False, 0.0, "Timeout"
    except Exception as e:
        return False, 0.0, str(e)[:200]

def cts_download(
    image_name: str,
    strategy: Optional[str] = None,
    threads: Optional[int] = None,
    env_state: Optional[Dict] = None,
    image_features: Optional[Dict] = None,
    clear_cache: bool = True
) -> Dict[str, Any]:
    
    start_total = time.perf_counter()
    
    # 0. 清理缓存
    if clear_cache:
        try:
            subprocess.run(["docker", "rmi", "-f", image_name], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except:
            pass

    # 1. 准备特征
    if env_state is None:
        env_state = _collect_env_state()
    if image_features is None:
        image_features = _get_image_features(image_name)
    
    # 补充物理特征 (必须与训练一致)
    img_size = image_features.get('total_size_mb', 100.0)
    env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
    env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
    env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
    env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)

    # 2. 决策
    final_algo = strategy or "zstd_l3"
    final_threads = threads or 4
    pred_time = 0.0
    
    if strategy is None:
        wrapper, engine = _get_cts_engine()
        if engine:
            try:
                config_dict, metrics, _ = engine.make_decision(
                    env_state=env_state,
                    image_features=image_features,
                    safety_threshold=0.7
                )
                final_algo = config_dict['algo_name']
                final_threads = config_dict['threads']
                pred_time = metrics.get('pred_time_s', 0.0)
                logger.info(f"🤖 [CTS] AI 决策: {final_algo} @ {final_threads} (Pred: {pred_time:.2f}s)")
            except Exception as e:
                logger.warning(f"决策失败，回退默认: {e}")
                final_algo = "zstd_l3"
                final_threads = 4

    # 3. 执行真实物理流程
    # 注意：这里我们分两种模式
    # Mode A: 如果是 Baseline 对比，我们直接 docker pull
    # Mode B: 如果是 CTS，我们跑 Save-Compress-Load 循环
    
    # 为了让 exp1 能对比，我们这里加一个逻辑：
    # 如果 strategy 显式传入 'baseline' 或 'gzip' 且 threads==1，我们直接 pull
    # 否则跑 CTS 流程
    
    use_native_pull = (strategy == "baseline") or (strategy == "gzip" and threads == 1)
    
    if use_native_pull:
        # 原生模式 (Baseline)
        logger.info(f"[Baseline] 执行原生 docker pull...")
        t0 = time.time()
        try:
            subprocess.run(["docker", "pull", image_name], check=True, timeout=600)
            total_time = time.time() - t0
            return {
                "strategy": "Baseline",
                "compression": "gzip",
                "threads": 1,
                "image": image_name,
                "time_s": round(total_time, 2),
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "strategy": "Baseline",
                "compression": "gzip",
                "threads": 1,
                "image": image_name,
                "time_s": 0,
                "success": False,
                "error": str(e)[:200]
            }
    else:
        # CTS 模式 (真实压缩流程)
        success, total_time, error = _run_cts_pipeline(image_name, final_algo, final_threads)
        
        return {
            "strategy": "CTS",
            "compression": final_algo,
            "threads": final_threads,
            "decision_source": "AI" if strategy is None else "Manual",
            "image": image_name,
            "time_s": round(total_time, 2),
            "pred_time_s": pred_time,
            "success": success,
            "error": error
        }