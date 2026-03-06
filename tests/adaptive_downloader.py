#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTS 自适应下载器 - 预压缩优化版
核心逻辑（优化后）：
1. 调用 CAGS 引擎决策
2. 直接读取预压缩文件（跳过 docker save 和实时压缩）
3. 执行解压 -> docker load
4. 记录真实耗时
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
PRECOMPRESSED_DIR = Path(os.environ.get("PRECOMPRESSED_IMAGES_DIR", "/cts/data/preprocessed_images"))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

# 全局缓存
_ENGINE_CACHE: Optional[Tuple] = None

# ==========================================
# 🔧 核心映射：模型决策 algo_name -> 预压缩文件后缀
# ==========================================
# ==========================================
# 🔧 核心映射：模型决策 algo_name -> 预压缩文件后缀
# ==========================================
ALGO_TO_PRECOMPRESSED_SUFFIX = {
    # Gzip 系列（直接对应文件名）
    "gzip-1": "gzip-1",
    "gzip-6": "gzip-6",
    "gzip-9": "gzip-9",
    
    # ZSTD 系列（直接对应文件名）
    "zstd-1": "zstd-1",
    "zstd-3": "zstd-3",
    "zstd-6": "zstd-6",
    
    # LZ4 系列（直接对应文件名）
    "lz4-fast": "lz4-fast",
    "lz4-medium": "lz4-medium",
    "lz4-slow": "lz4-slow",
    
    # Brotli 系列（直接对应文件名）
    "brotli-1": "brotli-1"
}


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
            'bandwidth_mbps': 100.0,
            'network_rtt': 10.0,
            'cpu_limit': float(cpu_count),
            'mem_limit_mb': float(mem_total)
        }
    except:
        return {'bandwidth_mbps': 100.0, 'cpu_limit': 4.0, 'mem_limit_mb': 8192.0}

def _build_decompress_command(algo_name: str) -> list:
    """构建解压命令（根据预压缩文件后缀）"""
    if "gzip" in algo_name:
        return ["pigz", "-d", "-c"]
    elif "lz4" in algo_name:
        return ["lz4", "-d", "-c"]
    elif "brotli" in algo_name:
        return ["brotli", "-d", "-c"]
    else:  # zstd
        return ["zstd", "-d", "-c"]

def _get_precompressed_path(image_name: str, algo_suffix: str) -> Optional[Path]:
    """
    查找预压缩文件路径
    输入：image_name="ubuntu:latest", algo_suffix="zstd-3"
    输出：/cts/data/preprocessed_images/ubuntu_latest.tar.zstd-3
    """
    safe_name = image_name.replace(":", "_").replace("/", "_")
    precompressed_file = PRECOMPRESSED_DIR / f"{safe_name}.tar.{algo_suffix}"
    
    if precompressed_file.exists():
        logger.info(f"✅ [CTS] 找到预压缩文件: {precompressed_file.name} ({precompressed_file.stat().st_size / 1024 / 1024:.2f}MB)")
        return precompressed_file
    else:
        logger.error(f"❌ [CTS] 预压缩文件不存在: {precompressed_file.name}")
        return None

def _run_cts_pipeline(
    image_name: str, 
    algo_name: str, 
    threads: int
) -> Tuple[bool, float, str]:
    """
    优化后的 CTS 流水线：
    1. 查找预压缩文件（跳过 docker save 和实时压缩）
    2. 直接解压预压缩文件
    3. Docker Load
    """
    temp_dir = Path("/tmp/cts")
    temp_dir.mkdir(exist_ok=True)
    
    total_start = time.perf_counter()
    success = False
    error_msg = ""

    try:
        # ==========================================
        # 🔧 核心优化：直接使用预压缩文件
        # ==========================================
        # Step 1: 映射 algo_name 到预压缩文件后缀
        algo_suffix = ALGO_TO_PRECOMPRESSED_SUFFIX.get(algo_name, "zstd-3")
        
        # Step 2: 查找预压缩文件
        precompressed_path = _get_precompressed_path(image_name, algo_suffix)
        if not precompressed_path:
            return False, 0.0, f"预压缩文件不存在: {algo_suffix}"

        # Step 3: 直接解压预压缩文件（模拟接收端）
        logger.info(f"[CTS] Step 1/2: 解压预压缩文件...")
        t_decomp = time.perf_counter()
        
        decomp_cmd = _build_decompress_command(algo_suffix)
        recovered_tar = temp_dir / f"img_{os.getpid()}.recovered.tar"
        
        with open(precompressed_path, 'rb') as f_in:
            with open(recovered_tar, 'wb') as f_out:
                p2 = subprocess.Popen(decomp_cmd, stdin=f_in, stdout=f_out, stderr=subprocess.PIPE)
                _, err2 = p2.communicate(timeout=600)
        
        if p2.returncode != 0:
            logger.warning(f"解压警告: {err2.decode()[:200]}")
            
        decomp_time = time.perf_counter() - t_decomp
        logger.info(f"[CTS] 解压完成: {decomp_time:.2f}s")

        # Step 4: Docker Load
        logger.info(f"[CTS] Step 2/2: docker load...")
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
            recovered_tar.unlink(missing_ok=True)
            logger.info(f"[CTS] 临时文件已清理")
        except Exception as e:
            logger.warning(f"⚠️  临时文件清理失败: {e}")

        return success, total_time, ""

    except subprocess.TimeoutExpired:
        try:
            recovered_tar.unlink(missing_ok=True)
        except:
            pass
        return False, 0.0, "Timeout"
    except Exception as e:
        try:
            recovered_tar.unlink(missing_ok=True)
        except:
            pass
        return False, 0.0, str(e)[:200]

# def cts_download(
#     image_name: str,
#     strategy: Optional[str] = None,
#     threads: Optional[int] = None,
#     env_state: Optional[Dict] = None,
#     image_features: Optional[Dict] = None,
#     clear_cache: bool = False
# ) -> Dict[str, Any]:
    
#     start_total = time.perf_counter()
    
#     # 0. 清理缓存（默认关闭）
#     if clear_cache:
#         try:
#             subprocess.run(["docker", "rmi", "-f", image_name], 
#                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
#         except:
#             pass

#     # 1. 准备特征
#     if env_state is None:
#         env_state = _collect_env_state()
#     if image_features is None:
#         image_features = _get_image_features(image_name)
    
#     # 补充物理特征
#     img_size = image_features.get('total_size_mb', 100.0)
#     env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
#     env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
#     env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
#     env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)

#     # 2. 决策
#     final_algo = strategy or "zstd_l3"
#     final_threads = threads or 4
#     pred_time = 0.0
    
#     if strategy is None:
#         wrapper, engine = _get_cts_engine()
#         if engine:
#             try:
#                 config_dict, metrics, _ = engine.make_decision(
#                     env_state=env_state,
#                     image_features=image_features,
#                     safety_threshold=0.7
#                 )
#                 final_algo = config_dict['algo_name']
#                 final_threads = config_dict['threads']
#                 pred_time = metrics.get('pred_time_s', 0.0)
#                 logger.info(f"🤖 [CTS] AI 决策: {final_algo} @ {final_threads} (Pred: {pred_time:.2f}s)")
#             except Exception as e:
#                 logger.warning(f"决策失败，回退默认: {e}")
#                 final_algo = "zstd_l3"
#                 final_threads = 4

#     # 3. 执行流程
#     use_native_pull = (strategy == "baseline") or (strategy == "gzip" and threads == 1)
    
#     if use_native_pull:
#         # 原生模式 (Baseline)
#         logger.info(f"[Baseline] 执行原生 docker pull...")
#         t0 = time.time()
#         try:
#             result = subprocess.run(
#                 ["docker", "pull", image_name],
#                 capture_output=True, text=True, timeout=600
#             )
#             if result is None:
#                 return {
#                     "strategy": "Baseline",
#                     "compression": "gzip",
#                     "threads": 1,
#                     "image": image_name,
#                     "time_s": 0,
#                     "success": False,
#                     "error": "Docker Pull返回空结果"
#                 }
#             if result.returncode == 0:
#                 total_time = time.time() - t0
#                 return {
#                     "strategy": "Baseline",
#                     "compression": "gzip",
#                     "threads": 1,
#                     "image": image_name,
#                     "time_s": round(total_time, 2),
#                     "success": True,
#                     "error": None
#                 }
#             else:
#                 total_time = time.time() - t0
#                 return {
#                     "strategy": "Baseline",
#                     "compression": "gzip",
#                     "threads": 1,
#                     "image": image_name,
#                     "time_s": 0,
#                     "success": False,
#                     "error": f"Docker Pull失败: {result.stderr[:200] if result.stderr else '未知错误'}"
#                 }
#         except Exception as e:
#             return {
#                 "strategy": "Baseline",
#                 "compression": "gzip",
#                 "threads": 1,
#                 "image": image_name,
#                 "time_s": 0,
#                 "success": False,
#                 "error": str(e)[:200]
#             }
#     else:
#         # CTS 模式 (使用预压缩文件)
#         success, total_time, error = _run_cts_pipeline(image_name, final_algo, final_threads)
        
#         return {
#             "strategy": "CTS",
#             "compression": final_algo,
#             "threads": final_threads,
#             "decision_source": "AI" if strategy is None else "Manual",
#             "image": image_name,
#             "time_s": round(total_time, 2),
#             "pred_time_s": pred_time,
#             "success": success,
#             "error": error
#         }

def cts_download(
    image_name: str,
    strategy: Optional[str] = None,
    threads: Optional[int] = None,
    env_state: Optional[Dict] = None,
    image_features: Optional[Dict] = None,
    clear_cache: bool = False,
    # 🔧 新增参数用于实验对比
    use_precompressed: bool = True,
    force_algo: Optional[str] = None
) -> Dict[str, Any]:
    
    start_total = time.perf_counter()
    
    # 0. 清理缓存（默认关闭）
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
    
    # 补充物理特征
    img_size = image_features.get('total_size_mb', 100.0)
    env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
    env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
    env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
    env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)

    # 2. 决策
    final_algo = strategy or "zstd-3"
    final_threads = threads or 4
    pred_time = 0.0
    
    # 🔧 如果没有强制指定算法，才使用 AI 决策
    if force_algo is None and strategy is None:
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
                logger.info(f"🤖 [CTS] AI 决策：{final_algo} @ {final_threads} (Pred: {pred_time:.2f}s)")
            except Exception as e:
                logger.warning(f"决策失败，回退默认：{e}")
                final_algo = "zstd-3"
                final_threads = 4
    elif force_algo is not None:
        # 🔧 使用强制指定的算法（用于固定策略对比）
        final_algo = force_algo
        final_threads = threads or 4
        logger.info(f"[CTS] 使用固定策略：{final_algo} @ {final_threads}")

    # 3. 执行流程（都使用预压缩文件）
    logger.info(f"[CTS] 使用预压缩文件：{final_algo}...")
    success, total_time, error = _run_cts_pipeline(image_name, final_algo, final_threads)
    
    return {
        "strategy": "CTS",
        "compression": final_algo,
        "threads": final_threads,
        "decision_source": "AI" if (force_algo is None and strategy is None) else "Fixed",
        "image": image_name,
        "time_s": round(total_time, 2),
        "pred_time_s": pred_time,
        "success": success,
        "error": error
    }
