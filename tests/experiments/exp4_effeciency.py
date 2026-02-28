#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4：效率实验（适配新版 adaptive_downloader）
核心目标：量化CTS系统自身的性能开销（模型推理/决策耗时）
"""
import os
import sys
import time
import yaml
import argparse
import psutil
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# ======================== 关键修改1：统一导入路径 ========================
# 🔧 修复：从 testbed 导入
from testbed import adaptive_downloader

LOG_DIR = Path("/cts/logs")
RESULT_DIR = Path("/cts/results")
LOG_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR.mkdir(exist_ok=True, parents=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "exp4_efficiency.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    config_paths = [Path("/cts/config.yaml"), Path("/cts/configs/global_config.yaml")]
    for p in config_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    return {"experiment": {"repeat_times": 10}}

def _complete_env_features(env_state: dict, image_features: dict):
    """🔧 新增：补全物理交叉特征（确保与训练一致）"""
    img_size = image_features.get('total_size_mb', 100.0)
    env_state = env_state.copy()
    env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
    env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
    env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
    env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)
    return env_state

def init_cts_system() -> Tuple[Optional[object], Optional[object]]:
    """初始化模型并预热"""
    try:
        from model_wrapper import CFTNetWrapper
        from cags_decision import CAGSDecisionEngine
        
        model_cfg_path = Path("/cts/configs/model_config.yaml")
        if not model_cfg_path.exists():
            logger.warning("模型配置不存在，无法测量决策开销")
            return None, None
        
        logger.info("📌 加载CTS模型并预热...")
        wrapper = CFTNetWrapper(str(model_cfg_path))
        engine = CAGSDecisionEngine(
            model=wrapper.model, scaler_c=wrapper.scaler_c, scaler_i=wrapper.scaler_i,
            enc=wrapper.enc, cols_c=wrapper.cols_c, cols_i=wrapper.cols_i, device=wrapper.device
        )
        
        # 预热
        dummy_env = _complete_env_features(
            {'bandwidth_mbps': 100, 'network_rtt': 20, 'cpu_limit': 8.0, 'mem_limit_mb': 8192},
            {'total_size_mb': 100}
        )
        for i in range(3):
            engine.make_decision(env_state=dummy_env, image_features={'total_size_mb': 100})
        
        logger.info("✅ CTS引擎初始化+预热完成")
        return wrapper, engine
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
        return None, None

def get_image_features_from_csv(image_name: str) -> dict:
    csv_path = Path("/cts/data/image_features_database.csv")
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            image_base = image_name.split(':')[0]
            match = df[df['image_name'].str.contains(image_base, na=False, case=False)]
            return match.iloc[0].to_dict() if not match.empty else {'total_size_mb': 100.0}
    except Exception as e:
        logger.error(f"获取特征失败: {e}")
    return {'total_size_mb': 100.0}

def collect_env_state(bandwidth: str, delay: str, cpus: float, memory: str) -> dict:
    bw = float(bandwidth.replace('mbit', '')) if 'mbit' in bandwidth else 100.0
    rtt = float(delay.replace('ms', '')) if 'ms' in delay else 20.0
    
    mem_unit = memory[-1].lower()
    mem_size = float(memory[:-1])
    mem_limit_mb = {
        'k': mem_size / 1024, 'm': mem_size, 
        'g': mem_size * 1024, 't': mem_size * 1024 * 1024
    }.get(mem_unit, 8192)
    
    return {
        'bandwidth_mbps': bw, 'network_rtt': rtt, 'packet_loss': 0.0,
        'cpu_limit': cpus, 'mem_limit_mb': mem_limit_mb
    }

def measure_cts_overhead(image: str, env_state: dict, wrapper, engine) -> Dict:
    """测量CTS单次决策的完整开销"""
    logger.info(f"\n🔍 效率测量 - 镜像：{image}")
    
    # 1. 测量特征读取
    t1 = time.perf_counter()
    features = get_image_features_from_csv(image)
    feat_time = (time.perf_counter() - t1) * 1000
    logger.info(f"  特征读取耗时：{feat_time:.4f}ms")
    
    # 补全特征
    full_env = _complete_env_features(env_state, features)
    
    # 2. 测量模型决策 (核心)
    decision_time = 0.0
    config_dict = {'algo_name': 'zstd_l3', 'threads': 4}
    if wrapper is not None and engine is not None:
        t2 = time.perf_counter()
        config_dict, metrics_dict, _ = engine.make_decision(
            env_state=full_env, 
            image_features=features
        )
        decision_time = (time.perf_counter() - t2) * 1000
        logger.info(f"  模型决策耗时：{decision_time:.4f}ms (决策: {config_dict['algo_name']})")
    
    # 3. 测量完整下载 (传入上面决策好的参数，避免下载器内部再次决策)
    t3 = time.perf_counter()
    total_time = 0.0
    success = False
    try:
        res = adaptive_downloader.cts_download(
            image_name=image,
            strategy=config_dict['algo_name'],  # 🔧 关键：传入固定策略
            threads=config_dict['threads'],
            env_state=full_env,
            image_features=features,
            clear_cache=True
        )
        total_time = res['time_s'] * 1000
        success = res.get('success', True)
    except Exception as e:
        pass

    total_overhead = feat_time + decision_time
    overhead_ratio = (total_overhead / total_time) * 100 if (total_time > 0) else 0.0
    
    return {
        'image_name': image,
        'feat_read_time_ms': round(feat_time, 4),
        'decision_time_ms': round(decision_time, 4),
        'total_overhead_ms': round(total_overhead, 4),
        'total_download_time_ms': round(total_time, 2),
        'overhead_ratio_pct': round(overhead_ratio, 4),
        'success': success
    }

def main():
    parser = argparse.ArgumentParser(description="实验4：效率实验")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bandwidth", default="100mbit")
    parser.add_argument("--delay", default="10ms")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cpus", type=float, default=4.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--repeat-times", type=int, default=10)
    # 🔧 修复：接收额外参数
    parser.add_argument("--packet-loss", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--jitter", type=str, default="0ms", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    config = load_config()
    repeat = args.repeat_times or config['experiment']['repeat_times']
    wrapper, engine = init_cts_system()
    images = [img.strip() for img in args.image_list.split(',') if img.strip()]
    env_state = collect_env_state(args.bandwidth, args.delay, args.cpus, args.memory)
    
    logger.info(f"\n🚀 开始实验4：效率实验")
    
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for img in images:
            res = measure_cts_overhead(img, env_state, wrapper, engine)
            res['round'] = r+1
            all_results.append(res)
            time.sleep(1)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp4_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 结果已保存：{csv_path}")
    
    # 简单汇总
    df_success = df[df['success'] == True]
    if not df_success.empty:
        logger.info(f"\n🎉 实验4完成！平均决策耗时: {df_success['decision_time_ms'].mean():.4f}ms")

if __name__ == "__main__":
    main()