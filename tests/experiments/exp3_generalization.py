#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验3：泛化性实验（适配新版 adaptive_downloader）
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
        logging.FileHandler(LOG_DIR / "exp3_generalization.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    config_paths = [Path("/cts/config.yaml"), Path("/cts/configs/global_config.yaml")]
    for p in config_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    return {"experiment": {"repeat_times": 1}}

def init_cts_system() -> Tuple[Optional[object], Optional[object]]:
    """初始化引擎（仅用于日志确认，实际决策交给下载器）"""
    try:
        from model_wrapper import CFTNetWrapper
        from cags_decision import CAGSDecisionEngine
        
        model_cfg_path = Path("/cts/configs/model_config.yaml")
        if not model_cfg_path.exists():
            return None, None
        
        wrapper = CFTNetWrapper(str(model_cfg_path))
        engine = CAGSDecisionEngine(
            model=wrapper.model, scaler_c=wrapper.scaler_c, scaler_i=wrapper.scaler_i,
            enc=wrapper.enc, cols_c=wrapper.cols_c, cols_i=wrapper.cols_i, device=wrapper.device
        )
        logger.info("✅ CTS决策引擎初始化成功")
        return wrapper, engine
    except Exception as e:
        logger.warning(f"⚠️  模型加载失败，将使用固定策略: {e}")
        return None, None

IMAGE_CATEGORIES = {
    "lightweight": ["alpine:latest", "busybox:latest"],
    "medium": ["nginx:alpine", "redis:alpine", "python:alpine"],
    "heavyweight": ["ubuntu:latest", "mysql:latest", "tensorflow:latest"]
}

def collect_env_state(bandwidth: str, delay: str, cpus: float, memory: str) -> dict:
    bw = float(bandwidth.replace('mbit', '')) if 'mbit' in bandwidth else 1000.0
    rtt = float(delay.replace('ms', '')) if 'ms' in delay else 1.0
    
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

def _complete_env_features(env_state: dict, image_features: dict):
    """🔧 新增：补全物理交叉特征"""
    img_size = image_features.get('total_size_mb', 100.0)
    env_state = env_state.copy()
    env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
    env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
    env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
    env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)
    return env_state

def get_image_features_from_csv(image_name: str) -> dict:
    csv_path = Path("/cts/data/image_features_database.csv")
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            image_base = image_name.split(':')[0]
            match = df[df['image_name'].str.contains(image_base, na=False, case=False)]
            return match.iloc[0].to_dict() if not match.empty else {'total_size_mb': 100.0}
    except Exception as e:
        logger.error(f"获取镜像特征失败: {e}")
    return {'total_size_mb': 100.0}

def run_generalization_test(image: str, img_category: str, env_state: dict, has_model: bool, round_num: int) -> Dict:
    """执行单镜像泛化性测试（简化逻辑，委托下载器决策）"""
    logger.info(f"\n🔍 泛化性测试 - 轮次{round_num} | 类型：{img_category} | 镜像：{image}")
    
    features = get_image_features_from_csv(image)
    full_env = _complete_env_features(env_state, features)
    
    try:
        # 🔧 关键修改：根据是否有模型，决定传 None 还是 固定策略
        if has_model:
            # 有模型：传 None，让下载器内部自动决策
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy=None,
                threads=None,
                env_state=full_env,
                image_features=features,
                clear_cache=True
            )
            used_algo = res.get('compression', 'AI_Auto')
            used_threads = res.get('threads', 0)
        else:
            # 无模型：固定启发式
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy="zstd_l3",
                threads=4,
                env_state=full_env,
                image_features=features,
                clear_cache=True
            )
            used_algo = "zstd_l3"
            used_threads = 4

        return {
            'image_category': img_category,
            'image_name': image,
            'round': round_num,
            'time_s': res['time_s'],
            'success': res.get('success', True),
            'img_size_mb': features.get('total_size_mb', 100.0),
            'compression': used_algo,
            'threads': used_threads,
            'error_msg': res.get('error', '')[:200]
        }
    
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return {
            'image_category': img_category,
            'image_name': image,
            'round': round_num,
            'time_s': 0.0,
            'success': False,
            'img_size_mb': 100.0,
            'compression': 'unknown',
            'threads': 0,
            'error_msg': str(e)[:200]
        }

def main():
    parser = argparse.ArgumentParser(description="实验3：泛化性实验")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bandwidth", default="1000mbit")
    parser.add_argument("--delay", default="1ms")
    parser.add_argument("--image-list", default="default")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cpus", type=float, default=4.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--repeat-times", type=int, default=1)
    # 🔧 修复：接收额外参数防止报错
    parser.add_argument("--packet-loss", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--jitter", type=str, default="0ms", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    config = load_config()
    repeat = args.repeat_times or config['experiment']['repeat_times']
    
    # 初始化引擎（仅用于判断是否有模型，不在这里做决策）
    wrapper, engine = init_cts_system()
    has_model = (wrapper is not None and engine is not None)
    
    env_state = collect_env_state(args.bandwidth, args.delay, args.cpus, args.memory)
    
    logger.info(f"\n🚀 开始实验3：泛化性实验")
    logger.info(f"场景：{args.scene_name} | 模型状态：{'✅ 已加载' if has_model else '❌ 未加载'}")
    
    # 确定测试镜像
    test_image_config = {}
    if args.image_list != "default" and args.image_list.strip() != "":
        test_image_config['custom'] = [img.strip() for img in args.image_list.split(',') if img.strip()]
    else:
        test_image_config = IMAGE_CATEGORIES
    
    # 执行实验
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for cat, imgs in test_image_config.items():
            for img in imgs:
                res = run_generalization_test(img, cat, env_state, has_model, r+1)
                all_results.append(res)
                time.sleep(2)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp3_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 结果已保存：{csv_path}")
    
    # 简单汇总打印
    success_df = df[df['success'] == True]
    if not success_df.empty:
        cat_summary = success_df.groupby('image_category')['time_s'].agg(['mean', 'count']).round(2)
        logger.info(f"\n🎉 实验3完成！简单汇总：\n{cat_summary}")

if __name__ == "__main__":
    main()