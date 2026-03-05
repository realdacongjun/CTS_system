#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 1：预压缩 + 智能选择 vs 固定策略
核心目标：证明在都使用预压缩文件的前提下，智能选择优于固定策略
"""
import os
import sys
import time
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import adaptive_downloader
from model_wrapper import CFTNetWrapper
from cags_decision import CAGSDecisionEngine

# 容器内固定路径
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
        logging.FileHandler(LOG_DIR / "exp1_fixed_vs_adaptive.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    config_paths = [
        Path("/cts/config.yaml"),
        Path("/cts/configs/global_config.yaml"),
    ]
    for p in config_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    return {"experiment": {"repeat_times": 1}}

def init_cts_system(config: dict):
    try:
        model_cfg_path = Path("/cts/configs/model_config.yaml")
        if not model_cfg_path.exists():
            return None, None
        
        wrapper = CFTNetWrapper(str(model_cfg_path))
        engine = CAGSDecisionEngine(
            model=wrapper.model,
            scaler_c=wrapper.scaler_c,
            scaler_i=wrapper.scaler_i,
            enc=wrapper.enc,
            cols_c=wrapper.cols_c,
            cols_i=wrapper.cols_i,
            device=wrapper.device
        )
        logger.info("✅ CTS 决策引擎初始化成功")
        return wrapper, engine
    except Exception as e:
        logger.error(f"❌ 模型加载失败：{e}", exc_info=True)
        return None, None

def get_image_features(image_name: str) -> dict:
    csv_path = Path("/cts/data/image_features_database.csv")
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

def collect_env_state(bandwidth: str, delay: str, cpus: float, memory: str) -> dict:
    bw = float(bandwidth.replace('mbit', '')) if 'mbit' in bandwidth else 100.0
    rtt = float(delay.replace('ms', '')) if 'ms' in delay else 20.0
    
    mem_unit = memory[-1].lower()
    mem_size = float(memory[:-1])
    mem_limit_mb = {'k': mem_size/1024, 'm': mem_size, 'g': mem_size*1024}.get(mem_unit, 8192)
    
    return {
        'bandwidth_mbps': bw,
        'network_rtt': rtt,
        'cpu_limit': cpus,
        'mem_limit_mb': mem_limit_mb
    }

def run_fixed_strategy(image: str, force_algo: str, threads: int, round_num: int) -> Dict:
    """运行固定策略测试（使用预压缩文件）"""
    logger.info(f"\n🔵 固定策略 - 轮次{round_num} | 镜像：{image} | 算法：{force_algo} @ {threads}线程")
    try:
        features = get_image_features(image)
        res = adaptive_downloader.cts_download(
            image_name=image,
            strategy=None,
            threads=threads,
            env_state={},
            image_features=features,
            clear_cache=True,
            use_precompressed=True,
            force_algo=force_algo
        )
        return {
            'strategy': f'Fixed-{force_algo}',
            'round': round_num,
            'image_name': image,
            'time_s': res.get('time_s', 0.0),
            'compression': force_algo,
            'threads': threads,
            'success': res.get('success', False),
            'error_msg': res.get('error', '')[:200]
        }
    except Exception as e:
        logger.error(f"固定策略测试失败：{e}", exc_info=True)
        return {
            'strategy': f'Fixed-{force_algo}',
            'round': round_num,
            'image_name': image,
            'time_s': 0.0,
            'compression': force_algo,
            'threads': threads,
            'success': False,
            'error_msg': str(e)[:200]
        }

def run_adaptive_cts(image: str, env_state: dict, wrapper, engine, round_num: int) -> Dict:
    """运行自适应 CTS 测试（AI 决策）"""
    logger.info(f"\n🟢 自适应 CTS - 轮次{round_num} | 镜像：{image}")
    try:
        features = get_image_features(image)
        
        # 补充特征
        img_size = features.get('total_size_mb', 100.0)
        env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
        env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
        env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
        env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)
        
        res = adaptive_downloader.cts_download(
            image_name=image,
            strategy=None,
            env_state=env_state,
            image_features=features,
            clear_cache=True,
            use_precompressed=True,
            force_algo=None
        )
        return {
            'strategy': 'Full-CTS',
            'round': round_num,
            'image_name': image,
            'time_s': res.get('time_s', 0.0),
            'compression': res.get('compression', 'unknown'),
            'threads': res.get('threads', 0),
            'decision_source': 'AI',
            'success': res.get('success', False),
            'error_msg': res.get('error', '')[:200]
        }
    except Exception as e:
        logger.error(f"自适应 CTS 测试失败：{e}", exc_info=True)
        return {
            'strategy': 'Full-CTS',
            'round': round_num,
            'image_name': image,
            'time_s': 0.0,
            'compression': 'unknown',
            'threads': 0,
            'decision_source': 'AI',
            'success': False,
            'error_msg': str(e)[:200]
        }

def main():
    parser = argparse.ArgumentParser(description="实验 1：固定策略 vs 自适应")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bandwidth", default="100mbit")
    parser.add_argument("--delay", default="10ms")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--cpus", type=float, default=6.0)
    parser.add_argument("--memory", default="12g")
    parser.add_argument("--repeat-times", type=int, default=3)
    
    args = parser.parse_args()

    config = load_config()
    repeat = args.repeat_times or config['experiment'].get('repeat_times', 3)
    images = args.image_list.split(',')
    wrapper, engine = init_cts_system(config)
    env_state = collect_env_state(args.bandwidth, args.delay, args.cpus, args.memory)
    
    logger.info(f"\n🚀 开始实验 1：固定策略 vs 自适应 CTS")
    logger.info(f"场景：{args.scene_name} | 带宽：{args.bandwidth} | 延迟：{args.delay}")
    logger.info(f"镜像列表：{images} | 重复次数：{repeat}")
    
    # 定义固定策略对比组
    fixed_strategies = [
        ('zstd-3', 4),
        ('gzip-6', 4),
        ('lz4-fast', 8),
    ]
    
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for img in images:
            # 运行固定策略
            for algo, threads in fixed_strategies:
                res = run_fixed_strategy(img, algo, threads, r+1)
                all_results.append(res)
                time.sleep(2)
            
            # 运行自适应 CTS
            res = run_adaptive_cts(img, env_state, wrapper, engine, r+1)
            all_results.append(res)
            time.sleep(3)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp1_fixed_vs_adaptive_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 实验结果已保存到：{csv_path}")
    
    # 汇总统计
    summary = {
        "experiment": "exp1_fixed_vs_adaptive",
        "scene": args.scene_name,
        "total_tests": len(all_results),
        "success_count": sum([1 for x in all_results if x['success']]),
        "avg_times": df.groupby('strategy')['time_s'].mean().to_dict(),
        "best_fixed": df[df['strategy'].str.contains('Fixed')].groupby('strategy')['time_s'].mean().min(),
        "avg_cts_time": df[df['strategy']=='Full-CTS']['time_s'].mean()
    }
    
    json_path = csv_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"📋 实验汇总已保存到：{json_path}")
    
    logger.info(f"\n🎉 实验 1 完成！")
    logger.info(f"最佳固定策略平均耗时：{summary['best_fixed']:.2f}s")
    logger.info(f"Full-CTS 平均耗时：{summary['avg_cts_time']:.2f}s")
    if summary['best_fixed'] > 0:
        improvement = (summary['best_fixed'] - summary['avg_cts_time']) / summary['best_fixed'] * 100
        logger.info(f"性能提升：{improvement:.1f}%")

if __name__ == "__main__":
    main()