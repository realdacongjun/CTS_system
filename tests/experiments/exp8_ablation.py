#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 3：消融实验
核心目标：定量分析 CFT-Net、CAGS、不确定性、预压缩等各模块的贡献
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import adaptive_downloader
from model_wrapper import CFTNetWrapper
from cags_decision import CAGSDecisionEngine

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
        logging.FileHandler(LOG_DIR / "exp3_ablation.log", encoding="utf-8")
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

def rule_based_decision(env_state: dict) -> Tuple[str, int]:
    """基于规则的决策"""
    bw = env_state.get('bandwidth_mbps', 100)
    if bw < 10:
        return 'zstd-6', 8
    elif bw > 500:
        return 'lz4-fast', 16
    else:
        return 'zstd-3', 4

def run_ablation_variant(image: str, env_state: dict, variant: str, 
                        wrapper, engine, round_num: int) -> Dict:
    """运行不同消融变体的测试"""
    logger.info(f"\n🔍 消融变体：{variant} - 轮次{round_num} | 镜像：{image}")
    
    try:
        features = get_image_features(image)
        
        # 补充特征
        img_size = features.get('total_size_mb', 100.0)
        env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
        env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
        env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
        env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)
        
        algo = 'unknown'
        threads = 0
        
        if variant == 'Full-CTS':
            # 完整系统（CFT-Net + CAGS + 预压缩）
            res = adaptive_downloader.cts_download(
                image_name=image, strategy=None, env_state=env_state,
                image_features=features, clear_cache=True,
                use_precompressed=True, force_algo=None
            )
            algo = res.get('compression', 'unknown')
            threads = res.get('threads', 0)
            
        elif variant == 'No-CFT-Net':
            # 无 CFT-Net：用规则预测替代
            algo, threads = rule_based_decision(env_state)
            res = adaptive_downloader.cts_download(
                image_name=image, strategy=None, threads=threads,
                env_state=env_state, image_features=features,
                clear_cache=True, use_precompressed=True, force_algo=algo
            )
            
        elif variant == 'No-CAGS':
            # 无 CAGS：只用 CFT-Net 预测 + 贪心选择（选 pred_time 最小的）
            if wrapper and engine:
                pred_df = engine._predict_batch(env_state, features)
                best = pred_df.loc[pred_df['pred_time'].idxmin()]
                algo = best['algo_name']
                threads = int(best['threads'])
            else:
                algo, threads = 'zstd-3', 4
            
            res = adaptive_downloader.cts_download(
                image_name=image, strategy=None, threads=threads,
                env_state=env_state, image_features=features,
                clear_cache=True, use_precompressed=True, force_algo=algo
            )
            
        elif variant == 'No-Uncertainty':
            # 无不确定性过滤：关闭 safety_threshold
            if wrapper and engine:
                config_dict, _, _ = engine.make_decision(
                    env_state=env_state,
                    image_features=features,
                    safety_threshold=1e6,  # 🔧 不 filtering
                    enable_uncertainty=False
                )
                algo = config_dict['algo_name']
                threads = config_dict['threads']
            else:
                algo, threads = 'zstd-3', 4
            
            res = adaptive_downloader.cts_download(
                image_name=image, strategy=None, threads=threads,
                env_state=env_state, image_features=features,
                clear_cache=True, use_precompressed=True, force_algo=algo
            )
            
        elif variant == 'No-DPC':
            # 无动态帕累托坍缩：固定使用 TRANSITION 模式
            if wrapper and engine:
                config_dict, metrics, _ = engine.make_decision(
                    env_state=env_state,
                    image_features=features,
                    safety_threshold=0.5,
                    enable_dpc=False  # 🔧 禁用 DPC
                )
                algo = config_dict['algo_name']
                threads = config_dict['threads']
            else:
                algo, threads = 'zstd-3', 4
            
            res = adaptive_downloader.cts_download(
                image_name=image, strategy=None, threads=threads,
                env_state=env_state, image_features=features,
                clear_cache=True, use_precompressed=True, force_algo=algo
            )
            
        elif variant == 'Fixed-Strategy':
            # 固定策略基线
            algo, threads = 'zstd-3', 4
            res = adaptive_downloader.cts_download(
                image_name=image, strategy=None, threads=threads,
                env_state=env_state, image_features=features,
                clear_cache=True, use_precompressed=True, force_algo=algo
            )
        
        return {
            'variant': variant,
            'round': round_num,
            'image_name': image,
            'time_s': res.get('time_s', 0.0),
            'compression': algo,
            'threads': threads,
            'success': res.get('success', False),
            'error_msg': res.get('error', '')[:200],
            'decision_source': variant
        }
        
    except Exception as e:
        logger.error(f"{variant}测试失败：{e}", exc_info=True)
        return {
            'variant': variant,
            'round': round_num,
            'image_name': image,
            'time_s': 0.0,
            'compression': 'unknown',
            'threads': 0,
            'success': False,
            'error_msg': str(e)[:200],
            'decision_source': variant
        }

def main():
    parser = argparse.ArgumentParser(description="实验 3：消融实验")
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
    
    logger.info(f"\n🚀 开始实验 3：消融实验")
    logger.info(f"场景：{args.scene_name} | 带宽：{args.bandwidth} | 延迟：{args.delay}")
    logger.info(f"镜像列表：{images} | 重复次数：{repeat}")
    
    # 定义消融变体
    ablation_variants = [
        'Full-CTS',       # 完整系统（基准）
        'No-CFT-Net',     # 无 CFT-Net（用规则替代）
        'No-CAGS',        # 无 CAGS（贪心选择）
        'No-Uncertainty', # 无不确定性过滤
        'No-DPC',         # 无动态帕累托坍缩
        'Fixed-Strategy', # 固定策略基线
    ]
    
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for img in images:
            for variant in ablation_variants:
                res = run_ablation_variant(img, env_state, variant, wrapper, engine, r+1)
                all_results.append(res)
                time.sleep(2)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp3_ablation_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 实验结果已保存到：{csv_path}")
    
    # 汇总统计（以 Full-CTS 为基准计算相对性能）
    full_cts_time = df[df['variant']=='Full-CTS']['time_s'].mean()
    
    summary = {
        "experiment": "exp3_ablation",
        "scene": args.scene_name,
        "variants": ablation_variants,
        "avg_times": df.groupby('variant')['time_s'].mean().to_dict(),
        "std_times": df.groupby('variant')['time_s'].std().to_dict(),
        "relative_performance": {},
        "baseline_time": full_cts_time
    }
    
    # 计算相对性能（相对于 Full-CTS）
    for variant in ablation_variants:
        avg_time = df[df['variant']==variant]['time_s'].mean()
        if full_cts_time > 0:
            rel_perf = (full_cts_time - avg_time) / full_cts_time * 100
            summary['relative_performance'][variant] = round(rel_perf, 2)
    
    json_path = csv_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"📋 实验汇总已保存到：{json_path}")
    
    logger.info(f"\n🎉 实验 3 完成！")
    logger.info(f"基准时间 (Full-CTS): {full_cts_time:.2f}s")
    for variant in ablation_variants:
        avg_time = summary['avg_times'].get(variant, 0)
        rel_perf = summary['relative_performance'].get(variant, 0)
        logger.info(f"{variant:15s}: {avg_time:6.2f}s (相对性能：{rel_perf:+5.1f}%)")

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

if __name__ == "__main__":
    main()