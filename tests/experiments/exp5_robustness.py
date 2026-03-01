#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验5：鲁棒性实验（适配新版 adaptive_downloader）
核心目标：验证CTS在弱网/资源受限场景下的稳定性
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接导入 tests 目录下的 adaptive_downloader
import adaptive_downloader

# ... 后面的代码保持不变 ...

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
        logging.FileHandler(LOG_DIR / "exp5_robustness.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    config_paths = [Path("/cts/config.yaml"), Path("/cts/configs/global_config.yaml")]
    for p in config_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    return {"experiment": {"repeat_times": 3}}

def init_cts_system() -> Tuple[Optional[object], Optional[object]]:
    """初始化引擎（仅用于判断是否有模型，实际决策委托下载器）"""
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
        logger.info("✅ CTS引擎初始化成功")
        return wrapper, engine
    except Exception as e:
        logger.warning(f"⚠️  模型加载失败，将对比固定策略: {e}")
        return None, None

def parse_memory_str(mem_str: str) -> float:
    mem_str = mem_str.strip().lower()
    mem_unit = mem_str[-1]
    mem_size = float(mem_str[:-1])
    unit_map = {'k': 1/1024, 'm': 1, 'g': 1024, 't': 1024*1024}
    return mem_size * unit_map.get(mem_unit, 1)

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
        logger.error(f"获取特征失败: {e}")
    return {'total_size_mb': 100.0}

ROBUSTNESS_SCENARIOS = {
    "weak_net": {"bandwidth": "1mbit", "delay": "500ms", "packet_loss": 5.0},
    "high_loss": {"bandwidth": "10mbit", "delay": "100ms", "packet_loss": 10.0},
    "resource_limit": {"bandwidth": "5mbit", "delay": "100ms", "cpu_limit": 1.0, "mem_limit": "2g"}
}

def build_robust_env_state(base_env: dict, scene_params: dict) -> dict:
    env_state = base_env.copy()
    if 'bandwidth' in scene_params:
        env_state['bandwidth_mbps'] = float(scene_params['bandwidth'].replace('mbit', ''))
    if 'delay' in scene_params:
        env_state['network_rtt'] = float(scene_params['delay'].replace('ms', ''))
    env_state['packet_loss'] = scene_params.get('packet_loss', 0.0)
    if 'cpu_limit' in scene_params:
        env_state['cpu_limit'] = scene_params['cpu_limit']
    if 'mem_limit' in scene_params:
        env_state['mem_limit_mb'] = parse_memory_str(scene_params['mem_limit'])
    return env_state

def run_single_download(
    image: str, 
    mode: str, 
    env_state: dict, 
    features: dict
) -> Tuple[bool, float, str]:
    """🔧 新增：统一单次下载封装"""
    try:
        if mode == "baseline":
            # 基线模式：原生 Docker Pull
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy="baseline",
                threads=1,
                env_state=env_state,
                image_features=features,
                clear_cache=True
            )
        elif mode == "cts_ai":
            # CTS模式：AI 自适应决策
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy=None,  # 触发 AI
                threads=None,
                env_state=env_state,
                image_features=features,
                clear_cache=True
            )
        else: # cts_fixed
            # CTS固定保守策略
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy="zstd_l1",
                threads=2,
                env_state=env_state,
                image_features=features,
                clear_cache=True
            )
        
        return res.get('success', False), res.get('time_s', 0.0), res.get('compression', 'unknown')
    
    except Exception as e:
        return False, 0.0, str(e)[:100]

def run_robustness_test(
    image: str, 
    robust_scenario: str, 
    env_state: dict, 
    has_model: bool, 
    round_num: int
) -> Dict:
    """执行鲁棒性测试（简化逻辑，对比清晰）"""
    logger.info(f"\n🔍 鲁棒性测试 - 轮次{round_num} | 场景：{robust_scenario} | 镜像：{image}")
    
    features = get_image_features_from_csv(image)
    full_env = _complete_env_features(env_state, features)
    
    # 1. 先跑 Baseline
    logger.info(f"  [1/2] 运行 Baseline...")
    baseline_success, baseline_time, _ = run_single_download(image, "baseline", full_env, features)
    
    # 2. 再跑 CTS
    logger.info(f"  [2/2] 运行 CTS...")
    cts_mode = "cts_ai" if has_model else "cts_fixed"
    cts_success, cts_time, cts_algo = run_single_download(image, cts_mode, full_env, features)
    
    logger.info(f"  结果: CTS({cts_success}/{cts_algo}) vs Baseline({baseline_success})")

    return {
        'robust_scenario': robust_scenario,
        'image_name': image,
        'round': round_num,
        'cts_time_s': cts_time,
        'cts_success': cts_success,
        'cts_algo': cts_algo,
        'baseline_time_s': baseline_time,
        'baseline_success': baseline_success
    }

def main():
    parser = argparse.ArgumentParser(description="实验5：鲁棒性实验")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bandwidth", default="5mbit")
    parser.add_argument("--delay", default="100ms")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--cpus", type=float, default=4.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--repeat-times", type=int, default=3)
    # 🔧 修复：接收额外参数
    parser.add_argument("--packet-loss", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--jitter", type=str, default="0ms", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    config = load_config()
    repeat = args.repeat_times or config['experiment']['repeat_times']
    wrapper, engine = init_cts_system()
    has_model = (wrapper is not None and engine is not None)
    images = [img.strip() for img in args.image_list.split(',') if img.strip()]
    
    base_env = {
        'bandwidth_mbps': float(args.bandwidth.replace('mbit', '')) if 'mbit' in args.bandwidth else 5.0,
        'network_rtt': float(args.delay.replace('ms', '')) if 'ms' in args.delay else 100.0,
        'packet_loss': 0.0,
        'cpu_limit': args.cpus,
        'mem_limit_mb': parse_memory_str(args.memory)
    }
    
    logger.info(f"\n🚀 开始实验5：鲁棒性实验")
    logger.info(f"CTS模式: {'✅ AI自适应' if has_model else '⚠️ 固定保守策略'}")
    
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for robust_scene, scene_params in ROBUSTNESS_SCENARIOS.items():
            env_state = build_robust_env_state(base_env, scene_params)
            logger.info(f"\n📌 子场景: {robust_scene}")
            
            for img in images:
                res = run_robustness_test(img, robust_scene, env_state, has_model, r+1)
                all_results.append(res)
                time.sleep(3) # 长间隔避免拥塞
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp5_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 结果已保存：{csv_path}")
    
    # 简单汇总打印
    if not df.empty:
        summary = df.groupby('robust_scenario').agg(
            cts_rate=('cts_success', 'mean'),
            base_rate=('baseline_success', 'mean')
        ).round(4) * 100
        logger.info(f"\n🎉 实验5完成！成功率汇总:\n{summary}")

if __name__ == "__main__":
    main()