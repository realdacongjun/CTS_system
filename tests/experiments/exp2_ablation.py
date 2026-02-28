#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验2：消融实验（适配新版 adaptive_downloader）
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
# 🔧 修复：从 testbed 导入，且不再需要单独的 baseline_downloader
from testbed import adaptive_downloader

# 容器内固定路径
LOG_DIR = Path("/cts/logs")
RESULT_DIR = Path("/cts/results")
LOG_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR.mkdir(exist_ok=True, parents=True)

# 日志配置
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "exp2_ablation.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ======================== 核心函数（保持逻辑，微调实现） ========================
def load_config() -> dict:
    config_paths = [Path("/cts/config.yaml"), Path("/cts/configs/global_config.yaml")]
    for p in config_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    return {"experiment": {"repeat_times": 1}}

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

ABLATION_VARIANTS = {
    "full_cts": "完整CTS（CFT-Net + CAGS + 自适应压缩）",
    "no_model": "无CFT-Net（仅规则决策 + 自适应压缩）",
    "no_cags": "无CAGS（CFT-Net + 固定压缩）",
    "no_compress": "无自适应压缩（原生Docker Pull）"
}

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

def init_cts_engine() -> Tuple[Optional[object], Optional[object]]:
    try:
        from model_wrapper import CFTNetWrapper
        from cags_decision import CAGSDecisionEngine
        
        model_cfg_path = Path("/cts/configs/model_config.yaml")
        if not model_cfg_path.exists():
            logger.warning("模型配置文件不存在，full_cts变体将跳过")
            return None, None
        
        wrapper = CFTNetWrapper(str(model_cfg_path))
        engine = CAGSDecisionEngine(
            model=wrapper.model, scaler_c=wrapper.scaler_c, scaler_i=wrapper.scaler_i,
            enc=wrapper.enc, cols_c=wrapper.cols_c, cols_i=wrapper.cols_i, device=wrapper.device
        )
        return wrapper, engine
    except Exception as e:
        logger.error(f"初始化CTS引擎失败: {e}", exc_info=True)
        return None, None

def _complete_env_features(env_state: dict, image_features: dict):
    """🔧 新增：补全物理交叉特征（确保与训练一致）"""
    img_size = image_features.get('total_size_mb', 100.0)
    env_state = env_state.copy()
    env_state['theoretical_time'] = img_size / (env_state.get('bandwidth_mbps', 100) / 8 + 1e-8)
    env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / (img_size + 1e-8)
    env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / (img_size + 1e-8)
    env_state['network_score'] = env_state.get('bandwidth_mbps', 100) / (env_state.get('network_rtt', 10) + 1e-8)
    return env_state

def run_ablation_variant(image: str, variant: str, env_state: dict, round_num: int) -> Dict:
    """执行单个消融变体测试（统一使用 adaptive_downloader）"""
    logger.info(f"\n🔍 消融测试 - 轮次{round_num} | 变体：{variant} | 镜像：{image}")
    
    # 准备特征
    features = get_image_features_from_csv(image)
    full_env = _complete_env_features(env_state, features)
    
    try:
        res = None
        
        if variant == "full_cts":
            # 完整CTS：不传 strategy，让下载器内部调 AI
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy=None,  # 🔧 关键：None 触发 AI 决策
                threads=None,
                env_state=full_env,
                image_features=features,
                clear_cache=True
            )
        
        elif variant == "no_model":
            # 无模型：固定启发式策略
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy="zstd_l3",
                threads=4,
                env_state=full_env,
                image_features=features,
                clear_cache=True
            )
        
        elif variant == "no_cags":
            # 无决策：固定一个较好的压缩参数
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy="zstd_l5",  # 固定压缩
                threads=6,           # 固定线程
                env_state=full_env,
                image_features=features,
                clear_cache=True
            )
        
        elif variant == "no_compress":
            # 无压缩：原生 Docker Pull
            # 🔧 关键：传入 "baseline" 触发新下载器的原生模式
            res = adaptive_downloader.cts_download(
                image_name=image,
                strategy="baseline",
                threads=1,
                env_state=full_env,
                image_features=features,
                clear_cache=True
            )
            # 补充标记
            res['compression'] = "none"

        # 统一返回结构
        return {
            'variant': variant,
            'variant_desc': ABLATION_VARIANTS[variant],
            'image_name': image,
            'round': round_num,
            'time_s': res['time_s'],
            'success': res.get('success', True),
            'error_msg': res.get('error', '')[:200],
            'compression': res.get('compression', 'unknown'),
            'threads': res.get('threads', 0)
        }
    
    except Exception as e:
        logger.error(f"变体{variant}测试失败: {e}", exc_info=True)
        return {
            'variant': variant,
            'variant_desc': ABLATION_VARIANTS[variant],
            'image_name': image,
            'round': round_num,
            'time_s': 0.0,
            'success': False,
            'error_msg': str(e)[:200],
            'compression': 'unknown',
            'threads': 0
        }

# ======================== 主逻辑（保持不变） ========================
def main():
    parser = argparse.ArgumentParser(description="实验2：消融实验")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bandwidth", default="100mbit")
    parser.add_argument("--delay", default="10ms")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cpus", type=float, default=4.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--repeat-times", type=int, default=1)
    # 接收但忽略 inner_runner 传来的额外参数
    parser.add_argument("--packet-loss", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--jitter", type=str, default="0ms", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    config = load_config()
    repeat = args.repeat_times or config['experiment']['repeat_times']
    images = args.image_list.split(',')
    env_state = collect_env_state(args.bandwidth, args.delay, args.cpus, args.memory)
    
    logger.info(f"\n🚀 开始实验2：消融实验")
    logger.info(f"场景：{args.scene_name} | 带宽：{args.bandwidth} | 延迟：{args.delay}")
    logger.info(f"消融变体：{list(ABLATION_VARIANTS.keys())}")
    
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for img in images:
            for variant in ABLATION_VARIANTS.keys():
                res = run_ablation_variant(img, variant, env_state, r+1)
                all_results.append(res)
                time.sleep(2)
    
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp2_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 结果已保存：{csv_path}")
    
    # 汇总分析
    success_df = df[df['success'] == True]
    if not success_df.empty:
        summary = success_df.groupby('variant')['time_s'].agg(['mean', 'std']).reset_index()
        summary['desc'] = summary['variant'].map(ABLATION_VARIANTS)
        
        logger.info(f"\n🎉 实验2完成！核心结论：")
        for _, row in summary.iterrows():
            logger.info(f"{row['desc']}: {row['mean']:.2f}s (±{row['std']:.2f}s)")

if __name__ == "__main__":
    main()