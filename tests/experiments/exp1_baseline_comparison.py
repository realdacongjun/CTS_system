#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验1：基线对比实验
核心目标：验证CTS自适应下载相对原生Docker Pull的性能提升
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

# 把 tests 目录（/cts）加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
# 🔧 新增：把 src 目录也加到 Python 路径，这样就能找到 model_wrapper 和 cags_decision
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 现在可以正常导入了
import adaptive_downloader
import baseline_downloader
from model_wrapper import CFTNetWrapper
from cags_decision import CAGSDecisionEngine # 修复：是cags_decision不是decision_engine
# ======================== 关键修改1：适配容器内路径 ========================
# 容器内固定路径（无需计算PROJECT_ROOT）
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
        logging.FileHandler(LOG_DIR / "exp1_baseline_comparison.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ======================== 复用原有核心函数（修复路径） ========================
def load_config() -> dict:
    """加载配置（适配容器内路径）"""
    config_paths = [
        Path("/cts/config.yaml"),
        Path("/cts/configs/global_config.yaml"),
        Path("/cts/configs/config.yaml")
    ]
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    # 默认配置（容器内）
    return {
        "paths": {"results_file": "/cts/results/exp1_temp.csv"},
        "experiment": {"repeat_times": 1}
    }



def init_cts_system(config: dict) -> Tuple[Optional[object], Optional[object]]:
    """初始化CTS决策引擎（修复导入路径）"""
    try:
        # 直接导入（容器内已添加src到PYTHONPATH）

        
        model_cfg_path = Path("/cts/configs/model_config.yaml")
        if not model_cfg_path.exists():
            logger.warning("模型配置文件不存在，使用默认策略")
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
        logger.info("✅ CTS决策引擎初始化成功")
        return wrapper, engine
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
        return None, None

def get_image_features_from_csv(image_name: str) -> dict:
    """获取镜像特征（适配容器内路径）"""
    csv_path = Path("/cts/data/image_features_database.csv")
    try:
        if not csv_path.exists():
            logger.warning(f"镜像特征库不存在，使用默认值: {csv_path}")
            return {'total_size_mb': 100.0}
        
        df = pd.read_csv(csv_path)
        # 模糊匹配镜像名（兼容tag）
        image_base = image_name.split(':')[0]
        match = df[df['image_name'].str.contains(image_base, na=False, case=False)]
        
        if match.empty:
            logger.warning(f"未找到镜像{image_name}的特征，使用默认值")
            return {'total_size_mb': 100.0}
        
        return match.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"获取镜像特征失败: {e}")
        return {'total_size_mb': 100.0}

# experiments/exp1_baseline_comparison.py
# ... (前面的import保持不变) ...

# 🔧 修复2：修正导入路径，从 testbed 导入，而非 tests
from testbed import adaptive_downloader, baseline_downloader
from model_wrapper import CFTNetWrapper
from cags_decision import CAGSDecisionEngine

# ... (LOG_DIR, RESULT_DIR, logger配置保持不变) ...

def cts_strategy_selector(image: str, env_state: dict, wrapper, engine) -> Tuple[str, int]:
    """CTS策略选择（核心逻辑保留，移除强制覆盖）"""
    if wrapper is None or engine is None:
        logger.warning("使用默认策略：zstd_l3, 4线程")
        return "zstd_l3", 4
    
    try:
        image_features = get_image_features_from_csv(image)
        img_size = image_features.get('total_size_mb', 100)
        
        env_state['theoretical_time'] = img_size / (env_state['bandwidth_mbps'] / 8 + 1e-8)
        env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / img_size
        env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / img_size
        env_state['network_score'] = env_state['bandwidth_mbps'] / (env_state['network_rtt'] + 1e-8)
        
        config_dict, metrics_dict, _ = engine.make_decision(
            env_state=env_state, 
            image_features=image_features, 
            safety_threshold=0.7,  # 🔧 修复：提高到0.7，更保守
            enable_uncertainty=True
        )
        
        algo_name = config_dict.get('algo_name', 'zstd_l3')
        threads = config_dict.get('threads', 4)
        
        # 🔧 修复3：移除强制覆盖逻辑，直接使用决策结果
        # final_method = 'gzip' if algo_name == 'gzip' else 'zstd_l3' <-- 删除这行
        final_method = algo_name
        
        logger.info(f"📌 决策结果：{image} → 算法={final_method}, 线程={threads}")
        return final_method, threads
    except Exception as e:
        logger.error(f"❌ 决策出错，使用默认策略: {e}", exc_info=True)
        return "zstd_l3", 4

def run_cts(image: str, env_state: dict, wrapper, engine, round_num: int) -> Dict:
    """运行CTS自适应测试"""
    logger.info(f"\n🟢 CTS测试 - 轮次{round_num} | 镜像：{image}")
    try:
        method, threads = cts_strategy_selector(image, env_state, wrapper, engine)
        features = get_image_features_from_csv(image)
        
        # 🔧 修复4：传递正确的参数给下载器
        res = adaptive_downloader.cts_download(
            image_name=image, 
            strategy=method,      # 传入决策出的算法
            threads=threads,       # 传入决策出的线程数
            env_state=env_state,
            image_features=features, 
            clear_cache=True
        )
        return {
            'strategy': 'cts', 
            'round': round_num, 
            'image_name': image,
            'time_s': res['time_s'], 
            'decision_algo': method, 
            'decision_threads': threads,
            'success': res['success'], 
            'error_msg': res.get('error', '')[:200]
        }
    except Exception as e:
        logger.error(f"CTS测试失败: {e}", exc_info=True)
        return {
            'strategy': 'cts', 
            'round': round_num, 
            'image_name': image,
            'time_s': 0.0, 
            'decision_algo': 'zstd_l3', 
            'decision_threads': 6,
            'success': False, 
            'error_msg': str(e)[:200]
        }



    # ... (后续 main 函数逻辑保持不变) ...

def collect_env_state(bandwidth: str, delay: str, cpus: float, memory: str) -> dict:
    """采集环境状态（修复：兼容传入的CPU/内存参数）"""
    # 解析网络参数
    bandwidth_mbps = float(bandwidth.replace('mbit', '')) if 'mbit' in bandwidth else 100.0
    network_rtt = float(delay.replace('ms', '')) if 'ms' in delay else 20.0
    
    # 解析内存参数（如8g → 8192mb）
    mem_unit = memory[-1].lower()
    mem_size = float(memory[:-1])
    mem_limit_mb = {
        'k': mem_size / 1024,
        'm': mem_size,
        'g': mem_size * 1024,
        't': mem_size * 1024 * 1024
    }.get(mem_unit, 8192)
    
    # 容器内资源采集（兼容传入参数）
    return {
        'bandwidth_mbps': bandwidth_mbps,
        'network_rtt': network_rtt,
        'packet_loss': 0.0,  # 可从命令行扩展
        'cpu_limit': cpus,  # 使用传入的CPU限制（容器内psutil不准）
        'mem_limit_mb': mem_limit_mb,  # 使用传入的内存限制
        'cpu_available': 100.0 - psutil.cpu_percent(interval=0.1),
        'mem_available': float(psutil.virtual_memory().available / 1024 / 1024)
    }

# 在 experiments/exp1_baseline_comparison.py 中

def run_baseline(image: str, round_num: int) -> Dict:
    """运行基线测试（调用 adaptive_downloader 的原生模式）"""
    logger.info(f"\n🔵 基线测试 - 轮次{round_num} | 镜像：{image}")
    try:
        # 🔧 关键修改：传入 strategy="baseline"，让下载器内部走 docker pull
        res = adaptive_downloader.cts_download(
            image_name=image, 
            strategy="baseline",  # 触发原生模式
            threads=1,
            env_state={}, 
            image_features={},
            clear_cache=True
        )
        return {
            'strategy': 'baseline', 
            'round': round_num, 
            'image_name': image,
            'time_s': res['time_s'], 
            'success': res['success'], 
            'error_msg': res.get('error', '')[:200]
        }
    except Exception as e:
        logger.error(f"基线测试失败: {e}", exc_info=True)
        return {
            'strategy': 'baseline', 
            'round': round_num, 
            'image_name': image,
            'time_s': 0.0, 
            'success': False, 
            'error_msg': str(e)[:200]
        }



def main():
    parser = argparse.ArgumentParser(description="实验1：基线对比")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bandwidth", default="100mbit")
    parser.add_argument("--delay", default="10ms")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cpus", type=float, default=4.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--repeat-times", type=int, default=1)
    # 🔧 修复5：接收 inner_runner 传来的但我们不用的参数，防止报错
    parser.add_argument("--packet-loss", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--jitter", type=str, default="0ms", help=argparse.SUPPRESS)
    parser.add_argument("--model-config-path", type=str, default=None, help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    # 初始化
    config = load_config()
    # 关键修复：命令行参数优先级 > 配置文件
    repeat = args.repeat_times or config['experiment']['repeat_times']
    images = args.image_list.split(',')
    wrapper, engine = init_cts_system(config)
    env_state = collect_env_state(args.bandwidth, args.delay, args.cpus, args.memory)
    
    logger.info(f"\n🚀 开始实验1：基线对比")
    logger.info(f"场景：{args.scene_name} | 带宽：{args.bandwidth} | 延迟：{args.delay}")
    logger.info(f"镜像列表：{images} | 重复次数：{repeat}")
    
    # 执行实验
    all_results = []
    for r in range(repeat):
        logger.info(f"\n==================== 轮次 {r+1}/{repeat} ====================")
        for img in images:
            # 运行基线测试
            bl_res = run_baseline(img, r+1)
            all_results.append(bl_res)
            time.sleep(2)  # 间隔避免资源竞争
            
            # 运行CTS测试
            cts_res = run_cts(img, env_state, wrapper, engine, r+1)
            all_results.append(cts_res)
            time.sleep(3)  # 间隔避免资源竞争
    
    # 保存结果（适配容器内路径）
    df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / f"exp1_{args.scene_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"📊 实验结果已保存到CSV：{csv_path}")
    
    # 输出汇总JSON（替换yaml为json，更通用）
    json_res = {
        "experiment": "exp1_baseline_comparison",
        "scene": args.scene_name,
        "bandwidth": args.bandwidth,
        "delay": args.delay,
        "total_tests": len(all_results),
        "success": sum([1 for x in all_results if x['success']]),
        "avg_baseline_time": df[df['strategy']=='baseline']['time_s'].mean(),
        "avg_cts_time": df[df['strategy']=='cts']['time_s'].mean(),
        # 避免除以0
        "acceleration_ratio": (df[df['strategy']=='baseline']['time_s'].mean() / df[df['strategy']=='cts']['time_s'].mean()) 
                              if df[df['strategy']=='cts']['time_s'].mean() > 0 else 0
    }
    json_path = csv_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_res, f, ensure_ascii=False, indent=2)
    logger.info(f"📋 实验汇总已保存到JSON：{json_path}")
    
    # 打印总结
    logger.info(f"\n🎉 实验1完成！")
    logger.info(f"基线平均耗时：{json_res['avg_baseline_time']:.2f}s")
    logger.info(f"CTS平均耗时：{json_res['avg_cts_time']:.2f}s")
    logger.info(f"性能提升：{json_res['acceleration_ratio']:.2f}倍")

if __name__ == "__main__":
    main()