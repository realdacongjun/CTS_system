#!/usr/bin/env python3
import os
import sys
import time
import yaml
import csv
import logging
import pandas as pd
from pathlib import Path

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入我们的模块
import baseline_downloader
import adaptive_downloader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 【核心】全局初始化 CTS 模型
# ==========================================
def init_cts_system(config: dict):
    """初始化 CFTNetWrapper 和 CAGSDecisionEngine"""
    try:
        from src.model_wrapper import CFTNetWrapper
        from src.decision_engine import CAGSDecisionEngine
        
        logger.info("🧠 正在加载 CTS 模型系统...")
        
        # 【修复】明确传入 model_config.yaml 的路径
        model_cfg_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        wrapper = CFTNetWrapper(str(model_cfg_path))
        
        # 2. 初始化决策引擎
        engine = CAGSDecisionEngine(
            model=wrapper.model,
            scaler_c=wrapper.scaler_c,
            scaler_i=wrapper.scaler_i,
            enc=wrapper.enc,
            cols_c=wrapper.cols_c,
            cols_i=wrapper.cols_i,
            device=wrapper.device
        )
        
        logger.info("✅ CTS 系统加载成功")
        return wrapper, engine
        
    except Exception as e:
        logger.error(f"❌ CTS 系统加载失败: {e}", exc_info=True)
        logger.warning("⚠️  将回退到简单规则策略")
        return None, None

def load_config():
    # 【修复】明确指向 configs/ 目录
    # 同时支持 config.yaml (新) 和 global_config.yaml (旧)
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / "global_config.yaml"
        
    if not config_path.exists():
        # 回退到 configs/config.yaml
        config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_image_features_from_csv(image_name: str, csv_path: str = "./data/image_features_database.csv") -> dict:
    """
    【辅助】从 CSV 读取镜像特征
    因为在本地测试中，我们需要 image_features 输入给模型
    """
    try:
        import pandas as pd
        if not Path(csv_path).exists():
            return {'total_size_mb': 100.0} # 默认值
            
        df = pd.read_csv(csv_path)
        # 简单匹配：找包含 image_name 的行
        # 注意：这里需要根据你的实际 CSV 格式调整匹配逻辑
        match = df[df['image_name'].str.contains(image_name.split(':')[0], na=False)]
        if not match.empty:
            return match.iloc[0].to_dict()
        return {'total_size_mb': 100.0}
    except:
        return {'total_size_mb': 100.0}

def cts_strategy_selector(
    image: str, 
    env_state: dict, 
    wrapper, 
    engine, 
    config: dict
) -> tuple:
    """
    【核心】使用 CFT-Net + CAGS 进行决策
    """
    if wrapper is None or engine is None:
        # 模型加载失败，回退到简单规则
        return "zstd_l3", 6

    try:
        # 1. 获取镜像特征 (从 CSV 或数据库)
        image_features = get_image_features_from_csv(image)
        
        # 2. 补充 env_state (如果缺少某些字段)
        # 确保有模型需要的物理交叉特征
        img_size = image_features.get('total_size_mb', 100)
        if 'theoretical_time' not in env_state:
            env_state['theoretical_time'] = img_size / (env_state['bandwidth_mbps'] / 8 + 1e-8)
        if 'cpu_to_size_ratio' not in env_state:
            env_state['cpu_to_size_ratio'] = env_state.get('cpu_limit', 4.0) / img_size
        if 'mem_to_size_ratio' not in env_state:
            env_state['mem_to_size_ratio'] = env_state.get('mem_limit_mb', 8192) / img_size
        if 'network_score' not in env_state:
            env_state['network_score'] = env_state['bandwidth_mbps'] / (env_state['network_rtt'] + 1e-8)

        # 3. CAGS 决策
        config_dict, metrics_dict, _ = engine.make_decision(
            env_state=env_state,
            image_features=image_features,
            safety_threshold=0.5,
            enable_uncertainty=True
        )
        
        # 4. 映射回我们的文件结构
        # config_dict['algo_name'] 可能是 'gzip', 'zstd'
        # 我们需要把它映射到 'gzip', 'zstd_l3', 'zstd_l10'
        algo_name = config_dict.get('algo_name', 'zstd')
        threads = config_dict.get('threads', 4)
        
        # 简单的映射逻辑 (根据你的实际情况调整)
        if algo_name == 'gzip':
            final_method = 'gzip'
        elif algo_name == 'zstd':
            # 可以根据不确定性进一步选择 level
            final_method = 'zstd_l3' # 或根据 metrics_dict 动态选 l3/l10
        else:
            final_method = 'zstd_l3'
            
        logger.info(f"   🤖 CTS决策: Algo={final_method}, Threads={threads}, Unc={metrics_dict.get('epistemic_unc', 0):.3f}")
        
        return final_method, threads

    except Exception as e:
        logger.error(f"   ❌ CTS决策出错: {e}", exc_info=False)
        return "zstd_l3", 6 # 出错回退

def collect_env_state() -> dict:
    """
    【辅助】采集当前环境状态
    在真实场景中，这由 EnvironmentController 完成
    这里我们模拟一些固定值或简单测量
    """
    import psutil
    # 这里简化处理，实际应该用 EnvironmentController
    return {
        'bandwidth_mbps': 100.0,  # 可以设为变量，模拟不同场景
        'network_rtt': 20.0,
        'packet_loss': 0.0,
        'cpu_limit': float(psutil.cpu_count(logical=True)),
        'mem_limit_mb': float(psutil.virtual_memory().total / 1024 / 1024),
        'cpu_available': 100.0 - psutil.cpu_percent(interval=0.1),
        'mem_available': float(psutil.virtual_memory().available / 1024 / 1024)
    }

def main():
    print("🚀 CTS 完整实验运行器 (含模型推理)")
    print("="*60)
    
    config = load_config()
    results_file = Path(config['paths']['results_file'])
    repeat = config['experiment']['repeat_times']
    images = config['images']
    
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 【核心】初始化模型
    wrapper, engine = init_cts_system(config)
    
    # 采集一次环境状态
    env_state = collect_env_state()
    logger.info(f"当前环境状态: {env_state}")
    
    all_results = []
    
    print(f"\n⚠️  请确保 proxy_server.py 正在另一个终端运行！")
    input("按 Enter 键继续...")
    
    for r in range(repeat):
        logger.info(f"\n🔄 实验轮次 {r+1}/{repeat}")
        
        for image in images:
            logger.info(f"\n🧪 测试镜像: {image}")
            
            # --- 1. 运行 Baseline ---
            logger.info("   运行 Baseline...")
            try:
                bl_res = baseline_downloader.pull_with_docker(image)
                bl_res['round'] = r + 1
                all_results.append(bl_res)
                logger.info(f"      完成: {bl_res['time_s']:.2f}s")
            except Exception as e:
                logger.error(f"      Baseline 失败: {e}")
            
            time.sleep(2)
            
            # --- 2. 运行 CTS (含模型决策) ---
            logger.info("   运行 CTS (通过 API)...")
            
            # 【核心】调用模型决策
            cts_method, cts_threads = cts_strategy_selector(
                image, env_state, wrapper, engine, config
            )
            
            # 获取镜像特征用于API调用
            image_features = get_image_features_from_csv(image)
            
            try:
                cts_res = adaptive_downloader.cts_download(
                    image, 
                    strategy=None,      # 设为 None，完全由 API 决定
                    threads=None,       # 设为 None，完全由 API 决定
                    env_state=env_state, # 传入环境状态
                    image_features=image_features, # 传入镜像特征
                    clear_cache=True
                )
                cts_res['round'] = r + 1
                # 记录决策细节
                cts_res['decision_algo'] = cts_method
                cts_res['decision_threads'] = cts_threads
                all_results.append(cts_res)
                logger.info(f"      完成: {cts_res['time_s']:.2f}s")
            except Exception as e:
                logger.error(f"      CTS 失败: {e}")
            
            time.sleep(3)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_csv(results_file, index=False, encoding='utf-8-sig')
    logger.info(f"\n💾 结果已保存至: {results_file}")
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 最终结果")
    print("="*60)
    if 'time_s' in df.columns and 'success' in df.columns:
        summary = df[df['success']].groupby(['strategy'])['time_s'].mean()
        print(summary)

if __name__ == "__main__":
    main()