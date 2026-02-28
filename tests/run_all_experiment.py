#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTS闭环实验总控脚本（适配新版架构）
核心修改：
1. 简化配置加载（适配 model_wrapper 的新逻辑）
2. 统一导入路径（去掉 src. 前缀）
3. 保留原有 Docker 编排能力
"""
import os
import sys
import subprocess
import argparse
import logging
import time
import yaml
import json
import psutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======================== 1. 路径配置 ========================
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TESTBED_DIR = PROJECT_ROOT / "testbed"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DATA_DIR = PROJECT_ROOT / "data"

for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ======================== 2. 日志配置 ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "run_all_experiments.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ======================== 3. 核心接口（适配新版） ========================
def load_config() -> dict:
    """加载配置文件（简化逻辑，优先 global_config）"""
    config_paths = [
        PROJECT_ROOT / "config.yaml",
        PROJECT_ROOT / "configs" / "global_config.yaml",
        PROJECT_ROOT / "configs" / "config.yaml"
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"加载配置文件：{config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    logger.warning("未找到配置文件，使用默认配置")
    return {
        "paths": {
            "results_file": str(RESULTS_DIR / "experiment_results.csv"),
            "image_features_csv": str(DATA_DIR / "image_features_database.csv")
        },
        "experiment": {
            "repeat_times": 1,
            "images": ["alpine:latest", "nginx:alpine"],
            "timeout": 3600
        },
        "scenes": {
            "S1_Normal": {"bandwidth": "100mbit", "delay": "10ms", "cpus": 4.0, "memory": "8g"},
            "S2_WeakNet": {"bandwidth": "5mbit", "delay": "100ms", "cpus": 2.0, "memory": "4g"}
        }
    }

def init_cts_system(config: dict) -> Tuple[Optional[object], Optional[object]]:
    """【适配】初始化CTS模型（简化配置加载）"""
    try:
        # 🔧 修改1：统一导入路径，去掉 src. 前缀
        from model_wrapper import CFTNetWrapper
        from cags_decision import CAGSDecisionEngine
        
        logger.info("🧠 正在加载 CTS 模型系统...")
        
        # 🔧 修改2：直接指向 model_config.yaml，不再依赖旧的嵌套配置
        model_cfg_path = CONFIG_DIR / "model_config.yaml"
        
        if not model_cfg_path.exists():
            logger.error(f"❌ 模型配置不存在: {model_cfg_path}")
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
        
        logger.info("✅ CTS 系统加载成功")
        return wrapper, engine
        
    except Exception as e:
        logger.error(f"❌ CTS 系统加载失败: {e}", exc_info=True)
        return None, None

def collect_env_state(scene_config: Dict = None) -> dict:
    """采集环境状态（保持原有逻辑）"""
    bandwidth_mbps = 100.0
    network_rtt = 20.0
    if scene_config:
        if 'bandwidth' in scene_config and 'mbit' in scene_config['bandwidth']:
            bandwidth_mbps = float(scene_config['bandwidth'].replace('mbit', ''))
        if 'delay' in scene_config and 'ms' in scene_config['delay']:
            network_rtt = float(scene_config['delay'].replace('ms', ''))
    
    return {
        'bandwidth_mbps': bandwidth_mbps,
        'network_rtt': network_rtt,
        'packet_loss': 0.0,
        'cpu_limit': float(psutil.cpu_count(logical=True)),
        'mem_limit_mb': float(psutil.virtual_memory().total / 1024 / 1024)
    }

def get_image_features_from_csv(image_name: str, config: dict) -> dict:
    """读取镜像特征（保持原有逻辑）"""
    csv_path = config['paths']['image_features_csv']
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            match = df[df['image_name'].str.contains(image_name.split(':')[0], na=False, case=False)]
            if not match.empty:
                return match.iloc[0].to_dict()
    except:
        pass
    return {'total_size_mb': 200.0}

# ======================== 4. 测试矩阵 ========================
def get_test_matrix(config: dict) -> Dict:
    """获取测试场景矩阵"""
    test_matrix = config.get('scenes', {})
    if not test_matrix:
        test_matrix = {
            "S1_Normal": {"cpus": 4.0, "memory": "8g", "bandwidth": "100mbit", "delay": "10ms"}
        }
    
    for scene_name in test_matrix:
        # 默认执行所有实验（exp1 到 exp5）
        test_matrix[scene_name]['experiments'] = test_matrix[scene_name].get(
            'experiments', 
            ["exp1_baseline_comparison", "exp2_ablation", "exp3_generalization", "exp4_efficiency", "exp5_robustness"]
        )
        test_matrix[scene_name]['timeout'] = test_matrix[scene_name].get('timeout', config['experiment']['timeout'])
        test_matrix[scene_name]['repeat_times'] = config['experiment']['repeat_times']
        test_matrix[scene_name]['images'] = config['experiment']['images']
    
    return test_matrix

# ======================== 5. Docker 编排（保持原有能力） ========================
def build_testbed_image(image_name: str = "cts-testbed:latest") -> bool:
    """构建测试床镜像"""
    dockerfile_path = TESTBED_DIR / "Dockerfile"
    if not dockerfile_path.exists():
        logger.error(f"Dockerfile不存在：{dockerfile_path}")
        return False

    logger.info(f"开始构建测试床镜像：{image_name}")
    try:
        build_cmd = ["docker", "build", "-t", image_name, str(TESTBED_DIR)]
        result = subprocess.run(
            build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, encoding="utf-8", timeout=600
        )
        logger.info(f"测试床镜像构建成功")
        return True
    except Exception as e:
        logger.error(f"测试床镜像构建失败：{e}")
        return False

def start_experiment_container(scene_name: str, scene_config: Dict, image_name: str = "cts-testbed:latest") -> Optional[str]:
    """启动实验容器"""
    container_name = f"cts-run-{scene_name}"
    logger.info(f"启动实验容器：{container_name}")

    try:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except:
        pass

    try:
        run_cmd = [
            "docker", "run", "-d", "--name", container_name,
            f"--cpus={scene_config['cpus']}", f"--memory={scene_config['memory']}",
            "--network=host", "--privileged",
            "-v", f"{PROJECT_ROOT}:/cts",
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            image_name, "sleep", "infinity"
        ]
        result = subprocess.run(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, encoding="utf-8")
        logger.info(f"容器启动成功")
        time.sleep(2)
        return container_name
    except Exception as e:
        logger.error(f"容器启动失败：{e}")
        return None

def execute_experiment_in_container(container_name: str, experiment_name: str, scene_config: Dict, config: dict) -> bool:
    """执行容器内实验（适配新的实验脚本命名）"""
    logger.info(f"在容器中执行实验：{experiment_name}")

    # 🔧 修改：确保实验脚本路径正确（指向 experiments/ 目录）
    script_path = f"/cts/experiments/{experiment_name}.py"
    
    # 构造命令
    exec_cmd = [
        "docker", "exec", container_name,
        "python", script_path,
        "--scene-name", scene_config.get("scene_name", scene_name),
        "--bandwidth", scene_config["bandwidth"],
        "--delay", scene_config["delay"],
        "--cpus", str(scene_config["cpus"]),
        "--memory", scene_config["memory"],
        "--image-list", ",".join(scene_config["images"]),
        "--repeat-times", str(scene_config["repeat_times"])
    ]

    try:
        result = subprocess.run(
            exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, encoding="utf-8",
            timeout=scene_config["timeout"]
        )
        logger.info(f"实验执行成功")
        return True
    except Exception as e:
        logger.error(f"实验执行失败：{e}")
        return False

def stop_experiment_container(container_name: str) -> bool:
    """停止容器"""
    logger.info(f"停止容器：{container_name}")
    try:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        time.sleep(3)
        return True
    except:
        return False

# ======================== 6. 结果汇总 ========================
def collect_and_summary_results(config: dict) -> None:
    """汇总结果"""
    logger.info("开始汇总所有实验结果...")
    all_results = []
    results_file = Path(config['paths']['results_file'])

    for result_file in RESULTS_DIR.glob("exp*.csv"):
        try:
            df = pd.read_csv(result_file, encoding='utf-8-sig')
            all_results.append(df)
        except Exception as e:
            logger.warning(f"读取结果文件失败：{e}")

    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存到：{results_file}")

# ======================== 7. 主流程 ========================
def run_all_experiments(skip_build: bool = False) -> bool:
    """主实验流程"""
    config = load_config()
    logger.info(f"📌 实验配置：重复次数={config['experiment']['repeat_times']}")

    if not skip_build:
        if not build_testbed_image():
            return False
    else:
        logger.info("跳过镜像构建")

    test_matrix = get_test_matrix(config)
    logger.info(f"实验场景：{list(test_matrix.keys())}")

    all_scenes_succeeded = True
    for scene_name, scene_config in test_matrix.items():
        logger.info(f"\n======= 场景：{scene_name} =======")
        scene_config['scene_name'] = scene_name
        
        container_name = start_experiment_container(scene_name, scene_config)
        if not container_name:
            all_scenes_succeeded = False
            continue

        for experiment_name in scene_config['experiments']:
            success = execute_experiment_in_container(
                container_name, experiment_name, scene_config, config
            )
            if not success:
                all_scenes_succeeded = False

        stop_experiment_container(container_name)

    collect_and_summary_results(config)
    return all_scenes_succeeded

def main():
    parser = argparse.ArgumentParser(description="CTS 实验总控")
    parser.add_argument("--skip-build", action="store_true", help="跳过 Docker 镜像构建")
    parser.add_argument("--repeat-times", type=int, default=None, help="覆盖配置中的重复次数")
    parser.add_argument("--images", type=str, default=None, help="覆盖配置中的镜像列表")
    args = parser.parse_args()

    config = load_config()
    if args.repeat_times:
        config['experiment']['repeat_times'] = args.repeat_times
    if args.images:
        config['experiment']['images'] = args.images.split(',')

    success = run_all_experiments(skip_build=args.skip_build)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()