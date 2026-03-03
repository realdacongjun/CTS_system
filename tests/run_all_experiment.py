#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTS闭环实验总控脚本（适配新版架构）
核心修改：
1. 简化配置加载（适配 model_wrapper 的新逻辑）
2. 统一导入路径（去掉 src. 前缀）
3. 适配 inner_runner.py 的参数逻辑（仅传 --scene-name 和 --config-path）
4. 对齐预压缩文件路径、镜像列表、场景参数
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


# 🔧 核心修正1：正确计算PROJECT_ROOT（适配你的目录结构）
# 假设该脚本在 /root/CTS_system/tests/ 下，PROJECT_ROOT = tests目录
PROJECT_ROOT = Path(__file__).parent.absolute()
# 宿主机的预压缩镜像目录（根目录的data/preprocessed_images）
HOST_PRECOMPRESSED_DATA_DIR = PROJECT_ROOT.parent / "data" / "preprocessed_images"
# 验证预压缩目录是否存在
if not HOST_PRECOMPRESSED_DATA_DIR.exists():
    raise FileNotFoundError(f"预压缩文件目录不存在：{HOST_PRECOMPRESSED_DATA_DIR}，请先执行prepare_images.py")

# 统一添加系统路径
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ======================== 1. 路径配置 ========================
TESTBED_DIR = PROJECT_ROOT / "testbed"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
# 容器内的data目录（同时映射宿主机tests/data和根目录data）
CONTAINER_DATA_DIR = PROJECT_ROOT / "data"

# 创建必要目录
for dir_path in [RESULTS_DIR, LOGS_DIR, CONTAINER_DATA_DIR]:
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
    
    logger.warning("未找到配置文件，使用默认配置（对齐你的实验参数）")
    return {
        "paths": {
            "results_file": str(RESULTS_DIR / "experiment_results.csv"),
            "image_features_csv": str(CONTAINER_DATA_DIR / "image_features_database.csv")
        },
        "experiment": {
            "repeat_times": 3,  # 对齐你的实验轮次（3次）
            "images": ["ubuntu:latest", "nginx:latest", "mysql:latest"],  # 对齐你的镜像列表
            "timeout": 3600  # 单个实验超时时间
        },
        "scenes": {
            # 对齐你的实验场景参数
            "S1_Normal": {"bandwidth": "100mbit", "delay": "10ms", "cpus": 6.0, "memory": "12g"},
            "S2_WeakNet": {"bandwidth": "5mbit", "delay": "100ms", "cpus": 2.0, "memory": "4g"},
            "S3_Robust": {"bandwidth": "1mbit", "delay": "500ms", "cpus": 1.0, "memory": "4g", "packet_loss": 10.0}
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
    packet_loss = 0.0
    if scene_config:
        if 'bandwidth' in scene_config and 'mbit' in scene_config['bandwidth']:
            bandwidth_mbps = float(scene_config['bandwidth'].replace('mbit', ''))
        if 'delay' in scene_config and 'ms' in scene_config['delay']:
            network_rtt = float(scene_config['delay'].replace('ms', ''))
        if 'packet_loss' in scene_config:
            packet_loss = float(scene_config['packet_loss'])
    
    return {
        'bandwidth_mbps': bandwidth_mbps,
        'network_rtt': network_rtt,
        'packet_loss': packet_loss,
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
    except Exception as e:
        logger.warning(f"读取镜像特征失败：{e}")
    return {'total_size_mb': 200.0}

# ======================== 4. 测试矩阵 ========================
def get_test_matrix(config: dict) -> Dict:
    """获取测试场景矩阵"""
    test_matrix = config.get('scenes', {})
    if not test_matrix:
        test_matrix = {
            "S1_Normal": {"cpus": 6.0, "memory": "12g", "bandwidth": "100mbit", "delay": "10ms"}
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
        logger.error(f"构建错误详情：{e.stderr if hasattr(e, 'stderr') else '无'}")
        return False

# def start_experiment_container(scene_name: str, scene_config: Dict, image_name: str = "cts-testbed:latest") -> Optional[str]:
#     """启动实验容器（核心修改：修正预压缩文件挂载路径）"""
#     container_name = f"cts-run-{scene_name}"
#     logger.info(f"启动实验容器：{container_name}")

#     # 先清理旧容器
#     try:
#         subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
#     except Exception as e:
#         logger.warning(f"清理旧容器失败：{e}")

#     try:
#         run_cmd = [
#             "docker", "run", "-d", "--name", container_name,
#             f"--cpus={scene_config['cpus']}", f"--memory={scene_config['memory']}",
#             "--network=host", "--privileged",  # 特权模式解决网络/权限问题
#             # 核心修正1：挂载项目根目录到/cts
#             "-v", f"{PROJECT_ROOT.parent}:/cts",  
#             # 核心修正2：预压缩文件挂载到实验脚本找的路径 /cts/precompressed_images
#             "-v", f"{HOST_PRECOMPRESSED_DATA_DIR}:/cts/precompressed_images",  
#             # 挂载docker sock，确保容器内可执行docker命令
#             "-v", "/var/run/docker.sock:/var/run/docker.sock",  
#             # 环境变量：指定预压缩文件路径
#             "-e", "PRECOMPRESSED_IMAGES_DIR=/cts/precompressed_images",
#             image_name, "sleep", "infinity"
#         ]
def start_experiment_container(scene_name: str, scene_config: Dict, image_name: str = "cts-testbed:latest") -> Optional[str]:
    """启动实验容器（核心修改：修正预压缩文件挂载路径）"""
    container_name = f"cts-run-{scene_name}"
    logger.info(f"启动实验容器：{container_name}")

    # 先清理旧容器
    try:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception as e:
        logger.warning(f"清理旧容器失败：{e}")

    try:
        run_cmd = [
            "docker", "run", "-d", "--name", container_name,
            f"--cpus={scene_config['cpus']}", f"--memory={scene_config['memory']}",
            "--network=host", "--privileged",
            # 🔧 核心修正：使用绝对路径挂载 tests目录到/cts
            "-v", f"{PROJECT_ROOT}:/cts",  
            # 预压缩文件挂载
            "-v", f"{HOST_PRECOMPRESSED_DATA_DIR}:/cts/precompressed_images",  
            "-v", "/var/run/docker.sock:/var/run/docker.sock",  
            "-e", "PRECOMPRESSED_IMAGES_DIR=/cts/precompressed_images",
            image_name, "sleep", "infinity"
        ]

        result = subprocess.run(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, encoding="utf-8")
        logger.info(f"容器启动成功，命令：{' '.join(run_cmd)}")
        time.sleep(2)  # 等待容器完全启动
        return container_name
    except Exception as e:
        logger.error(f"容器启动失败：{e}")
        logger.error(f"启动命令：{' '.join(run_cmd)}")
        logger.error(f"错误详情：{e.stderr if hasattr(e, 'stderr') else '无'}")
        return None



def execute_experiment_in_container(
    container_name: str, 
    scene_name: str,
    config: dict
) -> bool:
    """执行容器内实验（适配inner_runner.py的逻辑：单个场景只执行一次）"""
    logger.info(f"在容器{container_name}中执行场景{scene_name}的所有实验")

    # 🔧 最终终极修正：inner_runner.py 的准确路径
    exec_cmd = [
        "docker", "exec", "--user", "root",
        container_name,
        "python", "/cts/tests/testbed/inner_runner.py",  # ✅ 100%正确路径！
        "--scene-name", scene_name,
        "--config-path", "/cts/tests/configs/config.yaml"  # 确认config.yaml路径也正确
    ]

    try:
        result = subprocess.run(
            exec_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True, 
            encoding="utf-8",
            timeout=10800
        )
        logger.info(f"场景{scene_name}所有实验执行成功，输出摘要：{result.stdout[:500]}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"场景{scene_name}实验执行失败：{e}")
        logger.error(f"执行命令：{' '.join(exec_cmd)}")
        logger.error(f"标准输出：{e.stdout}")
        logger.error(f"错误输出：{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"场景{scene_name}实验执行异常：{e}", exc_info=True)
        return False


def stop_experiment_container(container_name: str) -> bool:
    """停止容器"""
    logger.info(f"停止容器：{container_name}")
    try:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        time.sleep(3)
        return True
    except Exception as e:
        logger.warning(f"停止容器失败：{e}")
        return False

# ======================== 6. 结果汇总 ========================
def collect_and_summary_results(config: dict) -> None:
    """汇总结果"""
    logger.info("开始汇总所有实验结果...")
    all_results = []
    results_file = Path(config['paths']['results_file'])

    # 加载inner_runner生成的场景结果
    for result_file in RESULTS_DIR.glob("*_inner_results.yaml"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                scene_results = yaml.safe_load(f)
                all_results.extend(scene_results)
            logger.info(f"读取场景结果文件：{result_file}，共{len(scene_results)}条记录")
        except Exception as e:
            logger.warning(f"读取结果文件失败：{result_file} - {e}")

    # 加载单个实验结果
    for result_file in RESULTS_DIR.glob("exp*.csv"):
        try:
            df = pd.read_csv(result_file, encoding='utf-8-sig')
            all_results.append(df.to_dict('records'))
            logger.info(f"读取实验结果文件：{result_file}，共{len(df)}条记录")
        except Exception as e:
            logger.warning(f"读取结果文件失败：{result_file} - {e}")

    if all_results:
        # 扁平化结果并保存
        flat_results = []
        for item in all_results:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)
        
        # 保存为CSV
        if flat_results:
            df_merged = pd.DataFrame(flat_results)
            df_merged.to_csv(results_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 所有结果已汇总保存到：{results_file}，共{len(df_merged)}条记录")
        else:
            logger.warning("⚠️ 无有效结果可汇总")
    else:
        logger.warning("⚠️ 未找到任何实验结果文件")

# ======================== 7. 主流程 ========================
def run_all_experiments(skip_build: bool = False, config: dict = None) -> bool:
    """主实验流程（适配inner_runner.py的逻辑）"""
    if config is None:
        config = load_config()
    logger.info(f"📌 实验配置：重复次数={config['experiment']['repeat_times']}，镜像列表={config['experiment']['images']}")

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
        
        # 启动容器
        container_name = start_experiment_container(scene_name, scene_config)
        if not container_name:
            logger.error(f"❌ 场景{scene_name}容器启动失败，跳过该场景")
            all_scenes_succeeded = False
            continue

        # 🔧 核心简化：单个场景只执行一次inner_runner.py（它会自动执行该场景下的所有实验）
        success = execute_experiment_in_container(
            container_name, 
            scene_name,
            config
        )
        if not success:
            logger.error(f"❌ 场景{scene_name}所有实验执行失败")
            all_scenes_succeeded = False
        else:
            logger.info(f"✅ 场景{scene_name}所有实验执行成功")

        # 停止容器
        stop_experiment_container(container_name)

    # 汇总结果
    collect_and_summary_results(config)
    return all_scenes_succeeded

def main():
    parser = argparse.ArgumentParser(description="CTS 实验总控")
    parser.add_argument("--skip-build", action="store_true", help="跳过 Docker 镜像构建")
    parser.add_argument("--repeat-times", type=int, default=None, help="覆盖配置中的重复次数")
    parser.add_argument("--images", type=str, default=None, help="覆盖配置中的镜像列表")
    args = parser.parse_args()

    # 加载并更新配置
    config = load_config()
    if args.repeat_times:
        config['experiment']['repeat_times'] = args.repeat_times
        logger.info(f"覆盖重复次数为：{args.repeat_times}")
    if args.images:
        config['experiment']['images'] = args.images.split(',')
        logger.info(f"覆盖镜像列表为：{config['experiment']['images']}")

    # 执行所有实验
    success = run_all_experiments(skip_build=args.skip_build, config=config)
    
    # 退出码：0成功，1失败
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()




# # 1. 回到 tests 目录
# cd /root/CTS_system/tests/

# # 2. 停止旧进程和容器
# pkill -f run_all_experiment.py 2>/dev/null || true
# docker stop $(docker ps -a | grep cts-run- | awk '{print $1}') 2>/dev/null || true
# docker rm -f $(docker ps -a | grep cts-run- | awk '{print $1}') 2>/dev/null || true

# # 3. 清空旧日志
# rm -f *.log

# # 4. 备份并清空旧 results
# mv /root/CTS_system/tests/results /root/CTS_system/tests/results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
# mkdir -p /root/CTS_system/tests/results

# # 5. 重新运行实验（只跑1轮，快速验证）
# nohup python run_all_experiment.py --skip-build --repeat-times 1 --images ubuntu:latest > full_experiment.log 2>&1 &

# # 6. 查看后台进程
# ps aux | grep run_all

# # 7. 跟踪日志
# tail -f full_experiment.log
