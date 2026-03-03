#!/usr/bin/env python3
"""
容器内实验执行器 - 最终版（适配10种压缩策略+你的model_config）
"""
import os
import sys
import logging
import subprocess
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# 容器内CTS根目录
CTS_ROOT = Path("/cts")
sys.path.append(str(CTS_ROOT))

# 日志配置（提前创建logs目录，适配model_config的日志路径）
LOG_DIR = CTS_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True, mode=0o755)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "inner_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inner_runner")

def check_compression_dependencies() -> bool:
    """
    关键修复：检查10种压缩策略所需的工具是否安装
    必须：gzip(默认)、lz4、zstd、brotli
    """
    required_tools = {
        "gzip": "gzip --version",
        "lz4": "lz4 --version",
        "zstd": "zstd --version",
        "brotli": "brotli --version"
    }
    missing_tools = []
    
    for tool, cmd in required_tools.items():
        try:
            subprocess.run(
                cmd.split(), capture_output=True, timeout=10, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error(f"缺少压缩工具：{', '.join(missing_tools)}，请在容器内执行：")
        logger.error("apt update && apt install -y gzip lz4 zstd brotli")
        return False
    logger.info("所有压缩工具依赖检查通过")
    return True

def load_scene_config(config_path: str, scene_name: str) -> Dict[str, Any]:
    """加载指定场景的配置（兼容全局config.yaml结构）"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 兼容两种场景配置写法（name/scene_name）
        for scene_key, scene in config.get("scenes", {}).items():
            if scene.get("name") == scene_name or scene_key == scene_name:
                # 补充全局实验配置
                scene["image_list"] = config["experiment"]["images"]
                scene["repeat_times"] = config["experiment"]["repeat_times"]
                scene["timeout"] = scene.get("timeout", config["experiment"]["timeout"])
                scene["scene_name"] = scene_name
                # 补充默认值（避免KeyError）
                scene["packet_loss"] = scene.get("packet_loss", 0.0)
                scene["jitter"] = scene.get("jitter", "0ms")  # 网络抖动（鲁棒性实验用）
                return scene
        
        logger.error(f"场景配置不存在: {scene_name}")
        return {}
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return {}

def setup_network(scene_cfg: Dict[str, Any]) -> bool:
    """设置容器内网络限制（修复丢包率/抖动处理）"""
    try:
        # 清理原有规则（忽略错误）
        subprocess.run(
            ["tc", "qdisc", "del", "dev", "eth0", "root"],
            capture_output=True,
            timeout=30,
            check=False
        )
        
        # 提取网络参数并验证
        bandwidth = scene_cfg.get("bandwidth", "100mbit")
        delay = scene_cfg.get("delay", "10ms")
        packet_loss = max(0.0, min(100.0, scene_cfg.get("packet_loss", 0.0)))  # 限制0-100%
        jitter = scene_cfg.get("jitter", "0ms")
        
        # 构建tc命令（支持丢包率+抖动）
        cmd = [
            "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
            "rate", bandwidth,
            "delay", delay, jitter  # 新增抖动支持
        ]
        
        # 仅当丢包率>0时添加
        if packet_loss > 0:
            cmd.extend(["loss", f"{packet_loss:.1f}%"])
        
        # 执行网络配置
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            logger.error(f"tc命令执行失败: {result.stderr}")
            return False
        
        logger.info(
            f"网络配置完成: 带宽={bandwidth} | 延迟={delay}+{jitter} | 丢包率={packet_loss}%"
        )
        return True
    except Exception as e:
        logger.error(f"网络配置失败: {e}")
        return False







def verify_model_config() -> bool:
    """
    验证model_config.yaml的完整性：
    1. 检查文件是否存在
    2. 修复preprocess_path笔误（运行时临时修正）
    3. 检查10种策略是否在候选列表中
    """
    model_cfg_path = "/cts/tests/configs/model_config.yaml"
    model_config_path = Path(model_cfg_path)
    if not model_config_path.exists():
        logger.error(f"模型配置文件不存在: {model_config_path}")
        return False
    
    # 加载并修复笔误
    try:
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        
        # 修复preprocess_path → preprocessing_path
        if "preprocess_path" in model_config and "preprocessing_path" not in model_config:
            model_config["preprocessing_path"] = model_config.pop("preprocess_path")
            # 保存修正后的配置（避免每次加载都出错）
            with open(model_config_path, "w", encoding="utf-8") as f:
                yaml.dump(model_config, f, ensure_ascii=False, indent=2)
            logger.warning("已自动修复model_config.yaml中的preprocess_path笔误")
        
        # 验证10种策略是否完整
        candidate_strategies = model_config.get("candidate_strategies", [])
        if len(candidate_strategies) != 10:
            logger.warning(f"候选策略数量不为10（当前{len(candidate_strategies)}）")
        else:
            logger.info("10种压缩策略配置验证通过")
        
        return True
    except Exception as e:
        logger.error(f"验证model_config失败: {e}")
        return False

# testbed/inner_runner.py
# ... (前面部分保持不变) ...

def run_experiment_script(experiment_name: str, scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    script_path = Path(CTS_ROOT / "tests" / "experiments" / f"{experiment_name}.py") 
    if not script_path.exists():
        error_msg = f"实验脚本不存在: {script_path}"
        logger.error(error_msg)
        return {"experiment": experiment_name, "scene": scene_cfg["scene_name"], "success": False, "error": error_msg}
    
    # 🔧 修复：移除 --packet-loss, --jitter, --model-config-path
    cmd = [
        "sudo","python", str(script_path),
        "--scene-name", scene_cfg["scene_name"],
        "--bandwidth", scene_cfg["bandwidth"],
        "--delay", scene_cfg["delay"],
        "--cpus", str(scene_cfg["cpus"]),
        "--memory", scene_cfg["memory"],
        "--image-list", ",".join(scene_cfg["image_list"]),
        "--repeat-times", str(scene_cfg["repeat_times"]),
        "--timeout", str(scene_cfg["timeout"])
    ]
    
    logger.info(f"启动实验脚本: {' '.join(cmd)}")
    
    # ... (后续 subprocess 执行逻辑保持不变) ...
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=scene_cfg["timeout"],
            cwd=str(CTS_ROOT)
        )
        
        if result.returncode == 0:
            logger.info(f"实验 {experiment_name} 执行成功")
            return {
                "experiment": experiment_name,
                "scene": scene_cfg["scene_name"],
                "success": True,
                "stdout": result.stdout[:1000],
                "stderr": ""
            }
        else:
            error_msg = result.stderr[:2000]
            logger.error(f"实验 {experiment_name} 执行失败: {error_msg}")
            return {
                "experiment": experiment_name,
                "scene": scene_cfg["scene_name"],
                "success": False,
                "error": error_msg
            }
    
    except subprocess.TimeoutExpired:
        error_msg = f"实验 {experiment_name} 超时（{scene_cfg['timeout']}s）"
        logger.error(error_msg)
        return {
            "experiment": experiment_name,
            "scene": scene_cfg["scene_name"],
            "success": False,
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"实验 {experiment_name} 异常: {str(e)[:200]}"
        logger.error(error_msg)
        return {
            "experiment": experiment_name,
            "scene": scene_cfg["scene_name"],
            "success": False,
            "error": error_msg
        }

def main():
    """主函数（新增依赖检查+模型配置验证）"""
    parser = argparse.ArgumentParser(description="容器内实验执行器")
    parser.add_argument("--scene-name", required=True, help="场景名称（如S1_Normal）")
    parser.add_argument("--config-path", default="/cts/config.yaml", help="全局配置文件路径")
    args = parser.parse_args()
    
    # 前置检查1：压缩工具依赖
    if not check_compression_dependencies():
        sys.exit(1)
    
    # 前置检查2：模型配置验证
    if not verify_model_config():
        sys.exit(1)
    
    # 加载场景配置
    scene_cfg = load_scene_config(args.config_path, args.scene_name)
    if not scene_cfg:
        logger.error("场景配置加载失败，退出")
        sys.exit(1)
    
    # 配置网络
    if not setup_network(scene_cfg):
        logger.error("网络配置失败，退出")
        sys.exit(1)
    
    # 执行实验
    experiment_list = scene_cfg.get("experiments", [])
    if not experiment_list:
        logger.warning("场景未配置任何实验，退出")
        sys.exit(0)
    
    all_results = []
    for exp_name in experiment_list:
        exp_result = run_experiment_script(exp_name, scene_cfg)
        all_results.append(exp_result)
        if not exp_result["success"]:
            logger.warning(f"实验 {exp_name} 失败，继续执行下一个")
    
    # 保存结果
    result_path = CTS_ROOT / "results" / f"{args.scene_name}_inner_results.yaml"
    result_path.parent.mkdir(exist_ok=True, mode=0o755)
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.dump(all_results, f, indent=2)



    logger.info(f"所有实验执行完成，结果保存至: {result_path}")
    
    # 检查失败实验
    failed_exps = [r for r in all_results if not r["success"]]
    if failed_exps:
        logger.error(f"共 {len(failed_exps)} 个实验失败")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()