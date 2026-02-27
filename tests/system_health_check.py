#!/usr/bin/env python3
"""
CTS 系统健康检查脚本
=====================
快速验证：
1. 目录结构
2. 配置文件
3. 核心模块导入
4. 模型加载
5. 代理连通性
6. 下载器接口
"""

import sys
import os
import time
import yaml
import requests
import logging
from pathlib import Path
from colorama import init, Fore, Style

# 初始化颜色输出
init(autoreset=True)

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def print_ok(msg: str):
    print(f"{Fore.GREEN}✅ {msg}")

def print_warn(msg: str):
    print(f"{Fore.YELLOW}⚠️  {msg}")

def print_err(msg: str):
    print(f"{Fore.RED}❌ {msg}")

def print_info(msg: str):
    print(f"{Fore.CYAN}ℹ️  {msg}")

def check_directories() -> bool:
    """检查必要目录"""
    print("\n" + "="*60)
    print("1. 检查目录结构")
    print("="*60)
    
    required_dirs = [
        "configs",
        "src",
        "data",
        "models",
        "results"
    ]
    
    all_good = True
    for d in required_dirs:
        p = PROJECT_ROOT / d
        if p.exists():
            print_ok(f"目录存在: {d}/")
        else:
            print_warn(f"目录不存在: {d}/ (将自动创建)")
            p.mkdir(parents=True, exist_ok=True)
    
    return True

def check_configs() -> dict:
    """检查并加载配置文件"""
    print("\n" + "="*60)
    print("2. 检查配置文件")
    print("="*60)
    
    configs = {}
    
    # 检查 model_config.yaml
    model_cfg_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    if model_cfg_path.exists():
        print_ok(f"找到模型配置: {model_cfg_path.name}")
        try:
            with open(model_cfg_path, 'r', encoding='utf-8') as f:
                configs['model'] = yaml.safe_load(f)
            print_ok("  模型配置加载成功")
        except Exception as e:
            print_err(f"  模型配置加载失败: {e}")
            return None
    else:
        print_err(f"缺少配置: {model_cfg_path}")
        return None
    
    # 检查 global_config.yaml 或 config.yaml
    global_cfg_path = PROJECT_ROOT / "configs" / "global_config.yaml"
    if not global_cfg_path.exists():
        global_cfg_path = PROJECT_ROOT / "config.yaml"
    
    if global_cfg_path.exists():
        print_ok(f"找到全局配置: {global_cfg_path.name}")
        try:
            with open(global_cfg_path, 'r', encoding='utf-8') as f:
                configs['global'] = yaml.safe_load(f)
            print_ok("  全局配置加载成功")
        except Exception as e:
            print_warn(f"  全局配置加载警告: {e}")
    else:
        print_warn("未找到全局配置文件 (可选)")
    
    return configs

def check_core_modules() -> bool:
    """检查核心模块导入"""
    print("\n" + "="*60)
    print("3. 检查核心模块导入")
    print("="*60)
    
    # 1. 检查 cts_model
    try:
        # 这里我们不实际导入整个模块（防止触发训练），只检查文件存在
        cts_model_path = PROJECT_ROOT / "src" / "cts_model.py"
        if cts_model_path.exists():
            print_ok("核心模型文件存在: src/cts_model.py")
        else:
            print_err("缺少: src/cts_model.py")
            return False
    except Exception as e:
        print_err(f"模型文件检查失败: {e}")
        return False

    # 2. 检查 model_wrapper
    try:
        from src.model_wrapper import CFTNetWrapper
        print_ok("成功导入: CFTNetWrapper")
    except Exception as e:
        print_err(f"导入失败: CFTNetWrapper")
        print(f"   错误: {e}")
        return False

    # 3. 检查 decision_engine
    try:
        from src.decision_engine import CAGSDecisionEngine
        print_ok("成功导入: CAGSDecisionEngine")
    except Exception as e:
        print_err(f"导入失败: CAGSDecisionEngine")
        print(f"   错误: {e}")
        return False

    # 4. 检查下载器
    try:
        import baseline_downloader
        print_ok("成功导入: baseline_downloader")
    except Exception as e:
        print_warn(f"导入警告: baseline_downloader ({e})")

    try:
        import adaptive_downloader
        print_ok("成功导入: adaptive_downloader")
    except Exception as e:
        print_warn(f"导入警告: adaptive_downloader ({e})")

    return True

def check_model_files(configs: dict) -> bool:
    """检查模型文件"""
    print("\n" + "="*60)
    print("4. 检查模型权重文件")
    print("="*60)
    
    if 'model' not in configs:
        print_warn("跳过模型检查 (无配置)")
        return True # 不算致命错误
    
    model_cfg = configs['model']
    
    # 解析路径 (相对于 configs/ 目录)
    base_dir = PROJECT_ROOT / "configs"
    
    model_path = base_dir / model_cfg['model']['model_path']
    preproc_path = base_dir / model_cfg['model']['preprocess_path']
    
    # 尝试直接相对于根目录
    if not model_path.exists():
        model_path = PROJECT_ROOT / model_cfg['model']['model_path']
    if not preproc_path.exists():
        preproc_path = PROJECT_ROOT / model_cfg['model']['preprocess_path']

    all_good = True
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_ok(f"模型权重存在: {model_path.name} ({size_mb:.2f} MB)")
    else:
        print_err(f"模型权重缺失: {model_path}")
        print_info("   提示：请确保已运行训练脚本，或修改 model_config.yaml 中的路径")
        all_good = False
    
    if preproc_path.exists():
        size_mb = preproc_path.stat().st_size / (1024 * 1024)
        print_ok(f"预处理器存在: {preproc_path.name} ({size_mb:.2f} MB)")
    else:
        print_warn(f"预处理器缺失: {preproc_path}")
        # 这个不算致命错误，可能是新训练流程
    
    return all_good

def test_model_wrapper_init(configs: dict) -> bool:
    """测试模型包装器初始化"""
    print("\n" + "="*60)
    print("5. 测试 CTS 系统初始化")
    print("="*60)
    
    try:
        from src.model_wrapper import CFTNetWrapper
        
        model_cfg_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        
        print_info("正在初始化 CFTNetWrapper (这可能需要几秒钟)...")
        wrapper = CFTNetWrapper(str(model_cfg_path))
        
        print_ok("CFTNetWrapper 初始化成功！")
        info = wrapper.get_model_info()
        print(f"   - 模型类: {info['model_class']}")
        print(f"   - 参数量: {info['total_parameters']:,}")
        print(f"   - 算法数: {info['num_algorithms']}")
        print(f"   - 设备: {info['device']}")
        
        # 测试决策引擎初始化
        try:
            from src.decision_engine import CAGSDecisionEngine
            engine = CAGSDecisionEngine(
                model=wrapper.model,
                scaler_c=wrapper.scaler_c,
                scaler_i=wrapper.scaler_i,
                enc=wrapper.enc,
                cols_c=wrapper.cols_c,
                cols_i=wrapper.cols_i,
                device=wrapper.device
            )
            print_ok("CAGSDecisionEngine 初始化成功！")
            return True
        except Exception as e:
            print_err(f"CAGSDecisionEngine 初始化失败: {e}")
            return False
            
    except FileNotFoundError as e:
        print_err(f"初始化失败 (文件缺失): {e}")
        return False
    except Exception as e:
        print_err(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_proxy_server() -> bool:
    """检查代理服务器连通性"""
    print("\n" + "="*60)
    print("6. 检查代理服务器 (可选)")
    print("="*60)
    
    proxy_url = "http://localhost:8000"
    
    try:
        print_info(f"尝试连接: {proxy_url}")
        r = requests.get(proxy_url, timeout=2)
        if r.status_code == 200 or r.status_code == 404:
            print_ok("代理服务器响应正常！")
            print_info("   提示：404 是正常的，因为我们在访问根目录")
            return True
        else:
            print_warn(f"代理服务器状态码异常: {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_warn("无法连接到代理服务器")
        print_info("   提示：如果还没准备好镜像测试，请先忽略此项")
        print_info("   运行命令: python proxy_server.py")
        return False
    except Exception as e:
        print_warn(f"代理检查异常: {e}")
        return False

def check_data_csv() -> bool:
    """检查镜像特征 CSV"""
    print("\n" + "="*60)
    print("7. 检查镜像特征数据库")
    print("="*60)
    
    csv_path = PROJECT_ROOT / "data" / "image_features_database.csv"
    
    if csv_path.exists():
        print_ok(f"CSV 存在: {csv_path.name}")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=5)
            print_ok(f"   CSV 读取成功，列数: {len(df.columns)}")
            if 'image_name' in df.columns and 'total_size_mb' in df.columns:
                print_ok("   必要列存在 (image_name, total_size_mb)")
                return True
            else:
                print_warn("   缺少必要列，请检查 CSV 格式")
                return False
        except Exception as e:
            print_warn(f"   CSV 读取警告: {e}")
            return False
    else:
        print_warn(f"CSV 不存在: {csv_path}")
        print_info("   提示：这是决策引擎读取镜像特征所需的")
        return False

def main():
    print(f"\n{Fore.BLUE}{Style.BRIGHT}🚀 CTS 系统健康检查")
    print(f"{Fore.BLUE}{Style.BRIGHT}{'='*60}")
    
    results = {}
    
    # 1. 目录
    results['目录结构'] = check_directories()
    
    # 2. 配置
    configs = check_configs()
    results['配置文件'] = (configs is not None)
    
    # 3. 模块
    results['核心模块'] = check_core_modules()
    
    # 4. 模型文件
    if configs:
        results['模型文件'] = check_model_files(configs)
    else:
        results['模型文件'] = False
    
    # 5. 系统初始化
    if configs and results['核心模块']:
        results['系统初始化'] = test_model_wrapper_init(configs)
    else:
        results['系统初始化'] = False
        print_warn("跳过系统初始化测试 (前置条件未满足)")
    
    # 6. 代理
    results['代理服务'] = check_proxy_server()
    
    # 7. 数据
    results['数据CSV'] = check_data_csv()
    
    # 总结
    print("\n" + "="*60)
    print(f"{Style.BRIGHT}📋 检查总结")
    print("="*60)
    
    all_passed = True
    for name, result in results.items():
        status = f"{Fore.GREEN}✅ 通过" if result else f"{Fore.RED}❌ 失败"
        print(f"  {name:15} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print(f"{Fore.GREEN}{Style.BRIGHT}🎉 所有核心检查通过！")
        print(f"\n下一步:")
        print(f"  1. 确保 proxy_server.py 正在运行")
        print(f"  2. 运行: python experiment_runner.py")
    else:
        print(f"{Fore.YELLOW}⚠️  部分检查未通过，请查看上面的错误信息。")
        print(f"\n提示:")
        print(f"  - '模型文件' 和 '代理服务' 失败不影响代码结构验证")
        print(f"  - 请优先解决 '核心模块' 和 '系统初始化' 的错误")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}检查被用户中断")
        sys.exit(0)