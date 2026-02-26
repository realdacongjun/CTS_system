import pandas as pd
import numpy as np
import logging
import random
import time
import sys
import os
import csv
from typing import Dict, Any, List

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnvironmentController
from src.executor import DockerExecutor
from src.model_wrapper import CFTNetWrapper
from src.decision_engine import CAGSDecisionEngine
from src.utils import calculate_statistics, save_json_result

def _load_image_feature_db(csv_path: str, logger) -> Dict[str, Dict[str, float]]:
    """加载镜像特征数据库（复用逻辑）"""
    feature_db = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"镜像特征数据库文件不存在: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = row['image_name']
            image_name_tag = full_name.split('/')[-1] if '/' in full_name else full_name
            
            feature_dict = {}
            for k, v in row.items():
                if k != 'image_name':
                    try:
                        feature_dict[k] = float(v)
                    except:
                        feature_dict[k] = 0.0
            
            feature_db[image_name_tag] = feature_dict
            feature_db[full_name] = feature_dict
    return feature_db

def _get_safe_image_features(
    image_key: str, 
    feature_db: Dict[str, Dict[str, float]], 
    required_cols: list,
    logger
) -> Dict[str, float]:
    """安全获取镜像特征（复用逻辑）"""
    features = feature_db.get(image_key)
    if not features and ':' in image_key:
        name_only = image_key.split(':')[0]
        for db_key in feature_db:
            if db_key.startswith(name_only):
                features = feature_db[db_key]
                break
    
    if not features:
        return {col: 0.0 for col in required_cols}
    
    return {col: features.get(col, 0.0) for col in required_cols}

def run(env_controller: EnvironmentController, 
        docker_executor: DockerExecutor,
        model_wrapper: CFTNetWrapper,  # 新增：传入模型
        global_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    实验三：动态鲁棒性测试
    
    验证目标：CTS在动态网络波动和极端OOD场景下的鲁棒性
    """
    logger = logging.getLogger(__name__)
    logger.info("🛡️ 开始执行实验三：动态鲁棒性测试")
    
    results = {
        'experiment_name': 'robustness_testing',
        'dynamic_fluctuation': {},
        'ood_extreme': {}
    }
    
    # ==========================================
    # 1. 初始化
    # ==========================================
    # 加载镜像特征库
    image_csv_path = global_config['global'].get('image_feature_csv', '')
    image_feature_db = _load_image_feature_db(image_csv_path, logger)
    
    # 初始化CAGS引擎
    cags_engine = CAGSDecisionEngine(
        model=model_wrapper.model,
        scaler_c=model_wrapper.scaler_c,
        scaler_i=model_wrapper.scaler_i,
        enc=model_wrapper.enc,
        cols_c=model_wrapper.cols_c,
        cols_i=model_wrapper.cols_i,
        device=model_wrapper.device
    )
    
    # 子场景A：动态网络波动测试
    logger.info("\n🔄 子场景A：动态网络波动测试")
    dynamic_results = _test_dynamic_fluctuation(
        env_controller, docker_executor, model_wrapper, cags_engine, 
        image_feature_db, global_config
    )
    results['dynamic_fluctuation'] = dynamic_results
    
    # 子场景B：OOD极端弱网测试
    logger.info("\n⚠️ 子场景B：OOD极端弱网测试")
    ood_results = _test_ood_extreme(
        env_controller, docker_executor, model_wrapper, cags_engine, 
        image_feature_db, global_config
    )
    results['ood_extreme'] = ood_results
    
    # 保存结果
    result_file = f"{global_config['global']['result_save_dir']}/exp3_robustness_results.json"
    save_json_result(results, result_file)
    
    logger.info("✅ 实验三完成")
    return results

def _test_dynamic_fluctuation(
    env_controller: EnvironmentController, 
    docker_executor: DockerExecutor,
    model_wrapper: CFTNetWrapper,
    cags_engine: CAGSDecisionEngine,
    image_feature_db: Dict,
    global_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    【核心修改】测试动态网络波动场景
    真正对比：Baseline vs CTS
    """
    logger = logging.getLogger(__name__)
    
    # 基准场景：边缘网络
    base_scenario = global_config['scenarios']['edge_network']
    baseline_config = global_config['baseline']
    images = global_config['images'][:1]  # 只测一个典型镜像加快速度
    image_info = images[0]
    image_key = f"{image_info['name']}:{image_info['tag']}"
    
    # 获取镜像特征
    image_features = _get_safe_image_features(
        image_key, image_feature_db, model_wrapper.cols_i, logger
    )
    image_size_mb = image_features.get('total_size_mb', image_info.get('size_mb', 100))
    
    results = {
        'baseline': {},
        'cts': {},
        'comparison': {}
    }
    
    # ==========================================
    # 1. 动态波动测试 (同时测 Baseline 和 CTS)
    # ==========================================
    logger.info("执行动态波动对比测试 (Baseline vs CTS)...")
    
    baseline_all_throughputs = []
    cts_all_throughputs = []
    baseline_all_success = []
    cts_all_success = []
    round_logs = []
    
    # 模拟10轮网络波动
    for round_num in range(10):
        # 随机生成网络参数
        bandwidth = random.uniform(10, 50)  # Mbps
        delay = random.uniform(20, 100)     # ms
        loss = random.uniform(0.005, 0.05)  # 0.5%-5%
        
        logger.info(f"  [轮次 {round_num+1}] BW={bandwidth:.1f}Mbps, Delay={delay:.1f}ms, Loss={loss*100:.1f}%")
        
        # 配置网络
        try:
            env_controller.init_network(bandwidth, delay, loss)
        except Exception as e:
            logger.warning(f"    ⚠️  网络配置失败: {e}")
        
        # 采集环境特征
        env_state = env_controller.collect_features()
        env_state.update({
            'cpu_limit': base_scenario.get('cpu_cores', 1.0),
            'mem_limit_mb': base_scenario.get('mem_limit_mb', 4096)
        })
        
        # --------------------------
        # 1.1 执行 Baseline 测试
        # --------------------------
        try:
            df_bl = docker_executor.pull_image(
                image_info['name'], image_info['tag'],
                baseline_config['compression'], baseline_config, repeat=2
            )
            df_bl['throughput_mbps'] = (image_size_mb * 8) / df_bl['total_time_s']
            
            bl_throughputs = df_bl[df_bl['success'] == 1]['throughput_mbps'].tolist()
            bl_success = df_bl['success'].tolist()
            
            baseline_all_throughputs.extend(bl_throughputs)
            baseline_all_success.extend(bl_success)
            logger.info(f"      Baseline: {np.mean(bl_throughputs):.2f} Mbps (成功: {sum(bl_success)}/{len(bl_success)})")
        except Exception as e:
            logger.error(f"      ❌ Baseline测试失败: {e}")
        
        # --------------------------
        # 1.2 执行 CTS 测试
        # --------------------------
        try:
            # CTS决策
            cts_config_dict, cts_metrics_dict, _ = cags_engine.make_decision(
                env_state=env_state,
                image_features=image_features,
                safety_threshold=0.5,
                enable_uncertainty=True,
                enable_dpc=True
            )
            
            # 构造拉取配置
            cts_pull_config = {
                'concurrent_downloads': cts_config_dict.get('threads', baseline_config['concurrent_downloads']),
                'cpu_quota': int(cts_config_dict['cpu_quota'] * 100000) if cts_config_dict.get('cpu_quota', 0) > 0 else -1,
                'chunk_size_kb': cts_config_dict.get('chunk_size_kb', 64)
            }
            
            # 执行拉取
            df_cts = docker_executor.pull_image(
                image_info['name'], image_info['tag'],
                cts_config_dict.get('algo_name', baseline_config['compression']),
                cts_pull_config, repeat=2
            )
            df_cts['throughput_mbps'] = (image_size_mb * 8) / df_cts['total_time_s']
            
            cts_throughputs = df_cts[df_cts['success'] == 1]['throughput_mbps'].tolist()
            cts_success = df_cts['success'].tolist()
            
            cts_all_throughputs.extend(cts_throughputs)
            cts_all_success.extend(cts_success)
            logger.info(f"      CTS:      {np.mean(cts_throughputs):.2f} Mbps (成功: {sum(cts_success)}/{len(cts_success)})")
            
            # 记录轮次日志
            round_logs.append({
                'round': round_num + 1,
                'bandwidth': bandwidth,
                'delay': delay,
                'loss': loss,
                'baseline_throughput': float(np.mean(bl_throughputs)) if bl_throughputs else 0,
                'cts_throughput': float(np.mean(cts_throughputs)) if cts_throughputs else 0,
                'cts_decision': cts_config_dict,
                'cts_uncertainty': cts_metrics_dict.get('epistemic_unc', 0)
            })
        except Exception as e:
            logger.error(f"      ❌ CTS测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理缓存
        try:
            env_controller.clear_cache()
        except:
            pass
        time.sleep(1)
    
    # ==========================================
    # 2. 统计分析
    # ==========================================
    if baseline_all_throughputs:
        bl_stats = calculate_statistics(baseline_all_throughputs)
        bl_cv = bl_stats['std'] / bl_stats['mean'] if bl_stats['mean'] != 0 else 0
        results['baseline'] = {
            'statistics': bl_stats,
            'throughput_cv': float(bl_cv),
            'success_rate': float(np.mean(baseline_all_success)),
            'all_throughputs': [float(x) for x in baseline_all_throughputs]
        }
    
    if cts_all_throughputs:
        cts_stats = calculate_statistics(cts_all_throughputs)
        cts_cv = cts_stats['std'] / cts_stats['mean'] if cts_stats['mean'] != 0 else 0
        results['cts'] = {
            'statistics': cts_stats,
            'throughput_cv': float(cts_cv),
            'success_rate': float(np.mean(cts_all_success)),
            'all_throughputs': [float(x) for x in cts_all_throughputs],
            'round_logs': round_logs
        }
    
    # 对比分析
    if baseline_all_throughputs and cts_all_throughputs:
        bl_mean = bl_stats['mean']
        cts_mean = cts_stats['mean']
        
        results['comparison'] = {
            'cts_throughput_gain': float(((cts_mean - bl_mean) / bl_mean * 100) if bl_mean != 0 else 0),
            'cts_cv_reduction': float(((bl_cv - cts_cv) / bl_cv * 100) if bl_cv != 0 else 0),
            'cts_success_rate_improvement': float(np.mean(cts_all_success) - np.mean(baseline_all_success))
        }
    
    return results

def _test_ood_extreme(
    env_controller: EnvironmentController,
    docker_executor: DockerExecutor,
    model_wrapper: CFTNetWrapper,
    cags_engine: CAGSDecisionEngine,
    image_feature_db: Dict,
    global_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    【核心修改】测试OOD极端弱网场景
    真正使用：认知不确定性 (epistemic_unc) 进行OOD检测
    """
    logger = logging.getLogger(__name__)
    
    baseline_config = global_config['baseline']
    images = global_config['images'][:1]
    image_info = images[0]
    image_key = f"{image_info['name']}:{image_info['tag']}"
    
    # 获取镜像特征
    image_features = _get_safe_image_features(
        image_key, image_feature_db, model_wrapper.cols_i, logger
    )
    image_size_mb = image_features.get('total_size_mb', image_info.get('size_mb', 100))
    
    results = {
        'extreme_scenarios': {},
        'ood_detection': {},
        'baseline_comparison': {}
    }
    
    # 定义极端场景（超出训练数据范围）
    extreme_scenarios = {
        'ultra_weak_network': {
            'name': '超弱网环境',
            'bandwidth_mbps': 0.5,    # 极低带宽
            'delay_ms': 300,          # 极高延迟
            'loss_rate': 0.2          # 20%丢包率
        },
        'burst_loss': {
            'name': '突发丢包环境',
            'bandwidth_mbps': 20,
            'delay_ms': 50,
            'loss_rate': 0.5          # 50%丢包率
        },
        'high_jitter': {
            'name': '高抖动环境',
            'bandwidth_mbps': 10,
            'delay_ms': 200,
            'loss_rate': 0.1
        }
    }
    
    scenario_results_list = []
    
    for scenario_key, scenario_config in extreme_scenarios.items():
        logger.info(f"\n测试极端场景: {scenario_config['name']}")
        
        # 配置网络
        try:
            env_controller.init_network(
                scenario_config['bandwidth_mbps'],
                scenario_config['delay_ms'],
                scenario_config['loss_rate']
            )
        except Exception as e:
            logger.warning(f"    ⚠️  网络配置失败: {e}")
        
        # 采集环境特征
        env_state = env_controller.collect_features()
        env_state.update({
            'cpu_limit': 1.0,
            'mem_limit_mb': 4096
        })
        
        scenario_data = {
            'scenario_key': scenario_key,
            'scenario_config': scenario_config,
            'baseline': {},
            'cts': {}
        }
        
        # --------------------------
        # 1. Baseline 测试
        # --------------------------
        try:
            df_bl = docker_executor.pull_image(
                image_info['name'], image_info['tag'],
                baseline_config['compression'], baseline_config, repeat=3
            )
            df_bl['throughput_mbps'] = (image_size_mb * 8) / df_bl['total_time_s']
            
            bl_success = df_bl['success'].mean()
            bl_throughput = df_bl[df_bl['success'] == 1]['throughput_mbps'].mean() if bl_success > 0 else 0
            
            scenario_data['baseline'] = {
                'throughput_mbps': float(bl_throughput),
                'success_rate': float(bl_success)
            }
            logger.info(f"    Baseline: {bl_throughput:.2f} Mbps, 成功率: {bl_success*100:.1f}%")
        except Exception as e:
            logger.error(f"    ❌ Baseline测试失败: {e}")
        
        # --------------------------
        # 2. CTS 测试 + OOD检测
        # --------------------------
        cts_epistemic_unc = 0.0
        try:
            # CTS决策 (获取不确定性)
            cts_config_dict, cts_metrics_dict, _ = cags_engine.make_decision(
                env_state=env_state,
                image_features=image_features,
                safety_threshold=0.5,
                enable_uncertainty=True,
                enable_dpc=True
            )
            
            # 【核心】获取认知不确定性用于OOD检测
            cts_epistemic_unc = cts_metrics_dict.get('epistemic_unc', 0.0)
            
            # OOD判断：不确定性 > 阈值 (例如 0.5) 认为是OOD
            ood_threshold = 0.5
            is_ood = cts_epistemic_unc > ood_threshold
            
            # 构造拉取配置
            cts_pull_config = {
                'concurrent_downloads': cts_config_dict.get('threads', baseline_config['concurrent_downloads']),
                'cpu_quota': int(cts_config_dict['cpu_quota'] * 100000) if cts_config_dict.get('cpu_quota', 0) > 0 else -1,
                'chunk_size_kb': cts_config_dict.get('chunk_size_kb', 64)
            }
            
            # 执行拉取
            df_cts = docker_executor.pull_image(
                image_info['name'], image_info['tag'],
                cts_config_dict.get('algo_name', baseline_config['compression']),
                cts_pull_config, repeat=3
            )
            df_cts['throughput_mbps'] = (image_size_mb * 8) / df_cts['total_time_s']
            
            cts_success = df_cts['success'].mean()
            cts_throughput = df_cts[df_cts['success'] == 1]['throughput_mbps'].mean() if cts_success > 0 else 0
            
            scenario_data['cts'] = {
                'throughput_mbps': float(cts_throughput),
                'success_rate': float(cts_success),
                'decision': cts_config_dict,
                'epistemic_uncertainty': float(cts_epistemic_unc),
                'is_ood_detected': bool(is_ood)
            }
            logger.info(f"    CTS:      {cts_throughput:.2f} Mbps, 成功率: {cts_success*100:.1f}%, "
                       f"不确定性: {cts_epistemic_unc:.3f}, OOD: {'✓' if is_ood else '✗'}")
            
        except Exception as e:
            logger.error(f"    ❌ CTS测试失败: {e}")
            import traceback
            traceback.print_exc()
            scenario_data['cts'] = {
                'epistemic_uncertainty': float(cts_epistemic_unc),
                'is_ood_detected': bool(cts_epistemic_unc > 0.5) if cts_epistemic_unc > 0 else False,
                'error': str(e)
            }
        
        results['extreme_scenarios'][scenario_key] = scenario_data
        scenario_results_list.append(scenario_data)
        
        # 清理缓存
        try:
            env_controller.clear_cache()
        except:
            pass
    
    # ==========================================
    # OOD检测统计
    # ==========================================
    # 假设这三个场景都应该被检测为OOD
    total_ood_scenarios = len(scenario_results_list)
    detected_ood = sum(1 for r in scenario_results_list if r['cts'].get('is_ood_detected', False))
    
    results['ood_detection'] = {
        'total_scenarios': total_ood_scenarios,
        'detected_count': detected_ood,
        'detection_rate': float(detected_ood / total_ood_scenarios if total_ood_scenarios > 0 else 0)
    }
    
    return results

def print_summary(results: Dict[str, Any]):
    """打印实验三结果摘要"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🛡️ 实验三结果摘要：动态鲁棒性")
    logger.info("="*60)
    
    # 动态波动结果
    if 'dynamic_fluctuation' in results:
        dyn = results['dynamic_fluctuation']
        logger.info("\n🔄 动态网络波动测试结果:")
        
        if 'baseline' in dyn and 'cts' in dyn:
            bl = dyn['baseline']
            cts = dyn['cts']
            
            logger.info(f"  【Baseline】")
            logger.info(f"    平均吞吐量: {bl['statistics']['mean']:.2f} Mbps")
            logger.info(f"    吞吐量变异系数 (CV): {bl['throughput_cv']:.3f}")
            logger.info(f"    成功率: {bl['success_rate']*100:.1f}%")
            
            logger.info(f"\n  【CTS】")
            logger.info(f"    平均吞吐量: {cts['statistics']['mean']:.2f} Mbps")
            logger.info(f"    吞吐量变异系数 (CV): {cts['throughput_cv']:.3f}")
            logger.info(f"    成功率: {cts['success_rate']*100:.1f}%")
            
            if 'comparison' in dyn:
                comp = dyn['comparison']
                logger.info(f"\n  【对比】")
                logger.info(f"    CTS 吞吐量提升: {comp['cts_throughput_gain']:+.2f}%")
                logger.info(f"    CTS 波动降低 (CV): {comp['cts_cv_reduction']:+.2f}%")
                logger.info(f"    CTS 成功率提升: {comp['cts_success_rate_improvement']*100:+.1f}%")
    
    # OOD极端测试结果
    if 'ood_extreme' in results:
        ood = results['ood_extreme']
        logger.info("\n⚠️ OOD极端场景测试结果:")
        
        if 'ood_detection' in ood:
            det = ood['ood_detection']
            logger.info(f"  OOD检测率: {det['detection_rate']*100:.1f}% ({det['detected_count']}/{det['total_scenarios']})")
        
        if 'extreme_scenarios' in ood:
            logger.info(f"\n  各场景详情:")
            for scenario_key, scenario_data in ood['extreme_scenarios'].items():
                cfg = scenario_data['scenario_config']
                bl = scenario_data.get('baseline', {})
                cts = scenario_data.get('cts', {})
                
                bl_tp = bl.get('throughput_mbps', 0)
                cts_tp = cts.get('throughput_mbps', 0)
                ood_mark = "✓" if cts.get('is_ood_detected', False) else "✗"
                unc = cts.get('epistemic_uncertainty', 0)
                
                logger.info(f"    {cfg['name']}:")
                logger.info(f"      网络: {cfg['bandwidth_mbps']}Mbps, {cfg['delay_ms']}ms, {cfg['loss_rate']*100}%丢包")
                logger.info(f"      Baseline: {bl_tp:.2f} Mbps | CTS: {cts_tp:.2f} Mbps")
                logger.info(f"      CTS不确定性: {unc:.3f} | OOD检测: {ood_mark}")