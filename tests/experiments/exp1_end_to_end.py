import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import time
import sys
import os
import csv

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnvironmentController
from src.executor import DockerExecutor
from src.model_wrapper import CFTNetWrapper
from src.decision_engine import CAGSDecisionEngine
from src.utils import calculate_statistics, perform_t_test, save_json_result

def _load_image_feature_db(csv_path: str, logger) -> Dict[str, Dict[str, float]]:
    """
    加载镜像特征数据库
    支持解析带仓库前缀的镜像名 (如 quay.io/centos/centos:stream9 -> centos:stream9)
    """
    feature_db = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"镜像特征数据库文件不存在: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = row['image_name']
            
            # 解析镜像名：处理带仓库前缀的情况
            if '/' in full_name:
                # 取最后一个 '/' 后面的部分作为 key
                image_name_tag = full_name.split('/')[-1]
            else:
                image_name_tag = full_name
            
            # 转换所有数值特征为 float
            feature_dict = {}
            for k, v in row.items():
                if k != 'image_name':
                    try:
                        feature_dict[k] = float(v)
                    except (ValueError, TypeError):
                        feature_dict[k] = 0.0  # 非数值特征填0
            
            feature_db[image_name_tag] = feature_dict
            feature_db[full_name] = feature_dict  # 同时存全名，防止匹配失败

    logger.info(f"✅ 镜像特征数据库加载成功: {len(feature_db)//2} 个唯一镜像 (含全名映射)")
    return feature_db

def _get_safe_image_features(
    image_key: str, 
    feature_db: Dict[str, Dict[str, float]], 
    required_cols: list,
    logger
) -> Dict[str, float]:
    """
    安全地从数据库获取特征，严格保证与训练时的 cols_i 一致
    """
    # 尝试匹配
    features = None
    if image_key in feature_db:
        features = feature_db[image_key]
    else:
        # 尝试不带 tag 匹配
        if ':' in image_key:
            name_only = image_key.split(':')[0]
            for db_key in feature_db.keys():
                if db_key.startswith(name_only):
                    features = feature_db[db_key]
                    logger.warning(f"    ⚠️  精确匹配失败，使用近似匹配: {db_key}")
                    break
    
    if features is None:
        logger.error(f"    ❌ 镜像特征完全缺失: {image_key}，使用全0默认值")
        return {col: 0.0 for col in required_cols}
    
    # 严格按 required_cols 顺序构造，缺失特征填0并警告
    final_features = {}
    missing_cols = []
    for col in required_cols:
        if col in features:
            final_features[col] = features[col]
        else:
            final_features[col] = 0.0
            missing_cols.append(col)
    
    if missing_cols:
        logger.warning(f"    ⚠️  以下特征缺失，已填0: {missing_cols}")
    
    return final_features

def run(env_controller: EnvironmentController, 
        docker_executor: DockerExecutor,
        model_wrapper: CFTNetWrapper,
        global_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    实验一：端到端性能基准测试
    
    验证目标：CTS闭环系统相比工业界通用静态基线的性能提升
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 开始执行实验一：端到端性能基准测试")
    
    results = {
        'experiment_name': 'end_to_end_benchmark',
        'scenarios': {},
        'overall_comparison': {},
        'statistical_tests': {}
    }
    
    # 获取测试配置
    scenarios = global_config['scenarios']
    images = global_config['images']
    baseline_config = global_config['baseline']
    repeat_times = global_config['global']['repeat_times']
    
    # ==========================================
    # 1. 加载镜像特征数据库 (核心修改点)
    # ==========================================
    image_csv_path = global_config['global'].get('image_feature_csv', '')
    if not image_csv_path:
        raise ValueError("请在 global_config.yaml 中配置 'image_feature_csv' 路径")
    
    image_feature_db = _load_image_feature_db(image_csv_path, logger)
    
    # ==========================================
    # 2. 初始化CAGS决策引擎
    # ==========================================
    cags_engine = CAGSDecisionEngine(
        model=model_wrapper.model,
        scaler_c=model_wrapper.scaler_c,
        scaler_i=model_wrapper.scaler_i,
        enc=model_wrapper.enc,
        cols_c=model_wrapper.cols_c,
        cols_i=model_wrapper.cols_i,
        device=model_wrapper.device
    )
    
    # 遍历所有场景
    for scenario_id, scenario_config in scenarios.items():
        logger.info(f"\n📋 测试场景: {scenario_config['name']}")
        
        # 配置网络环境
        try:
            env_controller.init_network(
                scenario_config['bandwidth_mbps'],
                scenario_config['delay_ms'],
                scenario_config['loss_rate']
            )
        except Exception as e:
            logger.warning(f"    ⚠️  网络配置失败 (可能是Windows环境)，跳过tc配置: {e}")
        
        scenario_results = {
            'scenario_info': scenario_config,
            'baseline_results': [],
            'cts_results': [],
            'cts_decisions': [],
            'comparison': {}
        }
        
        # 测试每个镜像
        for image_info in images:
            image_key = f"{image_info['name']}:{image_info['tag']}"
            logger.info(f"\n  📦 测试镜像: {image_key}")
            
            # ==========================================
            # 3. 构造特征 (核心修改点：从CSV查表)
            # ==========================================
            # 3.1 客户端环境特征
            env_state = env_controller.collect_features()
            # 补充场景固定特征 (确保与训练一致)
            env_state.update({
                'cpu_limit': scenario_config.get('cpu_cores', 1.0),
                'mem_limit_mb': scenario_config.get('mem_limit_mb', 4096)
            })
            
            # 3.2 镜像特征 (从数据库安全获取)
            image_features = _get_safe_image_features(
                image_key=image_key,
                feature_db=image_feature_db,
                required_cols=model_wrapper.cols_i,
                logger=logger
            )
            
            # 从特征中获取镜像大小 (用于计算吞吐量)
            image_size_mb = image_features.get('total_size_mb', image_info.get('size_mb', 100))
            
            # ==========================================
            # 4. 基线测试 (静态配置)
            # ==========================================
            logger.info("    🔧 执行基线测试...")
            try:
                baseline_df = docker_executor.pull_image(
                    image_info['name'],
                    image_info['tag'],
                    baseline_config['compression'],
                    baseline_config,
                    repeat=repeat_times
                )
                # 计算吞吐量 (MB * 8 / s = Mbps)
                baseline_df['throughput_mbps'] = (image_size_mb * 8) / baseline_df['total_time_s']
                baseline_df['test_type'] = 'baseline'
                scenario_results['baseline_results'].append(baseline_df)
            except Exception as e:
                logger.error(f"    ❌ 基线测试失败: {e}")
            
            # ==========================================
            # 5. CTS系统测试 (智能决策)
            # ==========================================
            logger.info("    🤖 执行CTS决策与测试...")
            cts_config_dict = baseline_config.copy()
            cts_metrics_dict = {}
            
            try:
                # 5.1 CAGS决策
                cts_config_dict, cts_metrics_dict, pareto_front = cags_engine.make_decision(
                    env_state=env_state,
                    image_features=image_features,
                    safety_threshold=0.5,
                    enable_uncertainty=True,
                    enable_dpc=True
                )
                # 记录决策
                scenario_results['cts_decisions'].append({
                    'image': image_key,
                    'decision': cts_config_dict,
                    'metrics': cts_metrics_dict,
                    'env_state': env_state
                })
                logger.info(f"       ✅ CTS决策: Algo={cts_config_dict['algo_name']}, "
                           f"Threads={cts_config_dict['threads']}, "
                           f"PredTime={cts_metrics_dict['pred_time_s']:.2f}s")
            except Exception as e:
                logger.error(f"       ❌ CTS决策失败，回退基线: {e}")
                import traceback
                traceback.print_exc()
            
            # 5.2 构造CTS拉取配置
            cts_pull_config = {
                'concurrent_downloads': cts_config_dict.get('threads', baseline_config['concurrent_downloads']),
                'cpu_quota': int(cts_config_dict['cpu_quota'] * 100000) if cts_config_dict.get('cpu_quota', 0) > 0 else -1,
                'chunk_size_kb': cts_config_dict.get('chunk_size_kb', 64)
            }
            
            # 5.3 执行CTS拉取
            try:
                cts_df = docker_executor.pull_image(
                    image_info['name'],
                    image_info['tag'],
                    cts_config_dict.get('algo_name', baseline_config['compression']),
                    cts_pull_config,
                    repeat=repeat_times
                )
                # 计算CTS吞吐量
                cts_df['throughput_mbps'] = (image_size_mb * 8) / cts_df['total_time_s']
                cts_df['test_type'] = 'cts'
                scenario_results['cts_results'].append(cts_df)
            except Exception as e:
                logger.error(f"    ❌ CTS拉取失败: {e}")
            
            # 清理缓存
            try:
                env_controller.clear_cache()
            except:
                pass
        
        # ==========================================
        # 6. 场景级结果统计
        # ==========================================
        if scenario_results['baseline_results'] and scenario_results['cts_results']:
            try:
                baseline_combined = pd.concat(scenario_results['baseline_results'], ignore_index=True)
                cts_combined = pd.concat(scenario_results['cts_results'], ignore_index=True)
                
                # 仅统计成功的拉取
                baseline_success = baseline_combined[baseline_combined['success'] == 1]
                cts_success = cts_combined[cts_combined['success'] == 1]
                
                if len(baseline_success) > 0 and len(cts_success) > 0:
                    baseline_stats = calculate_statistics(baseline_success['throughput_mbps'].tolist())
                    cts_stats = calculate_statistics(cts_success['throughput_mbps'].tolist())
                    
                    performance_gain = (
                        (cts_stats['mean'] - baseline_stats['mean']) / baseline_stats['mean'] * 100
                        if baseline_stats['mean'] != 0 else 0
                    )
                    
                    scenario_results['comparison'] = {
                        'baseline_stats': baseline_stats,
                        'cts_stats': cts_stats,
                        'performance_gain_percent': performance_gain,
                        'baseline_success_rate': float(baseline_combined['success'].mean()),
                        'cts_success_rate': float(cts_combined['success'].mean())
                    }
                    logger.info(f"    📊 场景结果: 基线={baseline_stats['mean']:.2f}Mbps, "
                               f"CTS={cts_stats['mean']:.2f}Mbps, "
                               f"提升={performance_gain:.2f}%")
            except Exception as e:
                logger.error(f"    ❌ 场景统计失败: {e}")
        
        results['scenarios'][scenario_id] = scenario_results
    
    # ==========================================
    # 7. 整体对比分析
    # ==========================================
    all_baseline_throughputs = []
    all_cts_throughputs = []
    
    for scenario_data in results['scenarios'].values():
        if 'comparison' in scenario_data and scenario_data['comparison']:
            baseline_mean = scenario_data['comparison']['baseline_stats']['mean']
            cts_mean = scenario_data['comparison']['cts_stats']['mean']
            all_baseline_throughputs.append(baseline_mean)
            all_cts_throughputs.append(cts_mean)
    
    if len(all_baseline_throughputs) >= 2 and len(all_cts_throughputs) >= 2:
        try:
            t_test_result = perform_t_test(all_cts_throughputs, all_baseline_throughputs)
            
            results['overall_comparison'] = {
                'avg_baseline_throughput': float(np.mean(all_baseline_throughputs)),
                'avg_cts_throughput': float(np.mean(all_cts_throughputs)),
                'overall_performance_gain': float(
                    (np.mean(all_cts_throughputs) - np.mean(all_baseline_throughputs)) / 
                    np.mean(all_baseline_throughputs) * 100
                ),
                'performance_gains_by_scenario': {
                    name: data['comparison']['performance_gain_percent']
                    for name, data in results['scenarios'].items()
                    if 'comparison' in data and data['comparison']
                }
            }
            results['statistical_tests'] = t_test_result
        except Exception as e:
            logger.error(f"❌ 整体统计失败: {e}")
    
    # 保存结果
    result_file = f"{global_config['global']['result_save_dir']}/exp1_end_to_end_results.json"
    save_json_result(results, result_file)
    
    logger.info("\n✅ 实验一完成")
    return results

def print_summary(results: Dict[str, Any]):
    """打印实验一结果摘要"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("📊 实验一结果摘要")
    logger.info("="*60)
    
    if 'overall_comparison' in results and results['overall_comparison']:
        comp = results['overall_comparison']
        logger.info(f"\n📈 整体性能:")
        logger.info(f"  平均基线吞吐量: {comp['avg_baseline_throughput']:.2f} Mbps")
        logger.info(f"  平均CTS吞吐量: {comp['avg_cts_throughput']:.2f} Mbps")
        logger.info(f"  🏆 整体性能提升: {comp['overall_performance_gain']:.2f}%")
        
        logger.info(f"\n📋 各场景性能提升:")
        for scenario, gain in comp.get('performance_gains_by_scenario', {}).items():
            logger.info(f"  {scenario}: {gain:.2f}%")
    
    if 'statistical_tests' in results and 'p_value' in results['statistical_tests']:
        stats = results['statistical_tests']
        logger.info(f"\n🔬 统计检验:")
        logger.info(f"  p值: {stats['p_value']:.6f}")
        logger.info(f"  显著性: {'✅ 是' if stats['significant'] else '❌ 否'} (α=0.05)")
    
    # 打印CTS决策示例
    if 'scenarios' in results:
        first_scenario_id = list(results['scenarios'].keys())[0]
        first_scenario = results['scenarios'][first_scenario_id]
        if 'cts_decisions' in first_scenario and first_scenario['cts_decisions']:
            logger.info(f"\n🤖 CTS决策示例 ({first_scenario_id}):")
            for dec in first_scenario['cts_decisions'][:2]:
                logger.info(f"  [镜像 {dec['image']}]")
                logger.info(f"    配置: {dec['decision']}")
                logger.info(f"    预测: {dec['metrics']['pred_time_s']:.2f}s, "
                           f"Unc: {dec['metrics']['uncertainty_s']:.2f}s")