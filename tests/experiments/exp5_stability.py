import pandas as pd
import numpy as np
import logging
import time
import json
import csv
import sys
import os
from typing import Dict, Any, List

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnvironmentController
from src.executor import DockerExecutor
from src.model_wrapper import CFTNetWrapper
from src.decision_engine import CAGSDecisionEngine
from src.utils import save_json_result  # 统一使用src.utils的

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('exp5_stability.log', encoding='utf-8')
    ]
)

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
    实验五：长期稳定性测试
    
    验证目标：CTS在高频连续执行下的稳定性表现
    """
    logger = logging.getLogger(__name__)
    logger.info("🔁 开始执行实验五：长期稳定性测试")
    
    results = {
        'experiment_name': 'long_term_stability',
        'baseline': {},
        'cts': {},
        'comparison': {}
    }
    
    # ==========================================
    # 1. 初始化
    # ==========================================
    # 选择云数据中心场景（高频率场景）
    scenario_config = global_config['scenarios']['cloud_datacenter']
    baseline_config = global_config['baseline']
    images = global_config['images'][:1]  # 测试一个典型镜像
    image_info = images[0]
    image_key = f"{image_info['name']}:{image_info['tag']}"
    
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
    
    # 获取镜像特征
    image_features = _get_safe_image_features(
        image_key, image_feature_db, model_wrapper.cols_i, logger
    )
    image_size_mb = image_features.get('total_size_mb', image_info.get('size_mb', 100))
    
    logger.info(f"📋 测试场景: {scenario_config['name']}")
    logger.info(f"📦 测试镜像: {image_key}")
    
    # 配置网络环境
    try:
        env_controller.init_network(
            scenario_config['bandwidth_mbps'],
            scenario_config['delay_ms'],
            scenario_config['loss_rate']
        )
    except Exception as e:
        logger.warning(f"    ⚠️  网络配置失败: {e}")
    
    # ==========================================
    # 2. 执行50次连续冷启动测试 (Baseline + CTS)
    # ==========================================
    logger.info("\n🔄 开始执行50次连续冷启动对比测试...")
    
    baseline_results = _execute_continuous_pulls(
        env_controller=env_controller,
        docker_executor=docker_executor,
        pull_config=baseline_config,
        compression=baseline_config['compression'],
        image_info=image_info,
        image_size_mb=image_size_mb,
        iterations=50,
        test_type='Baseline'
    )
    results['baseline'] = baseline_results
    
    cts_results = _execute_continuous_pulls(
        env_controller=env_controller,
        docker_executor=docker_executor,
        model_wrapper=model_wrapper,
        cags_engine=cags_engine,
        scenario_config=scenario_config,
        image_features=image_features,
        image_info=image_info,
        image_size_mb=image_size_mb,
        iterations=50,
        test_type='CTS'
    )
    results['cts'] = cts_results
    
    # ==========================================
    # 3. 性能衰减对比分析
    # ==========================================
    logger.info("\n📉 执行性能衰减对比分析...")
    decay_comparison = _analyze_performance_decay_comparison(baseline_results, cts_results)
    results['comparison']['performance_decay'] = decay_comparison
    
    # ==========================================
    # 4. 配置稳定性对比分析
    # ==========================================
    logger.info("\n🔧 执行配置稳定性对比分析...")
    config_comparison = _analyze_configuration_stability_comparison(baseline_results, cts_results)
    results['comparison']['configuration_stability'] = config_comparison
    
    # 保存结果
    result_file = f"{global_config['global']['result_save_dir']}/exp5_stability_results.json"
    save_json_result(results, result_file)
    
    logger.info("\n✅ 实验五完成")
    return results

def _execute_continuous_pulls(
    env_controller: EnvironmentController,
    docker_executor: DockerExecutor,
    image_info: Dict[str, Any],
    image_size_mb: float,
    iterations: int = 50,
    test_type: str = 'Baseline',
    # Baseline专用参数
    pull_config: Dict[str, Any] = None,
    compression: str = None,
    # CTS专用参数
    model_wrapper: CFTNetWrapper = None,
    cags_engine: CAGSDecisionEngine = None,
    scenario_config: Dict[str, Any] = None,
    image_features: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    【核心修改】执行连续拉取测试
    支持 Baseline 和 CTS 两种模式
    """
    logger = logging.getLogger(__name__)
    
    results = {
        'test_type': test_type,
        'iterations': iterations,
        'execution_log': [],
        'cts_decisions': [],  # CTS专用：记录每一轮决策
        'performance_metrics': {},
        'failure_log': []
    }
    
    all_throughputs = []
    all_latencies = []
    successful_executions = 0
    
    logger.info(f"[{test_type}] 开始执行 {iterations} 次连续拉取测试...")
    
    for i in range(iterations):
        try:
            logger.info(f"[{test_type}] [{i+1}/{iterations}] 执行第 {i+1} 次拉取...")
            
            # 每次都清理缓存确保冷启动
            env_controller.clear_cache()
            
            # ==========================================
            # 1. 构造拉取配置
            # ==========================================
            current_pull_config = None
            current_compression = None
            cts_decision_log = None
            
            if test_type == 'Baseline':
                current_pull_config = pull_config
                current_compression = compression
            else:
                # CTS模式：采集环境特征 + 决策
                env_state = env_controller.collect_features()
                env_state.update({
                    'cpu_limit': scenario_config.get('cpu_cores', 4.0),
                    'mem_limit_mb': scenario_config.get('mem_limit_mb', 8192),
                    # 补充物理交叉特征
                    'theoretical_time': image_size_mb / (env_state['bandwidth_mbps'] / 8 + 1e-8),
                    'cpu_to_size_ratio': env_state['cpu_limit'] / image_size_mb,
                    'mem_to_size_ratio': env_state['mem_limit_mb'] / image_size_mb,
                    'network_score': env_state['bandwidth_mbps'] / (env_state['network_rtt'] + 1e-8)
                })
                
                # CAGS决策
                cts_config_dict, cts_metrics_dict, _ = cags_engine.make_decision(
                    env_state=env_state,
                    image_features=image_features,
                    safety_threshold=0.5,
                    enable_uncertainty=True,
                    enable_dpc=True
                )
                
                # 构造拉取配置
                current_pull_config = {
                    'concurrent_downloads': cts_config_dict.get('threads', 16),
                    'cpu_quota': int(cts_config_dict['cpu_quota'] * 100000) if cts_config_dict.get('cpu_quota', 0) > 0 else -1,
                    'chunk_size_kb': cts_config_dict.get('chunk_size_kb', 64)
                }
                current_compression = cts_config_dict.get('algo_name', 'gzip')
                
                # 记录决策
                cts_decision_log = {
                    'iteration': i + 1,
                    'decision': cts_config_dict,
                    'metrics': cts_metrics_dict,
                    'env_state': env_state
                }
                results['cts_decisions'].append(cts_decision_log)
                logger.info(f"       CTS决策: {cts_config_dict}, 不确定性: {cts_metrics_dict['epistemic_unc']:.3f}")
            
            # ==========================================
            # 2. 执行拉取
            # ==========================================
            start_time = time.time()
            df = docker_executor.pull_image(
                image_info['name'],
                image_info['tag'],
                current_compression,
                current_pull_config,
                repeat=1
            )
            end_time = time.time()
            execution_time = end_time - start_time
            
            # ==========================================
            # 3. 收集结果
            # ==========================================
            if not df.empty and df.iloc[0]['success'] == 1:
                latency = df.iloc[0]['total_time_s']
                throughput = (image_size_mb * 8) / latency
                
                all_throughputs.append(throughput)
                all_latencies.append(latency)
                successful_executions += 1
                
                execution_record = {
                    'iteration': i + 1,
                    'success': True,
                    'throughput_mbps': round(throughput, 2),
                    'latency_seconds': round(latency, 2),
                    'execution_time': round(execution_time, 2),
                    'timestamp': time.time()
                }
            else:
                execution_record = {
                    'iteration': i + 1,
                    'success': False,
                    'error': df.iloc[0]['stderr'] if not df.empty else 'Unknown error',
                    'timestamp': time.time()
                }
                results['failure_log'].append(execution_record)
            
            results['execution_log'].append(execution_record)
            
            # 控制执行间隔
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"[{test_type}] [{i+1}/{iterations}] 执行异常: {e}")
            import traceback
            traceback.print_exc()
            failure_record = {
                'iteration': i + 1,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
            results['failure_log'].append(failure_record)
            results['execution_log'].append(failure_record)
    
    # ==========================================
    # 4. 计算总体性能指标
    # ==========================================
    success_rate = (successful_executions / iterations) * 100 if iterations > 0 else 0
    
    results['performance_metrics'] = {
        'total_executions': iterations,
        'successful_executions': successful_executions,
        'success_rate_percent': round(success_rate, 1),
        'avg_throughput_mbps': round(float(np.mean(all_throughputs)) if all_throughputs else 0, 2),
        'avg_latency_seconds': round(float(np.mean(all_latencies)) if all_latencies else 0, 2),
        'throughput_std': round(float(np.std(all_throughputs)) if all_throughputs else 0, 2),
        'latency_std': round(float(np.std(all_latencies)) if all_latencies else 0, 2)
    }
    
    # 分段性能分析（前10次 vs 后10次）
    if len(all_throughputs) >= 20:
        early_throughputs = all_throughputs[:10]
        late_throughputs = all_throughputs[-10:]
        early_latencies = all_latencies[:10]
        late_latencies = all_latencies[-10:]
        
        results['performance_metrics'].update({
            'early_avg_throughput': round(float(np.mean(early_throughputs)), 2),
            'late_avg_throughput': round(float(np.mean(late_throughputs)), 2),
            'early_avg_latency': round(float(np.mean(early_latencies)), 2),
            'late_avg_latency': round(float(np.mean(late_latencies)), 2)
        })
    
    logger.info(f"[{test_type}] 连续测试完成 - 成功率: {success_rate:.1f}%")
    return results

def _analyze_performance_decay_comparison(
    baseline_results: Dict[str, Any],
    cts_results: Dict[str, Any]
) -> Dict[str, Any]:
    """【核心修改】分析Baseline vs CTS的性能衰减对比"""
    logger = logging.getLogger(__name__)
    comparison = {
        'baseline': {},
        'cts': {},
        'cts_improvement': {}
    }
    
    # 分析Baseline
    bl_metrics = baseline_results.get('performance_metrics', {})
    if 'early_avg_throughput' in bl_metrics and bl_metrics['early_avg_throughput'] > 0:
        bl_decay = (
            (bl_metrics['early_avg_throughput'] - bl_metrics['late_avg_throughput']) / 
            bl_metrics['early_avg_throughput'] * 100
        )
        comparison['baseline'] = {
            'throughput_decay_percent': round(bl_decay, 2),
            'early_avg_throughput': bl_metrics['early_avg_throughput'],
            'late_avg_throughput': bl_metrics['late_avg_throughput'],
            'performance_stable': abs(bl_decay) < 5
        }
    
    # 分析CTS
    cts_metrics = cts_results.get('performance_metrics', {})
    if 'early_avg_throughput' in cts_metrics and cts_metrics['early_avg_throughput'] > 0:
        cts_decay = (
            (cts_metrics['early_avg_throughput'] - cts_metrics['late_avg_throughput']) / 
            cts_metrics['early_avg_throughput'] * 100
        )
        comparison['cts'] = {
            'throughput_decay_percent': round(cts_decay, 2),
            'early_avg_throughput': cts_metrics['early_avg_throughput'],
            'late_avg_throughput': cts_metrics['late_avg_throughput'],
            'performance_stable': abs(cts_decay) < 5
        }
    
    # 对比改进
    if comparison['baseline'] and comparison['cts']:
        bl_decay_abs = abs(comparison['baseline']['throughput_decay_percent'])
        cts_decay_abs = abs(comparison['cts']['throughput_decay_percent'])
        decay_reduction = (
            (bl_decay_abs - cts_decay_abs) / bl_decay_abs * 100
            if bl_decay_abs > 0 else 0
        )
        comparison['cts_improvement'] = {
            'decay_reduction_percent': round(decay_reduction, 2),
            'cts_more_stable': cts_decay_abs < bl_decay_abs
        }
    
    return comparison

def _analyze_configuration_stability_comparison(
    baseline_results: Dict[str, Any],
    cts_results: Dict[str, Any]
) -> Dict[str, Any]:
    """【核心修改】分析Baseline vs CTS的配置稳定性对比"""
    logger = logging.getLogger(__name__)
    comparison = {
        'baseline': {},
        'cts': {}
    }
    
    # Baseline配置稳定性：只看执行时间和吞吐量的变异系数
    bl_exec_log = baseline_results.get('execution_log', [])
    bl_success = [log for log in bl_exec_log if log.get('success', False)]
    if len(bl_success) > 10:
        bl_throughputs = [log['throughput_mbps'] for log in bl_success]
        bl_exec_times = [log['execution_time'] for log in bl_success]
        bl_tp_mean = np.mean(bl_throughputs)
        bl_tp_cv = np.std(bl_throughputs) / bl_tp_mean if bl_tp_mean != 0 else 0
        bl_time_mean = np.mean(bl_exec_times)
        bl_time_cv = np.std(bl_exec_times) / bl_time_mean if bl_time_mean != 0 else 0
        
        comparison['baseline'] = {
            'throughput_coefficient_of_variation': round(float(bl_tp_cv), 3),
            'execution_time_coefficient_of_variation': round(float(bl_time_cv), 3),
            'anomaly_count': _count_anomalies(bl_throughputs, bl_exec_times)
        }
    
    # CTS配置稳定性：额外看决策一致性和不确定性稳定性
    cts_exec_log = cts_results.get('execution_log', [])
    cts_decisions = cts_results.get('cts_decisions', [])
    cts_success = [log for log in cts_exec_log if log.get('success', False)]
    if len(cts_success) > 10 and len(cts_decisions) > 10:
        cts_throughputs = [log['throughput_mbps'] for log in cts_success]
        cts_exec_times = [log['execution_time'] for log in cts_success]
        cts_tp_mean = np.mean(cts_throughputs)
        cts_tp_cv = np.std(cts_throughputs) / cts_tp_mean if cts_tp_mean != 0 else 0
        cts_time_mean = np.mean(cts_exec_times)
        cts_time_cv = np.std(cts_exec_times) / cts_time_mean if cts_time_mean != 0 else 0
        
        # 决策一致性分析
        first_decision = cts_decisions[0]['decision']
        consistent_decisions = 0
        for dec in cts_decisions:
            if dec['decision'] == first_decision:
                consistent_decisions += 1
        decision_consistency_rate = consistent_decisions / len(cts_decisions) * 100
        
        # 不确定性稳定性分析
        uncertainties = [dec['metrics']['epistemic_unc'] for dec in cts_decisions]
        unc_mean = np.mean(uncertainties)
        unc_std = np.std(uncertainties)
        
        comparison['cts'] = {
            'throughput_coefficient_of_variation': round(float(cts_tp_cv), 3),
            'execution_time_coefficient_of_variation': round(float(cts_time_cv), 3),
            'anomaly_count': _count_anomalies(cts_throughputs, cts_exec_times),
            'decision_consistency_rate_percent': round(decision_consistency_rate, 1),
            'uncertainty_mean': round(float(unc_mean), 4),
            'uncertainty_std': round(float(unc_std), 4)
        }
    
    return comparison

def _count_anomalies(throughputs: List[float], execution_times: List[float]) -> int:
    """辅助函数：识别异常值（超过2个标准差）"""
    if len(throughputs) < 10:
        return 0
    
    tp_mean = np.mean(throughputs)
    tp_std = np.std(throughputs)
    time_mean = np.mean(execution_times)
    time_std = np.std(execution_times)
    
    anomaly_count = 0
    for tp, et in zip(throughputs, execution_times):
        if (abs(tp - tp_mean) > 2 * tp_std or
            abs(et - time_mean) > 2 * time_std):
            anomaly_count += 1
    
    return anomaly_count

def print_summary(results: Dict[str, Any]):
    """打印实验五结果摘要"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🔁 实验五结果摘要：长期稳定性测试")
    logger.info("="*60)
    
    # 总体性能对比
    if 'baseline' in results and 'cts' in results:
        bl = results['baseline'].get('performance_metrics', {})
        cts = results['cts'].get('performance_metrics', {})
        
        logger.info("\n📈 总体性能对比:")
        logger.info(f"  【Baseline】")
        logger.info(f"    总执行次数: {bl.get('total_executions', 0)}")
        logger.info(f"    成功次数: {bl.get('successful_executions', 0)}")
        logger.info(f"    成功率: {bl.get('success_rate_percent', 0):.1f}%")
        logger.info(f"    平均吞吐量: {bl.get('avg_throughput_mbps', 0):.2f} Mbps")
        logger.info(f"    吞吐量标准差: {bl.get('throughput_std', 0):.2f} Mbps")
        
        logger.info(f"\n  【CTS】")
        logger.info(f"    总执行次数: {cts.get('total_executions', 0)}")
        logger.info(f"    成功次数: {cts.get('successful_executions', 0)}")
        logger.info(f"    成功率: {cts.get('success_rate_percent', 0):.1f}%")
        logger.info(f"    平均吞吐量: {cts.get('avg_throughput_mbps', 0):.2f} Mbps")
        logger.info(f"    吞吐量标准差: {cts.get('throughput_std', 0):.2f} Mbps")
        
        if bl.get('avg_throughput_mbps', 0) > 0:
            tp_gain = (cts.get('avg_throughput_mbps', 0) - bl.get('avg_throughput_mbps', 0)) / bl.get('avg_throughput_mbps', 0) * 100
            logger.info(f"\n  【CTS改进】")
            logger.info(f"    吞吐量提升: {tp_gain:+.2f}%")
    
    # 性能衰减对比
    if 'comparison' in results and 'performance_decay' in results['comparison']:
        decay = results['comparison']['performance_decay']
        bl_decay = decay.get('baseline', {})
        cts_decay = decay.get('cts', {})
        cts_imp_decay = decay.get('cts_improvement', {})
        
        logger.info("\n📉 性能衰减对比:")
        if bl_decay:
            logger.info(f"  【Baseline】")
            logger.info(f"    前期平均吞吐量: {bl_decay.get('early_avg_throughput', 0):.2f} Mbps")
            logger.info(f"    后期平均吞吐量: {bl_decay.get('late_avg_throughput', 0):.2f} Mbps")
            logger.info(f"    吞吐量衰减: {bl_decay.get('throughput_decay_percent', 0):.2f}%")
            logger.info(f"    性能稳定: {'✅ 是' if bl_decay.get('performance_stable', False) else '❌ 否'}")
        
        if cts_decay:
            logger.info(f"\n  【CTS】")
            logger.info(f"    前期平均吞吐量: {cts_decay.get('early_avg_throughput', 0):.2f} Mbps")
            logger.info(f"    后期平均吞吐量: {cts_decay.get('late_avg_throughput', 0):.2f} Mbps")
            logger.info(f"    吞吐量衰减: {cts_decay.get('throughput_decay_percent', 0):.2f}%")
            logger.info(f"    性能稳定: {'✅ 是' if cts_decay.get('performance_stable', False) else '❌ 否'}")
        
        if cts_imp_decay:
            logger.info(f"\n  【CTS改进】")
            logger.info(f"    衰减降低: {cts_imp_decay.get('decay_reduction_percent', 0):.2f}%")
            logger.info(f"    CTS更稳定: {'✅ 是' if cts_imp_decay.get('cts_more_stable', False) else '❌ 否'}")
    
    # 配置稳定性对比
    if 'comparison' in results and 'configuration_stability' in results['comparison']:
        config = results['comparison']['configuration_stability']
        bl_config = config.get('baseline', {})
        cts_config = config.get('cts', {})
        
        logger.info("\n🔧 配置稳定性对比:")
        if bl_config:
            logger.info(f"  【Baseline】")
            logger.info(f"    吞吐量变异系数: {bl_config.get('throughput_coefficient_of_variation', 0):.3f}")
            logger.info(f"    异常次数: {bl_config.get('anomaly_count', 0)}")
        
        if cts_config:
            logger.info(f"\n  【CTS】")
            logger.info(f"    吞吐量变异系数: {cts_config.get('throughput_coefficient_of_variation', 0):.3f}")
            logger.info(f"    异常次数: {cts_config.get('anomaly_count', 0)}")
            logger.info(f"    决策一致性率: {cts_config.get('decision_consistency_rate_percent', 0):.1f}%")
            logger.info(f"    认知不确定性均值: {cts_config.get('uncertainty_mean', 0):.4f}")
            logger.info(f"    认知不确定性标准差: {cts_config.get('uncertainty_std', 0):.4f}")