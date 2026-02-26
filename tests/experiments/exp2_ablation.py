import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import sys
import os
import csv

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnvironmentController
from src.executor import DockerExecutor
from src.model_wrapper import CFTNetWrapper
from src.decision_engine import CAGSDecisionEngine
from src.utils import calculate_statistics, save_json_result

def _load_image_feature_db(csv_path: str, logger) -> Dict[str, Dict[str, float]]:
    """加载镜像特征数据库（复用实验一的逻辑）"""
    feature_db = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"镜像特征数据库文件不存在: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = row['image_name']
            if '/' in full_name:
                image_name_tag = full_name.split('/')[-1]
            else:
                image_name_tag = full_name
            
            feature_dict = {}
            for k, v in row.items():
                if k != 'image_name':
                    try:
                        feature_dict[k] = float(v)
                    except (ValueError, TypeError):
                        feature_dict[k] = 0.0
            
            feature_db[image_name_tag] = feature_dict
            feature_db[full_name] = feature_dict

    logger.info(f"✅ 镜像特征数据库加载成功")
    return feature_db

def _get_safe_image_features(
    image_key: str, 
    feature_db: Dict[str, Dict[str, float]], 
    required_cols: list,
    logger
) -> Dict[str, float]:
    """安全获取镜像特征（复用实验一的逻辑）"""
    features = None
    if image_key in feature_db:
        features = feature_db[image_key]
    else:
        if ':' in image_key:
            name_only = image_key.split(':')[0]
            for db_key in feature_db.keys():
                if db_key.startswith(name_only):
                    features = feature_db[db_key]
                    break
    
    if features is None:
        return {col: 0.0 for col in required_cols}
    
    final_features = {}
    for col in required_cols:
        final_features[col] = features.get(col, 0.0)
    return final_features

def _get_variant_config(
    variant_name: str,
    variant_def: Dict,
    env_state: Dict,
    image_features: Dict,
    cags_engine: CAGSDecisionEngine,
    baseline_config: Dict,
    logger
) -> Dict:
    """
    【核心修改】根据消融变体类型，生成对应的拉取配置
    真正实现四个变体的逻辑差异
    """
    if variant_name == 'V1_Static':
        # 变体1：纯静态基线
        return {
            'pull_config': baseline_config.copy(),
            'compression': baseline_config['compression'],
            'decision_log': 'Static baseline'
        }
    
    elif variant_name == 'V2_CFT_Only':
        # 变体2：仅CFT-Net预测
        # 逻辑：遍历所有压缩算法，用CFT预测耗时，选最快的那个，但不做帕累托优化
        best_algo = baseline_config['compression']
        best_pred_time = float('inf')
        
        try:
            # 简单遍历算法列表（这里简化为直接调用CAGS的批量预测，然后只选最快的）
            pred_df = cags_engine._predict_batch(env_state, image_features)
            # 按预测时间排序，选最快的
            best_row = pred_df.sort_values('pred_time').iloc[0]
            best_algo = best_row['algo_name']
            best_pred_time = best_row['pred_time']
            
            decision_log = f"CFT-only: chose {best_algo}, pred_time={best_pred_time:.2f}s"
            logger.info(f"       {decision_log}")
        except Exception as e:
            logger.warning(f"       ⚠️  CFT-only决策失败，回退基线: {e}")
            decision_log = "CFT-only failed, fallback to baseline"
        
        # 构造配置：仅改压缩算法，其他用基线
        pull_config = baseline_config.copy()
        return {
            'pull_config': pull_config,
            'compression': best_algo,
            'decision_log': decision_log
        }
    
    elif variant_name == 'V3_CFT_CAGS_NoUnc':
        # 变体3：CFT+CAGS，但关闭不确定性过滤
        try:
            cts_config_dict, cts_metrics_dict, _ = cags_engine.make_decision(
                env_state=env_state,
                image_features=image_features,
                safety_threshold=1e6,  # 关键：设极大值，关闭不确定性过滤
                enable_uncertainty=True,  # 逻辑上开启，但阈值极高
                enable_dpc=True
            )
            decision_log = f"CFT+CAGS (NoUnc): {cts_config_dict}"
            logger.info(f"       {decision_log}")
        except Exception as e:
            logger.warning(f"       ⚠️  CFT+CAGS (NoUnc)决策失败，回退基线: {e}")
            cts_config_dict = baseline_config.copy()
            decision_log = "CFT+CAGS (NoUnc) failed, fallback"
        
        pull_config = {
            'concurrent_downloads': cts_config_dict.get('threads', baseline_config['concurrent_downloads']),
            'cpu_quota': int(cts_config_dict['cpu_quota'] * 100000) if cts_config_dict.get('cpu_quota', 0) > 0 else -1,
            'chunk_size_kb': cts_config_dict.get('chunk_size_kb', 64)
        }
        return {
            'pull_config': pull_config,
            'compression': cts_config_dict.get('algo_name', baseline_config['compression']),
            'decision_log': decision_log
        }
    
    elif variant_name == 'V4_CTS_Full':
        # 变体4：CTS完整系统
        try:
            cts_config_dict, cts_metrics_dict, _ = cags_engine.make_decision(
                env_state=env_state,
                image_features=image_features,
                safety_threshold=0.5,
                enable_uncertainty=True,
                enable_dpc=True
            )
            decision_log = f"CTS-Full: {cts_config_dict}"
            logger.info(f"       {decision_log}")
        except Exception as e:
            logger.warning(f"       ⚠️  CTS-Full决策失败，回退基线: {e}")
            cts_config_dict = baseline_config.copy()
            decision_log = "CTS-Full failed, fallback"
        
        pull_config = {
            'concurrent_downloads': cts_config_dict.get('threads', baseline_config['concurrent_downloads']),
            'cpu_quota': int(cts_config_dict['cpu_quota'] * 100000) if cts_config_dict.get('cpu_quota', 0) > 0 else -1,
            'chunk_size_kb': cts_config_dict.get('chunk_size_kb', 64)
        }
        return {
            'pull_config': pull_config,
            'compression': cts_config_dict.get('algo_name', baseline_config['compression']),
            'decision_log': decision_log
        }
    
    else:
        # 默认回退
        return {
            'pull_config': baseline_config.copy(),
            'compression': baseline_config['compression'],
            'decision_log': 'Unknown variant, fallback'
        }

def run(env_controller: EnvironmentController, 
        docker_executor: DockerExecutor,
        model_wrapper: CFTNetWrapper,  # 新增：传入模型包装器
        global_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    实验二：协同增益消融实验
    
    验证目标：分解CFT-Net和CAGS各自的贡献及协同效应
    """
    logger = logging.getLogger(__name__)
    logger.info("🔬 开始执行实验二：协同增益消融实验")
    
    results = {
        'experiment_name': 'ablation_study',
        'edge_network_results': {},
        'variants_comparison': {},
        'decision_logs': []  # 新增：记录所有决策
    }
    
    # ==========================================
    # 1. 初始化
    # ==========================================
    # 选择边缘网络场景
    scenario_name = 'edge_network'
    scenario_config = global_config['scenarios'][scenario_name]
    images = global_config['images']
    baseline_config = global_config['baseline']
    repeat_times = global_config['global']['repeat_times']
    
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
    
    logger.info(f"📋 测试场景: {scenario_config['name']}")
    
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
    # 2. 定义四个对比变体
    # ==========================================
    variants = {
        'V1_Static': {
            'description': '工业界通用静态配置 (Baseline)',
            'use_model': False,
            'use_cags': False
        },
        'V2_CFT_Only': {
            'description': '仅使用CFT-Net预测最优压缩算法',
            'use_model': True,
            'use_cags': False
        },
        'V3_CFT_CAGS_NoUnc': {
            'description': 'CFT-Net + CAGS (关闭不确定性过滤)',
            'use_model': True,
            'use_cags': True
        },
        'V4_CTS_Full': {
            'description': 'CTS全模块协同 (完整系统)',
            'use_model': True,
            'use_cags': True
        }
    }
    
    variant_results = {}
    
    # ==========================================
    # 3. 测试每个变体
    # ==========================================
    for variant_name, variant_def in variants.items():
        logger.info(f"\n🧪 测试变体: {variant_name}")
        logger.info(f"   描述: {variant_def['description']}")
        
        all_variant_results = []
        variant_decisions = []
        
        # 测试所有镜像
        for image_info in images:
            image_key = f"{image_info['name']}:{image_info['tag']}"
            logger.info(f"\n  📦 测试镜像: {image_key}")
            
            # 构造特征
            env_state = env_controller.collect_features()
            env_state.update({
                'cpu_limit': scenario_config.get('cpu_cores', 1.0),
                'mem_limit_mb': scenario_config.get('mem_limit_mb', 4096)
            })
            image_features = _get_safe_image_features(
                image_key, image_feature_db, model_wrapper.cols_i, logger
            )
            image_size_mb = image_features.get('total_size_mb', image_info.get('size_mb', 100))
            
            # ==========================================
            # 核心：根据变体获取配置
            # ==========================================
            decision_output = _get_variant_config(
                variant_name=variant_name,
                variant_def=variant_def,
                env_state=env_state,
                image_features=image_features,
                cags_engine=cags_engine,
                baseline_config=baseline_config,
                logger=logger
            )
            
            # 记录决策
            variant_decisions.append({
                'variant': variant_name,
                'image': image_key,
                'decision_log': decision_output['decision_log']
            })
            
            # ==========================================
            # 执行拉取
            # ==========================================
            try:
                df = docker_executor.pull_image(
                    image_info['name'],
                    image_info['tag'],
                    decision_output['compression'],
                    decision_output['pull_config'],
                    repeat=repeat_times
                )
                # 计算吞吐量
                df['throughput_mbps'] = (image_size_mb * 8) / df['total_time_s']
                df['variant'] = variant_name
                df['image_name'] = image_info['name']
                df['decision_log'] = decision_output['decision_log']
                all_variant_results.append(df)
            except Exception as e:
                logger.error(f"    ❌ 拉取失败: {e}")
            
            # 清理缓存
            try:
                env_controller.clear_cache()
            except:
                pass
        
        # ==========================================
        # 变体级结果统计
        # ==========================================
        if all_variant_results:
            combined_df = pd.concat(all_variant_results, ignore_index=True)
            # 仅统计成功的
            success_df = combined_df[combined_df['success'] == 1]
            
            if len(success_df) > 0:
                stats = calculate_statistics(success_df['throughput_mbps'].tolist())
                
                variant_results[variant_name] = {
                    'config': variant_def,
                    'results_df': combined_df.to_dict('records'),
                    'statistics': stats,
                    'avg_throughput': stats['mean'],
                    'success_rate': float(combined_df['success'].mean()),
                    'decisions': variant_decisions
                }
                logger.info(f"    📊 变体结果: {stats['mean']:.2f} Mbps, "
                           f"成功率: {variant_results[variant_name]['success_rate']*100:.1f}%")
        
        results['decision_logs'].extend(variant_decisions)
    
    results['edge_network_results'] = variant_results
    
    # ==========================================
    # 4. 变体间对比分析
    # ==========================================
    if variant_results and 'V1_Static' in variant_results:
        v1_baseline = variant_results['V1_Static']['avg_throughput']
        
        comparison_data = {}
        for variant_name, variant_data in variant_results.items():
            throughput = variant_data['avg_throughput']
            gain_vs_v1 = ((throughput - v1_baseline) / v1_baseline * 100) if v1_baseline != 0 else 0
            comparison_data[variant_name] = {
                'avg_throughput': throughput,
                'gain_vs_static': gain_vs_v1,
                'success_rate': variant_data['success_rate']
            }
        
        results['variants_comparison'] = comparison_data
    
    # 保存结果
    result_file = f"{global_config['global']['result_save_dir']}/exp2_ablation_results.json"
    save_json_result(results, result_file)
    
    logger.info("\n✅ 实验二完成")
    return results

def print_summary(results: Dict[str, Any]):
    """打印实验二结果摘要"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🔬 实验二结果摘要：协同增益消融")
    logger.info("="*60)
    
    if 'variants_comparison' in results:
        comparison = results['variants_comparison']
        
        logger.info("\n📊 各变体性能对比:")
        for variant, data in comparison.items():
            logger.info(f"\n{variant}:")
            logger.info(f"  平均吞吐量: {data['avg_throughput']:.2f} Mbps")
            logger.info(f"  相比静态基线提升: {data['gain_vs_static']:.2f}%")
            logger.info(f"  成功率: {data['success_rate']*100:.1f}%")
        
        # 协同增益分析
        required_variants = ['V1_Static', 'V2_CFT_Only', 'V3_CFT_CAGS_NoUnc', 'V4_CTS_Full']
        if all(v in comparison for v in required_variants):
            cft_gain = comparison['V2_CFT_Only']['gain_vs_static']
            no_unc_gain = comparison['V3_CFT_CAGS_NoUnc']['gain_vs_static']
            full_gain = comparison['V4_CTS_Full']['gain_vs_static']
            
            # 计算各部分贡献
            cags_contribution = no_unc_gain - cft_gain
            uncertainty_contribution = full_gain - no_unc_gain
            total_synergy = full_gain - cft_gain
            
            logger.info(f"\n🤝 贡献分解分析:")
            logger.info(f"  1. CFT-Net 单独贡献:          {cft_gain:+.2f}%")
            logger.info(f"  2. CAGS 决策机制贡献:        {cags_contribution:+.2f}%")
            logger.info(f"  3. 不确定性过滤贡献:         {uncertainty_contribution:+.2f}%")
            logger.info(f"  -------------------------------------")
            logger.info(f"  CTS 完整系统总提升:          {full_gain:+.2f}%")
            logger.info(f"  协同增益 (1+2+3 > 1):      {total_synergy - cft_gain:+.2f}%")