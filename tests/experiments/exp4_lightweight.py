import psutil
import time
import logging
import torch
import pandas as pd
import numpy as np
import csv
from typing import Dict, Any
from src.utils import save_json_result
import sys
import os

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def run(global_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    实验四：轻量化部署验证
    
    验证目标：CTS决策引擎的资源消耗和部署可行性
    """
    logger = logging.getLogger(__name__)
    logger.info("⚙️ 开始执行实验四：轻量化部署验证")
    
    results = {
        'experiment_name': 'lightweight_deployment',
        'model_characteristics': {},
        'cags_engine_characteristics': {},
        'full_system_characteristics': {},
        'runtime_performance': {},
        'resource_consumption': {}
    }
    
    # ==========================================
    # 1. 初始化依赖
    # ==========================================
    try:
        from src.model_wrapper import CFTNetWrapper
        from src.decision_engine import CAGSDecisionEngine
    except ImportError as e:
        logger.error(f"❌ 依赖导入失败: {e}")
        results['error'] = str(e)
        return results
    
    # 加载镜像特征库（用于构造测试数据）
    image_csv_path = global_config['global'].get('image_feature_csv', '')
    try:
        image_feature_db = _load_image_feature_db(image_csv_path, logger)
    except Exception as e:
        logger.warning(f"⚠️  镜像特征库加载失败，使用默认测试数据: {e}")
        image_feature_db = {}
    
    # ==========================================
    # 2. 模型特性分析
    # ==========================================
    logger.info("分析模型特性...")
    model_chars = _analyze_model_characteristics(global_config)
    results['model_characteristics'] = model_chars
    
    # ==========================================
    # 3. CAGS引擎特性分析
    # ==========================================
    logger.info("分析CAGS引擎特性...")
    cags_chars = _analyze_cags_characteristics()
    results['cags_engine_characteristics'] = cags_chars
    
    # ==========================================
    # 4. 全系统特性分析
    # ==========================================
    logger.info("分析全系统特性...")
    full_chars = _analyze_full_system_characteristics(model_chars, cags_chars)
    results['full_system_characteristics'] = full_chars
    
    # ==========================================
    # 5. 运行时性能测试 (完整决策流程)
    # ==========================================
    logger.info("测试运行时性能...")
    runtime_perf = _test_runtime_performance(
        global_config, image_feature_db
    )
    results['runtime_performance'] = runtime_perf
    
    # ==========================================
    # 6. 资源消耗测试
    # ==========================================
    logger.info("测量资源消耗...")
    resource_usage = _measure_resource_consumption(
        global_config, image_feature_db
    )
    results['resource_consumption'] = resource_usage
    
    # 保存结果
    result_file = f"{global_config['global']['result_save_dir']}/exp4_lightweight_results.json"
    save_json_result(results, result_file)
    
    logger.info("✅ 实验四完成")
    return results

def _analyze_model_characteristics(global_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    【核心修改】分析 CompactCFTNetV2 模型特性
    """
    logger = logging.getLogger(__name__)
    try:
        from src.model_wrapper import CFTNetWrapper
        
        # 初始化模型包装器（自动加载正确的模型和预处理器）
        model_wrapper = CFTNetWrapper("./configs/model_config.yaml")
        model = model_wrapper.model
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小 (参数 + 预处理器)
        model_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        # 估算预处理器大小 (StandardScaler/LabelEncoder 很小，约 10KB)
        preprocessor_bytes = 10 * 1024
        total_model_bytes = model_params_bytes + preprocessor_bytes
        total_model_mb = total_model_bytes / (1024 * 1024)
        
        return {
            'model_class': 'CompactCFTNetV2',
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(total_params - trainable_params),
            'model_params_size_mb': round(model_params_bytes / (1024 * 1024), 3),
            'preprocessor_size_mb': round(preprocessor_bytes / (1024 * 1024), 3),
            'total_model_system_size_mb': round(total_model_mb, 2),
            'client_feature_dim': len(model_wrapper.cols_c),
            'image_feature_dim': len(model_wrapper.cols_i),
            'num_algorithms': len(model_wrapper.enc.classes_),
            'embed_dim': 64,  # 从训练代码 CONFIG 中获取
            'num_transformer_layers': 2,  # 从训练代码 CONFIG 中获取
            'device': str(model_wrapper.device)
        }
        
    except Exception as e:
        logger.error(f"❌ 模型特性分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def _analyze_cags_characteristics() -> Dict[str, Any]:
    """分析 CAGS 决策引擎特性"""
    logger = logging.getLogger(__name__)
    try:
        # 估算决策引擎代码大小 (Python 源码约 50KB)
        cags_code_bytes = 50 * 1024
        cags_code_mb = cags_code_bytes / (1024 * 1024)
        
        # 决策空间大小
        threads_list = [1, 2, 4, 8, 16]
        cpu_quota_list = [0.2, 0.5, 1.0, 2.0]
        chunk_size_list = [64, 256, 1024]
        # 假设算法数为 3
        num_algos = 3
        decision_space_size = len(threads_list) * len(cpu_quota_list) * len(chunk_size_list) * num_algos
        
        return {
            'engine_name': 'CAGS (Confidence-Aware Adaptive Greedy Search)',
            'code_size_mb': round(cags_code_mb, 3),
            'decision_space_size': int(decision_space_size),
            'core_modules': [
                'DPC (Dynamic Pareto Collapse) Detection',
                'CID (Confidence Interval Dominance) Sorting',
                'AKT (Adaptive Knee Tracking) Selection'
            ],
            'supports_uncertainty_filtering': True,
            'supports_dynamic_scenario_adaptation': True
        }
        
    except Exception as e:
        logger.error(f"❌ CAGS引擎特性分析失败: {e}")
        return {'error': str(e)}

def _analyze_full_system_characteristics(
    model_chars: Dict, 
    cags_chars: Dict
) -> Dict[str, Any]:
    """分析全系统特性"""
    logger = logging.getLogger(__name__)
    try:
        if 'error' in model_chars or 'error' in cags_chars:
            return {'error': 'Model or CAGS analysis failed'}
        
        total_size_mb = (
            model_chars.get('total_model_system_size_mb', 0) + 
            cags_chars.get('code_size_mb', 0)
        )
        
        # 轻量化判断标准
        is_lightweight_for_edge = total_size_mb < 50  # 边缘设备 < 50MB
        is_lightweight_for_iot = total_size_mb < 10    # IoT设备 < 10MB
        
        return {
            'total_system_size_mb': round(total_size_mb, 2),
            'is_lightweight_for_edge': bool(is_lightweight_for_edge),
            'is_lightweight_for_iot': bool(is_lightweight_for_iot),
            'lightweight_judgment_criteria': {
                'edge_device_threshold_mb': 50,
                'iot_device_threshold_mb': 10
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 全系统特性分析失败: {e}")
        return {'error': str(e)}

def _test_runtime_performance(
    global_config: Dict[str, Any],
    image_feature_db: Dict
) -> Dict[str, Any]:
    """
    【核心修改】测试完整 CAGS 决策流程的运行时性能
    分解：模型预测时间 vs CAGS决策时间
    """
    logger = logging.getLogger(__name__)
    try:
        from src.model_wrapper import CFTNetWrapper
        from src.decision_engine import CAGSDecisionEngine
        
        # 初始化
        model_wrapper = CFTNetWrapper("./configs/model_config.yaml")
        cags_engine = CAGSDecisionEngine(
            model=model_wrapper.model,
            scaler_c=model_wrapper.scaler_c,
            scaler_i=model_wrapper.scaler_i,
            enc=model_wrapper.enc,
            cols_c=model_wrapper.cols_c,
            cols_i=model_wrapper.cols_i,
            device=model_wrapper.device
        )
        
        # 构造真实的测试特征
        # 1. 客户端环境特征（边缘网络）
        base_scenario = global_config['scenarios']['edge_network']
        env_state = {
            'bandwidth_mbps': base_scenario['bandwidth_mbps'],
            'network_rtt': base_scenario['delay_ms'],
            'packet_loss': base_scenario['loss_rate'],
            'cpu_limit': base_scenario.get('cpu_cores', 1.0),
            'mem_limit_mb': base_scenario.get('mem_limit_mb', 4096),
            'cpu_available': 70.0,
            'mem_available': 3000.0
        }
        # 补充物理交叉特征（用于模型预测）
        env_state['theoretical_time'] = 100.0 / (env_state['bandwidth_mbps'] / 8 + 1e-8)
        env_state['cpu_to_size_ratio'] = env_state['cpu_limit'] / 100.0
        env_state['mem_to_size_ratio'] = env_state['mem_limit_mb'] / 100.0
        env_state['network_score'] = env_state['bandwidth_mbps'] / (env_state['network_rtt'] + 1e-8)
        
        # 2. 镜像特征（从CSV取 nginx:latest）
        image_key = 'nginx:latest'
        image_features = _get_safe_image_features(
            image_key, image_feature_db, model_wrapper.cols_i, logger
        )
        
        # 预热 (10次完整决策)
        logger.info("    预热中...")
        for _ in range(10):
            cags_engine.make_decision(env_state, image_features)
        
        # ==========================================
        # 性能测试 (1000次完整决策)
        # ==========================================
        logger.info("    执行1000次完整决策测试...")
        
        model_pred_latencies = []  # 仅模型预测时间
        cags_total_latencies = []  # 完整决策时间 (预测+排序+选择)
        
        for i in range(1000):
            # 1. 测量完整决策时间
            start_total = time.perf_counter()
            cags_engine.make_decision(env_state, image_features)
            end_total = time.perf_counter()
            cags_total_latencies.append((end_total - start_total) * 1000)  # ms
            
            # 2. 单独测量模型预测时间 (批量预测)
            start_pred = time.perf_counter()
            cags_engine._predict_batch(env_state, image_features)
            end_pred = time.perf_counter()
            model_pred_latencies.append((end_pred - start_pred) * 1000)  # ms
        
        # ==========================================
        # 统计分析
        # ==========================================
        cags_total_arr = np.array(cags_total_latencies)
        model_pred_arr = np.array(model_pred_latencies)
        cags_overhead_arr = cags_total_arr - model_pred_arr  # CAGS排序+选择时间
        
        # 决策开销占比分析 (用实验一的基线平均传输时间 10s 作为参考)
        avg_transmission_time_s = 10.0
        avg_cags_total_ms = np.mean(cags_total_arr)
        decision_overhead_ratio = (avg_cags_total_ms / 1000) / avg_transmission_time_s * 100
        
        return {
            'test_type': 'Full CAGS Decision Flow',
            'test_iterations': 1000,
            'test_image': image_key,
            'test_scenario': 'Edge Network',
            
            # 完整决策时间
            'cags_total': {
                'average_ms': round(float(np.mean(cags_total_arr)), 3),
                'p50_ms': round(float(np.percentile(cags_total_arr, 50)), 3),
                'p95_ms': round(float(np.percentile(cags_total_arr, 95)), 3),
                'p99_ms': round(float(np.percentile(cags_total_arr, 99)), 3),
                'std_ms': round(float(np.std(cags_total_arr)), 3),
                'throughput_decisions_per_second': int(1000 / np.mean(cags_total_arr))
            },
            
            # 分解：模型预测时间
            'model_prediction': {
                'average_ms': round(float(np.mean(model_pred_arr)), 3),
                'percentage_of_total': round(float(np.mean(model_pred_arr) / np.mean(cags_total_arr) * 100), 1)
            },
            
            # 分解：CAGS排序+选择时间
            'cags_overhead': {
                'average_ms': round(float(np.mean(cags_overhead_arr)), 3),
                'percentage_of_total': round(float(np.mean(cags_overhead_arr) / np.mean(cags_total_arr) * 100), 1)
            },
            
            # 关键：决策开销占比
            'decision_overhead_analysis': {
                'reference_transmission_time_s': avg_transmission_time_s,
                'decision_overhead_ratio_percent': round(decision_overhead_ratio, 4),
                'is_overhead_acceptable': bool(decision_overhead_ratio < 1.0)  # <1% 合格
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 运行时性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def _measure_resource_consumption(
    global_config: Dict[str, Any],
    image_feature_db: Dict
) -> Dict[str, Any]:
    """
    【核心修改】测量完整系统的资源消耗
    """
    logger = logging.getLogger(__name__)
    try:
        from src.model_wrapper import CFTNetWrapper
        from src.decision_engine import CAGSDecisionEngine
        import tracemalloc
        
        # 获取当前进程
        process = psutil.Process()
        
        # ==========================================
        # 1. 测量初始资源 (加载系统前)
        # ==========================================
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        initial_cpu_percent = process.cpu_percent(interval=0.1)
        
        # ==========================================
        # 2. 加载完整系统
        # ==========================================
        logger.info("    加载完整系统...")
        model_wrapper = CFTNetWrapper("./configs/model_config.yaml")
        cags_engine = CAGSDecisionEngine(
            model=model_wrapper.model,
            scaler_c=model_wrapper.scaler_c,
            scaler_i=model_wrapper.scaler_i,
            enc=model_wrapper.enc,
            cols_c=model_wrapper.cols_c,
            cols_i=model_wrapper.cols_i,
            device=model_wrapper.device
        )
        
        # 构造测试特征
        base_scenario = global_config['scenarios']['edge_network']
        env_state = {
            'bandwidth_mbps': base_scenario['bandwidth_mbps'],
            'network_rtt': base_scenario['delay_ms'],
            'packet_loss': base_scenario['loss_rate'],
            'cpu_limit': base_scenario.get('cpu_cores', 1.0),
            'mem_limit_mb': base_scenario.get('mem_limit_mb', 4096),
            'cpu_available': 70.0,
            'mem_available': 3000.0,
            'theoretical_time': 100.0 / (base_scenario['bandwidth_mbps'] / 8 + 1e-8),
            'cpu_to_size_ratio': base_scenario.get('cpu_cores', 1.0) / 100.0,
            'mem_to_size_ratio': base_scenario.get('mem_limit_mb', 4096) / 100.0,
            'network_score': base_scenario['bandwidth_mbps'] / (base_scenario['delay_ms'] + 1e-8)
        }
        image_key = 'nginx:latest'
        image_features = _get_safe_image_features(
            image_key, image_feature_db, model_wrapper.cols_i, logger
        )
        
        # 预热
        for _ in range(10):
            cags_engine.make_decision(env_state, image_features)
        
        # ==========================================
        # 3. 测量加载后的资源
        # ==========================================
        loaded_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_load_mb = loaded_memory_mb - initial_memory_mb
        
        # ==========================================
        # 4. 测量执行决策时的资源
        # ==========================================
        logger.info("    执行100次决策并测量资源...")
        
        # 启动内存跟踪
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        # 测量CPU时间
        initial_cpu_times = process.cpu_times()
        
        # 执行100次决策
        for _ in range(100):
            cags_engine.make_decision(env_state, image_features)
        
        # 停止测量
        final_cpu_times = process.cpu_times()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # 计算内存增量
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_exec_mb = final_memory_mb - loaded_memory_mb
        
        # 计算内存分配峰值
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        peak_memory_alloc_mb = round(top_stats[0].size_diff / 1024 / 1024, 3) if top_stats else 0.0
        
        # 计算CPU时间
        cpu_user_time_s = final_cpu_times.user - initial_cpu_times.user
        cpu_system_time_s = final_cpu_times.system - initial_cpu_times.system
        cpu_total_time_s = cpu_user_time_s + cpu_system_time_s
        
        # 计算平均CPU使用率
        # 执行100次决策的总时间
        exec_start = time.perf_counter()
        for _ in range(100):
            cags_engine.make_decision(env_state, image_features)
        exec_end = time.perf_counter()
        exec_total_time_s = exec_end - exec_start
        avg_cpu_percent = (cpu_total_time_s / exec_total_time_s) * 100 if exec_total_time_s > 0 else 0
        
        return {
            'initial_memory_mb': round(initial_memory_mb, 2),
            'loaded_system_memory_mb': round(loaded_memory_mb, 2),
            'memory_increase_on_load_mb': round(memory_increase_load_mb, 2),
            'memory_increase_on_execution_mb': round(memory_increase_exec_mb, 2),
            'peak_memory_allocation_mb': round(peak_memory_alloc_mb, 3),
            'cpu_user_time_seconds': round(cpu_user_time_s, 4),
            'cpu_system_time_seconds': round(cpu_system_time_s, 4),
            'cpu_total_time_seconds': round(cpu_total_time_s, 4),
            'average_cpu_percent_during_execution': round(avg_cpu_percent, 2),
            'test_iterations_for_resource': 100
        }
        
    except Exception as e:
        logger.error(f"❌ 资源消耗测量失败: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def print_summary(results: Dict[str, Any]):
    """打印实验四结果摘要"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("⚙️ 实验四结果摘要：轻量化部署验证")
    logger.info("="*60)
    
    # 全系统特性
    if 'full_system_characteristics' in results and 'error' not in results['full_system_characteristics']:
        full = results['full_system_characteristics']
        model = results.get('model_characteristics', {})
        cags = results.get('cags_engine_characteristics', {})
        
        logger.info("\n📦 全系统特性:")
        logger.info(f"  总大小: {full['total_system_size_mb']:.2f} MB")
        logger.info(f"    - 模型系统: {model.get('total_model_system_size_mb', 0):.2f} MB")
        logger.info(f"    - CAGS引擎: {cags.get('code_size_mb', 0):.3f} MB")
        logger.info(f"  边缘设备轻量化: {'✅ 是' if full['is_lightweight_for_edge'] else '❌ 否'}")
        logger.info(f"  IoT设备轻量化: {'✅ 是' if full['is_lightweight_for_iot'] else '❌ 否'}")
    
    # 模型特性
    if 'model_characteristics' in results and 'error' not in results['model_characteristics']:
        chars = results['model_characteristics']
        logger.info("\n🤖 模型特性 (CompactCFTNetV2):")
        logger.info(f"  总参数量: {chars['total_parameters']:,}")
        logger.info(f"  可训练参数量: {chars['trainable_parameters']:,}")
        logger.info(f"  客户端特征维度: {chars['client_feature_dim']}")
        logger.info(f"  镜像特征维度: {chars['image_feature_dim']}")
        logger.info(f"  Transformer层数: {chars['num_transformer_layers']}")
    
    # 运行时性能
    if 'runtime_performance' in results and 'error' not in results['runtime_performance']:
        perf = results['runtime_performance']
        logger.info("\n⚡ 运行时性能 (完整CAGS决策):")
        logger.info(f"  测试次数: {perf['test_iterations']}")
        logger.info(f"  平均决策延迟: {perf['cags_total']['average_ms']:.3f} ms")
        logger.info(f"  P95决策延迟: {perf['cags_total']['p95_ms']:.3f} ms")
        logger.info(f"  P99决策延迟: {perf['cags_total']['p99_ms']:.3f} ms")
        logger.info(f"  决策吞吐量: {perf['cags_total']['throughput_decisions_per_second']} 决策/秒")
        
        logger.info(f"\n  时间分解:")
        logger.info(f"    模型预测: {perf['model_prediction']['average_ms']:.3f} ms ({perf['model_prediction']['percentage_of_total']}%)")
        logger.info(f"    CAGS排序+选择: {perf['cags_overhead']['average_ms']:.3f} ms ({perf['cags_overhead']['percentage_of_total']}%)")
        
        if 'decision_overhead_analysis' in perf:
            overhead = perf['decision_overhead_analysis']
            logger.info(f"\n  🎯 决策开销占比分析:")
            logger.info(f"    参考传输时间: {overhead['reference_transmission_time_s']} s")
            logger.info(f"    决策开销占比: {overhead['decision_overhead_ratio_percent']:.4f}%")
            logger.info(f"    开销是否可接受: {'✅ 是' if overhead['is_overhead_acceptable'] else '❌ 否'} (<1%标准)")
    
    # 资源消耗
    if 'resource_consumption' in results and 'error' not in results['resource_consumption']:
        res = results['resource_consumption']
        logger.info("\n💾 资源消耗:")
        logger.info(f"  系统加载后内存: {res['loaded_system_memory_mb']:.2f} MB")
        logger.info(f"  内存增量(加载): {res['memory_increase_on_load_mb']:.2f} MB")
        logger.info(f"  内存增量(执行100次): {res['memory_increase_on_execution_mb']:.2f} MB")
        logger.info(f"  峰值内存分配: {res['peak_memory_allocation_mb']:.3f} MB")
        logger.info(f"  平均CPU使用率(执行时): {res['average_cpu_percent_during_execution']:.2f}%")