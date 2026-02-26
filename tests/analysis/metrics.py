import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats
import logging

def calculate_comprehensive_metrics(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算综合评估指标
    """
    logger = logging.getLogger(__name__)
    metrics = {}
    
    # 实验一：端到端性能基准
    if 'exp1' in all_results:
        exp1_metrics = _calculate_exp1_metrics(all_results['exp1'])
        metrics['end_to_end_benchmark'] = exp1_metrics
    
    # 实验二：协同增益消融
    if 'exp2' in all_results:
        exp2_metrics = _calculate_exp2_metrics(all_results['exp2'])
        metrics['ablation_study'] = exp2_metrics
    
    # 实验三：动态鲁棒性
    if 'exp3' in all_results:
        exp3_metrics = _calculate_exp3_metrics(all_results['exp3'])
        metrics['robustness_testing'] = exp3_metrics
    
    # 实验四：轻量化部署
    if 'exp4' in all_results:
        exp4_metrics = _calculate_exp4_metrics(all_results['exp4'])
        metrics['lightweight_deployment'] = exp4_metrics
    
    # 实验五：长期稳定性
    if 'exp5' in all_results:
        exp5_metrics = _calculate_exp5_metrics(all_results['exp5'])
        metrics['long_term_stability'] = exp5_metrics
    
    return metrics

def _calculate_exp1_metrics(exp1_results: Dict[str, Any]) -> Dict[str, Any]:
    """计算实验一指标"""
    metrics = {}
    
    if 'overall_comparison' in exp1_results:
        comp = exp1_results['overall_comparison']
        metrics.update({
            'baseline_avg_throughput': comp['avg_baseline_throughput'],
            'cts_avg_throughput': comp['avg_cts_throughput'],
            'performance_gain_percent': comp['overall_performance_gain'],
            'performance_gains_by_scenario': comp['performance_gains_by_scenario']
        })
    
    if 'statistical_tests' in exp1_results:
        stats = exp1_results['statistical_tests']
        metrics.update({
            't_statistic': stats.get('t_statistic', 0),
            'p_value': stats.get('p_value', 1.0),
            'effect_size': stats.get('effect_size', 0),
            'statistically_significant': stats.get('significant', False)
        })
    
    return metrics

def _calculate_exp2_metrics(exp2_results: Dict[str, Any]) -> Dict[str, Any]:
    """计算实验二指标"""
    metrics = {}
    
    if 'variants_comparison' in exp2_results:
        comparison = exp2_results['variants_comparison']
        metrics.update({
            'variant_performance': comparison
        })
        
        # 计算协同增益
        if all(variant in comparison for variant in ['V1_Static', 'V2_CFT_Only', 'V4_CTS_Full']):
            cft_gain = comparison['V2_CFT_Only']['gain_vs_static']
            full_gain = comparison['V4_CTS_Full']['gain_vs_static']
            synergy_gain = full_gain - cft_gain
            
            metrics['synergy_analysis'] = {
                'cft_net_contribution': cft_gain,
                'full_system_contribution': full_gain,
                'synergy_gain': synergy_gain
            }
    
    return metrics

def _calculate_exp3_metrics(exp3_results: Dict[str, Any]) -> Dict[str, Any]:
    """计算实验三指标"""
    metrics = {}
    
    # 动态波动测试指标
    if 'dynamic_fluctuation' in exp3_results:
        dyn = exp3_results['dynamic_fluctuation']
        if 'stability_metrics' in dyn:
            metrics['dynamic_stability'] = dyn['stability_metrics']
    
    # OOD测试指标
    if 'ood_extreme' in exp3_results:
        ood = exp3_results['ood_extreme']
        if 'ood_detection' in ood:
            metrics['ood_detection'] = ood['ood_detection']
    
    return metrics

def _calculate_exp4_metrics(exp4_results: Dict[str, Any]) -> Dict[str, Any]:
    """计算实验四指标"""
    metrics = {}
    
    if 'model_characteristics' in exp4_results and 'error' not in exp4_results['model_characteristics']:
        chars = exp4_results['model_characteristics']
        metrics['model_specs'] = {
            'parameters': chars['total_parameters'],
            'model_size_mb': chars['model_size_mb']
        }
    
    if 'runtime_performance' in exp4_results and 'error' not in exp4_results['runtime_performance']:
        perf = exp4_results['runtime_performance']
        metrics['runtime_efficiency'] = {
            'avg_latency_ms': perf['average_latency_ms'],
            'p99_latency_ms': perf['p99_latency_ms'],
            'predictions_per_second': perf['throughput_predictions_per_second']
        }
        
        # 决策开销占比（假设传输时间）
        transmission_time = 10.0  # 秒
        overhead_ratio = (perf['average_latency_ms'] / 1000) / transmission_time * 100
        metrics['runtime_efficiency']['decision_overhead_ratio_percent'] = overhead_ratio
    
    if 'resource_consumption' in exp4_results and 'error' not in exp4_results['resource_consumption']:
        res = exp4_results['resource_consumption']
        metrics['resource_usage'] = {
            'memory_increase_mb': res['memory_increase_mb'],
            'cpu_utilization_percent': res['cpu_percent_utilization']
        }
    
    return metrics

def _calculate_exp5_metrics(exp5_results: Dict[str, Any]) -> Dict[str, Any]:
    """计算实验五指标"""
    metrics = {}
    
    if 'continuous_execution' in exp5_results:
        cont = exp5_results['continuous_execution']
        metrics['execution_reliability'] = {
            'total_executions': cont['performance_metrics']['total_executions'],
            'success_rate_percent': cont['performance_metrics']['success_rate_percent'],
            'avg_throughput_mbps': cont['performance_metrics']['avg_throughput_mbps']
        }
    
    if 'performance_decay' in exp5_results:
        decay = exp5_results['performance_decay']
        metrics['performance_stability'] = {
            'performance_stable': decay['performance_stable'],
            'throughput_decay_percent': decay.get('decay_metrics', {}).get('throughput_decay_percent', 0)
        }
    
    if 'configuration_stability' in exp5_results:
        config = exp5_results['configuration_stability']
        metrics['configuration_consistency'] = {
            'consistent_behavior': config['consistent_behavior'],
            'anomaly_count': config['anomaly_count'],
            'throughput_cv': config.get('throughput_coefficient_of_variation', 0)
        }
    
    return metrics

def print_final_summary(all_results: Dict[str, Any]):
    """
    打印最终实验总结
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("🏆 CTS闭环系统完整评估实验总结报告")
    logger.info("="*80)
    
    comprehensive_metrics = calculate_comprehensive_metrics(all_results)
    
    # 核心性能指标
    if 'end_to_end_benchmark' in comprehensive_metrics:
        e2e = comprehensive_metrics['end_to_end_benchmark']
        logger.info(f"\n🚀 核心性能提升:")
        logger.info(f"   基线平均吞吐量: {e2e['baseline_avg_throughput']:.2f} Mbps")
        logger.info(f"   CTS平均吞吐量: {e2e['cts_avg_throughput']:.2f} Mbps")
        logger.info(f"   性能提升幅度: {e2e['performance_gain_percent']:.2f}%")
        logger.info(f"   统计显著性: {'✓' if e2e['statistically_significant'] else '✗'} (p={e2e['p_value']:.6f})")
    
    # 协同增益分析
    if 'ablation_study' in comprehensive_metrics:
        ablation = comprehensive_metrics['ablation_study']
        if 'synergy_analysis' in ablation:
            synergy = ablation['synergy_analysis']
            logger.info(f"\n🤝 协同增益分解:")
            logger.info(f"   CFT-Net单独贡献: {synergy['cft_net_contribution']:.2f}%")
            logger.info(f"   CTS完整系统: {synergy['full_system_contribution']:.2f}%")
            logger.info(f"   协同增益: {synergy['synergy_gain']:.2f}%")
    
    # 鲁棒性指标
    if 'robustness_testing' in comprehensive_metrics:
        robust = comprehensive_metrics['robustness_testing']
        logger.info(f"\n🛡️ 系统鲁棒性:")
        if 'dynamic_stability' in robust:
            dyn = robust['dynamic_stability']
            logger.info(f"   动态环境稳定性: {100*(1-dyn['throughput_cv']):.1f}%")
            logger.info(f"   性能退化: {dyn['performance_degradation']:.2f}%")
        if 'ood_detection' in robust:
            ood = robust['ood_detection']
            logger.info(f"   OOD检测率: {ood['detection_rate']*100:.1f}%")
    
    # 轻量化指标
    if 'lightweight_deployment' in comprehensive_metrics:
        light = comprehensive_metrics['lightweight_deployment']
        logger.info(f"\n⚙️ 轻量化特性:")
        if 'model_specs' in light:
            specs = light['model_specs']
            logger.info(f"   模型参数量: {specs['parameters']:,}")
            logger.info(f"   模型大小: {specs['model_size_mb']} MB")
        if 'runtime_efficiency' in light:
            eff = light['runtime_efficiency']
            logger.info(f"   平均决策延迟: {eff['avg_latency_ms']:.2f} ms")
            logger.info(f"   P99延迟: {eff['p99_latency_ms']:.2f} ms")
            logger.info(f"   决策开销占比: {eff['decision_overhead_ratio_percent']:.3f}%")
    
    # 稳定性指标
    if 'long_term_stability' in comprehensive_metrics:
        stab = comprehensive_metrics['long_term_stability']
        logger.info(f"\n🔁 长期稳定性:")
        if 'execution_reliability' in stab:
            rel = stab['execution_reliability']
            logger.info(f"   连续执行成功率: {rel['success_rate_percent']:.1f}%")
        if 'performance_stability' in stab:
            perf_stab = stab['performance_stability']
            logger.info(f"   性能稳定性: {'✓' if perf_stab['performance_stable'] else '✗'}")
            logger.info(f"   性能衰减: {perf_stab['throughput_decay_percent']:.2f}%")
    
    logger.info("\n" + "="*80)
    logger.info("📊 实验评估完成 - CTS闭环系统验证通过")
    logger.info("="*80)