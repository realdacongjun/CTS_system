"""
日志分析工具
功能：实时监控实验状态，分析实验日志，检测异常情况
输入：实验日志文件
输出：分析报告和异常检测结果
"""

import os
import json
import re
import time
import sqlite3
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import pandas as pd


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir: str = "/tmp/exp_data"):
        """
        初始化日志分析器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        self.db_path = os.path.join(log_dir, "experiment_manifest.db")
        self.status_counts = {
            'COMPLETED': 0,
            'FAILED': 0,
            'PENDING': 0,
            'PROCESSING': 0
        }
        
    def analyze_logs(self) -> Dict[str, Any]:
        """分析日志文件"""
        print(f"开始分析日志目录: {self.log_dir}")
        
        # 重置统计计数
        self.status_counts = {
            'COMPLETED': 0,
            'FAILED': 0,
            'PENDING': 0,
            'PROCESSING': 0
        }
        
        # 从数据库获取实验记录
        all_records = []
        error_records = []
        
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query("SELECT * FROM experiments", conn)
                conn.close()
                
                # 转换为字典列表
                all_records = df.to_dict('records')
                
                # 筛选错误记录
                error_records = [r for r in all_records if r.get('error_msg')]
                
                # 统计状态
                for record in all_records:
                    status = record.get('status', 'UNKNOWN').upper()
                    if status in self.status_counts:
                        self.status_counts[status] += 1
                    else:
                        self.status_counts['PENDING'] += 1
                        
            except Exception as e:
                print(f"从数据库读取实验记录失败: {e}")
        
        # 从JSON文件获取实验数据（作为备用）
        log_files = [f for f in os.listdir(self.log_dir) 
                    if f.endswith('.json') and f not in ['aggregated_results.json', 'completed_experiments.json']]
        
        for filename in log_files:
            filepath = os.path.join(self.log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # 如果数据不是列表，转换为列表
                    if not isinstance(data, list):
                        data = [data]
                    
                    for record in data:
                        all_records.append(record)
                        
                        # 检查是否有错误
                        if 'error_msg' in record and record['error_msg']:
                            error_records.append(record)
                            
                        # 统计状态
                        status = record.get('status', 'COMPLETED').upper()
                        if status in self.status_counts:
                            self.status_counts[status] += 1
                        else:
                            self.status_counts['COMPLETED'] += 1
                            
            except Exception as e:
                print(f"读取文件 {filename} 失败: {e}")

        # 生成分析结果
        analysis_result = {
            'total_records': len(all_records),
            'error_records': len(error_records),
            'status_counts': self.status_counts,
            'error_rate': len(error_records) / len(all_records) if all_records else 0,
            'error_records_list': error_records[:10],  # 只返回前10个错误记录
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查是否存在大量失败
        if analysis_result['error_rate'] > 0.1:  # 错误率超过10%
            analysis_result['alert'] = 'HIGH_ERROR_RATE'
            analysis_result['message'] = f'警告: 错误率过高 ({analysis_result["error_rate"]:.2%})'
        
        return analysis_result
    
    def generate_status_report(self) -> Dict[str, Any]:
        """生成状态报告"""
        analysis = self.analyze_logs()
        
        # 从数据库获取更详细的统计信息
        config_stats = {}
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                
                # 查询按配置分组的统计信息
                query = """
                SELECT 
                    profile_id, 
                    image_name, 
                    method,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(duration) as avg_duration,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration
                FROM experiments 
                GROUP BY profile_id, image_name, method
                """
                
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                for _, row in df.iterrows():
                    config_key = f"{row['profile_id']}_{row['image_name']}_{row['method']}"
                    config_stats[config_key] = {
                        'total': int(row['total']),
                        'completed': int(row['completed']),
                        'failed': int(row['failed']),
                        'avg_duration': row['avg_duration'],
                        'min_duration': row['min_duration'],
                        'max_duration': row['max_duration']
                    }
        
        report = {
            'analysis': analysis,
            'config_stats': config_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """检测实验中的异常数据"""
        print("正在检测异常数据...")
        
        # 从数据库获取实验数据
        if not os.path.exists(self.db_path):
            return {'anomalies': [], 'message': '数据库不存在'}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 查询所有完成的实验
            query = """
            SELECT 
                profile_id, 
                image_name, 
                method,
                duration,
                experiment_id,
                host_cpu_load,
                host_memory_usage
            FROM experiments 
            WHERE status = 'completed'
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # 按配置分组计算统计信息
            grouped = df.groupby(['profile_id', 'image_name', 'method'])
            
            anomalies = []
            for (profile, image, method), group in grouped:
                durations = group['duration'].values
                
                if len(durations) >= 3:  # 需要至少3个数据点才能计算方差
                    mean_duration = durations.mean()
                    std_duration = durations.std()
                    
                    # 标记超过均值2个标准差的数据点为异常
                    for idx, duration in enumerate(durations):
                        if std_duration > 0 and abs(duration - mean_duration) > 2 * std_duration:
                            anomaly_record = group.iloc[idx]
                            anomalies.append({
                                'experiment_id': anomaly_record['experiment_id'],
                                'profile_id': profile,
                                'image_name': image,
                                'method': method,
                                'duration': duration,
                                'mean_duration': mean_duration,
                                'std_duration': std_duration,
                                'z_score': abs(duration - mean_duration) / std_duration if std_duration > 0 else 0,
                                'host_cpu_load': anomaly_record['host_cpu_load'],
                                'host_memory_usage': anomaly_record['host_memory_usage']
                            })
            
            print(f"检测到 {len(anomalies)} 个异常数据点")
            return {
                'anomalies': anomalies,
                'total_experiments': len(df),
                'anomaly_rate': len(anomalies) / len(df) if len(df) > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"检测异常数据失败: {e}")
            return {'anomalies': [], 'error': str(e)}
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """保存分析报告"""
        if not filename:
            filename = f"log_analysis_report_{int(time.time())}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"分析报告已保存至: {filepath}")
        return filepath
    
    def monitor_realtime(self, interval: int = 30, duration: int = None) -> List[Dict[str, Any]]:
        """
        实时监控日志状态
        
        Args:
            interval: 监控间隔（秒）
            duration: 监控持续时间（秒），None表示持续监控直到手动停止
        """
        print(f"开始实时监控，间隔 {interval} 秒...")
        
        start_time = time.time()
        reports = []
        
        try:
            while True:
                if duration and (time.time() - start_time) >= duration:
                    break
                
                report = self.generate_status_report()
                reports.append(report)
                
                # 检查是否有需要警告的情况
                if 'alert' in report['analysis']:
                    print(f"⚠️  {report['analysis']['message']}")
                
                # 每次监控后等待指定间隔
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n实时监控被用户中断")
        
        return reports


def main():
    """主函数，运行日志分析"""
    print("启动日志分析工具...")
    
    # 创建分析器
    analyzer = LogAnalyzer()
    
    # 生成状态报告
    report = analyzer.generate_status_report()
    
    # 检测异常数据
    anomalies = analyzer.detect_anomalies()
    
    # 合并报告
    full_report = {
        'status_report': report,
        'anomaly_detection': anomalies,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存报告
    report_path = analyzer.save_report(full_report)
    
    # 打印摘要
    analysis = report['analysis']
    print(f"\n=== 日志分析摘要 ===")
    print(f"总记录数: {analysis['total_records']}")
    print(f"错误记录数: {analysis['error_records']}")
    print(f"错误率: {analysis['error_rate']:.2%}")
    print(f"状态统计: {analysis['status_counts']}")
    
    if 'alert' in analysis:
        print(f"⚠️  {analysis['message']}")
    
    print(f"\n异常检测结果:")
    print(f"异常数据点: {len(anomalies['anomalies'])}")
    print(f"异常率: {anomalies['anomaly_rate']:.2%}")
    
    print(f"\n分析报告已保存至: {report_path}")
    
    # 询问是否启动实时监控
    response = input("\n是否启动实时监控？(y/n): ").lower()
    if response == 'y':
        print("启动实时监控（按 Ctrl+C 停止）...")
        analyzer.monitor_realtime(interval=10)


if __name__ == "__main__":
    main()