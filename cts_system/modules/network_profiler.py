"""
网络被动探测模块
通过观察客户端的网络行为来构建客户端能力画像
"""

import time
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import os


@dataclass
class NetworkProfile:
    """网络能力画像"""
    node_id: str  # 节点标识符
    avg_bandwidth_mbps: float = 0.0  # 平均带宽
    avg_rtt_ms: float = 0.0  # 平均RTT
    packet_loss_rate: float = 0.0  # 丢包率
    connection_stability: float = 1.0  # 连接稳定性
    last_updated: float = 0.0  # 最后更新时间
    transfer_count: int = 0  # 传输次数


class NetworkProfiler:
    """网络被动探测器"""
    
    def __init__(self, profile_storage_path: str = "profiles/network_profiles.json"):
        self.profile_storage_path = profile_storage_path
        self.profiles: Dict[str, NetworkProfile] = {}
        self.transfer_sessions: Dict[str, Dict[str, Any]] = {}
        self._load_profiles()
    
    def _get_node_id(self, client_ip: str, tls_fingerprint: Optional[str] = None) -> str:
        """
        生成节点标识符
        
        Args:
            client_ip: 客户端IP地址
            tls_fingerprint: TLS指纹（如果有）
            
        Returns:
            节点标识符
        """
        if tls_fingerprint:
            identifier = f"{client_ip}:{tls_fingerprint}"
        else:
            identifier = client_ip
            
        # 使用哈希确保标识符长度一致
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def start_transfer_session(self, session_id: str, client_ip: str, 
                             tls_fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """
        开始传输会话
        
        Args:
            session_id: 会话ID
            client_ip: 客户端IP
            tls_fingerprint: TLS指纹
            
        Returns:
            客户端网络配置
        """
        node_id = self._get_node_id(client_ip, tls_fingerprint)
        
        # 查找节点历史画像
        profile = self.profiles.get(node_id)
        
        session_info = {
            "session_id": session_id,
            "node_id": node_id,
            "client_ip": client_ip,
            "start_time": time.time(),
            "bytes_transferred": 0,
            "transfer_events": [],
            "initial_profile": asdict(profile) if profile else None
        }
        
        self.transfer_sessions[session_id] = session_info
        
        # 基于历史画像或默认值生成客户端配置
        if profile:
            client_profile = {
                "bandwidth_mbps": profile.avg_bandwidth_mbps,
                "network_rtt": profile.avg_rtt_ms,
                "packet_loss_rate": profile.packet_loss_rate,
                "connection_stability": profile.connection_stability
            }
        else:
            # 默认保守配置
            client_profile = {
                "bandwidth_mbps": 10.0,  # 默认10Mbps
                "network_rtt": 50.0,      # 默认50ms
                "packet_loss_rate": 0.01, # 默认1%丢包率
                "connection_stability": 0.9 # 默认稳定性0.9
            }
        
        return client_profile
    
    def record_transfer_progress(self, session_id: str, bytes_sent: int, 
                               timestamp: Optional[float] = None):
        """
        记录传输进度
        
        Args:
            session_id: 会话ID
            bytes_sent: 已发送字节数
            timestamp: 时间戳
        """
        if session_id not in self.transfer_sessions:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        session = self.transfer_sessions[session_id]
        session["bytes_transferred"] = bytes_sent
        session["transfer_events"].append({
            "timestamp": timestamp,
            "bytes_sent": bytes_sent
        })
    
    def end_transfer_session(self, session_id: str) -> NetworkProfile:
        """
        结束传输会话并更新画像
        
        Args:
            session_id: 会话ID
            
        Returns:
            更新后的网络画像
        """
        if session_id not in self.transfer_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.transfer_sessions[session_id]
        node_id = session["node_id"]
        
        # 分析传输行为
        metrics = self._analyze_transfer_behavior(session)
        
        # 更新或创建画像
        if node_id in self.profiles:
            profile = self.profiles[node_id]
            self._update_profile(profile, metrics)
        else:
            profile = NetworkProfile(
                node_id=node_id,
                avg_bandwidth_mbps=metrics["bandwidth_mbps"],
                avg_rtt_ms=metrics["avg_rtt_ms"],
                packet_loss_rate=metrics["packet_loss_rate"],
                connection_stability=metrics["connection_stability"],
                transfer_count=1
            )
            self.profiles[node_id] = profile
            
        profile.last_updated = time.time()
        
        # 保存画像
        self._save_profiles()
        
        # 清理会话
        del self.transfer_sessions[session_id]
        
        return profile
    
    def _analyze_transfer_behavior(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        分析传输行为
        
        Args:
            session: 会话信息
            
        Returns:
            分析指标
        """
        events = session["transfer_events"]
        if len(events) < 2:
            # 数据不足，返回默认值
            return {
                "bandwidth_mbps": 10.0,
                "avg_rtt_ms": 50.0,
                "packet_loss_rate": 0.01,
                "connection_stability": 0.9
            }
        
        # 计算带宽
        total_bytes = events[-1]["bytes_sent"] - events[0]["bytes_sent"]
        total_time = events[-1]["timestamp"] - events[0]["timestamp"]
        bandwidth_mbps = (total_bytes * 8) / (total_time * 1_000_000) if total_time > 0 else 10.0
        
        # 计算RTT（简化模型）
        avg_rtt_ms = 50.0  # 简化处理
        
        # 计算连接稳定性（基于传输连续性）
        interruptions = 0
        for i in range(1, len(events)):
            if events[i]["bytes_sent"] < events[i-1]["bytes_sent"]:
                interruptions += 1
                
        connection_stability = max(0.1, 1.0 - (interruptions / len(events)))
        
        # 丢包率（简化模型）
        packet_loss_rate = 0.01  # 简化处理
        
        return {
            "bandwidth_mbps": bandwidth_mbps,
            "avg_rtt_ms": avg_rtt_ms,
            "packet_loss_rate": packet_loss_rate,
            "connection_stability": connection_stability
        }
    
    def _update_profile(self, profile: NetworkProfile, new_metrics: Dict[str, float]):
        """
        更新画像
        
        Args:
            profile: 现有画像
            new_metrics: 新指标
        """
        # 使用移动平均更新指标
        alpha = 0.3  # 学习率
        
        profile.avg_bandwidth_mbps = (
            alpha * new_metrics["bandwidth_mbps"] + 
            (1 - alpha) * profile.avg_bandwidth_mbps
        )
        
        profile.avg_rtt_ms = (
            alpha * new_metrics["avg_rtt_ms"] + 
            (1 - alpha) * profile.avg_rtt_ms
        )
        
        profile.packet_loss_rate = (
            alpha * new_metrics["packet_loss_rate"] + 
            (1 - alpha) * profile.packet_loss_rate
        )
        
        profile.connection_stability = (
            alpha * new_metrics["connection_stability"] + 
            (1 - alpha) * profile.connection_stability
        )
        
        profile.transfer_count += 1
    
    def get_client_profile_for_decision(self, node_id: str) -> Dict[str, Any]:
        """
        获取用于决策的客户端配置
        
        Args:
            node_id: 节点ID
            
        Returns:
            客户端配置
        """
        profile = self.profiles.get(node_id)
        if not profile:
            # 返回默认配置
            return {
                "cpu_score": 1000,
                "bandwidth_mbps": 10.0,
                "decompression_speed": {
                    "gzip": 50,
                    "zstd": 100
                },
                "latency_requirement": 400
            }
        
        # 基于网络画像估算解压能力
        # 假设网络较好的节点通常计算能力也较强
        cpu_score = int(1000 + (profile.avg_bandwidth_mbps - 10) * 50)
        cpu_score = max(500, min(5000, cpu_score))  # 限制在合理范围内
        
        return {
            "cpu_score": cpu_score,
            "bandwidth_mbps": profile.avg_bandwidth_mbps,
            "decompression_speed": {
                "gzip": max(20, profile.avg_bandwidth_mbps * 5),  # 简化映射
                "zstd": max(40, profile.avg_bandwidth_mbps * 10)
            },
            "latency_requirement": 400
        }
    
    def _load_profiles(self):
        """加载已保存的画像"""
        try:
            if os.path.exists(self.profile_storage_path):
                with open(self.profile_storage_path, 'r') as f:
                    data = json.load(f)
                    for node_id, profile_data in data.items():
                        self.profiles[node_id] = NetworkProfile(**profile_data)
        except Exception as e:
            print(f"加载网络画像失败: {e}")
    
    def _save_profiles(self):
        """保存画像到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.profile_storage_path), exist_ok=True)
            
            # 转换为可序列化格式
            data = {}
            for node_id, profile in self.profiles.items():
                data[node_id] = asdict(profile)
            
            with open(self.profile_storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"保存网络画像失败: {e}")

    def cleanup_expired_profiles(self, expiry_days: int = 30):
        """
        清理过期画像
        
        Args:
            expiry_days: 过期天数
        """
        current_time = time.time()
        expiry_seconds = expiry_days * 24 * 3600
        
        expired_nodes = []
        for node_id, profile in self.profiles.items():
            if current_time - profile.last_updated > expiry_seconds:
                expired_nodes.append(node_id)
        
        for node_id in expired_nodes:
            del self.profiles[node_id]
        
        if expired_nodes:
            self._save_profiles()


def main():
    """测试网络被动探测器"""
    profiler = NetworkProfiler()
    
    # 模拟客户端会话
    client_ip = "192.168.1.100"
    session_id = "test_session_001"
    
    # 开始会话
    print("开始传输会话...")
    client_profile = profiler.start_transfer_session(session_id, client_ip)
    print(f"客户端配置: {client_profile}")
    
    # 模拟传输过程
    print("\n模拟传输过程...")
    start_time = time.time()
    for i in range(10):
        bytes_sent = (i + 1) * 1024 * 1024  # 每秒1MB
        timestamp = start_time + i
        profiler.record_transfer_progress(session_id, bytes_sent, timestamp)
        time.sleep(0.1)
    
    # 结束会话
    print("\n结束传输会话...")
    profile = profiler.end_transfer_session(session_id)
    print(f"更新后的网络画像: {asdict(profile)}")
    
    # 获取决策用配置
    decision_profile = profiler.get_client_profile_for_decision(profile.node_id)
    print(f"决策用配置: {decision_profile}")


if __name__ == "__main__":
    main()