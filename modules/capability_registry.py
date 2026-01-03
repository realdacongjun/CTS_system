"""
能力注册模块
实现节点能力画像的注册、存储和管理
"""

import json
import os
import hashlib
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class NodeCapability:
    """节点能力画像"""
    node_id: str
    cpu_score: int = 1000
    bandwidth_mbps: float = 10.0
    decompression_speed: Dict[str, float] = None
    system_info: Dict[str, Any] = None
    last_updated: float = 0.0
    confidence: float = 0.5  # 置信度
    # 新增客户端画像字段
    network_rtt: float = 0.0      # 网络往返时间 (ms)
    disk_io_speed: float = 0.0    # 磁盘I/O速度 (MB/s)
    memory_size: int = 0          # 内存大小 (MB)


class CapabilityRegistry:
    """能力注册中心"""
    
    def __init__(self, registry_path: str = "registry/capabilities.json"):
        self.registry_path = registry_path
        self.capabilities: Dict[str, NodeCapability] = {}
        self.lock = Lock()
        self._load_registry()
    
    def register_capability(self, node_id: str, capability_data: Dict[str, Any]) -> bool:
        """
        注册节点能力
        
        Args:
            node_id: 节点ID
            capability_data: 能力数据
            
        Returns:
            是否注册成功
        """
        with self.lock:
            # 创建或更新能力画像
            if node_id in self.capabilities:
                capability = self.capabilities[node_id]
                # 更新现有画像
                capability.cpu_score = capability_data.get("cpu_score", capability.cpu_score)
                capability.bandwidth_mbps = capability_data.get("bandwidth_mbps", capability.bandwidth_mbps)
                capability.decompression_speed = capability_data.get("decompression_speed", capability.decompression_speed)
                capability.system_info = capability_data.get("system_info", capability.system_info)
                capability.confidence = min(1.0, capability.confidence + 0.1)  # 增加置信度
                # 更新新增字段
                capability.network_rtt = capability_data.get("network_rtt", capability.network_rtt)
                capability.disk_io_speed = capability_data.get("disk_io_speed", capability.disk_io_speed)
                capability.memory_size = capability_data.get("memory_size", capability.memory_size)
            else:
                # 创建新画像
                capability = NodeCapability(
                    node_id=node_id,
                    cpu_score=capability_data.get("cpu_score", 1000),
                    bandwidth_mbps=capability_data.get("bandwidth_mbps", 10.0),
                    decompression_speed=capability_data.get("decompression_speed", {"gzip": 50, "zstd": 100}),
                    system_info=capability_data.get("system_info"),
                    confidence=0.8,
                    # 初始化新增字段
                    network_rtt=capability_data.get("network_rtt", 0.0),
                    disk_io_speed=capability_data.get("disk_io_speed", 0.0),
                    memory_size=capability_data.get("memory_size", 0)
                )
                self.capabilities[node_id] = capability
            
            capability.last_updated = time.time()
            self._save_registry()
            return True
    
    def get_capability(self, node_id: str) -> Optional[NodeCapability]:
        """
        获取节点能力画像
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点能力画像或None
        """
        with self.lock:
            return self.capabilities.get(node_id)
    
    def generate_node_id(self, client_info: Dict[str, Any]) -> str:
        """
        根据客户端信息生成节点ID
        
        Args:
            client_info: 客户端信息
            
        Returns:
            节点ID
        """
        # 基于客户端信息生成唯一标识
        info_str = f"{client_info.get('ip', '')}:{client_info.get('user_agent', '')}"
        return hashlib.sha256(info_str.encode()).hexdigest()[:16]
    
    def get_client_profile_for_decision(self, node_id: str) -> Dict[str, Any]:
        """
        获取用于决策的客户端配置
        
        Args:
            node_id: 节点ID
            
        Returns:
            客户端配置
        """
        capability = self.get_capability(node_id)
        if not capability:
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
        
        return {
            "node_id": capability.node_id,
            "cpu_score": capability.cpu_score,
            "bandwidth_mbps": capability.bandwidth_mbps,
            "decompression_speed": capability.decompression_speed or {"gzip": 50, "zstd": 100},
            "network_rtt": capability.network_rtt,
            "disk_io_speed": capability.disk_io_speed,
            "memory_size": capability.memory_size,
            "confidence": capability.confidence,
            "latency_requirement": 400
        }
    
    def _load_registry(self):
        """从文件加载能力注册表"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_id, cap_data in data.items():
                        # 直接从字典重建对象，dataclass会处理类型
                        capability = NodeCapability(**cap_data)
                        self.capabilities[node_id] = capability
            except Exception as e:
                print(f"加载能力注册表失败: {e}")
    
    def _save_registry(self):
        """保存能力注册表到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            
            data = {}
            for node_id, capability in self.capabilities.items():
                data[node_id] = asdict(capability)
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存能力注册表失败: {e}")
    
    def cleanup_expired(self, expiry_days: int = 30):
        """
        清理过期的能力画像
        
        Args:
            expiry_days: 过期天数
        """
        with self.lock:
            current_time = time.time()
            expiry_seconds = expiry_days * 24 * 3600
            
            expired_nodes = []
            for node_id, capability in self.capabilities.items():
                if current_time - capability.last_updated > expiry_seconds:
                    expired_nodes.append(node_id)
            
            for node_id in expired_nodes:
                del self.capabilities[node_id]
            
            if expired_nodes:
                self._save_registry()

    def get_all_capabilities(self) -> Dict[str, NodeCapability]:
        """
        获取所有节点能力画像
        
        Returns:
            所有节点能力画像字典
        """
        with self.lock:
            return self.capabilities.copy()


def main():
    """测试能力注册中心"""
    registry = CapabilityRegistry()
    
    # 生成节点ID
    client_info = {
        "ip": "192.168.1.100",
        "user_agent": "DockerClient/20.10.0"
    }
    node_id = registry.generate_node_id(client_info)
    print(f"生成节点ID: {node_id}")
    
    # 注册能力
    capability_data = {
        "cpu_score": 3200,
        "bandwidth_mbps": 45.5,
        "decompression_speed": {
            "gzip": 180,
            "zstd": 420
        },
        "system_info": {
            "os": "Ubuntu 20.04",
            "arch": "x86_64"
        }
    }
    
    print("注册节点能力...")
    registry.register_capability(node_id, capability_data)
    
    # 获取能力
    print("获取节点能力...")
    capability = registry.get_capability(node_id)
    if capability:
        print(f"节点能力: {asdict(capability)}")
    
    # 获取决策配置
    decision_profile = registry.get_client_profile_for_decision(node_id)
    print(f"决策配置: {decision_profile}")


if __name__ == "__main__":
    main()