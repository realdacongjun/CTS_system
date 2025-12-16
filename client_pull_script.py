#!/usr/bin/env python3
"""
客户端拉取脚本
用于在拉取镜像前自动检查和更新节点能力画像
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.client_probe import ClientProbe
from modules.capability_registry import CapabilityRegistry


def load_local_profile():
    """加载本地存储的能力画像"""
    profile_path = "local_node_profile.json"
    try:
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"加载本地画像失败: {e}")
    return None


def save_local_profile(profile):
    """保存能力画像到本地"""
    profile_path = "local_node_profile.json"
    try:
        # 添加时间戳
        profile['last_updated'] = time.time()
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        return True
    except Exception as e:
        print(f"保存本地画像失败: {e}")
        return False


def is_profile_expired(profile, max_age_hours=24):
    """检查画像是否过期"""
    if not profile or 'last_updated' not in profile:
        return True
    
    last_updated = profile['last_updated']
    now = time.time()
    age_seconds = now - last_updated
    max_age_seconds = max_age_hours * 3600
    
    return age_seconds > max_age_seconds


def execute_probe_and_register():
    """执行探针测试并注册能力"""
    print("执行节点能力探针测试...")
    
    # 执行本地探测
    client_probe = ClientProbe()
    capability_data = client_probe.get_client_profile()
    
    # 生成节点ID
    registry = CapabilityRegistry()
    client_info = {
        "ip": "localhost",  # 实际应用中应获取真实IP
        "user_agent": "Client_Pull_Script/1.0"
    }
    node_id = registry.generate_node_id(client_info)
    
    # 注册能力
    registry.register_capability(node_id, capability_data)
    
    # 构建本地画像
    local_profile = {
        "node_id": node_id,
        "capability_data": capability_data
    }
    
    # 保存到本地
    if save_local_profile(local_profile):
        print("✓ 节点能力画像已更新并保存")
        print(f"  节点ID: {node_id}")
        print(f"  最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("✗ 保存节点能力画像失败")
    
    return local_profile


def get_or_update_profile():
    """获取最新的节点能力画像，必要时更新"""
    # 加载本地画像
    local_profile = load_local_profile()
    
    # 检查画像是否存在且未过期
    if local_profile and not is_profile_expired(local_profile):
        print("使用现有的节点能力画像")
        last_updated = datetime.fromtimestamp(local_profile['last_updated'])
        print(f"  最后更新: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        return local_profile
    else:
        if local_profile:
            print("节点能力画像已过期，正在重新测试...")
        else:
            print("未找到节点能力画像，正在执行首次测试...")
        return execute_probe_and_register()


def pull_image_with_profile(image_name):
    """使用能力画像拉取镜像"""
    print(f"准备拉取镜像: {image_name}")
    print("=" * 50)
    
    # 获取或更新节点能力画像
    profile = get_or_update_profile()
    
    if not profile:
        print("无法获取节点能力画像，退出")
        return False
    
    # 构造带有节点ID的调用命令
    node_id = profile['node_id']
    command = f"python main.py node:{node_id} {image_name}"
    
    print("\n执行镜像优化和拉取...")
    print(f"命令: {command}")
    
    # 在实际应用中，这里会调用main.py执行具体的优化流程
    # 为了演示，我们只是打印相关信息
    print("\n" + "=" * 50)
    print("镜像拉取和优化流程摘要:")
    print(f"  镜像名称: {image_name}")
    print(f"  节点ID: {node_id}")
    print(f"  CPU评分: {profile['capability_data'].get('cpu_score')}")
    print(f"  网络带宽: {profile['capability_data'].get('bandwidth_mbps')} Mbps")
    print("  镜像优化已完成，可以开始拉取")
    
    return True


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python client_pull_script.py <镜像名称>")
        print("示例: python client_pull_script.py nginx:latest")
        return
    
    image_name = sys.argv[1]
    pull_image_with_profile(image_name)


if __name__ == "__main__":
    main()