"""
实验编排模块
功能：自动化控制大规模实验流程，管理容器环境和数据收集
输入：实验配置参数
输出：实验结果数据
"""

import os
import subprocess
import time
import json
import uuid
import docker
from typing import Dict, Any, List
from .config import get_client_capabilities, get_image_profiles, get_compression_config
import threading
import signal
import sys
from pathlib import Path
import shutil
import sqlite3
import tempfile
import subprocess


class ExperimentOrchestrator:
    CLIENT_PROFILES = {
        "C1": {"cpu": 0.2, "memory": "512m", "bandwidth": "2mbit", "rtt": "200ms", "disk_io": "5mb/s"},
        "C2": {"cpu": 0.5, "memory": "1g", "bandwidth": "20mbit", "rtt": "50ms", "disk_io": "10mb/s"},
        "C3": {"cpu": 1.0, "memory": "2g", "bandwidth": "5mbit", "rtt": "100ms", "disk_io": "50mb/s"},
        "C4": {"cpu": 1.5, "memory": "2g", "bandwidth": "50mbit", "rtt": "20ms", "disk_io": "150mb/s"},
        "C5": {"cpu": 0.8, "memory": "2g", "bandwidth": "100mbit", "rtt": "10ms", "disk_io": "200mb/s"},
        "C6": {"cpu": 4.0, "memory": "4g", "bandwidth": "500mbit", "rtt": "5ms", "disk_io": "500mb/s"}
    }
    
    def __init__(self, cloud_mode=False):
        self.cloud_mode = cloud_mode
        self.docker_client = docker.from_env()
        self.check_environment()

    def check_environment(self):
        """环境依赖预检 - 根据项目规范必须执行"""
        # 检查tc命令
        if not self.cloud_mode and not shutil.which("tc"):
            print("警告: tc命令不可用，将自动切换到云模式")
            self.cloud_mode = True

        # 检查sudo权限
        if not self.cloud_mode:
            try:
                subprocess.run(["sudo", "-n", "true"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("警告: 无sudo权限，无法使用tc。切换到云模式")
                self.cloud_mode = True

        # 检查Pumba（云模式必需）
        if self.cloud_mode and not shutil.which("pumba"):
            raise RuntimeError("云模式需要Pumba，但未安装。请安装: https://github.com/alexei-led/pumba")

    def run_experiment(self, client_type, image_name, algorithm):
        """执行单个实验，确保资源清理闭环"""
        container = None
        temp_dir = None
        try:
            # 1. 清理缓存 - 保障性能测量准确性
            subprocess.run(["sync"], check=True)
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            
            # 2. 启动容器并挂载临时目录
            container, temp_dir = self.start_container(client_type)
            
            # 3. 应用网络限制
            if not self.cloud_mode:
                self.apply_tc_rules(container, client_type)
            else:
                self.apply_pumba_rules(container, client_type)
            
            # 4. 执行实验
            result = self.execute_pull(container, image_name, algorithm)
            
            # 5. 验证数据质量
            return self.validate_experiment_data(result)
        finally:
            # 6. 清理资源（确保执行）
            if container:
                self.cleanup_container(container)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if not self.cloud_mode:
                self.cleanup_tc_rules(client_type)

    def validate_experiment_data(self, result):
        """数据质量校验 - 根据实验数据质量控制规范"""
        target_bandwidth = self.CLIENT_PROFILES[result['client_type']]['bandwidth']
        actual_bandwidth = result.get('actual_bandwidth', 0)
        
        # 带宽偏差检查
        if actual_bandwidth < 0.5 * target_bandwidth or actual_bandwidth > 1.5 * target_bandwidth:
            result['is_noisy_data'] = True
            result['noise_reason'] = f"带宽偏差过大: 目标{target_bandwidth}, 实测{actual_bandwidth}"

        # 带宽波动率检查
        bandwidth_std = result.get('bandwidth_std', 0)
        bandwidth_mean = result.get('bandwidth_mean', 0)
        if bandwidth_mean > 0:
            cv = bandwidth_std / bandwidth_mean
            if cv > 0.3:
                result['is_noisy_data'] = True
                result['noise_reason'] = f"带宽波动率过高: CV={cv:.2f}"
        
        return result

    def start_container(self, client_type):
        """启动容器并挂载临时目录 - 解决存储管理问题"""
        profile = self.CLIENT_PROFILES[client_type]
        temp_dir = tempfile.mkdtemp(prefix=f"exp_{client_type}_")
        
        container = self.docker_client.containers.run(
            "cts-system/client-agent:latest",
            detach=True,
            volumes={temp_dir: {'bind': '/tmp/experiment', 'mode': 'rw'}},
            nano_cpus=int(profile['cpu'] * 1e9),
            mem_limit=profile['memory'],
            # 其他资源限制...
        )
        return container, temp_dir

    def apply_tc_rules(self, container, client_type):
        """应用tc规则 - 本地模式"""
        veth_name = self.get_container_veth_interface(container.id)
        if not veth_name:
            raise RuntimeError("无法找到容器的veth接口")
        
        subprocess.run([
            "sudo", "tc", "qdisc", "del", "dev", veth_name, "root"
        ], check=True)
        
        subprocess.run([
            "sudo", "tc", "qdisc", "add", "dev", veth_name, "root", "netem",
            "rate", self.CLIENT_PROFILES[client_type]['bandwidth'],
            "delay", self.CLIENT_PROFILES[client_type]['rtt']
        ], check=True)

    def apply_pumba_rules(self, container, client_type):
        """应用Pumba规则 - 云模式"""
        subprocess.run([
            "pumba", "netem", "--duration", "60s", "delay",
            "--time", self.CLIENT_PROFILES[client_type]['rtt'],
            "--rate", self.CLIENT_PROFILES[client_type]['bandwidth'],
            "container", container.id
        ], check=True)

    def execute_pull(self, container, image_name, algorithm):
        """执行拉取操作 - 核心实验逻辑"""
        result = container.exec_run(
            f"python3 /app/client_pull_script.py {image_name} --method {algorithm}",
            stdout=True,
            stderr=True,
            detach=False
        )
        
        output = result.output.decode('utf-8')
        exit_code = result.exit_code
        
        if exit_code != 0:
            raise RuntimeError(f"拉取失败: {output}")
        
        # 解析输出
        for line in reversed(output.strip().split('\n')):
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except:
                    continue
        
        raise RuntimeError("未找到有效的JSON输出")

    def cleanup_container(self, container):
        """清理容器"""
        container.stop()
        container.remove()

    def cleanup_tc_rules(self, client_type):
        """清理tc规则"""
        veth_name = self.get_container_veth_interface(container.id)
        if veth_name:
            subprocess.run([
                "sudo", "tc", "qdisc", "del", "dev", veth_name, "root"
            ], check=True)

    def get_container_veth_interface(self, container_id):
        """
        通用、稳健的方法：通过 iflink 索引查找容器在宿主机侧对应的 veth 网卡名称
        """
        try:
            # 1. 在容器内部获取 eth0 对应的宿主机网卡索引号 (iflink)
            # 执行: cat /sys/class/net/eth0/iflink
            cmd_get_iflink = f"docker exec {container_id} cat /sys/class/net/eth0/iflink"
            iflink_idx = subprocess.check_output(cmd_get_iflink, shell=True).decode().strip()
            
            # 2. 在宿主机侧遍历所有网络接口，寻找索引号匹配的那个
            # 宿主机的网卡索引信息存放在 /sys/class/net/<iface>/ifindex
            for iface in os.listdir('/sys/class/net/'):
                ifindex_path = f'/sys/class/net/{iface}/ifindex'
                if os.path.exists(ifindex_path):
                    with open(ifindex_path, 'r') as f:
                        if f.read().strip() == iflink_idx:
                            # 排除掉回环网卡等干扰
                            if iface != 'lo':
                                return iface
                                
            return None
        except Exception as e:
            print(f"❌ 定位容器 {container_id} 的 veth 接口失败: {e}")
            return None

    def _setup_tc_limit(self, veth_name: str, bandwidth_mbps: int, network_rtt: int):
        """
        使用 netem 直接控制网卡速率和延迟 (比 TBF 更适合云环境虚拟网卡)
        """
        try:
            # 清理旧规则
            subprocess.run(f"sudo tc qdisc del dev {veth_name} root", shell=True, capture_output=True)
            
            # 核心优化：直接使用 netem 的 rate 参数，简单稳健
            # 计算延迟的一半作为平滑值 (optional)
            tc_cmd = (f"sudo tc qdisc add dev {veth_name} root netem "
                      f"rate {bandwidth_mbps}mbit "
                      f"delay {network_rtt}ms 2ms distribution normal")
            
            subprocess.run(tc_cmd, shell=True, check=True)
            print(f"成功设置网络限制: {bandwidth_mbps}Mbps, {network_rtt}ms延迟")
            return True
        except Exception as e:
            print(f"❌ Tc 设置失败 [{veth_name}]: {e}")
            return False

    def _setup_pumba_limit(self, container_id: str, bandwidth_mbps: int, network_rtt: int):
        """
        使用Pumba设置网络限制（云模式下）
        """
        try:
            # 停止可能存在的pumba进程
            subprocess.run(['pkill', '-f', f'pumba.*{container_id}'], capture_output=True)
            
            # 使用pumba设置网络限制
            cmd = [
                'pumba', 'netem', '--duration', '60s', 'delay',
                f'--time={network_rtt}ms', '--jitter=2ms',
                '--rate', f'{bandwidth_mbps}mbit',
                'container', container_id
            ]
            
            subprocess.Popen(cmd)
            print(f"成功设置Pumba网络限制: {bandwidth_mbps}Mbps, {network_rtt}ms延迟")
            return True
        except Exception as e:
            print(f"❌ Pumba 设置失败 [{container_id}]: {e}")
            return False

    def _get_container_veth_safe(self, container_id):
        """改进的 veth 查找：增加了重试机制，防止容器启动瞬间网卡未挂载"""
        for _ in range(3):  # 最多重试3次
            veth = self.get_container_veth_interface(container_id)
            if veth: 
                return veth
            time.sleep(1)
        return None

    def __init__(self, 
                 registry_url: str = "localhost:5000",
                 data_dir: str = "/tmp/exp_data",
                 container_image: str = "cts_client:latest",
                 cloud_mode: bool = False):
        """
        初始化实验编排器
        
        Args:
            registry_url: 镜像仓库地址
            data_dir: 实验数据目录
            container_image: 客户端容器镜像
            cloud_mode: 是否在云服务器模式下运行
        """
        self.registry_url = registry_url
        self.data_dir = data_dir
        self.container_image = container_image
        self.client = docker.from_env()
        self.cloud_mode = cloud_mode  # 云服务器模式标志
        
        # 检查环境兼容性
        self._check_environment()
        
        # 从配置获取实验参数
        self.client_profiles = get_client_capabilities()['profiles']
        self.target_images = get_image_profiles()
        self.compression_methods = get_compression_config()['algorithms']
        
        # 创建数据目录
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建实验记录数据库
        self.db_path = os.path.join(self.data_dir, "experiment_manifest.db")
        self._init_database()
        
        # 记录已完成的实验
        self.completed_experiments = self._load_completed_experiments()
        
        # 用于优雅退出
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _check_environment(self):
        """检查环境兼容性"""
        import platform
        
        # 检查tc命令是否存在 (仅在Linux环境下)
        if platform.system().lower() == "linux" and not self.cloud_mode:
            result = subprocess.run(['which', 'tc'], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ 未找到tc命令，如果在云环境运行请启用cloud_mode=True")
        elif not self.cloud_mode:
            print("⚠️ 当前系统不是Linux，跳过tc命令检查")
        
        # 检查Pumba命令是否存在
        if platform.system().lower() == "linux":
            result = subprocess.run(['which', 'pumba'], capture_output=True, text=True)
            if result.returncode != 0 and self.cloud_mode:
                print("⚠️ 未找到Pumba命令，云模式下网络仿真可能失败")
        elif self.cloud_mode:
            print("⚠️ 当前系统不是Linux，跳过Pumba命令检查")
        
        # 检查iperf3命令是否存在 (仅在Linux环境下)
        if platform.system().lower() == "linux" and not self.cloud_mode:
            result = subprocess.run(['which', 'iperf3'], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ 未找到iperf3命令，网络校准功能可能受限")
        elif not self.cloud_mode:
            print("⚠️ 当前系统不是Linux，跳过iperf3命令检查")
        
        # 检查是否具有sudo权限 (仅在Linux环境下)
        if platform.system().lower() == "linux" and not self.cloud_mode:
            result = subprocess.run(['sudo', '-n', 'true'], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ 当前用户没有sudo权限，网络仿真可能失败")
        elif not self.cloud_mode:
            print("⚠️ 当前系统不是Linux，跳过sudo权限检查")
    
    def _init_database(self):
        """初始化实验记录数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建实验记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                profile_id TEXT,
                image_name TEXT,
                method TEXT,
                replication INTEGER,
                status TEXT,
                start_time REAL,
                end_time REAL,
                cost_total REAL,
                host_cpu_load REAL,
                host_memory_usage REAL,
                host_disk_io REAL,
                decompression_time REAL,
                error_msg TEXT,
                is_noisy_data INTEGER,
                actual_bandwidth REAL,
                bandwidth_std REAL,
                avg_cpu_usage REAL,
                peak_memory REAL,
                compressed_size INTEGER,
                uncompressed_size INTEGER,
                layer_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建实验配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE,
                config_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"收到信号 {signum}，正在优雅退出...")
        self.running = False
        # 清理资源
        self._cleanup_containers()
    
    def _cleanup_containers(self):
        """清理所有实验容器"""
        try:
            containers = self.client.containers.list(all=True, filters={"ancestor": self.container_image})
            for container in containers:
                try:
                    # 清理可能残留的tc规则
                    if not self.cloud_mode:
                        veth = self.get_container_veth_interface(container.id)
                        if veth:
                            subprocess.run(f"sudo tc qdisc del dev {veth} root 2>/dev/null", shell=True)
                    
                    # 清理pumba进程
                    subprocess.run(['pkill', '-f', f'pumba.*{container.id}'], capture_output=True)
                    
                    container.stop()
                    container.remove()
                    print(f"已清理容器: {container.short_id}")
                except:
                    pass
        except Exception as e:
            print(f"清理容器时出错: {e}")
    
    def _load_completed_experiments(self) -> set:
        """从数据库加载已完成的实验记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT uuid FROM experiments WHERE status IN ('SUCCESS', 'ABNORMAL')")
            completed = {row[0] for row in cursor.fetchall()}
            print(f"已加载 {len(completed)} 个已完成的实验记录")
            return completed
        except Exception as e:
            print(f"加载已完成实验记录失败: {e}")
            return set()
        finally:
            conn.close()
    
    def _setup_emulated_container(self, profile: Dict[str, Any]) -> docker.models.containers.Container:
        """根据配置启动模拟客户端容器"""
        profile_name = profile['name']
        print(f"正在启动容器环境: {profile_name}")
        
        # 构建容器启动参数
        container_params = {
            'image': self.container_image,
            'volumes': {
                self.data_dir: {'bind': '/experiment_data', 'mode': 'rw'},
                '/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'rw'}
            },
            'network_mode': 'bridge',
            'environment': {
                'CLIENT_PROFILE_NAME': profile_name,
                'CPU_SCORE': str(profile.get('cpu_limit', profile.get('cpu_score', 1000))),
                'BANDWIDTH_MBPS': str(profile.get('bw_rate', profile.get('bandwidth_mbps', 10))),
                'NETWORK_RTT': str(profile.get('latency', profile.get('network_rtt', 100))),
                'DISK_IO_SPEED': str(profile.get('disk_read', profile.get('disk_io_speed', 50))),
                'MEMORY_SIZE': str(profile.get('mem_limit', profile.get('memory_size', 1))),
                'DECOMP_SPEED_GZIP': str(profile.get('decompression_speed', {}).get('gzip', 100)),
                'DECOMP_SPEED_ZSTD': str(profile.get('decompression_speed', {}).get('zstd', 100)),
                'DECOMP_SPEED_LZ4': str(profile.get('decompression_speed', {}).get('lz4', 100)),
                'LATENCY_REQUIREMENT': str(profile.get('latency_requirement', 100)),
                'REGISTRY_URL': self.registry_url
            },
            'mem_limit': f"{profile.get('mem_limit', profile.get('memory_size', '512m'))}",
            'detach': True,
            'tty': True,
            'stdin_open': True
        }
        
        # Windows Docker环境需要特殊处理CPU参数
        import platform
        if platform.system().lower() != "windows":
            # 非Windows系统使用nano_cpus限制
            container_params['nano_cpus'] = int(profile.get('cpu_limit', profile.get('cpu_score', 1000)) / 1000 * 1e9)
        else:
            # Windows系统使用不同的CPU限制方式
            cpu_limit = profile.get('cpu_limit', profile.get('cpu_score', 1000))
            # 将CPU分数转换为Docker兼容的CPU周期数
            # 这里我们按比例转换，1000分对应1个CPU核心
            # 添加最小值限制，避免CPU配额过小导致的错误
            container_params['cpu_period'] = 100000  # 100ms
            container_params['cpu_quota'] = max(1000, int((cpu_limit / 1000) * 100000))  # 按比例分配CPU时间，最小为1000
        
        # 启动容器
        container = self.client.containers.run(**container_params)
        
        # 应用网络限制
        if not self.cloud_mode:
            self._apply_network_limit(container.id, profile)
        else:
            # 云模式下使用Pumba
            profile_bw = profile.get('bw_rate', profile.get('bandwidth_mbps', 10))
            profile_rtt = profile.get('latency', profile.get('network_rtt', 100))
            self._setup_pumba_limit(container.id, profile_bw, profile_rtt)
        
        print(f"容器 {container.id[:12]} 已启动，应用配置: {profile_name}")
        return container

    def _apply_network_limit(self, container_id: str, profile: Dict[str, Any]):
        """应用网络限制（带宽、延迟等）"""
        try:
            # 获取容器的veth接口名称
            veth_name = self._get_container_veth_safe(container_id)
            
            if veth_name:
                # 设置网络限制
                success = self._setup_tc_limit(
                    veth_name,
                    profile['bandwidth_mbps'],
                    profile['network_rtt']
                )
                
                if success:
                    print(f"为容器 {container_id[:12]} 成功应用网络限制: {profile['bandwidth_mbps']}Mbps, {profile['network_rtt']}ms")
                else:
                    print(f"为容器 {container_id[:12]} 应用网络限制失败")
            else:
                print(f"无法找到容器 {container_id[:12]} 的veth接口，跳过网络限制设置")
        except Exception as e:
            print(f"应用网络限制失败: {e}")

    def _clear_system_cache(self):
        """清理系统缓存，确保实验准确性"""
        try:
            # 清理文件系统缓存
            subprocess.run(['sync'], check=True)
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            print("系统缓存已清理")
        except Exception as e:
            print(f"清理系统缓存失败: {e}")
    
    def _save_experiment_record(self, experiment_record: Dict[str, Any]):
        """保存实验记录到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO experiments 
                (uuid, profile_id, image_name, method, replication, status, 
                 start_time, end_time, cost_total, host_cpu_load, host_memory_usage, 
                 host_disk_io, decompression_time, error_msg, is_noisy_data,
                 actual_bandwidth, bandwidth_std, avg_cpu_usage, peak_memory,
                 compressed_size, uncompressed_size, layer_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_record.get('uuid'),
                experiment_record.get('profile_id'),
                experiment_record.get('image_name'),
                experiment_record.get('method'),
                experiment_record.get('replication'),
                experiment_record.get('status', 'PENDING'),
                experiment_record.get('start_time'),
                experiment_record.get('end_time'),
                experiment_record.get('cost_total'),
                experiment_record.get('host_cpu_load'),
                experiment_record.get('host_memory_usage'),
                experiment_record.get('host_disk_io'),
                experiment_record.get('decompression_time'),
                experiment_record.get('error_msg'),
                experiment_record.get('is_noisy_data', 0),
                experiment_record.get('actual_bandwidth'),
                experiment_record.get('bandwidth_std'),
                experiment_record.get('avg_cpu_usage'),
                experiment_record.get('peak_memory'),
                experiment_record.get('compressed_size'),
                experiment_record.get('uncompressed_size'),
                experiment_record.get('layer_count')
            ))
            
            conn.commit()
        except Exception as e:
            print(f"保存实验记录失败: {e}")
        finally:
            conn.close()
    
    def _get_host_runtime_metrics(self):
        """获取宿主机运行时指标"""
        return {
            'host_cpu_load': self._get_host_cpu_load(),
            'host_memory_usage': self._get_host_memory_usage(),
            'host_disk_io': self._get_host_disk_io()
        }
    
    def _get_host_disk_io(self) -> float:
        """获取宿主机磁盘IO使用率"""
        try:
            # 使用iostat获取磁盘IO信息（如果没有安装iostat，则返回0）
            result = subprocess.run(['iostat', '-d', '1', '2'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # 解析最后一部分数据
                for line in reversed(lines):
                    if line and not line.startswith('Device') and not line.startswith('Linux'):
                        parts = line.split()
                        if len(parts) >= 3:
                            # 返回磁盘IO的平均值（简化处理）
                            return float(0.0)  # 实际应用中需要更复杂的解析
            return 0.0
        except Exception:
            return 0.0

    def _collect_experiment_metrics(self, 
                        profile_id: str, 
                        image_name: str, 
                        method: str, 
                        rep: int, 
                        start_time: float, 
                        end_time: float,
                        decompression_time: float = 0,
                        error_msg: str = None,
                        monitoring_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """收集实验指标"""
        # 生成唯一实验ID
        exp_uuid = f"{profile_id}_{image_name}_{method}_rep{rep}_{uuid.uuid4().hex[:8]}"
        
        # 默认监控数据
        if monitoring_data is None:
            monitoring_data = {
                'actual_bandwidth': 0,
                'bandwidth_std': 0,
                'avg_cpu_usage': 0,
                'peak_memory': 0,
                'is_noisy_data': False,
                'compressed_size': 0,
                'uncompressed_size': 0,
                'layer_count': 0
            }
        
        # 判断是否为噪声数据
        target_bandwidth = next((p['bandwidth_mbps'] for p in self.client_profiles if p['name'] == profile_id), 0)
        bandwidth_deviation = abs(monitoring_data['actual_bandwidth'] - target_bandwidth) / target_bandwidth if target_bandwidth > 0 else 0
        cv = monitoring_data['bandwidth_std'] / monitoring_data['actual_bandwidth'] if monitoring_data['actual_bandwidth'] > 0 else 0
        is_noisy = bandwidth_deviation > 0.5 or cv > 0.3
        
        # 创建实验记录
        experiment_record = {
            'uuid': exp_uuid,
            'profile_id': profile_id,
            'image_name': image_name,
            'method': method,
            'replication': rep,
            'start_time': start_time,
            'end_time': end_time,
            'cost_total': end_time - start_time,
            'timestamp': time.time(),
            'host_cpu_load': self._get_host_cpu_load(),
            'host_memory_usage': self._get_host_memory_usage(),
            'host_disk_io': self._get_host_disk_io(),
            'decompression_time': decompression_time,
            'decompression_performance': {
                'decompression_time': decompression_time,
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_io': 0.0,
                'method': method
            },
            'error_msg': error_msg,
            'status': 'ABNORMAL' if error_msg or decompression_time < 0.01 else 'SUCCESS',  # 解压时间小于10ms标记为异常
            'is_noisy_data': is_noisy,
            'actual_bandwidth': monitoring_data['actual_bandwidth'],
            'bandwidth_std': monitoring_data['bandwidth_std'],
            'avg_cpu_usage': monitoring_data['avg_cpu_usage'],
            'peak_memory': monitoring_data['peak_memory'],
            'compressed_size': monitoring_data['compressed_size'],
            'uncompressed_size': monitoring_data['uncompressed_size'],
            'layer_count': monitoring_data['layer_count']
        }
        
        # 保存实验数据到独立文件
        filename = f"{exp_uuid}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(experiment_record, f, indent=2)
        
        # 保存到数据库
        self._save_experiment_record(experiment_record)
        
        return experiment_record
    
    def collect_experiment_metrics(self, 
                        profile_id: str, 
                        image_name: str, 
                        method: str, 
                        rep: int, 
                        start_time: float, 
                        end_time: float,
                        decompression_time: float = 0,
                        error_msg: str = None,
                        monitoring_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """公共方法：收集实验指标"""
        return self._collect_experiment_metrics(
            profile_id, image_name, method, rep, start_time, end_time, decompression_time, error_msg, monitoring_data
        )

    def _get_host_cpu_load(self) -> float:
        """获取宿主机CPU负载"""
        try:
            # 获取CPU使用率
            with open('/proc/loadavg', 'r') as f:
                loadavg = f.read().strip().split()[0]
                return float(loadavg)
        except Exception:
            return 0.0
    
    def _get_host_memory_usage(self) -> float:
        """获取宿主机内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _monitor_container_resources(self, container_id: str, duration: float, interval: float = 0.5):
        """
        实时监控容器资源使用情况
        """
        # 初始化存储数据的列表
        bandwidth_data = []
        cpu_usage_data = []
        memory_usage_data = []
        
        # 获取初始网络统计数据
        initial_stats = self.client.containers.get(container_id).stats(stream=False)
        initial_net = initial_stats.get('networks', {})
        initial_rx_bytes = sum([net.get('rx_bytes', 0) for net in initial_net.values()])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # 获取当前时间点的容器统计信息
                current_stats = self.client.containers.get(container_id).stats(stream=False)
                
                # 计算当前带宽使用情况
                current_net = current_stats.get('networks', {})
                current_rx_bytes = sum([net.get('rx_bytes', 0) for net in current_net.values()])
                
                # 计算实际带宽 (Mbps)
                time_delta = interval
                bytes_delta = current_rx_bytes - initial_rx_bytes
                bandwidth_mbps = (bytes_delta * 8) / (1024 * 1024) / time_delta  # 转换为Mbps
                
                # 更新初始值
                initial_rx_bytes = current_rx_bytes
                
                # 获取CPU使用率
                cpu_percent = 0.0
                cpu_stats = current_stats.get('cpu_stats', {})
                precpu_stats = current_stats.get('precpu_stats', {})
                
                cpu_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
                online_cpus = cpu_stats.get('online_cpus', len(cpu_stats.get('cpu_usage', {}).get('percpu_usage', [1])))
                
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0
                
                # 获取内存使用情况
                mem_stats = current_stats.get('memory_stats', {})
                mem_usage = mem_stats.get('usage', 0) / (1024 * 1024)  # 转换为MB
                
                # 存储数据
                bandwidth_data.append(bandwidth_mbps)
                cpu_usage_data.append(cpu_percent)
                memory_usage_data.append(mem_usage)
                
                # 等待下一个采样点
                time.sleep(interval)
                
            except Exception as e:
                print(f"监控容器 {container_id} 时出错: {e}")
                break
        
        # 计算统计特征
        if bandwidth_data:
            avg_bandwidth = sum(bandwidth_data) / len(bandwidth_data)
            bandwidth_std = (sum((x - avg_bandwidth) ** 2 for x in bandwidth_data) / len(bandwidth_data)) ** 0.5 if len(bandwidth_data) > 1 else 0
        else:
            avg_bandwidth = 0
            bandwidth_std = 0
        
        if cpu_usage_data:
            avg_cpu = sum(cpu_usage_data) / len(cpu_usage_data)
        else:
            avg_cpu = 0
        
        if memory_usage_data:
            peak_memory = max(memory_usage_data)
        else:
            peak_memory = 0
        
        return {
            'actual_bandwidth': avg_bandwidth,
            'bandwidth_std': bandwidth_std,
            'avg_cpu_usage': avg_cpu,
            'peak_memory': peak_memory
        }
    
    def run_profiled_experiment(self, 
                              container: docker.models.containers.Container, 
                              image_name: str, 
                              method: str,
                              profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行带监控的实验
        """
        try:
            # 准备命令
            if self.cloud_mode:
                print(f"云模式: 使用当前环境执行实验 - 镜像: {image_name}, 方法: {method}")
                command = f"python3 /app/client_pull_script.py {self.registry_url}/{image_name} --method {method}"
            else:
                command = f"python3 /app/client_pull_script.py {self.registry_url}/{image_name} --method {method}"
            
            # 启动监控线程
            monitoring_result = {}
            monitor_thread = threading.Thread(
                target=self._monitor_container_resources,
                args=(container.id, 60, 0.5)
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 启动监控线程
            monitor_thread.start()
            
            # 执行拉取实验
            result = container.exec_run(
                cmd=command,
                stdout=True,
                stderr=True,
                detach=False
            )
            
            # 解析执行结果
            output = result.output.decode('utf-8')
            exit_code = result.exit_code
            
            # 等待监控线程完成
            monitor_thread.join(timeout=5)  # 最多等待5秒
            
            # 获取监控数据
            monitoring_data = self._monitor_container_resources(container.id, 0.1, 0.5)  # 短时间快速采样获取最终数据
            
            # 获取镜像静态特征（模拟，实际应用中需要从配置中获取）
            image_features = self._get_image_features(image_name)
            
            # 增加：从标准输出中精准提取最后一行 JSON
            perf_data = {}
            for line in reversed(output.strip().split('\n')):
                if line.startswith('{') and line.endswith('}'):
                    try:
                        perf_data = json.loads(line)
                        break
                    except: 
                        continue
            
            if not perf_data:
                return {
                    "status": "ABNORMAL", 
                    "error": "No JSON found in output", 
                    "raw": output,
                    "actual_bandwidth": monitoring_data['actual_bandwidth'],
                    "bandwidth_std": monitoring_data['bandwidth_std'],
                    "avg_cpu_usage": monitoring_data['avg_cpu_usage'],
                    "peak_memory": monitoring_data['peak_memory'],
                    "is_noisy_data": False,
                    "compressed_size": image_features['compressed_size'],
                    "uncompressed_size": image_features['uncompressed_size'],
                    "layer_count": image_features['layer_count'],
                    "image_name": image_name,  # 确保包含image_name字段
                    "method": method,  # 确保包含method字段
                    "profile_id": profile['name']  # 确保包含profile_id字段
                }
                
            # 记录额外的解压性能（宿主机视角）
            host_metrics = self._get_host_runtime_metrics()
            perf_data.update(host_metrics)
            
            # 合并监控数据
            perf_data.update(monitoring_data)
            perf_data.update(image_features)
            
            # 确保包含关键字段
            perf_data['image_name'] = image_name
            perf_data['method'] = method
            perf_data['profile_id'] = profile['name']
            
            # 计算是否为噪声数据
            target_bandwidth = profile.get('bandwidth_mbps', 0)
            bandwidth_deviation = abs(monitoring_data['actual_bandwidth'] - target_bandwidth) / target_bandwidth if target_bandwidth > 0 else 0
            cv = monitoring_data['bandwidth_std'] / monitoring_data['actual_bandwidth'] if monitoring_data['actual_bandwidth'] > 0 else 0
            perf_data['is_noisy_data'] = bandwidth_deviation > 0.5 or cv > 0.3
            
            return perf_data
            
        except Exception as e:
            return {
                'status': 'ABNORMAL',
                'error': str(e),
                'raw_output': '',
                'image_name': image_name,  # 确保包含image_name字段
                'method': method,  # 确保包含method字段
                'profile_id': profile['name']  # 确保包含profile_id字段
            }
    
    def _get_image_features(self, image_name: str) -> Dict[str, Any]:
        """
        获取镜像静态特征
        """
        # 模拟获取镜像特征，实际应用中应从配置或镜像分析中获取
        return {
            'compressed_size': 100000000,  # 100MB
            'uncompressed_size': 250000000,  # 250MB
            'layer_count': 5
        }
    
    def run_experiment_matrix(self, 
                            replications: int = 3, 
                            parallel: bool = False) -> List[Dict[str, Any]]:
        """
        运行实验矩阵
        实现三级循环：客户端配置 -> 目标镜像 -> 压缩方法 -> 重复次数
        """
        print("开始运行实验矩阵...")
        print(f"客户端配置数: {len(self.client_profiles)}")
        print(f"目标镜像数: {len(self.target_images)}")
        print(f"压缩方法数: {len(self.compression_methods)}")
        print(f"重复次数: {replications}")
        total = len(self.client_profiles) * len(self.target_images) * len(self.compression_methods) * replications
        print(f"🚀 启动正式实验矩阵 | 预估总量: {total}")
        
        all_results = []
        completed_count = 0
        total_experiments = total
        
        for profile in self.client_profiles:
            if not self.running:
                break
                
            print(f"\n=== 开始处理客户端配置: {profile['name']} ({profile['description']}) ===")
            
            # 1. 启动容器前，先清理一遍 Docker 系统缓存
            os.system("docker system prune -f > /dev/null")
            
            # 启动模拟容器
            container = self._setup_emulated_container(profile)
            if not container: 
                print(f"容器启动失败: {profile['name']}")
                continue
            
            # 2. 获取 veth 并设置网络限制
            if not self.cloud_mode:
                veth = self._get_container_veth_safe(container.id)
                if veth:
                    self._setup_tc_limit(veth, profile['bandwidth_mbps'], profile['network_rtt'])
            else:
                # 云模式下使用Pumba
                self._setup_pumba_limit(container.id, profile['bandwidth_mbps'], profile['network_rtt'])
            
            try:
                for image in self.target_images:
                    if not self.running:
                        break
                        
                    for method in self.compression_methods:
                        if not self.running:
                            break
                            
                        for rep in range(replications):
                            if not self.running:
                                break
                                
                            # 生成实验UUID
                            exp_id = f"{profile['name']}_{image['name']}_{method}_rep{rep}"
                            
                            # 检查是否已完成
                            if exp_id in self.completed_experiments:
                                print(f"跳过已完成实验: {exp_id}")
                                completed_count += 1
                                continue
                            
                            print(f"执行实验 {completed_count + 1}/{total_experiments}: {exp_id}")
                            
                            # 3. 核心：每次拉取前必须删除容器内的镜像缓存
                            # 模拟"冷启动"拉取
                            img_url = f"{self.registry_url}/{image['name']}"
                            try:
                                container.exec_run(f"docker rmi -f {img_url}", detach=False)
                            except:
                                pass  # 忽略清理错误
                            
                            # 4. 清理系统缓存
                            self._clear_system_cache()
                            
                            # 执行带监控的实验
                            result = self.run_profiled_experiment(container, img_url, method, profile)
                            
                            # 记录实验结果
                            experiment_record = self._collect_experiment_metrics(
                                profile['name'], 
                                image['name'], 
                                method, 
                                rep, 
                                time.time(),  # start_time
                                time.time(),  # end_time
                                result.get('decompression_time', 0),
                                result.get('error'),
                                result  # 传递监控数据
                            )
                            
                            # 合并结果
                            experiment_record.update(result)
                            
                            # 标记实验完成
                            self.completed_experiments.add(exp_id)
                            all_results.append(experiment_record)
                            
                            completed_count += 1
                            
                            # 添加小延迟，避免系统过载
                            time.sleep(0.5)
                            
            finally:
                # 停止并删除容器
                try:
                    container.stop()
                    container.remove()
                    print(f"容器已清理: {profile['name']}")
                except:
                    pass
        
        print(f"\n实验矩阵完成！共执行 {len(all_results)} 个实验")
        return all_results
    
    def aggregate_results(self) -> Dict[str, Any]:
        """聚合实验结果"""
        print("正在聚合实验结果...")
        
        results = []
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and f != 'completed_experiments.json']
        
        for filename in data_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"读取文件 {filename} 失败: {e}")
        
        # 按实验配置分组
        grouped_results = {}
        for result in results:
            key = f"{result.get('profile_id', 'unknown')}_{result.get('image_name', 'unknown')}_{result.get('method', 'unknown')}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # 计算每组的统计信息
        aggregated = {}
        for key, group in grouped_results.items():
            # 修复原代码中的bug：使用cost_total而不是未定义的duration
            durations = [r['cost_total'] for r in group if r.get('status') == 'SUCCESS']
            
            if durations and len(durations) > 0:
                mean_duration = sum(durations) / len(durations)
                
                # 计算方差，用于检测异常数据
                variance = sum((x - mean_duration) ** 2 for x in durations) / len(durations) if len(durations) > 1 else 0
                std_dev = variance ** 0.5
                
                # 计算变异系数
                cv = (std_dev / mean_duration) * 100 if mean_duration > 0 else 0  # 变异系数
                
                # 标记变异系数过大的组
                high_cv = cv > 15  # 变异系数超过15%
                
                # 修复原代码中的bug：定义valid_durations
                valid_durations = durations
                
                aggregated[key] = {
                    'count': len(valid_durations),
                    'mean_duration': mean_duration,
                    'min_duration': min(valid_durations),
                    'max_duration': max(valid_durations),
                    'std_deviation': std_dev,
                    'variance': variance,
                    'cv': cv,  # 变异系数
                    'high_cv': high_cv,  # 高变异系数标记
                    'replications': valid_durations
                }
        
        print(f"聚合了 {len(aggregated)} 组实验结果")
        return aggregated


def main():
    """主函数，运行实验矩阵"""
    print("启动实验编排系统...")
    
    # 创建编排器
    orchestrator = ExperimentOrchestrator(cloud_mode=True)  # 默认使用云模式
    
    # 运行实验矩阵
    results = orchestrator.run_experiment_matrix(replications=3)
    
    # 聚合结果
    aggregated = orchestrator.aggregate_results()
    
    # 保存聚合结果
    output_file = os.path.join(orchestrator.data_dir, "aggregated_results.json")
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"聚合结果已保存至: {output_file}")
    print("实验编排完成！")


if __name__ == "__main__":
    main()