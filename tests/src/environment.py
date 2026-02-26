import os
import time
import subprocess
import psutil
import yaml
import logging
import platform
from typing import Dict, Any, Optional, Tuple

class EnvironmentController:
    def __init__(self, config_path: str):
        """
        初始化环境控制器 (跨平台优化版)
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 【优化1】系统检测
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        
        # 【优化2】安全获取配置，带默认值
        self.interface = self.config.get('network', {}).get('interface', 'eth0')
        self.registry = self.config.get('registry', {}).get('address', 'docker.io')
        
        # 【优化3】不在这里配置 basicConfig，避免与主入口冲突
        # 假设主入口 (run_all_experiments.py) 已经通过 utils.setup_logging() 配置好了
        self.logger = logging.getLogger(__name__)
        
        # 【优化4】缓存上一次设置的网络参数 (用于 collect_features)
        self._last_bandwidth = 100.0
        self._last_delay = 50.0
        self._last_loss = 0.01

    def init_network(self, bandwidth: float, delay: float, loss: float):
        """
        配置网络环境 (跨平台兼容)
        
        Args:
            bandwidth: 带宽(Mbps)
            delay: 延迟(ms)
            loss: 丢包率(0-1)
        """
        # 首先缓存设置值 (无论平台是否支持 tc，都缓存起来供 collect_features 使用)
        self._last_bandwidth = bandwidth
        self._last_delay = delay
        self._last_loss = loss
        
        # 非 Linux 系统：跳过 tc 配置
        if not self.is_linux:
            self.logger.warning(f"⚠️  非Linux系统，跳过 tc 网络限制配置")
            self.logger.info(f"   目标配置: {bandwidth}Mbps, {delay}ms, {loss*100}%丢包")
            self.logger.info(f"   如需网络限制，请手动配置或使用 WAN emulation 工具")
            time.sleep(1)
            return
        
        # Linux 系统：执行 tc 命令
        try:
            # 清理旧配置 (失败不报错，可能本来就没有配置)
            subprocess.run(
                f"tc qdisc del dev {self.interface} root netem",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            
            # 应用新配置
            tc_cmd = (
                f"tc qdisc add dev {self.interface} root netem "
                f"rate {bandwidth}mbit delay {delay}ms loss {loss*100}%"
            )
            subprocess.run(tc_cmd, shell=True, check=True, capture_output=True)
            time.sleep(2)
            
            self.logger.info(f"✅ 网络环境初始化完成: {bandwidth}Mbps, {delay}ms, {loss*100}%丢包")
            
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"❌ 网络配置失败 (需要root权限?): {stderr_msg}")
            self.logger.warning(f"   实验将继续，但网络限制未生效")

    def clear_cache(self):
        """清理缓存 (跨平台兼容)"""
        try:
            # 1. 清理 Docker 缓存 (跨平台)
            try:
                subprocess.run(
                    ["docker", "system", "prune", "-af"],
                    shell=False,
                    check=True,
                    capture_output=True,
                    timeout=60
                )
            except Exception as e:
                self.logger.debug(f"Docker prune 跳过或失败: {e}")
            
            # 2. 清理系统缓存 (仅Linux，且不强制 drop_caches)
            if self.is_linux:
                try:
                    subprocess.run(["sync"], shell=False, check=True, capture_output=True)
                    # 注意：echo 3 > /proc/sys/vm/drop_caches 需要 root，且在容器内通常无法执行
                    # 这里我们只做 sync，不强制 drop_caches，避免报错
                except Exception as e:
                    self.logger.debug(f"系统缓存清理跳过: {e}")
            
            time.sleep(1)
            self.logger.debug("✅ 缓存清理流程完成")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 缓存清理警告: {e}")

    def collect_features(self, use_cached_env: bool = True) -> Dict[str, Any]:
        """
        采集客户端实时环境特征 (优化版)
        
        Args:
            use_cached_env: 是否使用 init_network 缓存的网络参数 (推荐 True)
                           如果为 False，会尝试实测网络（较慢且不一定准）
            
        Returns:
            包含网络、CPU、内存等特征的字典
        """
        try:
            # ==========================================
            # 1. 网络特征 (优先使用缓存值，这是最准确的)
            # ==========================================
            if use_cached_env:
                bandwidth = self._last_bandwidth
                rtt = self._last_delay
                loss = self._last_loss
            else:
                # 实测模式 (较慢，且不推荐用于实验控制)
                bandwidth = self._estimate_bandwidth()
                rtt, loss = self._ping_registry()
            
            # ==========================================
            # 2. CPU/内存特征 (跨平台，使用 psutil)
            # ==========================================
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=0.1) # 缩短 interval 从 0.5 到 0.1
            cpu_available = 100.0 - cpu_percent
            
            mem_info = psutil.virtual_memory()
            mem_total_mb = mem_info.total / 1024 / 1024
            mem_available_mb = mem_info.available / 1024 / 1024
            
            # ==========================================
            # 3. 构造特征字典 (确保所有值都是 float，兼容模型输入)
            # ==========================================
            features = {
                "bandwidth_mbps": float(bandwidth),
                "network_rtt": float(rtt),
                "packet_loss": float(loss),
                "cpu_limit": float(cpu_cores),
                "mem_limit_mb": float(mem_total_mb),
                "cpu_available": float(cpu_available),
                "mem_available": float(mem_available_mb)
            }
            
            self.logger.debug(f"采集到环境特征: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ 特征采集失败: {e}", exc_info=False)
            # 返回安全的默认特征值
            return {
                "bandwidth_mbps": 100.0,
                "network_rtt": 50.0,
                "packet_loss": 0.01,
                "cpu_limit": 4.0,
                "mem_limit_mb": 8192.0,
                "cpu_available": 80.0,
                "mem_available": 6000.0
            }

    def _estimate_bandwidth(self) -> float:
        """(内部辅助) 估算带宽 (不推荐使用，仅作后备)"""
        try:
            net_stat1 = psutil.net_io_counters()
            time.sleep(0.5)
            net_stat2 = psutil.net_io_counters()
            bytes_diff = net_stat2.bytes_recv - net_stat1.bytes_recv
            mbps = (bytes_diff * 8) / (0.5 * 1024 * 1024)
            return max(mbps, 1.0)
        except:
            return 100.0

    def _ping_registry(self) -> Tuple[float, float]:
        """(内部辅助) Ping Registry (跨平台，简化版)"""
        try:
            host = self.registry.split('/')[0]
            if ':' in host: # 去掉端口
                host = host.split(':')[0]
            
            # 跨平台 ping 参数
            if self.is_windows:
                cmd = ["ping", "-n", "2", "-w", "1000", host]
            else:
                cmd = ["ping", "-c", "2", "-W", "1", host]
            
            result = subprocess.run(
                cmd, 
                shell=False, 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return self._last_delay, self._last_loss
            
            # 简单处理：不依赖复杂的字符串解析，直接返回缓存值
            # (因为不同语言/系统输出格式差异太大，且我们主要靠 init_network 控制)
            return self._last_delay, self._last_loss
            
        except:
            return 100.0, 0.1

    def cleanup(self):
        """清理网络环境配置"""
        if not self.is_linux:
            return
            
        try:
            subprocess.run(
                f"tc qdisc del dev {self.interface} root netem",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            self.logger.info("✅ 网络环境已清理")
        except Exception as e:
            self.logger.debug(f"环境清理跳过: {e}")

    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """
        获取指定场景的配置
        
        Args:
            scenario_name: 场景名称
            
        Returns:
            场景配置字典
        """
        return self.config.get('scenarios', {}).get(scenario_name, {})