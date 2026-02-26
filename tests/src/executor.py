import time
import subprocess
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import platform

class DockerExecutor:
    def __init__(self, registry_address: Optional[str] = None):
        """
        初始化Docker执行器 (优化版)
        
        Args:
            registry_address: 私有仓库地址 (可选，None则不拉取前缀)
        """
        self.registry = registry_address
        self.is_windows = platform.system() == "Windows"
        
        # 设置日志
        self.logger = logging.getLogger(__name__)

    def pull_image(
        self, 
        image_name: str, 
        image_tag: str, 
        compression: Optional[str] = None, # 【修改】compression 改为可选
        config: Optional[Dict[str, Any]] = None, # 【修改】config 改为可选
        repeat: int = 5,
        image_size_mb: float = 100.0 # 【新增】直接传入镜像大小
    ) -> pd.DataFrame:
        """
        执行 docker pull 操作 (简化版)
        
        Args:
            image_name: 镜像名称 (如 'nginx')
            image_tag: 镜像标签 (如 'latest')
            compression: 压缩算法 (仅用于记录，不修改镜像名)
            config: 执行配置 (仅用于记录，不实际限制Docker)
            repeat: 重复次数
            image_size_mb: 镜像大小 (用于计算吞吐量)
            
        Returns:
            包含执行结果的 DataFrame
        """
        # 【修改】构造镜像名：如果有 registry 才加前缀
        if self.registry:
            image_full = f"{self.registry}/{image_name}:{image_tag}"
        else:
            image_full = f"{image_name}:{image_tag}"
        
        # 记录实际使用的配置
        actual_concurrent = config.get('concurrent_downloads', 3) if config else 3
        
        results = []
        
        for i in range(repeat):
            try:
                # 【修改】清理缓存：只做 docker prune，跳过 drop_caches
                self._clear_docker_cache()
                
                # 【修改】构建命令：移除复杂的CPU限制逻辑
                # 注意：docker pull 原生支持 --max-concurrent-downloads
                cmd = [
                    "docker", "pull",
                    "--max-concurrent-downloads", str(actual_concurrent),
                    "--quiet", # 减少输出
                    image_full
                ]
                
                # 执行并计时
                start_time = time.perf_counter()
                
                # 【修改】规范的 subprocess 调用
                proc = subprocess.run(
                    cmd,
                    shell=False, # 【关键】不使用 shell=True
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )
                
                end_time = time.perf_counter()
                total_time = end_time - start_time
                success = 1 if proc.returncode == 0 else 0
                
                # 【修改】计算吞吐量：使用传入的 image_size_mb
                throughput_mbps = (image_size_mb * 8) / total_time if (total_time > 0 and success) else 0
                
                results.append({
                    "repeat": i + 1,
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "compression": compression,
                    "concurrent_downloads": actual_concurrent,
                    "total_time_s": round(total_time, 4),
                    "throughput_mbps": round(throughput_mbps, 2),
                    "success": success,
                    "stderr": proc.stderr.strip() if not success else "",
                    "stdout": proc.stdout.strip()[:200] if success else ""
                })
                
                status = "✅ 成功" if success else "❌ 失败"
                self.logger.info(f"[{i+1}/{repeat}] {status} 拉取 {image_full}, 耗时: {total_time:.2f}s")
                
                # 失败时等待
                if not success:
                    time.sleep(2)
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"[{i+1}/{repeat}] ⏰ 拉取超时: {image_full}")
                results.append({
                    "repeat": i + 1,
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "compression": compression,
                    "total_time_s": 300.0,
                    "throughput_mbps": 0,
                    "success": 0,
                    "stderr": "Timeout after 300 seconds",
                    "stdout": ""
                })
                
            except Exception as e:
                self.logger.error(f"[{i+1}/{repeat}] ❌ 执行异常: {e}", exc_info=False)
                results.append({
                    "repeat": i + 1,
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "compression": compression,
                    "total_time_s": 0,
                    "throughput_mbps": 0,
                    "success": 0,
                    "stderr": str(e),
                    "stdout": ""
                })
        
        return pd.DataFrame(results)

    def _clear_docker_cache(self):
        """清理Docker缓存 (跨平台兼容)"""
        try:
            # 1. 清理 Docker 镜像/容器缓存
            subprocess.run(
                ["docker", "system", "prune", "-af"],
                shell=False,
                check=True,
                capture_output=True,
                timeout=60
            )
            
            # 2. Linux特有：清理页缓存 (可选，失败不报错)
            if not self.is_windows:
                try:
                    # 需要 root 权限，失败则跳过
                    subprocess.run(
                        ["sync"], shell=False, check=True, capture_output=True
                    )
                    # 注意：echo 3 > ... 需要 root，且在容器内通常无法执行
                    # 这里我们只做 sync，不强制 drop_caches
                except:
                    pass
            
            time.sleep(1)
        except Exception as e:
            # 忽略清理错误，不影响主实验
            self.logger.debug(f"缓存清理跳过: {e}")
            pass