import os
import sys
import time
import json
import sqlite3
import logging
import subprocess
import shutil
import docker
import numpy as np
from datetime import datetime
from config import CLIENT_PROFILES, TARGET_IMAGES, COMPRESSION_METHODS, REPETITIONS, DB_PATH, TEMP_DIR, CLIENT_IMAGE

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentOrchestrator:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.conn = sqlite3.connect(DB_PATH)
        self._init_db()
        self._check_dependencies()
        
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

    def _init_db(self):
        """初始化SQLite数据库"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image TEXT,
                client_profile TEXT,
                method TEXT,
                rep_id INTEGER,
                status TEXT, -- 'SUCCESS', 'FAILED', 'ABNORMAL'
                download_time REAL,
                decomp_time REAL,
                total_time REAL,
                cpu_usage REAL,
                mem_usage REAL,
                compressed_size INTEGER,
                original_size INTEGER,
                bandwidth_measured REAL,
                is_noise BOOLEAN,
                error_msg TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image, client_profile, method, rep_id)
            )
        ''')
        self.conn.commit()

    def _check_dependencies(self):
        """检查环境依赖"""
        try:
            subprocess.run(['pumba', '--version'], check=True, stdout=subprocess.PIPE)
            subprocess.run(['tc', '-V'], check=True, stdout=subprocess.PIPE)
            logger.info("✅ 环境依赖检查通过 (Docker, Pumba, tc)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ 缺少必要的依赖工具 (Pumba 或 tc)，请先安装。")
            sys.exit(1)

    def _clear_system_cache(self):
        """清理系统缓存以保证实验准确性"""
        try:
            subprocess.run('sync', shell=True)
            subprocess.run('echo 3 > /proc/sys/vm/drop_caches', shell=True)
        except Exception as e:
            logger.warning(f"无法清理系统缓存 (可能需要sudo): {e}")

    def is_experiment_done(self, image, profile, method, rep):
        """检查实验是否已经完成（断点续跑）"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT count(*) FROM experiments 
            WHERE image=? AND client_profile=? AND method=? AND rep_id=? AND status='SUCCESS'
        ''', (image, profile, method, rep))
        return cursor.fetchone()[0] > 0

    def prepare_image_payload(self, image_name, method_name):
        """
        1. 拉取镜像
        2. 导出为Tar
        3. 压缩
        返回: (压缩文件路径, 原始大小, 压缩后大小)
        """
        safe_img_name = image_name.replace(':', '_').replace('/', '_')
        raw_tar_path = os.path.join(TEMP_DIR, f"{safe_img_name}.tar")
        
        # 1. 拉取镜像
        logger.info(f"正在拉取镜像: {image_name}")
        self.docker_client.images.pull(image_name)
        
        # 2. 导出为Tar (模拟提取镜像层)
        # 注意：真实场景可能需要提取特定Layer，这里为了简化模拟，导出整个Image Tar作为payload
        image = self.docker_client.images.get(image_name)
        with open(raw_tar_path, 'wb') as f:
            for chunk in image.save():
                f.write(chunk)
        
        original_size = os.path.getsize(raw_tar_path)
        
        # 3. 压缩
        cmd_args = COMPRESSION_METHODS[method_name]
        # 构造输出文件名 (例如 .tar.gz, .tar.zst)
        if 'gzip' in method_name: ext = '.gz'
        elif 'zstd' in method_name: ext = '.zst'
        elif 'lz4' in method_name: ext = '.lz4'
        elif 'brotli' in method_name: ext = '.br'
        else: ext = '.dat'
        
        compressed_path = raw_tar_path + ext
        
        # 执行压缩命令
        logger.info(f"正在压缩 ({method_name}): {raw_tar_path} -> {compressed_path}")
        start_time = time.time()
        
        # 针对不同工具的命令适配
        if 'gzip' in method_name:
            with open(raw_tar_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                subprocess.run(cmd_args, stdin=f_in, stdout=f_out, check=True)
        elif 'brotli' in method_name:
             with open(raw_tar_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                subprocess.run(cmd_args, stdin=f_in, stdout=f_out, check=True)
        else:
            # zstd 和 lz4 支持直接文件参数
            subprocess.run(cmd_args + [raw_tar_path, '-o', compressed_path], check=True)
            
        compressed_size = os.path.getsize(compressed_path)
        
        # 清理原始tar，只保留压缩包
        if os.path.exists(raw_tar_path):
            os.remove(raw_tar_path)
            
        return compressed_path, original_size, compressed_size


    def setup_client_container(self, profile_name):
        """启动特定配置的客户端容器"""
        config = CLIENT_PROFILES[profile_name]
        container_name = f"cts_worker_{profile_name}"
        
        # 清理旧容器
        try:
            old = self.docker_client.containers.get(container_name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass

        # 启动新容器 (应用 CPU/Mem 限制)
        # 启动新容器 (应用 CPU/Mem 限制 + 网络权限)
        container = self.docker_client.containers.run(
            CLIENT_IMAGE,
            name=container_name,
            detach=True,
            tty=True,
            nano_cpus=int(config['cpu'] * 1e9),
            mem_limit=config['mem'],
            # === 【新增下面这一行】 ===
            cap_add=['NET_ADMIN'], 
            # =========================
            volumes={TEMP_DIR: {'bind': '/data', 'mode': 'rw'}}, 
            command="tail -f /dev/null"
        )
        
        # 应用网络仿真 (Pumba)
        # 注意: 需要在宿主机安装 pumba 二进制文件
        logger.info(f"应用网络限制 ({profile_name}): BW={config['bw']}, Delay={config['delay']}")
        pumba_cmd = [
            "pumba", "netem",
            "--interface", "eth0",
            "--duration", "5m", 
            "rate", "--rate", config['bw'],
            "delay", "--time", config['delay'], "--jitter", "5ms", "--correlation", "0",
            container_name
        ]
        subprocess.run(pumba_cmd, check=True)
        
        return container

    def run_agent_in_container(self, container, compressed_file, method_name):
        """在容器内执行解压测试"""
        filename = os.path.basename(compressed_file)
        container_path = f"/data/{filename}"
        
        # 构造容器内命令
        # 假设 client_agent.py 接受: python3 client_agent.py --file <path> --method <method>
        cmd = f"python3 /app/client_agent.py --file {container_path} --method {method_name}"
        
        exec_result = container.exec_run(cmd)
        output = exec_result.output.decode('utf-8')
        
        if exec_result.exit_code != 0:
            raise Exception(f"Agent Execution Failed: {output}")
            
        # 解析最后一行 JSON 输出
        try:
            json_str = output.strip().split('\n')[-1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON output: {output}")

    def save_result(self, image, profile, method, rep, data, error=None):
        """保存结果到数据库"""
        is_noise = False
        status = 'SUCCESS'
        
        # ... (前略)
        if error:
            status = 'FAILED'
        else:
            # === 修改这里 ===
            # 原代码: target_bw_mbps = float(CLIENT_PROFILES[profile]['bw'].replace('m', '')) 
            # 新代码: 去掉 'mbit' 后转浮点数
            target_bw_mbps = float(CLIENT_PROFILES[profile]['bw'].replace('mbit', '')) 
            
            measured_bw = data.get('bandwidth_measured', 0)
            # ... (后略)
            
            # 1. 带宽偏差检查 (>50%)
            if abs(measured_bw - target_bw_mbps) / target_bw_mbps > 0.5:
                is_noise = True
                status = 'ABNORMAL'
            
            # 2. 解压时间过短 (<10ms)
            if data.get('decomp_time', 0) < 0.01:
                is_noise = True
                status = 'ABNORMAL'

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO experiments 
            (image, client_profile, method, rep_id, status, download_time, decomp_time, 
             total_time, cpu_usage, mem_usage, compressed_size, original_size, 
             bandwidth_measured, is_noise, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image, profile, method, rep, status,
            data.get('download_time', 0), data.get('decomp_time', 0),
            data.get('total_time', 0), data.get('cpu_usage', 0),
            data.get('mem_usage', 0), data.get('compressed_size', 0),
            data.get('original_size', 0), data.get('bandwidth_measured', 0),
            is_noise, str(error) if error else None
        ))
        self.conn.commit()
        logger.info(f"实验结果已保存: {status}")

    def run_matrix(self):
        """执行完整实验矩阵"""
        logger.info(f"开始运行实验矩阵: {len(TARGET_IMAGES)}镜像 x {len(CLIENT_PROFILES)}客户端 x {len(COMPRESSION_METHODS)}算法")
        
        # 1. 外层循环：镜像 (最耗时的资源，尽量少切换)
        for image in TARGET_IMAGES:
            try:
                # 2. 中层循环：客户端画像
                for profile_name in CLIENT_PROFILES.keys():
                    
                    container = None
                    try:
                        # 启动特定环境的容器
                        container = self.setup_client_container(profile_name)
                        
                        # 3. 内层循环：压缩算法
                        for method in COMPRESSION_METHODS.keys():
                            
                            # 准备数据 payload (宿主机压缩)
                            # 优化: 可以在Rep循环外做，但为了模拟每次请求，放在这里
                            comp_path, orig_size, comp_size = self.prepare_image_payload(image, method)
                            
                            # 4. 重复实验
                            for rep in range(REPETITIONS):
                                if self.is_experiment_done(image, profile_name, method, rep):
                                    logger.info(f"⏭️ 跳过已完成实验: {image} | {profile_name} | {method} | Rep{rep}")
                                    continue
                                
                                logger.info(f"▶️ 执行实验: {image} | {profile_name} | {method} | Rep{rep}")
                                self._clear_system_cache()
                                
                                try:
                                    # 执行核心测试
                                    result_data = self.run_agent_in_container(container, comp_path, method)
                                    
                                    # 补充宿主机已知的数据
                                    result_data['original_size'] = orig_size
                                    result_data['compressed_size'] = comp_size
                                    
                                    self.save_result(image, profile_name, method, rep, result_data)
                                    
                                except Exception as e:
                                    logger.error(f"❌ 实验失败: {e}")
                                    self.save_result(image, profile_name, method, rep, {}, error=e)
                                
                                time.sleep(1) # 冷却
                            
                            # 清理当次压缩文件
                            if os.path.exists(comp_path):
                                os.remove(comp_path)
                                
                    finally:
                        if container:
                            container.remove(force=True)
                            
                # 镜像层级清理: 完成一个镜像的所有实验后，删除本地镜像以释放空间
                self.docker_client.images.remove(image, force=True)
                logger.info(f"🧹 清理本地镜像: {image}")
                
            except Exception as e:
                logger.critical(f"🔥 镜像层级严重错误 ({image}): {e}")

if __name__ == "__main__":
    if os.geteuid() != 0:
        logger.warning("建议以 root 权限运行，否则 Pumba 和 缓存清理 可能失效。")
    
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_matrix()