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
        
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

        self.network_name = "cts_experiment_net"
        self.server_container_name = "cts_image_server"
        self.server_ip = "cts_image_server"
        self.server_port = 8000
        
        self._setup_infrastructure()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image TEXT,
                client_profile TEXT,
                method TEXT,
                rep_id INTEGER,
                status TEXT,
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
        try:
            subprocess.run(['lz4', '--version'], check=True, stdout=subprocess.PIPE)
            logger.info("✅ 环境依赖检查通过")
        except:
            logger.error("❌ 宿主机缺少 lz4 工具")
            sys.exit(1)

    def _setup_infrastructure(self):
        logger.info("🏗️  正在搭建实验网络架构...")
        try:
            self.docker_client.networks.get(self.network_name).remove()
        except: pass
        self.network = self.docker_client.networks.create(self.network_name, driver="bridge")

        try:
            self.docker_client.containers.get(self.server_container_name).remove(force=True)
        except: pass
        
        logger.info("🔵 启动镜像服务器 (Image Server)...")
        self.server = self.docker_client.containers.run(
            CLIENT_IMAGE,
            name=self.server_container_name,
            network=self.network_name,
            detach=True,
            cap_add=["NET_ADMIN"],
            volumes={TEMP_DIR: {'bind': '/data', 'mode': 'ro'}},
            command=f"python3 -m http.server {self.server_port} --directory /data"
        )
        time.sleep(2) 

    def update_server_network(self, bw, delay):
        self.server.exec_run("tc qdisc del dev eth0 root")
        cmd = f"tc qdisc add dev eth0 root netem rate {bw} delay {delay}"
        exit_code, output = self.server.exec_run(cmd)
        
        if exit_code != 0:
            logger.error(f"❌ 网络配置失败: {output.decode()}")
        else:
            logger.info(f"🌐 网络环境已更新: {bw} / {delay}")

    def is_experiment_done(self, image, profile, method, rep):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT count(*) FROM experiments 
            WHERE image=? AND client_profile=? AND method=? AND rep_id=? AND status='SUCCESS'
        ''', (image, profile, method, rep))
        return cursor.fetchone()[0] > 0

    def _pull_and_save_raw_tar(self, image_name):
        safe_img_name = image_name.replace(':', '_').replace('/', '_')
        raw_tar_path = os.path.join(TEMP_DIR, f"{safe_img_name}_raw.tar")
        
        if os.path.exists(raw_tar_path) and os.path.getsize(raw_tar_path) > 1000:
             return raw_tar_path, os.path.getsize(raw_tar_path)

        try:
            logger.info(f"⬇️  正在拉取镜像: {image_name}")
            self.docker_client.images.pull(image_name)
            
            logger.info(f"💾 正在导出: {safe_img_name}")
            image = self.docker_client.images.get(image_name)
            with open(raw_tar_path, 'wb') as f:
                for chunk in image.save():
                    f.write(chunk)
            return raw_tar_path, os.path.getsize(raw_tar_path)
        except Exception as e:
            if os.path.exists(raw_tar_path): os.remove(raw_tar_path)
            raise e

    def _create_compressed_payload(self, raw_tar_path, method_name):
        cmd_args = COMPRESSION_METHODS[method_name]
        if 'gzip' in method_name: ext = '.gz'
        elif 'zstd' in method_name: ext = '.zst'
        elif 'lz4' in method_name: ext = '.lz4'
        elif 'brotli' in method_name: ext = '.br'
        else: ext = '.dat'
        
        compressed_path = raw_tar_path + ext
        if os.path.exists(compressed_path):
             return compressed_path, os.path.getsize(compressed_path)

        try:
            if 'lz4' in method_name:
                subprocess.run(cmd_args + [raw_tar_path, compressed_path], check=True)
            elif 'zstd' in method_name:
                subprocess.run(cmd_args + [raw_tar_path, '-o', compressed_path], check=True)
            else:
                 with open(raw_tar_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                    subprocess.run(cmd_args, stdin=f_in, stdout=f_out, check=True)

            return compressed_path, os.path.getsize(compressed_path)
        except Exception as e:
            if os.path.exists(compressed_path): os.remove(compressed_path)
            raise e

    def run_agent_in_container(self, profile_name, compressed_file, method_name):
        config = CLIENT_PROFILES[profile_name]
        filename = os.path.basename(compressed_file)
        target_url = f"http://{self.server_ip}:{self.server_port}/{filename}"
        container_name = f"cts_worker_{profile_name}"

        # 【修复点 1】: 预先清理可能残留的同名容器，防止Conflict
        try:
            self.docker_client.containers.get(container_name).remove(force=True)
        except: pass

        container = None
        try:
            # 【修复点 2】: 路径改为当前目录 os.path.abspath("client_agent.py")
            container = self.docker_client.containers.run(
                CLIENT_IMAGE,
                name=container_name,
                network=self.network_name,
                detach=True,
                nano_cpus=int(config['cpu'] * 1e9),
                mem_limit=config['mem'],
                volumes={
                    os.path.abspath("client_agent.py"): {'bind': '/app/client_agent.py', 'mode': 'ro'}
                },
                command="tail -f /dev/null"
            )
            
            cmd = f"python3 /app/client_agent.py {target_url} --method {method_name}"
            exec_result = container.exec_run(cmd)
            output = exec_result.output.decode('utf-8', errors='ignore')
            
            if exec_result.exit_code != 0:
                raise Exception(f"Agent Error: {output[-300:]}")
            
            return json.loads(output.strip().split('\n')[-1])

        finally:
            if container: 
                try: container.remove(force=True)
                except: pass

    def save_result(self, image, profile, method, rep, data, error=None):
        status = 'FAILED' if error else 'SUCCESS'
        is_noise = False
        
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
        if status == 'SUCCESS':
            logger.info(f"✅ 完成: Rep{rep} | DL={data.get('download_time',0):.2f}s | Decomp={data.get('decomp_time',0):.2f}s")
        else:
            logger.warning(f"❌ 失败: {method} | {error}")

    def cleanup(self):
        try: self.server.remove(force=True)
        except: pass
        try: self.network.remove()
        except: pass
        logger.info("🧹 实验资源已清理")

    def run_matrix(self):
        logger.info(f"🚀 开始全真网络仿真实验...")
        try:
            for image in TARGET_IMAGES:
                try:
                    raw_path, raw_size = self._pull_and_save_raw_tar(image)
                    
                    for profile_name in CLIENT_PROFILES.keys():
                        config = CLIENT_PROFILES[profile_name]
                        self.update_server_network(config['bw'], config['delay'])
                        
                        for method in COMPRESSION_METHODS.keys():
                            try:
                                needed_reps = []
                                for r in range(REPETITIONS):
                                    if not self.is_experiment_done(image, profile_name, method, r):
                                        needed_reps.append(r)
                                
                                if not needed_reps: continue

                                comp_path, comp_size = self._create_compressed_payload(raw_path, method)
                                
                                for rep in needed_reps:
                                    logger.info(f"▶️  {image} | {profile_name} | {method} | Rep{rep}")
                                    try:
                                        result = self.run_agent_in_container(profile_name, comp_path, method)
                                        result.update({'original_size': raw_size, 'compressed_size': comp_size})
                                        self.save_result(image, profile_name, method, rep, result)
                                    except Exception as e:
                                        self.save_result(image, profile_name, method, rep, {}, error=e)
                                    
                                    time.sleep(1)
                            finally:
                                if comp_path and os.path.exists(comp_path):
                                    os.remove(comp_path)
                except Exception as e:
                    logger.critical(f"🔥 镜像级错误 ({image}): {e}")
                finally:
                    if 'raw_path' in locals() and os.path.exists(raw_path):
                        os.remove(raw_path)
                    try: self.docker_client.images.remove(image, force=True)
                    except: pass  
        finally:
            self.cleanup()

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    try:
        orchestrator.run_matrix()
    except KeyboardInterrupt:
        orchestrator.cleanup()
