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
        
        # 确保临时目录存在
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

        # === 新增：网络架构初始化 ===
        self.network_name = "cts_experiment_net"
        self.server_container_name = "cts_image_server"
        self.server_ip = "cts_image_server" # Docker DNS 会自动解析容器名
        self.server_port = 8000
        
        self._setup_infrastructure()

    def _init_db(self):
        # 数据库结构保持不变
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
        # 只需要检查 lz4，tc 现在在容器里跑，宿主机不需要装
        try:
            subprocess.run(['lz4', '--version'], check=True, stdout=subprocess.PIPE)
            logger.info("✅ 环境依赖检查通过")
        except:
            logger.error("❌ 宿主机缺少 lz4 工具")
            sys.exit(1)

    def _setup_infrastructure(self):
        """搭建实验基础设施：网络 + 服务端容器"""
        logger.info("🏗️  正在搭建实验网络架构...")
        
        # 1. 创建专用网络
        try:
            self.docker_client.networks.get(self.network_name).remove()
        except: pass
        self.network = self.docker_client.networks.create(self.network_name, driver="bridge")

        # 2. 启动服务端容器 (长期运行)
        try:
            self.docker_client.containers.get(self.server_container_name).remove(force=True)
        except: pass
        
        logger.info("🔵 启动镜像服务器 (Image Server)...")
        # 我们直接用 cts_client_image 充当服务器，因为它里面有 python
        self.server = self.docker_client.containers.run(
            CLIENT_IMAGE,
            name=self.server_container_name,
            network=self.network_name,
            detach=True,
            cap_add=["NET_ADMIN"], # 必须有这个权限才能运行 tc
            volumes={TEMP_DIR: {'bind': '/data', 'mode': 'ro'}}, # 只读挂载数据
            # 启动 HTTP Server，根目录为 /data
            command=f"python3 -m http.server {self.server_port} --directory /data"
        )
        # 确保服务器起来了
        time.sleep(2) 

    def update_server_network(self, bw, delay):
        """动态调整服务端的上传限制"""
        # 先删除旧规则 (容错)
        self.server.exec_run("tc qdisc del dev eth0 root")
        
        # 添加新规则 (Netem 同时控制带宽和延迟)
        # 这里的 rate 是限制服务器的【上传速度】，也就是客户端的【下载速度】
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
        # 这一步和原来一样，宿主机负责准备原始数据
        safe_img_name = image_name.replace(':', '_').replace('/', '_')
        raw_tar_path = os.path.join(TEMP_DIR, f"{safe_img_name}_raw.tar")
        
        # 如果文件已存在且大小正常，跳过拉取（断点续传优化）
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
        # 压缩逻辑和原来一样，生成的文件会在 TEMP_DIR 里
        # 因为 Server 挂载了 TEMP_DIR，所以 Client 马上就能通过 HTTP 下载到它
        cmd_args = COMPRESSION_METHODS[method_name]
        if 'gzip' in method_name: ext = '.gz'
        elif 'zstd' in method_name: ext = '.zst'
        elif 'lz4' in method_name: ext = '.lz4'
        elif 'brotli' in method_name: ext = '.br'
        else: ext = '.dat'
        
        compressed_path = raw_tar_path + ext
        # 简单缓存检查
        if os.path.exists(compressed_path):
             return compressed_path, os.path.getsize(compressed_path)

        try:
            # ... (压缩代码保持不变，照抄你之前的逻辑) ...
            # 为了节省篇幅，这里假设你保留了之前的 subprocess 压缩逻辑
            # 务必把之前的 _create_compressed_payload 完整逻辑放在这里
            # 注意：lz4 需要 input output 格式
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
        """启动 Client 容器 -> 下载 -> 解压"""
        config = CLIENT_PROFILES[profile_name]
        filename = os.path.basename(compressed_file)
        
        # 构造下载链接：http://cts_image_server:8000/文件名
        target_url = f"http://{self.server_ip}:{self.server_port}/{filename}"
        
        container = None
        try:
            # 启动 Client 容器
            container = self.docker_client.containers.run(
                CLIENT_IMAGE,
                name=f"cts_worker_{profile_name}",
                network=self.network_name, # 加入同一网络
                detach=True,
                nano_cpus=int(config['cpu'] * 1e9),
                mem_limit=config['mem'],
                # ⚠️ 关键点：挂载最新的 client_agent.py 脚本进去
                volumes={
                    os.path.abspath("ml_training/client_agent.py"): {'bind': '/app/client_agent.py', 'mode': 'ro'}
                },
                command="tail -f /dev/null"
            )
            
            # 运行脚本
            # 注意：这里不需要传本地路径了，传 URL
            cmd = f"python3 /app/client_agent.py {target_url} --method {method_name}"
            
            exec_result = container.exec_run(cmd)
            output = exec_result.output.decode('utf-8', errors='ignore')
            
            if exec_result.exit_code != 0:
                raise Exception(f"Agent Error: {output[-300:]}")
            
            return json.loads(output.strip().split('\n')[-1])

        finally:
            if container: container.remove(force=True)

    def save_result(self, image, profile, method, rep, data, error=None):
        # 保持不变
        status = 'FAILED' if error else 'SUCCESS'
        is_noise = False
        # ... (照抄之前的 save_result) ...
        # 这里为了确保代码完整性，请保留原有的 database insert 逻辑
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
        """清理所有资源"""
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
                        # 1. 在 Server 端应用当前 Profile 的网络限制
                        config = CLIENT_PROFILES[profile_name]
                        self.update_server_network(config['bw'], config['delay'])
                        
                        for method in COMPRESSION_METHODS.keys():
                            try:
                                # 检查是否已完成
                                needed_reps = []
                                for r in range(REPETITIONS):
                                    if not self.is_experiment_done(image, profile_name, method, r):
                                        needed_reps.append(r)
                                
                                if not needed_reps: continue

                                # 2. 准备压缩包
                                comp_path, comp_size = self._create_compressed_payload(raw_path, method)
                                
                                # 3. 跑实验
                                for rep in needed_reps:
                                    logger.info(f"▶️  {image} | {profile_name} | {method} | Rep{rep}")
                                    try:
                                        # 这一步会自动启动 Client 去下载
                                        result = self.run_agent_in_container(profile_name, comp_path, method)
                                        result.update({'original_size': raw_size, 'compressed_size': comp_size})
                                        self.save_result(image, profile_name, method, rep, result)
                                    except Exception as e:
                                        self.save_result(image, profile_name, method, rep, {}, error=e)
                                    
                                    # 稍微歇一下，防止 Docker 网络堵死
                                    time.sleep(1)

                            finally:
                                # 只有当所有 Profile 都跑完这个 method，才删文件？
                                # 现在的逻辑是跑完一个 method 就删，这样其实也没事，反正生成很快
                                if comp_path and os.path.exists(comp_path):
                                    os.remove(comp_path)

                except Exception as e:
                    logger.critical(f"🔥 镜像级错误 ({image}): {e}")
                finally:
                    # 清理原始 tar
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