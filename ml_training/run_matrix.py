
import os
import sys
import time
import json
import sqlite3
import logging
import subprocess
import shutil
import docker
import uuid
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
        required_tools = ['tar', 'gzip', 'zstd', 'lz4', 'brotli']
        missing = []
        for tool in required_tools:
            if not shutil.which(tool):
                missing.append(tool)
        
        if missing:
            logger.error(f"❌ 宿主机缺少必要工具: {', '.join(missing)}")
            logger.error("请安装: yum install -y tar gzip zstd lz4 brotli")
            sys.exit(1)
        logger.info("✅ 环境依赖检查通过")

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
        # 1. 清理旧规则 (忽略错误)
        self.server.exec_run("tc qdisc del dev eth0 root", check=False)
        
        # 2. TBF 限速 (Token Bucket Filter)
        cmd_tbf = f"tc qdisc add dev eth0 root handle 1: tbf rate {bw} burst 32kbit latency 400ms"
        exit_code, output = self.server.exec_run(cmd_tbf)
        if exit_code != 0:
            logger.error(f"❌ TBF限速失败: {output.decode()}")
            return

        # 3. Netem 延迟 (挂载到 TBF 下)
        cmd_netem = f"tc qdisc add dev eth0 parent 1:1 handle 10: netem delay {delay}"
        exit_code, output = self.server.exec_run(cmd_netem)
        
        if exit_code != 0:
            logger.error(f"❌ Netem延迟失败: {output.decode()}")
        else:
            logger.info(f"🌐 网络更新: {bw} + {delay}")

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
        
        # 即使文件存在，也检查大小，防止是坏文件
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
            
            # 【优化】导出后立即删除 Docker 里的镜像层，节省空间
            # self.docker_client.images.remove(image_name, force=True) 
            # (注：暂不立即删，因为后面可能还要用 info，统一在循环末尾删)
            
            return raw_tar_path, os.path.getsize(raw_tar_path)
        except Exception as e:
            if os.path.exists(raw_tar_path): os.remove(raw_tar_path)
            raise e

    def _create_compressed_payload(self, raw_tar_path, method_name):
        cmd_parts = COMPRESSION_METHODS[method_name]
        prog = cmd_parts[0]
        args = " ".join(cmd_parts[1:]) 
        
        if 'gzip' in method_name: ext = '.tar.gz'
        elif 'zstd' in method_name: ext = '.tar.zst'
        elif 'lz4' in method_name: ext = '.tar.lz4'
        elif 'brotli' in method_name: ext = '.tar.br'
        else: ext = '.tar'
        
        compressed_path = raw_tar_path + ext
        
        if os.path.exists(compressed_path):
             return compressed_path, os.path.getsize(compressed_path)

        try:
            # 构造 tar 命令: tar -I 'gzip -1' -cf out.tar.gz in.tar
            tar_cmd = ['tar', '-I', f"{prog} {args}", '-cf', compressed_path, raw_tar_path]
            subprocess.run(tar_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return compressed_path, os.path.getsize(compressed_path)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown error"
            logger.error(f"Compression failed for {method_name}: {error_msg}")
            if os.path.exists(compressed_path): os.remove(compressed_path)
            raise e

    def run_agent_in_container(self, profile_name, compressed_file, method_name):
        config = CLIENT_PROFILES[profile_name]
        filename = os.path.basename(compressed_file)
        target_url = f"http://{self.server_ip}:{self.server_port}/{filename}"
        
        # 【关键修复】UUID 随机后缀，防止容器重名冲突
        random_suffix = uuid.uuid4().hex[:6]
        container_name = f"cts_worker_{profile_name}_{random_suffix}"

        container = None
        try:
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
            
            # 【优化】带超时限制 (300秒) + 结果写入文件
            cmd = f"timeout 300 python3 /app/client_agent.py {target_url} --method {method_name}"
            exec_res = container.exec_run(f"sh -c '{cmd}'")
            
            if exec_res.exit_code != 0:
                err_log = exec_res.output.decode('utf-8', errors='ignore')
                raise Exception(f"Agent Execution Failed: {err_log[-200:]}")

            # 读取结果文件
            cat_res = container.exec_run("cat /tmp/result.json")
            if cat_res.exit_code != 0:
                 raise Exception("Result file not found")
                 
            return json.loads(cat_res.output.decode('utf-8').strip())

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
            logger.info(f"✅ 完成: {profile} | {method} | DL={data.get('download_time',0):.2f}s | Decomp={data.get('decomp_time',0):.2f}s")
        else:
            logger.warning(f"❌ 失败: {method} | {error}")

    def force_cleanup_images(self):
        """【关键修复】激进的清理逻辑，防止磁盘爆满"""
        logger.info("🧹 执行深度清理...")
        try:
            # 1. 尝试清理所有已停止的容器
            self.docker_client.containers.prune()
            # 2. 清理悬空的镜像 (dangling)
            self.docker_client.images.prune()
            # 3. 显式清理目标镜像 (在 config.TARGET_IMAGES 里的)
            for img in TARGET_IMAGES:
                try:
                    self.docker_client.images.remove(img, force=True)
                except: pass
        except Exception as e:
            logger.warning(f"清理过程遇到非致命错误: {e}")

    def cleanup(self):
        try: self.server.remove(force=True)
        except: pass
        try: self.network.remove()
        except: pass
        # 执行深度清理
        self.force_cleanup_images()
        logger.info("🧹 实验资源已清理完毕")

    def run_matrix(self):
        logger.info(f"🚀 开始全真网络仿真实验 (最终版)...")
        try:
            for image in TARGET_IMAGES:
                try:
                    raw_path = None
                    raw_path, raw_size = self._pull_and_save_raw_tar(image)
                    
                    for profile_name in CLIENT_PROFILES.keys():
                        config = CLIENT_PROFILES[profile_name]
                        self.update_server_network(config['bw'], config['delay'])
                        
                        for method in COMPRESSION_METHODS.keys():
                            comp_path = None
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
                            except Exception as e:
                                logger.error(f"处理压缩包失败: {e}")
                            finally:
                                if comp_path and os.path.exists(comp_path):
                                    os.remove(comp_path)
                except Exception as e:
                    logger.critical(f"🔥 镜像级错误 ({image}): {e}")
                finally:
                    # 跑完一个镜像，立马删掉原始文件，释放 40G 硬盘的压力
                    if raw_path and os.path.exists(raw_path):
                        os.remove(raw_path)
                    try: 
                        # 尝试立刻删除该镜像的 Docker 层
                        self.docker_client.images.remove(image, force=True)
                        logger.info(f"🗑️ 已清理镜像层: {image}")
                    except: pass  
        finally:
            self.cleanup()

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    try:
        orchestrator.run_matrix()
    except KeyboardInterrupt:
        orchestrator.cleanup()
