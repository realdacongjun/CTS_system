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
# 确保 config.py 里有这些变量
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
        
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

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
            subprocess.run(['tc', '-V'], check=True, stdout=subprocess.PIPE)
            # 检查 lz4 是否可用
            subprocess.run(['lz4', '--version'], check=True, stdout=subprocess.PIPE)
            logger.info("✅ 环境依赖检查通过 (Docker, tc, lz4)")
        except Exception:
            logger.error("❌ 缺少必要依赖，请确保安装了: iproute-tc, lz4, zstd")
            sys.exit(1)

    def _clear_system_cache(self):
        try:
            subprocess.run('sync', shell=True)
            subprocess.run('echo 3 > /proc/sys/vm/drop_caches', shell=True, stderr=subprocess.DEVNULL)
        except:
            pass

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
        
        try:
            logger.info(f"⬇️  正在拉取镜像: {image_name}")
            self.docker_client.images.pull(image_name)
            
            logger.info(f"💾 正在导出为原始Tar: {raw_tar_path}")
            image = self.docker_client.images.get(image_name)
            with open(raw_tar_path, 'wb') as f:
                for chunk in image.save():
                    f.write(chunk)
            
            return raw_tar_path, os.path.getsize(raw_tar_path)
        except Exception as e:
            if os.path.exists(raw_tar_path):
                os.remove(raw_tar_path)
            raise e

    # === 【核心修复】修复了 lz4 参数问题 ===
    def _create_compressed_payload(self, raw_tar_path, method_name):
        cmd_args = COMPRESSION_METHODS[method_name]
        
        if 'gzip' in method_name: ext = '.gz'
        elif 'zstd' in method_name: ext = '.zst'
        elif 'lz4' in method_name: ext = '.lz4'
        elif 'brotli' in method_name: ext = '.br'
        else: ext = '.dat'
        
        compressed_path = raw_tar_path + ext
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

        try:
            if 'gzip' in method_name or 'brotli' in method_name:
                with open(raw_tar_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                    subprocess.run(cmd_args, stdin=f_in, stdout=f_out, check=True)
            elif 'zstd' in method_name:
                subprocess.run(cmd_args + [raw_tar_path, '-o', compressed_path], check=True)
            elif 'lz4' in method_name:
                # 【修复点】lz4 不加 -o，直接 input output
                subprocess.run(cmd_args + [raw_tar_path, compressed_path], check=True)
            else:
                subprocess.run(cmd_args + [raw_tar_path, compressed_path], check=True)
            
            return compressed_path, os.path.getsize(compressed_path)
        except Exception as e:
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            raise e

    def setup_client_container(self, profile_name):
        config = CLIENT_PROFILES[profile_name]
        container_name = f"cts_worker_{profile_name}"
        
        try:
            self.docker_client.containers.get(container_name).remove(force=True)
        except:
            pass

        # === 【建议】如果 Mongo 总是失败，请在这里临时调大内存，例如 '1g' ===
        mem_limit = config['mem'] 
        
        container = self.docker_client.containers.run(
            CLIENT_IMAGE,
            name=container_name,
            detach=True,
            tty=True,
            nano_cpus=int(config['cpu'] * 1e9),
            mem_limit=mem_limit,  # 注意这里
            cap_add=['NET_ADMIN'], 
            volumes={TEMP_DIR: {'bind': '/data', 'mode': 'rw'}}, 
            command="tail -f /dev/null"
        )
        
        logger.info(f"🌐 配置网络 ({profile_name}): BW={config['bw']}, Delay={config['delay']}")
        tc_cmd = f"tc qdisc add dev eth0 root netem rate {config['bw']} delay {config['delay']}"
        container.exec_run(tc_cmd)
        
        return container

    def run_agent_in_container(self, container, compressed_file, method_name):
        filename = os.path.basename(compressed_file)
        container_path = f"/data/{filename}"
        
        if 'lz4' in method_name: base_method = 'lz4'
        elif 'brotli' in method_name: base_method = 'brotli'
        else: base_method = method_name.split('-')[0]

        cmd = f"python3 /app/client_agent.py {container_path} --method {base_method}"
        
        exec_result = container.exec_run(cmd)
        output = exec_result.output.decode('utf-8', errors='ignore')
        
        if exec_result.exit_code != 0:
            # 抛出详细错误，方便调试 OOM
            raise Exception(f"Agent Failed (Code {exec_result.exit_code}): {output[-300:]}")
            
        try:
            return json.loads(output.strip().split('\n')[-1])
        except:
            raise Exception(f"Invalid JSON: {output[-100:]}")

    def save_result(self, image, profile, method, rep, data, error=None):
        status = 'FAILED' if error else 'SUCCESS'
        is_noise = False
        
        if not error and data.get('decomp_time', 0) < 0.001:
            status = 'ABNORMAL'
            is_noise = True

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
            logger.info(f"✅ 完成: {method} | Rep{rep} | T={data.get('total_time',0):.2f}s")
        else:
            logger.warning(f"❌ 失败: {method} | Rep{rep}")

    def run_matrix(self):
        logger.info(f"🚀 开始运行实验矩阵...")
        
        for image in TARGET_IMAGES:
            raw_tar_path = None
            try:
                raw_tar_path, original_size = self._pull_and_save_raw_tar(image)
                
                for profile_name in CLIENT_PROFILES.keys():
                    container = None
                    try:
                        container = self.setup_client_container(profile_name)
                        
                        for method in COMPRESSION_METHODS.keys():
                            # === 【重要】增加 try 避免一个算法炸全家 ===
                            try:
                                compressed_path = None
                                
                                # 检查是否跳过
                                all_done = True
                                for rep in range(REPETITIONS):
                                    if not self.is_experiment_done(image, profile_name, method, rep):
                                        all_done = False; break
                                if all_done:
                                    # logger.info(f"⏭️  跳过: {method}"); 
                                    continue

                                logger.info(f"📦 正在压缩 ({method})...")
                                compressed_path, compressed_size = self._create_compressed_payload(raw_tar_path, method)
                                
                                for rep in range(REPETITIONS):
                                    if self.is_experiment_done(image, profile_name, method, rep): continue
                                    
                                    self._clear_system_cache()
                                    try:
                                        result_data = self.run_agent_in_container(container, compressed_path, method)
                                        result_data.update({'original_size': original_size, 'compressed_size': compressed_size})
                                        self.save_result(image, profile_name, method, rep, result_data)
                                    except Exception as e:
                                        # 【重要】打印具体错误信息到控制台
                                        logger.error(f"⚠️ 实验异常详情: {e}")
                                        self.save_result(image, profile_name, method, rep, {}, error=e)
                                    
                                    time.sleep(0.5)

                            except Exception as e:
                                logger.error(f"⚠️ 算法级错误 ({method}): {e}")
                            
                            finally:
                                if compressed_path and os.path.exists(compressed_path):
                                    os.remove(compressed_path)

                    finally:
                        if container: container.remove(force=True)

            except Exception as e:
                logger.critical(f"🔥 镜像级致命错误 ({image}): {e}")
            
            finally:
                if raw_tar_path and os.path.exists(raw_tar_path):
                    os.remove(raw_tar_path)
                try:
                    self.docker_client.images.remove(image, force=True)
                    logger.info(f"🧹 已清理镜像: {image}")
                except: pass
        
        logger.info("🎉 所有实验结束")

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_matrix()