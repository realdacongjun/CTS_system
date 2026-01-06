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
        
        # 确保临时目录存在且为空 (防止上次残留)
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
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
            # Pumba 实际上在你的新逻辑里没用到，用的是 tc，但保留检查也无妨
            # subprocess.run(['pumba', '--version'], check=True, stdout=subprocess.PIPE) 
            subprocess.run(['tc', '-V'], check=True, stdout=subprocess.PIPE)
            logger.info("✅ 环境依赖检查通过 (Docker, tc)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ 缺少必要的依赖工具 (tc)，请先安装: sudo apt install iproute2")
            sys.exit(1)

    def _clear_system_cache(self):
        """清理系统缓存以保证实验准确性"""
        try:
            subprocess.run('sync', shell=True)
            # 需要sudo权限，如果报错则忽略
            subprocess.run('echo 3 > /proc/sys/vm/drop_caches', shell=True, stderr=subprocess.DEVNULL)
        except Exception:
            pass # 忽略权限错误

    def is_experiment_done(self, image, profile, method, rep):
        """检查实验是否已经完成（断点续跑）"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT count(*) FROM experiments 
            WHERE image=? AND client_profile=? AND method=? AND rep_id=? AND status='SUCCESS'
        ''', (image, profile, method, rep))
        return cursor.fetchone()[0] > 0

    # === [优化1] 拆分: 只负责拉取和导出原始tar ===
    def _pull_and_save_raw_tar(self, image_name):
        """拉取镜像并保存为未压缩的tar文件"""
        safe_img_name = image_name.replace(':', '_').replace('/', '_')
        raw_tar_path = os.path.join(TEMP_DIR, f"{safe_img_name}_raw.tar")
        
        try:
            # 1. 拉取
            logger.info(f"⬇️  正在拉取镜像: {image_name}")
            self.docker_client.images.pull(image_name)
            
            # 2. 导出
            logger.info(f"💾 正在导出为原始Tar: {raw_tar_path}")
            image = self.docker_client.images.get(image_name)
            with open(raw_tar_path, 'wb') as f:
                for chunk in image.save():
                    f.write(chunk)
            
            original_size = os.path.getsize(raw_tar_path)
            return raw_tar_path, original_size
        except Exception as e:
            logger.error(f"镜像准备失败: {e}")
            # 如果失败，尝试清理
            if os.path.exists(raw_tar_path):
                os.remove(raw_tar_path)
            raise e

    # === [优化2] 拆分: 只负责压缩 ===
    def _create_compressed_payload(self, raw_tar_path, method_name):
        """基于已有的raw tar创建压缩包"""
        cmd_args = COMPRESSION_METHODS[method_name]
        
        # 构造输出文件名
        if 'gzip' in method_name: ext = '.gz'
        elif 'zstd' in method_name: ext = '.zst'
        elif 'lz4' in method_name: ext = '.lz4'
        elif 'brotli' in method_name: ext = '.br'
        else: ext = '.dat'
        
        compressed_path = raw_tar_path + ext
        
        # 如果文件已存在（比如上次中断），先删除
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

        logger.info(f"📦 正在压缩 ({method_name})...")
        
        # 执行压缩
        try:
            if 'gzip' in method_name:
                with open(raw_tar_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                    subprocess.run(cmd_args, stdin=f_in, stdout=f_out, check=True)
            elif 'brotli' in method_name:
                 with open(raw_tar_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                    subprocess.run(cmd_args, stdin=f_in, stdout=f_out, check=True)
            else:
                # zstd 和 lz4
                subprocess.run(cmd_args + [raw_tar_path, '-o', compressed_path], check=True)
            
            compressed_size = os.path.getsize(compressed_path)
            return compressed_path, compressed_size
        except Exception as e:
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            raise e

    def setup_client_container(self, profile_name):
        """启动并配置客户端容器"""
        config = CLIENT_PROFILES[profile_name]
        container_name = f"cts_worker_{profile_name}"
        
        try:
            old = self.docker_client.containers.get(container_name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass

        # 启动容器
        container = self.docker_client.containers.run(
            CLIENT_IMAGE,
            name=container_name,
            detach=True,
            tty=True,
            nano_cpus=int(config['cpu'] * 1e9),
            mem_limit=config['mem'],
            cap_add=['NET_ADMIN'], 
            volumes={TEMP_DIR: {'bind': '/data', 'mode': 'rw'}}, 
            command="tail -f /dev/null"
        )
        
        # 应用TC
        logger.info(f"🌐 配置网络 ({profile_name}): BW={config['bw']}, Delay={config['delay']}")
        tc_cmd = f"tc qdisc add dev eth0 root netem rate {config['bw']} delay {config['delay']}"
        
        exit_code, output = container.exec_run(tc_cmd)
        if exit_code != 0:
            # 尝试重置后重试
            container.exec_run("tc qdisc del dev eth0 root")
            container.exec_run(tc_cmd)
        
        return container

    def run_agent_in_container(self, container, compressed_file, method_name):
        """在容器内执行解压测试"""
        filename = os.path.basename(compressed_file)
        container_path = f"/data/{filename}"
        
        # 参数清洗
        if 'lz4' in method_name:
            base_method = 'lz4'
        elif 'brotli' in method_name:
            base_method = 'brotli'
        else:
            base_method = method_name.split('-')[0]

        cmd = f"python3 /app/client_agent.py {container_path} --method {base_method}"
        
        # 增加超时控制，防止死锁
        try:
            # exec_run 不支持 timeout 参数，这里依赖 agent 内部逻辑
            # 如果需要强杀，可以用 python 的 threading Timer，但这里简化处理
            exec_result = container.exec_run(cmd)
        except Exception as e:
            raise Exception(f"Docker Exec Failed: {e}")

        output = exec_result.output.decode('utf-8', errors='ignore')
        
        if exec_result.exit_code != 0:
            # logger.error(f"Agent Error Output: {output}") # 不要在控制台刷屏报错
            raise Exception(f"Agent Execution Failed: {output[-200:]}") # 只记录最后200字符
            
        try:
            # 寻找最后一行有效的 JSON
            lines = output.strip().split('\n')
            json_str = lines[-1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON output: {output[-100:]}")

    def save_result(self, image, profile, method, rep, data, error=None):
        """保存结果到数据库"""
        is_noise = False
        status = 'SUCCESS'
        
        if error:
            status = 'FAILED'
        else:
            # 数据校验
            try:
                target_bw_mbps = float(CLIENT_PROFILES[profile]['bw'].replace('mbit', '')) 
                measured_bw = data.get('bandwidth_measured', 0)
                if data.get('decomp_time', 0) < 0.001: # 极短时间视为异常
                    is_noise = True
                    status = 'ABNORMAL'
            except:
                pass

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
        
        # 简略日志
        if status == 'SUCCESS':
            logger.info(f"✅ 完成: {method} | Rep{rep} | T={data.get('total_time',0):.2f}s")
        else:
            logger.warning(f"❌ 失败: {method} | Rep{rep}")

    def run_matrix(self):
        """执行完整实验矩阵 (40GB硬盘优化版)"""
        logger.info(f"🚀 开始运行实验矩阵 (串行模式)...")
        
        # 1. 外层循环：镜像 (处理完一个删一个)
        for image in TARGET_IMAGES:
            raw_tar_path = None
            try:
                # === 阶段 A: 准备原料 (占用最大空间) ===
                # 检查是否所有 Profile + Method 都跑完了，如果是，直接跳过拉取
                # (这里简化处理，总是拉取，依赖 is_experiment_done 跳过具体 Rep)
                
                raw_tar_path, original_size = self._pull_and_save_raw_tar(image)
                
                # 2. 中层循环：客户端画像
                for profile_name in CLIENT_PROFILES.keys():
                    container = None
                    try:
                        container = self.setup_client_container(profile_name)
                        
                        # 3. 内层循环：压缩算法
                        for method in COMPRESSION_METHODS.keys():
                            
                            # === 阶段 B: 生产压缩包 (占用较小空间) ===
                            compressed_path = None
                            try:
                                # 检查是否所有 Rep 都跑完了
                                all_reps_done = True
                                for rep in range(REPETITIONS):
                                    if not self.is_experiment_done(image, profile_name, method, rep):
                                        all_reps_done = False
                                        break
                                
                                if all_reps_done:
                                    logger.info(f"⏭️  跳过已完成组: {image} | {profile_name} | {method}")
                                    continue

                                # 只有需要跑实验时，才进行压缩
                                compressed_path, compressed_size = self._create_compressed_payload(raw_tar_path, method)
                                
                                # 4. 重复实验
                                for rep in range(REPETITIONS):
                                    if self.is_experiment_done(image, profile_name, method, rep):
                                        continue
                                    
                                    self._clear_system_cache()
                                    try:
                                        result_data = self.run_agent_in_container(container, compressed_path, method)
                                        # 补充数据
                                        result_data['original_size'] = original_size
                                        result_data['compressed_size'] = compressed_size
                                        
                                        self.save_result(image, profile_name, method, rep, result_data)
                                    except Exception as e:
                                        self.save_result(image, profile_name, method, rep, {}, error=e)
                                    
                                    time.sleep(0.5) 

                            finally:
                                # === [关键优化] 用完即删压缩包 ===
                                if compressed_path and os.path.exists(compressed_path):
                                    os.remove(compressed_path)
                                    # logger.info(f"🗑️  已删除临时压缩包: {os.path.basename(compressed_path)}")

                    finally:
                        if container:
                            container.remove(force=True)

            except Exception as e:
                logger.critical(f"🔥 镜像级致命错误 ({image}): {e}")
            
            finally:
                # === [关键优化] 彻底清理镜像 ===
                # 1. 删除原始大 Tar
                if raw_tar_path and os.path.exists(raw_tar_path):
                    os.remove(raw_tar_path)
                    logger.info(f"🗑️  已删除原始Tar: {image}")
                
                # 2. 删除 Docker 镜像
                try:
                    self.docker_client.images.remove(image, force=True)
                    logger.info(f"🧹 已卸载 Docker 镜像: {image}")
                except:
                    pass
                
                # 3. 强力清理残留 (Prune)
                try:
                    self.docker_client.images.prune()
                except:
                    pass

        logger.info("🎉 所有实验执行完毕！")

if __name__ == "__main__":
    if os.geteuid() != 0:
        logger.warning("⚠️ 建议以 root 权限运行，否则 tc 网络限制可能失效。")
    
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_matrix()