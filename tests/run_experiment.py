import os
import sys
import time
import json
import pickle
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import psutil
from scipy import stats

# ==========================================
# 0. 配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cts_optimized_0218_2125_seed42.pth")
PREP_PATH = os.path.join(BASE_DIR, "models", "preprocessing_objects_optimized.pkl")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# 【修改】使用自定义镜像
WORKER_IMAGE = "cts-worker:ready"  # 你构建的镜像名
FALLBACK_IMAGE = "docker:25-dind"  # 备用官方镜像

TEST_IMAGES = [
    "alpine:3.19",      
    "nginx:1.25-alpine", 
    "ubuntu:22.04"       
]

BASELINE_STATIC_CONFIG = {
    "IoT_Weak": {"threads": 1, "cpu_quota": 0.2, "chunk_size_kb": 64, "algo_name": "gzip"},
    "Edge_Net": {"threads": 4, "cpu_quota": 1.0, "chunk_size_kb": 256, "algo_name": "gzip"},
    "Cloud_DC": {"threads": 16, "cpu_quota": 2.0, "chunk_size_kb": 1024, "algo_name": "gzip"},
    "Edge_Ablation": {"threads": 4, "cpu_quota": 1.0, "chunk_size_kb": 256, "algo_name": "gzip"}
}

# ==========================================
# 1. 导入
# ==========================================
try:
    from cts_model import CompactCFTNetV2, CONFIG as MODEL_CONFIG
    from cags_decision import CAGSDecisionEngine
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 基础设施（最终版 - 支持自定义镜像）
# ==========================================
class LabInfrastructure:
    def __init__(self):
        self.net_name = "cts-net"
        self.client_container = "cts-client-worker"
        self.worker_image = WORKER_IMAGE
        self.using_custom_image = False  # 标记是否使用自定义镜像
        
        # 【关键】如果使用自定义镜像，直接设为支持全功能
        self.skopeo_supports_threads = False
        self.skopeo_supports_chunk = False
        
        self._create_network()
        self._init_model()
        self.current_worker_cpu_max = 8.0

    def _create_network(self):
        """自动创建Docker网络"""
        try:
            result = subprocess.run(
                f"docker network inspect {self.net_name}",
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"✅ 网络 {self.net_name} 已存在")
                return True
        except:
            pass
        
        print(f"🔧 创建Docker网络: {self.net_name}")
        try:
            subprocess.run(
                f"docker network create --driver bridge {self.net_name}",
                shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"✅ 网络创建成功")
            time.sleep(1)
            return True
        except Exception as e:
            print(f"❌ 网络创建失败: {e}")
            return False

    def _init_model(self):
        """加载CTS模型"""
        print("🧠 加载CTS模型...")
        
        if not os.path.exists(PREP_PATH):
            raise FileNotFoundError(f"缺少预处理文件: {PREP_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"缺少模型文件: {MODEL_PATH}")
            
        with open(PREP_PATH, 'rb') as f:
            prep = pickle.load(f)
            
        self.scaler_c = prep['scaler_c']
        self.scaler_i = prep['scaler_i']
        self.enc = prep['enc']
        self.cols_c = prep['cols_c']
        self.cols_i = prep['cols_i']

        self.model = CompactCFTNetV2(
            client_feats=len(self.cols_c),
            image_feats=len(self.cols_i),
            num_algos=len(self.enc.classes_),
            embed_dim=MODEL_CONFIG.get("embed_dim", 64)
        )
        
        state_dict = torch.load(MODEL_PATH, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.engine = CAGSDecisionEngine(
            self.model, self.scaler_c, self.scaler_i,
            self.enc, self.cols_c, self.cols_i, device
        )
        
        print(f"✅ 模型加载完成")

    def _get_bridge_iface(self) -> Optional[str]:
        """获取网桥接口"""
        try:
            res = subprocess.check_output(
                f"docker network inspect {self.net_name}", 
                shell=True, text=True
            )
            net_data = json.loads(res)
            
            bridge_id = net_data[0].get('Id', '')[:12]
            if bridge_id:
                return f"br-{bridge_id}"
            
            options = net_data[0].get('Options', {})
            bridge_name = options.get('com.docker.network.bridge.name')
            if bridge_name:
                return bridge_name
                
            return None
        except Exception as e:
            print(f"⚠️  获取网桥失败: {e}")
            return None

    def set_network(self, bw_mbps: float, delay_ms: float, loss_pct: float) -> bool:
        """配置网络tc规则"""
        iface = self._get_bridge_iface()
        if not iface:
            print("⚠️  跳过网络配置（无法获取网桥）")
            return False
        
        subprocess.run(
            f"sudo tc qdisc del dev {iface} root 2>/dev/null", 
            shell=True, stderr=subprocess.DEVNULL
        )
        
        cmd = (
            f"sudo tc qdisc add dev {iface} root handle 1: htb default 11 && "
            f"sudo tc class add dev {iface} parent 1: classid 1:1 htb rate {bw_mbps}mbit ceil {bw_mbps}mbit && "
            f"sudo tc qdisc add dev {iface} parent 1:1 handle 10: "
            f"netem delay {delay_ms}ms loss {loss_pct}% limit 1000"
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   🌐 网络: {bw_mbps}Mbps | {delay_ms}ms | {loss_pct}%丢包")
            return True
        except Exception as e:
            print(f"   ⚠️  网络配置失败: {e}")
            return False

    def start_worker(self, max_cpu: float, mem: str) -> bool:
        """【最终版】启动工作容器，优先使用自定义镜像"""
        self.current_worker_cpu_max = max_cpu
        
        # 清理旧容器
        subprocess.run(
            f"docker rm -f {self.client_container} 2>/dev/null",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # 【方案A】先尝试使用自定义镜像（已预装skopeo）
        print(f"   🐳 尝试自定义镜像 {self.worker_image}...")
        cmd = (
            f"docker run -d --name {self.client_container} "
            f"--network {self.net_name} --cpus={max_cpu} --memory={mem} "
            f"--privileged {self.worker_image}"
        )
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            
            # 检查容器是否真的启动了
            time.sleep(2)
            check = subprocess.run(
                f"docker ps -q -f name={self.client_container}",
                shell=True, capture_output=True, text=True
            )
            
            if check.returncode == 0 and check.stdout.strip():
                print(f"   ✅ 自定义镜像启动成功")
                self.using_custom_image = True
                # 【关键】自定义镜像已预装新版skopeo，直接设为支持
                self.skopeo_supports_threads = True
                self.skopeo_supports_chunk = True
                print(f"   📦 skopeo: threads=True, chunk=True (预装)")
                return True
                
        except Exception as e:
            print(f"   ⚠️  自定义镜像失败: {e}")
        
        # 【方案B】回退到官方镜像，需要在线安装skopeo
        print(f"   🐳 回退到官方镜像 {FALLBACK_IMAGE}...")
        self.worker_image = FALLBACK_IMAGE
        
        cmd = (
            f"docker run -d --name {self.client_container} "
            f"--network {self.net_name} --cpus={max_cpu} --memory={mem} "
            f"--privileged {self.worker_image}"
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            print(f"   🐳 官方镜像启动中...", end="", flush=True)
            
            # 等待Docker-in-Docker就绪
            for i in range(30):
                time.sleep(1)
                check = subprocess.run(
                    f"docker exec {self.client_container} docker version",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if check.returncode == 0:
                    print(f" OK ({i+1}s)")
                    break
            else:
                print(" 超时")
                return False
            
            # 在线安装skopeo
            self._install_skopeo_fallback()
            return True
            
        except Exception as e:
            print(f"\n   ❌ 容器启动失败: {e}")
            return False

    def _install_skopeo_fallback(self):
        """【备选】官方镜像在线安装skopeo（网络不好时可能失败）"""
        print(f"   📦 在线安装skopeo...", end="", flush=True)
        
        # 尝试几个镜像源，每个30秒
        mirrors = [
            "https://mirrors.aliyun.com/alpine/edge/community",
            "https://mirrors.tuna.tsinghua.edu.cn/alpine/edge/community",
        ]
        
        for mirror in mirrors:
            cmd = (
                f"docker exec {self.client_container} sh -c '"
                f"sed -i \"s|dl-cdn.alpinelinux.org|{mirror.replace('https://', '').replace('/alpine/edge/community', '')}|g\" /etc/apk/repositories 2>/dev/null; "
                f"echo \"{mirror}\" >> /etc/apk/repositories; "
                f"apk update -q && apk add --no-cache skopeo'"
            )
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, 
                                      text=True, timeout=30)
                if result.returncode == 0:
                    print(" OK")
                    self._check_skopeo_capabilities()
                    return
            except:
                continue
        
        # 最后尝试默认源
        try:
            result = subprocess.run(
                f"docker exec {self.client_container} apk add --no-cache skopeo",
                shell=True, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(" OK(默认源)")
                self._check_skopeo_capabilities()
                return
        except:
            pass
        
        # 都失败了，进入兼容模式
        print(" 失败，进入【CPU控制模式】")
        self.skopeo_supports_threads = False
        self.skopeo_supports_chunk = False

    def _check_skopeo_capabilities(self):
        """检测skopeo支持的参数"""
        try:
            result = subprocess.run(
                f"docker exec {self.client_container} skopeo copy --help 2>&1",
                shell=True, capture_output=True, text=True, timeout=10
            )
            help_text = result.stdout + result.stderr
            self.skopeo_supports_threads = "--max-concurrent-downloads" in help_text
            self.skopeo_supports_chunk = "--chunk-size" in help_text
            print(f" (threads={self.skopeo_supports_threads}, chunk={self.skopeo_supports_chunk})")
        except:
            self.skopeo_supports_threads = False
            self.skopeo_supports_chunk = False
            print(" (检测失败，设为False)")

    def _set_cpu_quota(self, quota: float):
        """动态调整CPU配额"""
        quota = min(quota, self.current_worker_cpu_max)
        try:
            subprocess.run(
                f"docker update --cpus {quota} {self.client_container}",
                shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(0.3)
        except Exception as e:
            print(f"   ⚠️  CPU调整失败: {e}")

    def pull_image(self, image: str, config: Dict, timeout: int = 300) -> Tuple[float, bool, float]:
        """执行镜像拉取 - 根据skopeo能力动态构建命令"""
        self._set_cpu_quota(config['cpu_quota'])
        
        # 构建命令
        flags = ["--override-os linux", "--override-arch amd64"]
        
        if self.skopeo_supports_threads:
            flags.append(f"--max-concurrent-downloads {int(config['threads'])}")
        
        if self.skopeo_supports_chunk:
            flags.append(f"--chunk-size {int(config['chunk_size_kb'])}KB")
        
        flags.append(f"--compression-format {config['algo_name']}")
        
        skopeo_cmd = f"skopeo copy {' '.join(flags)} docker://docker.io/{image} docker-daemon:{image}"
        
        # 清理镜像
        subprocess.run(
            f"docker exec {self.client_container} docker rmi {image} 2>/dev/null",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
        
        # 执行
        start = time.time()
        peak_cpu = 0.0
        
        proc = subprocess.Popen(
            f"docker exec {self.client_container} {skopeo_cmd}",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        while proc.poll() is None:
            try:
                stats = subprocess.check_output(
                    f"docker stats {self.client_container} --no-stream --format '{{{{.CPUPerc}}}}'",
                    shell=True, text=True, timeout=1
                ).strip().replace('%', '')
                if stats:
                    peak_cpu = max(peak_cpu, float(stats))
            except:
                pass
            
            time.sleep(0.2)
            
            if time.time() - start > timeout:
                proc.kill()
                print(f"      ⏱️  超时({timeout}s)")
                break
        
        stdout, stderr = proc.communicate()
        elapsed = time.time() - start
        success = (proc.returncode == 0)
        
        if not success and stderr:
            err_msg = stderr.decode() if isinstance(stderr, bytes) else str(stderr)
            print(f"      ❌ {err_msg[:200].replace(chr(10), ' ')}")
        
        self._set_cpu_quota(self.current_worker_cpu_max)
        
        return elapsed, success, peak_cpu

    def get_image_features(self, image: str) -> Dict:
        """获取镜像特征"""
        size_map = {
            "alpine:3.19": 5,
            "nginx:1.25-alpine": 15,
            "ubuntu:22.04": 80
        }
        size = size_map.get(image, 50)
        
        return {
            'total_size_mb': size,
            'avg_layer_entropy': 0.65,
            'entropy_std': 0.2,
            'layer_count': 10,
            'size_std_mb': 5.0,
            'text_ratio': 0.3,
            'zero_ratio': 0.05
        }

    def cleanup(self, keep_network: bool = True):
        """清理环境"""
        print("\n🧹 清理环境...")
        
        subprocess.run(
            f"docker rm -f {self.client_container} 2>/dev/null",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("   ✅ 容器已删除")
        
        iface = self._get_bridge_iface()
        if iface:
            subprocess.run(
                f"sudo tc qdisc del dev {iface} root 2>/dev/null",
                shell=True, stderr=subprocess.DEVNULL
            )
            print("   ✅ 网络规则已清除")
        
        if not keep_network:
            subprocess.run(
                f"docker network rm {self.net_name} 2>/dev/null",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("   ✅ 网络已删除")
        
        print("   ✅ 环境清理完成")

# ==========================================
# 3. 实验编排器
# ==========================================
class ExperimentOrchestrator:
    def __init__(self, infra: LabInfrastructure):
        self.infra = infra
        self.results = []
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.repeat = 3

    def _log(self, exp: str, scene: str, method: str, image: str,
             time_s: float, ok: bool, cpu: float, cfg: Dict, extra: Optional[Dict] = None):
        """记录实验结果"""
        extra = extra or {}
        size = self.infra.get_image_features(image)['total_size_mb']
        
        # 失败时吞吐量显示N/A
        throughput = 0.0 if not ok else (size * 8) / max(time_s, 0.001)
        
        row = {
            'timestamp': datetime.now().isoformat(),
            'experiment': exp,
            'scene': scene,
            'method': method,
            'image': image,
            'image_size_mb': size,
            'time_s': time_s,
            'success': ok,
            'cpu_pct': cpu,
            'throughput_mbps': throughput,
            'cpu_efficiency': 0.0 if not ok else size / (cpu * time_s / 100 + 0.001),
            'config': json.dumps(cfg),
            'skopeo_threads_support': self.infra.skopeo_supports_threads,
            'skopeo_chunk_support': self.infra.skopeo_supports_chunk,
            'using_custom_image': self.infra.using_custom_image,
        }
        row.update(extra)
        self.results.append(row)
        
        status = "✅" if ok else "❌"
        tp_str = f"{throughput:7.2f}" if ok else "   N/A"
        print(f"   {status} {method:12s} | {image:20s} | {time_s:6.2f}s | {tp_str}Mbps")

    def _save(self):
        """保存结果"""
        if not self.results:
            print("⚠️  无数据可保存")
            return None
            
        df = pd.DataFrame(self.results)
        path = os.path.join(RESULT_DIR, f"results_{self.start_time}.csv")
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\n💾 保存: {path} ({len(df)}条记录)")
        return path

    def exp1_end2end(self):
        print("\n" + "="*70)
        print("🧪 实验1: 端到端性能基准")
        print("="*70)
        
        scenes = [
            {"name": "IoT_Weak", "cpu": 2.0, "mem": "2g", "bw": 2, "delay": 100, "loss": 2},
            {"name": "Edge_Net", "cpu": 4.0, "mem": "8g", "bw": 20, "delay": 30, "loss": 0.5},
            {"name": "Cloud_DC", "cpu": 8.0, "mem": "16g", "bw": 800, "delay": 1, "loss": 0},
        ]
        
        for s in scenes:
            print(f"\n📍 场景: {s['name']}")
            
            if not self.infra.start_worker(s['cpu'], s['mem']):
                continue
            self.infra.set_network(s['bw'], s['delay'], s['loss'])
            
            env = {
                'bandwidth_mbps': s['bw'],
                'cpu_limit': s['cpu'],
                'network_rtt': s['delay'],
                'mem_limit_mb': int(s['mem'][:-1]) * 1024
            }
            
            for r in range(self.repeat):
                print(f"\n   [重复 {r+1}/{self.repeat}]")
                for img in TEST_IMAGES:
                    feat = self.infra.get_image_features(img)
                    
                    base_cfg = BASELINE_STATIC_CONFIG[s['name']]
                    t, ok, cpu = self.infra.pull_image(img, base_cfg)
                    self._log("Exp1", s['name'], "Static", img, t, ok, cpu, base_cfg)
                    
                    try:
                        cts_cfg, metrics, _ = self.infra.engine.make_decision(env, feat)
                    except Exception as e:
                        print(f"      ⚠️ 决策失败: {e}")
                        cts_cfg = base_cfg
                        metrics = {}
                    
                    t, ok, cpu = self.infra.pull_image(img, cts_cfg)
                    self._log("Exp1", s['name'], "CTS", img, t, ok, cpu, cts_cfg, metrics)
                    time.sleep(1)
        
        self._save()

    def exp2_ablation(self):
        print("\n" + "="*70)
        print("🧪 实验2: 消融实验")
        print("="*70)
        
        scene = {"name": "Edge", "cpu": 4.0, "mem": "8g", "bw": 20, "delay": 30, "loss": 0.5}
        
        if not self.infra.start_worker(scene['cpu'], scene['mem']):
            return
        self.infra.set_network(scene['bw'], scene['delay'], scene['loss'])
        
        env = {
            'bandwidth_mbps': scene['bw'],
            'cpu_limit': scene['cpu'],
            'network_rtt': scene['delay'],
            'mem_limit_mb': 8192
        }
        
        variants = [
            ("Static", False, False),
            ("CFT_Only", False, False),
            ("CFT+Unc", True, False),
            ("CFT+CAGS", False, True),
            ("CFT+Unc+CAGS", True, True),
        ]
        
        for name, unc, dpc in variants:
            print(f"\n📍 变体: {name}")
            for r in range(self.repeat):
                for img in TEST_IMAGES:
                    feat = self.infra.get_image_features(img)
                    
                    if name == "Static":
                        cfg = BASELINE_STATIC_CONFIG['Edge_Ablation']
                        meta = {}
                    else:
                        try:
                            cfg, meta, _ = self.infra.engine.make_decision(
                                env, feat,
                                enable_uncertainty=unc,
                                enable_dpc=dpc
                            )
                        except Exception as e:
                            print(f"      失败: {e}")
                            cfg = BASELINE_STATIC_CONFIG['Edge_Ablation']
                            meta = {}
                    
                    t, ok, cpu = self.infra.pull_image(img, cfg)
                    self._log("Exp2", scene['name'], name, img, t, ok, cpu, cfg, meta)
        
        self._save()

    def exp3_robust(self):
        print("\n" + "="*70)
        print("🧪 实验3: 鲁棒性测试")
        print("="*70)
        
        if not self.infra.start_worker(4.0, "8g"):
            return
        
        img = "ubuntu:22.04"
        feat = self.infra.get_image_features(img)
        base_env = {'cpu_limit': 4.0, 'mem_limit_mb': 8192, 'network_rtt': 30}
        
        print("\n📍 动态波动场景")
        np.random.seed(42)
        for i in range(15):
            bw = np.random.uniform(2, 25)
            loss = np.random.uniform(0, 5)
            self.infra.set_network(bw, 30, loss)
            env = {**base_env, 'bandwidth_mbps': bw}
            
            t, ok, cpu = self.infra.pull_image(img, BASELINE_STATIC_CONFIG['Edge_Net'])
            self._log("Exp3", "Dynamic", "Static", img, t, ok, cpu, 
                     BASELINE_STATIC_CONFIG['Edge_Net'], {"bw": round(bw, 1)})
            
            cfg, meta, _ = self.infra.engine.make_decision(env, feat)
            t, ok, cpu = self.infra.pull_image(img, cfg)
            self._log("Exp3", "Dynamic", "CTS", img, t, ok, cpu, cfg, 
                     {**meta, "bw": round(bw, 1)})
        
        print("\n📍 OOD极端弱网")
        self.infra.set_network(0.5, 500, 15)
        env = {**base_env, 'bandwidth_mbps': 0.5, 'network_rtt': 500}
        
        for _ in range(5):
            t, ok, cpu = self.infra.pull_image(img, BASELINE_STATIC_CONFIG['IoT_Weak'])
            self._log("Exp3", "OOD", "Static", img, t, ok, cpu, 
                     BASELINE_STATIC_CONFIG['IoT_Weak'])
            
            cfg, meta, _ = self.infra.engine.make_decision(env, feat)
            t, ok, cpu = self.infra.pull_image(img, cfg)
            self._log("Exp3", "OOD", "CTS", img, t, ok, cpu, cfg, meta)
        
        self._save()

    def exp4_light(self):
        print("\n" + "="*70)
        print("🧪 实验4: 轻量化部署")
        print("="*70)
        
        if not self.infra.start_worker(2.0, "2g"):
            return
        self.infra.set_network(2, 100, 2)
        
        env = {'bandwidth_mbps': 2, 'cpu_limit': 2.0, 'network_rtt': 100, 'mem_limit_mb': 2048}
        feat = self.infra.get_image_features("alpine:3.19")
        
        print("   测试决策延迟...")
        latencies = []
        for _ in range(50):
            t0 = time.time()
            self.infra.engine.make_decision(env, feat)
            latencies.append((time.time() - t0) * 1000)
        
        stats = {
            'latency_mean_ms': round(np.mean(latencies), 2),
            'latency_p99_ms': round(np.percentile(latencies, 99), 2),
            'model_params': sum(p.numel() for p in self.infra.model.parameters()),
            'memory_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        }
        print(f"   📊 延迟: {stats['latency_mean_ms']}ms (P99: {stats['latency_p99_ms']}ms)")
        
        for _ in range(self.repeat):
            cfg, meta, _ = self.infra.engine.make_decision(env, feat)
            t, ok, cpu = self.infra.pull_image("alpine:3.19", cfg)
            self._log("Exp4", "IoT", "CTS", "alpine:3.19", t, ok, cpu, cfg, {**meta, **stats})
        
        self._save()

    def exp5_stability(self):
        print("\n" + "="*70)
        print("🧪 实验5: 长稳压力测试")
        print("="*70)
        
        if not self.infra.start_worker(8.0, "16g"):
            return
        self.infra.set_network(800, 1, 0)
        
        env = {'bandwidth_mbps': 800, 'cpu_limit': 8.0, 'network_rtt': 1, 'mem_limit_mb': 16384}
        feat = self.infra.get_image_features("nginx:1.25-alpine")
        
        times = []
        for i in range(50):
            print(f"\r   迭代 {i+1}/50", end="")
            cfg, meta, _ = self.infra.engine.make_decision(env, feat)
            t, ok, cpu = self.infra.pull_image("nginx:1.25-alpine", cfg)
            times.append(t)
            self._log("Exp5", "Stability", "CTS", "nginx:1.25-alpine", t, ok, cpu, cfg, {"iter": i+1})
            time.sleep(0.5)
        
        print(f"\n   稳定性: 首10次={np.mean(times[:10]):.2f}s, 末10次={np.mean(times[-10:]):.2f}s")
        self._save()

    def report(self, csv_path: str):
        """生成统计报告"""
        print("\n" + "="*70)
        print("📊 统计报告")
        print("="*70)
        
        df = pd.read_csv(csv_path)
        
        # 镜像使用情况
        custom_count = df[df['using_custom_image'] == True].shape[0]
        print(f"\n🔧 镜像使用情况: 自定义镜像={custom_count}次")
        
        # Exp1分析
        exp1 = df[df['experiment'] == 'Exp1']
        if len(exp1) > 0:
            print("\n🎯 端到端性能提升:")
            for scene in exp1['scene'].unique():
                s_df = exp1[exp1['scene'] == scene]
                base = s_df[s_df['method'] == 'Static']
                cts = s_df[s_df['method'] == 'CTS']
                
                base_ok = base[base['success'] == True]
                cts_ok = cts[cts['success'] == True]
                
                if len(base_ok) > 0 and len(cts_ok) > 0:
                    gain = (cts_ok['throughput_mbps'].mean() - base_ok['throughput_mbps'].mean()) / base_ok['throughput_mbps'].mean() * 100
                    _, p = stats.ttest_ind(base_ok['throughput_mbps'], cts_ok['throughput_mbps'])
                    sig = "✅" if p < 0.05 else "❌"
                    print(f"   [{scene:12s}] +{gain:6.2f}% | p={p:.4f} {sig}")
                else:
                    print(f"   [{scene:12s}] 数据不足 (base={len(base_ok)}, cts={len(cts_ok)})")
        
        print(f"\n🔧 skopeo支持情况:")
        print(f"   threads: {self.infra.skopeo_supports_threads}")
        print(f"   chunk:   {self.infra.skopeo_supports_chunk}")

# ==========================================
# 4. 主函数
# ==========================================
def main():
    print("🚀 CTS闭环实验平台 v3.0 (自定义镜像版)")
    print("="*70)
    
    # 检查自定义镜像是否存在
    check = subprocess.run(
        f"docker images {WORKER_IMAGE} -q",
        shell=True, capture_output=True, text=True
    )
    if check.returncode != 0 or not check.stdout.strip():
        print(f"⚠️  警告: 自定义镜像 {WORKER_IMAGE} 不存在")
        print(f"   请先构建镜像: docker build -t {WORKER_IMAGE} .")
        print(f"   将使用备用官方镜像（可能需要在线安装skopeo）")
    else:
        print(f"✅ 自定义镜像 {WORKER_IMAGE} 已就绪")
    
    # 检查sudo
    try:
        subprocess.run("sudo -v", shell=True, check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ sudo权限正常")
    except:
        print("❌ 需要sudo权限: sudo python3 run_master.py")
        sys.exit(1)
    
    # 初始化
    try:
        infra = LabInfrastructure()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    orch = ExperimentOrchestrator(infra)
    
    # 菜单
    menu = {
        "0": ("全部实验", [orch.exp1_end2end, orch.exp2_ablation, 
                         orch.exp3_robust, orch.exp4_light, orch.exp5_stability]),
        "1": ("实验1-端到端", [orch.exp1_end2end]),
        "2": ("实验2-消融", [orch.exp2_ablation]),
        "3": ("实验3-鲁棒性", [orch.exp3_robust]),
        "4": ("实验4-轻量化", [orch.exp4_light]),
        "5": ("实验5-长稳", [orch.exp5_stability]),
    }
    
    if sys.stdin.isatty():
        print("\n选择实验:")
        for k, (name, _) in menu.items():
            print(f"  {k}. {name}")
        choice = input("> ").strip() or "0"
    else:
        choice = "0"
        print("自动执行全部实验")
    
    # 执行
    try:
        for fn in menu.get(choice, menu["0"])[1]:
            fn()
        
        if orch.results:
            path = orch._save()
            if path:
                orch.report(path)
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        infra.cleanup()

if __name__ == "__main__":
    main()