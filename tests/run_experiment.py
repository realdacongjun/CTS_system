# import os
# import sys
# import time
# import json
# import pickle
# import subprocess
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# from typing import Dict, List, Tuple
# import psutil
# from scipy import stats

# # ==========================================
# # 0. 核心配置 (仅需修改这里的路径)
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "cts_optimized_0218_2125_seed42.pth")
# PREP_PATH = os.path.join(BASE_DIR, "models", "preprocessing_objects_optimized.pkl")
# RESULT_DIR = os.path.join(BASE_DIR, "results")
# os.makedirs(RESULT_DIR, exist_ok=True)

# # 【科研严谨性】实验镜像列表（固定大小，避免镜像本身成为干扰变量）
# TEST_IMAGES = [
#     "alpine:3.19",      # ~5MB  小镜像
#     "nginx:1.25-alpine", # ~10MB 中小镜像
#     "ubuntu:22.04"       # ~70MB 中大型镜像
# ]

# # 【科研严谨性】Static基线固定参数（完全对齐工业界默认配置）
# BASELINE_STATIC_CONFIG = {
#     "IoT_Weak": {"threads": 1, "cpu_quota": 0.2, "chunk_size_kb": 64, "algo_name": "gzip"},
#     "Edge_Net": {"threads": 4, "cpu_quota": 1.0, "chunk_size_kb": 256, "algo_name": "gzip"},
#     "Cloud_DC": {"threads": 16, "cpu_quota": 4.0, "chunk_size_kb": 1024, "algo_name": "gzip"},
#     "Edge_Ablation": {"threads": 4, "cpu_quota": 1.0, "chunk_size_kb": 256, "algo_name": "gzip"}
# }

# # ==========================================
# # 1. 导入你的CTS系统（无需修改）
# # ==========================================
# try:
#     from cts_model import CompactCFTNetV2, CONFIG
#     from cags_decision import CAGSDecisionEngine
# except ImportError as e:
#     print(f"❌ 无法导入模型模块: {e}")
#     print("   请确保 cts_model.py 和 cags_decision.py 在当前目录。")
#     sys.exit(1)

# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==========================================
# # 2. 【闭环升级】基础设施控制类（核心修改在这里）
# # ==========================================
# class LabInfrastructure:
#     def __init__(self):
#         self.net_name = "cts-net"
#         self.client_container = "cts-client-worker"
#         self._init_model()
#         self.current_worker_cpu_max = 8.0 # 容器最大CPU配额，用于动态调整

#     def _init_model(self):
#         """加载CTS模型和决策引擎（完全复用你的代码，无修改）"""
#         print("🧠 正在加载CTS模型与决策引擎...")
#         with open(PREP_PATH, 'rb') as f:
#             prep = pickle.load(f)
#         self.scaler_c = prep['scaler_c']
#         self.scaler_i = prep['scaler_i']
#         self.enc = prep['enc']
#         self.cols_c = prep['cols_c']
#         self.cols_i = prep['cols_i']

#         self.model = CompactCFTNetV2(len(self.cols_c), len(self.cols_i), len(self.enc.classes_), CONFIG["embed_dim"])
#         self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#         self.model = self.model.to(device)
#         self.model.eval()
        
#         self.engine = CAGSDecisionEngine(self.model, self.scaler_c, self.scaler_i, 
#                                           self.enc, self.cols_c, self.cols_i, device)
#         print(f"   ✅ 模型加载完成，参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

#     def _get_bridge_iface(self):
#         """获取Docker网络对应的宿主机网桥（无修改）"""
#         try:
#             res = subprocess.check_output(f"docker network inspect {self.net_name}", shell=True, text=True)
#             net_data = json.loads(res)
#             bridge_id = net_data[0]['Id'][:12]
#             return f"br-{bridge_id}"
#         except:
#             print("⚠️  未找到Docker网桥，网络配置将失效")
#             return None

#     def set_network(self, bw_mbps: float, delay_ms: float, loss_pct: float):
#         """配置网络瓶颈（无修改，仅优化了稳定性）"""
#         iface = self._get_bridge_iface()
#         if not iface:
#             return False
        
#         # 清除旧规则
#         subprocess.run(f"sudo tc qdisc del dev {iface} root", shell=True, stderr=subprocess.DEVNULL)
        
#         # 应用新规则（HTB+Netem，精准模拟链路）
#         cmd = (
#             f"sudo tc qdisc add dev {iface} root handle 1: htb default 11 && "
#             f"sudo tc class add dev {iface} parent 1: classid 1:1 htb rate {bw_mbps}mbit ceil {bw_mbps}mbit && "
#             f"sudo tc qdisc add dev {iface} parent 1:1 handle 10: netem delay {delay_ms}ms loss {loss_pct}% limit 1000"
#         )
#         try:
#             subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"   🌐 网络约束已生效: {bw_mbps}Mbps | {delay_ms}ms延迟 | {loss_pct}%丢包")
#             return True
#         except Exception as e:
#             print(f"   ⚠️  网络配置失败，请检查sudo权限: {e}")
#             return False

#     def start_worker(self, max_cpu_cores: float, mem_gb: str):
#         """
#         【闭环升级】启动资源受限的工作容器
#         核心改动：预留最大CPU配额，用于后续动态调整（对应CAGS的cpu_quota决策）
#         """
#         self.current_worker_cpu_max = max_cpu_cores
#         # 清理旧容器
#         subprocess.run(f"docker stop {self.client_container}", shell=True, stderr=subprocess.DEVNULL)
#         subprocess.run(f"docker rm {self.client_container}", shell=True, stderr=subprocess.DEVNULL)
        
#         # 启动DinD容器，预留最大CPU配额
#         cmd = (
#             f"docker run -d --name {self.client_container} "
#             f"--network {self.net_name} "
#             f"--cpus={max_cpu_cores} "
#             f"--memory={mem_gb} "
#             f"--privileged "
#             f"docker:25-dind"
#         )
#         subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
#         time.sleep(5) # 等待Docker-in-Docker启动
        
#         # 【闭环升级】自动安装skopeo，无需构建自定义镜像
#         subprocess.run(
#             f"docker exec {self.client_container} apk add --no-cache skopeo",
#             shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )
#         print(f"   🐳 工作容器启动完成: 最大CPU={max_cpu_cores}核 | 内存={mem_gb} | skopeo已安装")

#     def _set_worker_cpu_quota(self, cpu_quota: float):
#         """【闭环升级】动态调整容器CPU配额，对应CAGS的cpu_quota决策"""
#         # 限制不超过容器最大配额
#         cpu_quota = min(cpu_quota, self.current_worker_cpu_max)
#         subprocess.run(
#             f"docker update --cpus {cpu_quota} {self.client_container}",
#             shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )
#         time.sleep(0.5)

#     def pull_image_with_config(self, image_name: str, config: Dict) -> Tuple[float, bool, float]:
#         """
#         【闭环升级】核心执行函数：根据CAGS决策配置执行镜像拉取
#         完全打通「决策参数→系统执行」的因果链路
#         :param image_name: 镜像名
#         :param config: CAGS决策输出的配置字典，格式为：
#             {"threads": int, "cpu_quota": float, "chunk_size_kb": int, "algo_name": str}
#         :return: (耗时秒, 是否成功, 峰值CPU使用率)
#         """
#         # 1. 【因果绑定】动态调整CPU配额
#         self._set_worker_cpu_quota(config['cpu_quota'])
        
#         # 2. 【因果绑定】构造skopeo执行命令，完全映射决策参数
#         skopeo_cmd = (
#             f"skopeo copy "
#             f"--override-os linux "
#             f"--override-arch amd64 "
#             f"--max-concurrent-downloads {config['threads']} "
#             f"--chunk-size {config['chunk_size_kb']}KB "
#             f"--compression-format {config['algo_name']} "
#             f"docker://docker.io/{image_name} "
#             f"docker-daemon:{image_name}"
#         )
        
#         # 3. 清理本地镜像，确保冷启动拉取，避免缓存干扰
#         subprocess.run(
#             f"docker exec {self.client_container} docker rmi {image_name}",
#             shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )
#         time.sleep(1)
        
#         # 4. 执行拉取并监控资源
#         start_time = time.time()
#         success = False
#         peak_cpu = 0.0
        
#         # 异步执行拉取
#         proc = subprocess.Popen(
#             f"docker exec {self.client_container} {skopeo_cmd}",
#             shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
        
#         # 高频采样CPU使用率，确保数据准确
#         while proc.poll() is None:
#             try:
#                 stats = subprocess.check_output(
#                     f"docker stats {self.client_container} --no-stream --format json",
#                     shell=True, text=True
#                 )
#                 stats_json = json.loads(stats)
#                 cpu_str = stats_json['CPUPerc'].replace('%', '')
#                 if cpu_str:
#                     peak_cpu = max(peak_cpu, float(cpu_str))
#             except:
#                 pass
#             time.sleep(0.2)
        
#         # 5. 结果处理
#         elapsed = time.time() - start_time
#         stdout, stderr = proc.communicate()
#         success = (proc.returncode == 0)
        
#         if not success:
#             print(f"      ❌ 拉取失败，错误日志: {stderr[:200]}")
        
#         # 6. 重置CPU配额为最大值，避免影响后续实验
#         self._set_worker_cpu_quota(self.current_worker_cpu_max)
        
#         return elapsed, success, peak_cpu

#     def get_image_features(self, image_name: str) -> Dict:
#         """镜像特征映射（可替换为你的真实特征数据库）"""
#         size_map = {"alpine:3.19": 5, "nginx:1.25-alpine": 15, "ubuntu:22.04": 80}
#         size_mb = size_map.get(image_name, 50)
#         return {
#             'total_size_mb': size_mb,
#             'avg_layer_entropy': 0.65,
#             'entropy_std': 0.2,
#             'layer_count': 10,
#             'size_std_mb': 5,
#             'text_ratio': 0.3,
#             'zero_ratio': 0.05
#         }

#     def cleanup(self):
#         """环境清理"""
#         print("\n🧹 正在清理实验环境...")
#         subprocess.run(f"docker stop {self.client_container}", shell=True, stderr=subprocess.DEVNULL)
#         subprocess.run(f"docker rm {self.client_container}", shell=True, stderr=subprocess.DEVNULL)
#         # 重置网络
#         iface = self._get_bridge_iface()
#         if iface:
#             subprocess.run(f"sudo tc qdisc del dev {iface} root", shell=True, stderr=subprocess.DEVNULL)
#         print("   ✅ 环境清理完成")

# # ==========================================
# # 3. 【科研级】五大实验闭环实现
# # ==========================================
# class ExperimentOrchestrator:
#     def __init__(self, infra: LabInfrastructure):
#         self.infra = infra
#         self.all_results = []
#         self.exp_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.repeat_times = 5 # 【科研严谨性】每个实验重复5次，取95%置信区间

#     def _log_result(self, exp_name: str, scene: str, method: str, image: str, 
#                     time_s: float, success: bool, cpu_usage: float, config: Dict, extra: Dict = {}):
#         """【科研严谨性】全维度数据记录，包含执行配置，确保可复现"""
#         img_size = self.infra.get_image_features(image)['total_size_mb']
#         row = {
#             "timestamp": datetime.now().isoformat(),
#             "experiment": exp_name,
#             "scene": scene,
#             "method": method,
#             "image": image,
#             "image_size_mb": img_size,
#             "time_s": time_s,
#             "success": success,
#             "cpu_usage_pct": cpu_usage,
#             "throughput_mbps": (img_size * 8) / (time_s if time_s > 0 else 1e-8),
#             "cpu_efficiency": img_size / (cpu_usage * time_s / 100 + 1e-8), # MB/CPU·s
#             "exec_config": json.dumps(config), # 记录执行配置，确保可复现
#         }
#         row.update(extra)
#         self.all_results.append(row)
#         print(f"   📝 记录完成: {method} | {image} | 耗时{time_s:.2f}s | 吞吐量{row['throughput_mbps']:.2f}Mbps | {'✅' if success else '❌'}")

#     def _save_results(self):
#         """保存实验数据，自动备份"""
#         df = pd.DataFrame(self.all_results)
#         csv_path = os.path.join(RESULT_DIR, f"cts_closed_loop_exp_data_{self.exp_start_time}.csv")
#         df.to_csv(csv_path, index=False, encoding='utf-8-sig')
#         print(f"\n💾 全量实验数据已保存至: {csv_path}")
#         return csv_path

#     # ==========================================
#     # 实验一：端到端核心性能基准实验（闭环版）
#     # 【核心验证】完整CTS闭环相对工业基线的端到端性能增益
#     # ==========================================
#     def run_exp1_end2end(self):
#         print("\n" + "="*80)
#         print("🧪 实验一：端到端核心性能基准实验（闭环验证）")
#         print("="*80)
#         print("【科研严谨性】控制变量：仅执行配置不同，硬件/网络/镜像/软件栈完全固定")
        
#         # 三大场景定义，完全对齐论文
#         scenes = [
#             {"name": "IoT_Weak", "max_cpu": 2.0, "mem": "2g", "bw": 2, "delay": 100, "loss": 2},
#             {"name": "Edge_Net", "max_cpu": 4.0, "mem": "8g", "bw": 20, "delay": 30, "loss": 0.5},
#             {"name": "Cloud_DC",  "max_cpu": 8.0, "mem": "16g", "bw": 800, "delay": 1, "loss": 0},
#         ]
        
#         for scene in scenes:
#             scene_name = scene['name']
#             print(f"\n📍 开始场景：{scene_name}")
            
#             # 1. 固定环境配置（控制变量）
#             self.infra.start_worker(scene['max_cpu'], scene['mem'])
#             self.infra.set_network(scene['bw'], scene['delay'], scene['loss'])
            
#             # 2. 构造环境状态，输入CTS决策引擎
#             env_state = {
#                 'bandwidth_mbps': scene['bw'], 
#                 'cpu_limit': scene['max_cpu'], 
#                 'network_rtt': scene['delay'], 
#                 'mem_limit_mb': int(scene['mem'].replace('g', '')) * 1024
#             }
            
#             # 3. 重复实验，确保统计显著性
#             for repeat in range(self.repeat_times):
#                 print(f"\n   --- 第{repeat+1}/{self.repeat_times}次重复实验 ---")
#                 for img in TEST_IMAGES:
#                     img_feat = self.infra.get_image_features(img)
                    
#                     # --- 对照组：工业界静态保守配置 ---
#                     print(f"\n   🔹 对照组：Static基线")
#                     static_config = BASELINE_STATIC_CONFIG[scene_name]
#                     time_s, success, cpu = self.infra.pull_image_with_config(img, static_config)
#                     self._log_result("Exp1", scene_name, "Static_Baseline", img, time_s, success, cpu, static_config)
                    
#                     # --- 实验组：CTS完整闭环系统 ---
#                     print(f"\n   🔸 实验组：CTS完整闭环")
#                     try:
#                         cts_config, cts_metrics, _ = self.infra.engine.make_decision(env_state, img_feat)
#                         print(f"      CTS决策结果: {cts_config}")
#                     except Exception as e:
#                         print(f"      ⚠️  决策失败，回退到基线: {e}")
#                         cts_config = static_config
                    
#                     # 【因果绑定】用CTS决策的配置执行拉取
#                     time_s, success, cpu = self.infra.pull_image_with_config(img, cts_config)
#                     self._log_result("Exp1", scene_name, "CTS_Full_Closed_Loop", img, time_s, success, cpu, cts_config, cts_metrics)
                    
#                     time.sleep(2) # 避免网络拥塞
        
#         # 保存数据
#         self._save_results()
#         print(f"\n✅ 实验一完成，已验证CTS闭环的端到端性能增益")

#     # ==========================================
#     # 实验二：双创新点协同增益消融实验（闭环版）
#     # 【核心验证】两大创新点的1+1>2协同效应，精准量化每个组件的贡献
#     # ==========================================
#     def run_exp2_ablation(self):
#         print("\n" + "="*80)
#         print("🧪 实验二：双创新点协同增益消融实验（闭环验证）")
#         print("="*80)
#         print("【科研严谨性】递进式消融，仅关闭特定组件，其余所有变量完全固定")
        
#         # 固定边缘场景（最能体现多目标优化的协同效应）
#         scene_name = "Edge_Ablation"
#         scene_config = {"max_cpu": 4.0, "mem": "8g", "bw": 20, "delay": 30, "loss": 0.5}
#         env_state = {
#             'bandwidth_mbps': scene_config['bw'], 
#             'cpu_limit': scene_config['max_cpu'], 
#             'network_rtt': scene_config['delay'], 
#             'mem_limit_mb': int(scene_config['mem'].replace('g', '')) * 1024
#         }
        
#         # 启动环境
#         self.infra.start_worker(scene_config['max_cpu'], scene_config['mem'])
#         self.infra.set_network(scene_config['bw'], scene_config['delay'], scene_config['loss'])
        
#         # 【核心】6组递进式消融变体，完全对齐论文
#         ablation_variants = [
#             {
#                 "name": "Variant1_Static_Baseline",
#                 "desc": "极简基线：静态固定配置",
#                 "enable_cft": False,
#                 "enable_uncertainty": False,
#                 "enable_dpc": False,
#                 "enable_cags": False
#             },
#             {
#                 "name": "Variant2_CFT_Point_NSGA3",
#                 "desc": "CFT点估计+传统NSGA-III决策",
#                 "enable_cft": True,
#                 "enable_uncertainty": False,
#                 "enable_dpc": False,
#                 "enable_cags": False
#             },
#             {
#                 "name": "Variant3_CFT_Full_NSGA3",
#                 "desc": "CFT全功能+传统NSGA-III决策",
#                 "enable_cft": True,
#                 "enable_uncertainty": True,
#                 "enable_dpc": False,
#                 "enable_cags": False
#             },
#             {
#                 "name": "Variant4_CFT_Point_CAGS",
#                 "desc": "CFT点估计+完整CAGS决策",
#                 "enable_cft": True,
#                 "enable_uncertainty": False,
#                 "enable_dpc": True,
#                 "enable_cags": True
#             },
#             {
#                 "name": "Variant5_CFT_Full_CAGS_NoDPC",
#                 "desc": "CFT全功能+CAGS无DPC约束",
#                 "enable_cft": True,
#                 "enable_uncertainty": True,
#                 "enable_dpc": False,
#                 "enable_cags": True
#             },
#             {
#                 "name": "Variant6_CTS_Full_Closed_Loop",
#                 "desc": "完整CTS闭环（CFT全功能+DPC+CAGS）",
#                 "enable_cft": True,
#                 "enable_uncertainty": True,
#                 "enable_dpc": True,
#                 "enable_cags": True
#             }
#         ]
        
#         # 执行消融实验
#         for variant in ablation_variants:
#             print(f"\n📍 开始消融变体：{variant['name']} | {variant['desc']}")
#             for repeat in range(self.repeat_times):
#                 for img in TEST_IMAGES:
#                     img_feat = self.infra.get_image_features(img)
                    
#                     # 根据变体开关生成配置
#                     if not variant['enable_cft']:
#                         # 静态基线
#                         config = BASELINE_STATIC_CONFIG[scene_name]
#                     else:
#                         # 调用决策引擎，根据变体开关控制功能
#                         try:
#                             config, metrics, _ = self.infra.engine.make_decision(
#                                 env_state, img_feat,
#                                 enable_uncertainty=variant['enable_uncertainty'],
#                                 enable_dpc=variant['enable_dpc']
#                             )
#                         except Exception as e:
#                             print(f"      ⚠️  决策失败，回退到基线: {e}")
#                             config = BASELINE_STATIC_CONFIG[scene_name]
                    
#                     # 执行拉取
#                     time_s, success, cpu = self.infra.pull_image_with_config(img, config)
#                     self._log_result("Exp2", scene_name, variant['name'], img, time_s, success, cpu, config, {"variant_desc": variant['desc']})
        
#         self._save_results()
#         print(f"\n✅ 实验二完成，已量化两大创新点的协同增益")

#     # ==========================================
#     # 实验三：动态环境鲁棒性与风险防控实验（闭环版）
#     # 【核心验证】CTS闭环在动态波动/极端OOD场景下的鲁棒性与风险防控能力
#     # ==========================================
#     def run_exp3_robustness(self):
#         print("\n" + "="*80)
#         print("🧪 实验三：动态环境鲁棒性与风险防控实验（闭环验证）")
#         print("="*80)
        
#         # 固定硬件环境
#         self.infra.start_worker(4.0, "8g")
#         test_img = "ubuntu:22.04"
#         img_feat = self.infra.get_image_features(test_img)
#         base_env = {"bandwidth_mbps": 20, "cpu_limit": 4.0, "network_rtt": 30, "mem_limit_mb": 8192}
        
#         # 场景1：网络动态波动场景
#         print(f"\n📍 场景1：网络动态波动场景")
#         np.random.seed(42)
#         for i in range(20):
#             # 随机波动带宽和丢包率
#             bw = np.random.uniform(1, 20)
#             loss = np.random.uniform(0, 8)
#             self.infra.set_network(bw, 30, loss)
#             env_state = {**base_env, "bandwidth_mbps": bw, "network_rtt": 30}
            
#             # 静态基线
#             static_config = BASELINE_STATIC_CONFIG['Edge_Net']
#             time_s, success, cpu = self.infra.pull_image_with_config(test_img, static_config)
#             self._log_result("Exp3", "Dynamic_Fluctuation", "Static_Baseline", test_img, time_s, success, cpu, static_config, {"bandwidth": bw, "loss_pct": loss})
            
#             # CTS闭环
#             cts_config, cts_metrics, _ = self.infra.engine.make_decision(env_state, img_feat)
#             time_s, success, cpu = self.infra.pull_image_with_config(test_img, cts_config)
#             self._log_result("Exp3", "Dynamic_Fluctuation", "CTS_Full_Closed_Loop", test_img, time_s, success, cpu, cts_config, {**cts_metrics, "bandwidth": bw, "loss_pct": loss})
        
#         # 场景2：OOD极端弱网场景
#         print(f"\n📍 场景2：OOD极端弱网场景")
#         ood_env = {"bandwidth_mbps": 0.8, "delay": 400, "loss": 12}
#         self.infra.set_network(**ood_env)
#         env_state = {**base_env, "bandwidth_mbps": ood_env['bandwidth_mbps'], "network_rtt": ood_env['delay']}
        
#         for repeat in range(10):
#             # 静态基线
#             static_config = BASELINE_STATIC_CONFIG['IoT_Weak']
#             time_s, success, cpu = self.infra.pull_image_with_config(test_img, static_config)
#             self._log_result("Exp3", "OOD_Extreme_Weaknet", "Static_Baseline", test_img, time_s, success, cpu, static_config)
            
#             # CTS闭环
#             cts_config, cts_metrics, _ = self.infra.engine.make_decision(env_state, img_feat)
#             time_s, success, cpu = self.infra.pull_image_with_config(test_img, cts_config)
#             self._log_result("Exp3", "OOD_Extreme_Weaknet", "CTS_Full_Closed_Loop", test_img, time_s, success, cpu, cts_config, cts_metrics)
        
#         self._save_results()
#         print(f"\n✅ 实验三完成，已验证CTS闭环的鲁棒性与风险防控能力")

#     # ==========================================
#     # 实验四：端侧轻量化部署验证实验（闭环版）
#     # 【核心验证】CTS闭环在低算力端侧设备的部署可行性
#     # ==========================================
#     def run_exp4_lightweight(self):
#         print("\n" + "="*80)
#         print("🧪 实验四：端侧轻量化部署验证实验")
#         print("="*80)
        
#         # 模拟IoT端侧2核2G环境
#         self.infra.start_worker(2.0, "2g")
#         self.infra.set_network(2, 100, 2)
#         env_state = {"bandwidth_mbps": 2, "cpu_limit": 2.0, "network_rtt": 100, "mem_limit_mb": 2048}
#         img_feat = self.infra.get_image_features("alpine:3.19")
        
#         # 1. 模型参数量统计
#         num_params = sum(p.numel() for p in self.infra.model.parameters() if p.requires_grad)
#         model_size_mb = num_params * 4 / 1024 / 1024 # FP32模型大小
        
#         # 2. 端侧推理+决策延迟测试（100次采样，统计P99延迟）
#         latencies = []
#         for _ in range(100):
#             start = time.time()
#             self.infra.engine.make_decision(env_state, img_feat)
#             latencies.append((time.time() - start) * 1000) # 转ms
        
#         latency_mean = np.mean(latencies)
#         latency_p99 = np.percentile(latencies, 99)
        
#         # 3. 端侧内存占用测试
#         memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 # MB
        
#         # 4. 端侧实际执行效果验证
#         for repeat in range(self.repeat_times):
#             cts_config, cts_metrics, _ = self.infra.engine.make_decision(env_state, img_feat)
#             time_s, success, cpu = self.infra.pull_image_with_config("alpine:3.19", cts_config)
#             self._log_result("Exp4", "IoT_End_Device", "CTS_Lightweight", "alpine:3.19", time_s, success, cpu, cts_config, {
#                 "model_params": num_params,
#                 "model_size_mb": model_size_mb,
#                 "latency_mean_ms": latency_mean,
#                 "latency_p99_ms": latency_p99,
#                 "memory_usage_mb": memory_usage
#             })
        
#         # 打印结果
#         print(f"\n📊 轻量化验证结果:")
#         print(f"   模型参数量: {num_params:,}")
#         print(f"   模型大小: {model_size_mb:.2f}MB")
#         print(f"   平均推理+决策延迟: {latency_mean:.2f}ms")
#         print(f"   P99延迟: {latency_p99:.2f}ms")
#         print(f"   内存占用: {memory_usage:.2f}MB")
        
#         self._save_results()
#         print(f"\n✅ 实验四完成，已验证CTS闭环的端侧轻量化部署能力")

#     # ==========================================
#     # 实验五：长稳压力测试实验（闭环版）
#     # 【核心验证】CTS闭环在长时间高并发下的稳定性与一致性
#     # ==========================================
#     def run_exp5_stability(self):
#         print("\n" + "="*80)
#         print("🧪 实验五：长稳压力测试实验")
#         print("="*80)
        
#         # 云数据中心场景，8核16G
#         self.infra.start_worker(8.0, "16g")
#         self.infra.set_network(800, 1, 0)
#         test_img = "nginx:1.25-alpine"
#         env_state = {"bandwidth_mbps": 800, "cpu_limit": 8.0, "network_rtt": 1, "mem_limit_mb": 16384}
#         img_feat = self.infra.get_image_features(test_img)
        
#         # 连续100次循环拉取，模拟生产环境高频率调用
#         total_iterations = 100
#         success_count = 0
#         performance_decay = []
        
#         for i in range(total_iterations):
#             print(f"\n📍 迭代 {i+1}/{total_iterations}")
#             cts_config, cts_metrics, _ = self.infra.engine.make_decision(env_state, img_feat)
#             time_s, success, cpu = self.infra.pull_image_with_config(test_img, cts_config)
            
#             if success:
#                 success_count += 1
#             performance_decay.append(time_s)
            
#             self._log_result("Exp5", "Long_Stability_Test", "CTS_Full_Closed_Loop", test_img, time_s, success, cpu, cts_config, {
#                 "iteration": i+1,
#                 "total_iterations": total_iterations
#             })
            
#             time.sleep(1)
        
#         # 统计结果
#         success_rate = success_count / total_iterations * 100
#         first_10_avg = np.mean(performance_decay[:10])
#         last_10_avg = np.mean(performance_decay[-10:])
#         decay_rate = (last_10_avg - first_10_avg) / first_10_avg * 100
        
#         print(f"\n📊 长稳测试结果:")
#         print(f"   累计迭代次数: {total_iterations}")
#         print(f"   累计成功率: {success_rate:.2f}%")
#         print(f"   首尾性能衰减率: {decay_rate:.2f}%")
        
#         self._save_results()
#         print(f"\n✅ 实验五完成，已验证CTS闭环的长稳运行能力")

#     # ==========================================
#     # 科研级结果汇总与统计检验
#     # ==========================================
#     def generate_statistical_report(self, csv_path):
#         print("\n" + "="*80)
#         print("📊 生成科研级统计分析报告")
#         print("="*80)
        
#         df = pd.read_csv(csv_path)
#         report = {}
        
#         # 实验一核心增益统计
#         if 'Exp1' in df['experiment'].values:
#             exp1_df = df[df['experiment'] == 'Exp1']
#             report['exp1'] = {}
#             for scene in exp1_df['scene'].unique():
#                 scene_df = exp1_df[exp1_df['scene'] == scene]
#                 static_df = scene_df[scene_df['method'] == 'Static_Baseline']
#                 cts_df = scene_df[scene_df['method'] == 'CTS_Full_Closed_Loop']
                
#                 # 核心指标
#                 tp_gain = (cts_df['throughput_mbps'].mean() - static_df['throughput_mbps'].mean()) / static_df['throughput_mbps'].mean() * 100
#                 cpu_eff_gain = (cts_df['cpu_efficiency'].mean() - static_df['cpu_efficiency'].mean()) / static_df['cpu_efficiency'].mean() * 100
#                 failure_rate_drop = static_df['success'].mean() - cts_df['success'].mean()
                
#                 # 统计显著性检验（t检验）
#                 t_stat, p_value = stats.ttest_ind(static_df['throughput_mbps'], cts_df['throughput_mbps'])
                
#                 report['exp1'][scene] = {
#                     "throughput_gain_pct": round(tp_gain, 2),
#                     "cpu_efficiency_gain_pct": round(cpu_eff_gain, 2),
#                     "failure_rate_drop_pct": round(failure_rate_drop * 100, 2),
#                     "t_test_p_value": round(p_value, 4),
#                     "is_significant": p_value < 0.05
#                 }
        
#         # 保存报告
#         report_path = os.path.join(RESULT_DIR, f"statistical_report_{self.exp_start_time}.json")
#         with open(report_path, 'w', encoding='utf-8') as f:
#             json.dump(report, f, indent=2, ensure_ascii=False)
        
#         # 打印核心结论
#         print("\n🎯 核心实验结论:")
#         for scene, res in report.get('exp1', {}).items():
#             print(f"\n[{scene}]")
#             print(f"   吞吐量提升: {res['throughput_gain_pct']}%")
#             print(f"   CPU效率提升: {res['cpu_efficiency_gain_pct']}%")
#             print(f"   失败率降低: {res['failure_rate_drop_pct']}%")
#             print(f"   统计显著性: {'✅ p<0.05' if res['is_significant'] else '❌ 不显著'}")
        
#         print(f"\n📄 完整统计报告已保存至: {report_path}")
#         return report

# # ==========================================
# # 4. 主入口
# # ==========================================
# def main():
#     print("🚀 CTS闭环系统五大实验科研级编排平台启动")
#     print("【核心特性】决策-执行完全绑定 | 严格控制变量 | 统计显著性检验 | 全量数据可复现")
    
#     # 检查sudo权限（必须，用于tc网络配置）
#     try:
#         subprocess.run("sudo -v", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print("✅ 获取sudo权限成功，网络配置功能正常")
#     except:
#         print("❌ 无法获取sudo权限，脚本将退出")
#         print("   请使用 sudo python3 run_master.py 运行脚本")
#         sys.exit(1)

#     # 初始化基础设施
#     infra = LabInfrastructure()
#     orch = ExperimentOrchestrator(infra)
    
#     # ==========================================
#     # 【升级】全功能菜单逻辑
#     # ==========================================
#     import sys
    
#     # 检测是否为交互式终端
#     is_interactive = sys.stdin.isatty()
    
#     if is_interactive:
#         # 交互式模式：显示详细菜单
#         print("\n" + "="*60)
#         print("请选择要运行的实验:")
#         print("  0. 运行全部五大实验 (预计4-6小时)")
#         print("-" * 40)
#         print("  1. 仅运行 [实验一]: 端到端核心性能基准")
#         print("  2. 仅运行 [实验二]: 双创新点协同增益消融")
#         print("  3. 仅运行 [实验三]: 动态环境鲁棒性与风险防控")
#         print("  4. 仅运行 [实验四]: 端侧轻量化部署验证")
#         print("  5. 仅运行 [实验五]: 长稳压力测试")
#         print("-" * 40)
#         print("  组合模式:")
#         print("  12. 实验一 + 实验二 (核心创新验证)")
#         print("  45. 实验四 + 实验五 (部署与稳定性)")
#         print("="*60)
        
#         choice = input("\n请输入选项编号 [1]: ").strip() or "1"
#     else:
#         # 非交互式模式 (后台运行)
#         print("\n🔧 检测到非交互式环境 (后台运行)")
#         # ==========================================
#         # 【修改区】在这里设置后台默认运行的实验:
#         # 可选值: "0", "1", "2", "3", "4", "5", "12", "45"
#         # ==========================================
#         default_choice = "0" 
#         print(f"   自动执行选项: {default_choice}")
#         choice = default_choice
    
#     # ==========================================
#     # 执行路由
#     # ==========================================
#     try:
#         if choice == "0":
#             print("\n🏃 开始运行全部五大实验...")
#             orch.run_exp1_end2end()
#             orch.run_exp2_ablation()
#             orch.run_exp3_robustness()
#             orch.run_exp4_lightweight()
#             orch.run_exp5_stability()
            
#         elif choice == "1":
#             print("\n🏃 开始运行 [实验一]: 端到端核心性能基准...")
#             orch.run_exp1_end2end()
            
#         elif choice == "2":
#             print("\n🏃 开始运行 [实验二]: 双创新点协同增益消融...")
#             orch.run_exp2_ablation()
            
#         elif choice == "3":
#             print("\n🏃 开始运行 [实验三]: 动态环境鲁棒性与风险防控...")
#             orch.run_exp3_robustness()
            
#         elif choice == "4":
#             print("\n🏃 开始运行 [实验四]: 端侧轻量化部署验证...")
#             orch.run_exp4_lightweight()
            
#         elif choice == "5":
#             print("\n🏃 开始运行 [实验五]: 长稳压力测试...")
#             orch.run_exp5_stability()
            
#         elif choice == "12":
#             print("\n🏃 开始运行 [实验一 + 实验二]...")
#             orch.run_exp1_end2end()
#             orch.run_exp2_ablation()
            
#         elif choice == "45":
#             print("\n🏃 开始运行 [实验四 + 实验五]...")
#             orch.run_exp4_lightweight()
#             orch.run_exp5_stability()
            
#         else:
#             print(f"❌ 未知选项: {choice}，默认运行实验一")
#             orch.run_exp1_end2end()
            
#         # 生成统计报告
#         csv_file = orch._save_results()
#         orch.generate_statistical_report(csv_file)
        
#     except KeyboardInterrupt:
#         print("\n⏹️ 用户中断，已保存当前所有实验数据")
#         orch._save_results()
#     except Exception as e:
#         print(f"\n❌ 实验执行出错: {e}")
#         import traceback
#         traceback.print_exc()
#         orch._save_results()
#     finally:
#         infra.cleanup()
#         print("\n🏁 实验流程结束")

# if __name__ == "__main__":
#     main()
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
# 2. 基础设施（完整修复版）
# ==========================================
class LabInfrastructure:
    def __init__(self):
        self.net_name = "cts-net"
        self.client_container = "cts-client-worker"
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
        """启动工作容器并安装skopeo"""
        self.current_worker_cpu_max = max_cpu
        
        subprocess.run(
            f"docker rm -f {self.client_container} 2>/dev/null",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        cmd = (
            f"docker run -d --name {self.client_container} "
            f"--network {self.net_name} --cpus={max_cpu} --memory={mem} "
            f"--privileged docker:25-dind"
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            print(f"   🐳 容器启动中...", end="", flush=True)
            
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
            
            self._install_skopeo()
            return True
            
        except Exception as e:
            print(f"\n   ❌ 容器启动失败: {e}")
            return False

    def _install_skopeo(self):
        """【修复】安装skopeo并严格检测版本能力"""
        print(f"   📦 安装skopeo...", end="", flush=True)
        
        # 尝试edge仓库安装新版
        install_cmd = (
            f"docker exec {self.client_container} sh -c '"
            f"echo \"https://dl-cdn.alpinelinux.org/alpine/edge/community\" >> /etc/apk/repositories && "
            f"apk update -q 2>/dev/null && "
            f"apk add --no-cache skopeo -q'"
        )
        
        result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        install_ok = (result.returncode == 0)
        if not install_ok:
            print(f"\n   ⚠️  edge源失败，尝试默认源...", end="", flush=True)
            fallback = f"docker exec {self.client_container} apk add --no-cache skopeo -q"
            result2 = subprocess.run(fallback, shell=True, capture_output=True, text=True, timeout=60)
            install_ok = (result2.returncode == 0)
        
        if not install_ok:
            print(f"\n   ❌ skopeo安装失败")
            self.skopeo_supports_threads = False
            self.skopeo_supports_chunk = False
            return
        
        # 【关键修复】严格检测 - 实际运行help命令
        print(f" 检测中...", end="", flush=True)
        help_result = subprocess.run(
            f"docker exec {self.client_container} skopeo copy --help 2>&1",
            shell=True, capture_output=True, text=True
        )
        
        help_text = help_result.stdout + help_result.stderr
        
        # 严格检查标志
        self.skopeo_supports_threads = "--max-concurrent-downloads" in help_text
        self.skopeo_supports_chunk = "--chunk-size" in help_text
        
        print(f" threads={self.skopeo_supports_threads}, chunk={self.skopeo_supports_chunk}")

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
        """
        【完整修复】执行镜像拉取 - 严格根据检测到的能力构建命令
        """
        self._set_cpu_quota(config['cpu_quota'])
        
        # 【关键修复】严格根据检测到的能力构建命令
        flags = ["--override-os linux", "--override-arch amd64"]
        
        # 调试信息
        print(f"      🔧 skopeo: threads={self.skopeo_supports_threads}, chunk={self.skopeo_supports_chunk}")
        
        if self.skopeo_supports_threads:
            flags.append(f"--max-concurrent-downloads {int(config['threads'])}")
        else:
            print(f"      ⚠️  跳过 --max-concurrent-downloads (不支持)")
        
        if self.skopeo_supports_chunk:
            flags.append(f"--chunk-size {int(config['chunk_size_kb'])}KB")
        else:
            print(f"      ⚠️  跳过 --chunk-size (不支持)")
        
        flags.append(f"--compression-format {config['algo_name']}")
        
        skopeo_cmd = f"skopeo copy {' '.join(flags)} docker://docker.io/{image} docker-daemon:{image}"
        
        # 清理镜像
        subprocess.run(
            f"docker exec {self.client_container} docker rmi {image} 2>/dev/null",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
        
        # 【修复】正确计时
        start = time.time()
        peak_cpu = 0.0
        
        proc = subprocess.Popen(
            f"docker exec {self.client_container} {skopeo_cmd}",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # 监控CPU
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
            
            # 超时检查
            if time.time() - start > timeout:
                proc.kill()
                print(f"      ⏱️  超时({timeout}s)")
                break
        
        # 等待进程结束
        stdout, stderr = proc.communicate()
        elapsed = time.time() - start
        success = (proc.returncode == 0)
        
        # 处理错误
        err_msg = ""
        if stderr:
            err_msg = stderr.decode() if isinstance(stderr, bytes) else str(stderr)
            if not success:
                err_short = err_msg[:200].replace('\n', ' ')
                print(f"      ❌ {err_short}")
        
        self._set_cpu_quota(self.current_worker_cpu_max)
        
        # 【修复】如果失败且时间过短，可能是立即失败，标记为0
        if not success and elapsed < 2.0:
            print(f"      ⚠️  快速失败 (可能命令错误)，耗时={elapsed:.2f}s")
        
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
        
        # 【修复】如果失败，吞吐量标记为0避免虚假高速率
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
        
        exp1 = df[df['experiment'] == 'Exp1']
        if len(exp1) > 0:
            print("\n🎯 端到端性能提升:")
            for scene in exp1['scene'].unique():
                s_df = exp1[exp1['scene'] == scene]
                base = s_df[s_df['method'] == 'Static']
                cts = s_df[s_df['method'] == 'CTS']
                
                # 只统计成功的
                base_ok = base[base['success'] == True]
                cts_ok = cts[cts['success'] == True]
                
                if len(base_ok) > 0 and len(cts_ok) > 0:
                    gain = (cts_ok['throughput_mbps'].mean() - base_ok['throughput_mbps'].mean()) / base_ok['throughput_mbps'].mean() * 100
                    _, p = stats.ttest_ind(base_ok['throughput_mbps'], cts_ok['throughput_mbps'])
                    sig = "✅" if p < 0.05 else "❌"
                    print(f"   [{scene:12s}] +{gain:6.2f}% | p={p:.4f} {sig}")
                else:
                    print(f"   [{scene:12s}] 数据不足 (base={len(base_ok)}, cts={len(cts_ok)})")
        
        print("\n🔧 skopeo支持情况:")
        print(f"   threads: {self.infra.skopeo_supports_threads}")
        print(f"   chunk:   {self.infra.skopeo_supports_chunk}")

# ==========================================
# 4. 主函数
# ==========================================
def main():
    print("🚀 CTS闭环实验平台 v2.1 (网络+skopeo修复版)")
    
    try:
        subprocess.run("sudo -v", shell=True, check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ sudo权限正常")
    except:
        print("❌ 需要sudo权限: sudo python3 run_master.py")
        sys.exit(1)
    
    try:
        infra = LabInfrastructure()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    orch = ExperimentOrchestrator(infra)
    
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