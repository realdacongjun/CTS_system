import math
import numpy as np

# ==============================================================================
# 战略层：修复数学计算逻辑
# ==============================================================================
class CAGSStrategyLayer:
    def __init__(self, alpha=1.0, beta=0.5, gamma=2.0, uncertainty_weight=5.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight
        
        # 调整分片大小范围以适应真实环境
        self.chunk_sizes = [256*1024, 512*1024, 1024*1024, 2*1024*1024, 4*1024*1024]
        self.concurrencies = [1, 2, 4, 8]
        
        # 压缩算法列表
        self.compression_algorithms = [
            'gzip-1', 'gzip-6', 'gzip-9',
            'zstd-1', 'zstd-3', 'zstd-6', 'zstd-19',
            'lz4-fast', 'lz4-medium', 'lz4-slow',
            'brotli-1', 'brotli-6', 'brotli-11'
        ]

    def optimize(self, predicted_bw_mbps, predicted_loss_rate, client_cpu_load, model_uncertainty=0.0):
        best_cost = float('inf')
        best_config = (1024*1024, 1)

        bw_bps = predicted_bw_mbps * 1024 * 1024 / 8.0
        MTU = 1460 
        risk_amplifier = 1.0 + (self.uncertainty_weight * model_uncertainty)

        for s in self.chunk_sizes:
            for n in self.concurrencies:
                # 1. 传输时间 (考虑并发收益递减)
                effective_bw = bw_bps * (n ** 0.9)
                t_trans = s / effective_bw

                # 2. [修复点 1] 优化 CPU 负载计算公式
                # 原问题：(1024*1024/s) 过于激进。
                # 新逻辑：base_load (上下文切换) + io_overhead (系统调用开销)
                # s 越小，单位数据量的系统调用次数越多，但我们给它一个更平滑的系数。
                syscall_overhead = 0.005 * (1024*1024 / s) # 降低系数
                thread_overhead = 0.02 * n
                task_load = thread_overhead + syscall_overhead
                
                # 限制最大负载不能超过 1.0
                current_total_load = min(0.99, client_cpu_load + task_load)
                
                # 指数势垒 (保持不变，这是核心约束)
                c_cpu = math.exp(4 * current_total_load) 

                # 3. 风险成本
                num_packets = s / MTU
                prob_fail = 1 - (1 - predicted_loss_rate) ** num_packets
                r_risk = (prob_fail * t_trans * 10) * risk_amplifier

                cost = self.alpha * t_trans + self.beta * c_cpu + self.gamma * r_risk

                if cost < best_cost:
                    best_cost = cost
                    best_config = (s, n)

        return best_config, best_cost
    
    def predict_compression_times(self, client_profile, image_profile):
        """
        预测不同压缩算法的总时间并排序
        """
        method_times = {}
        
        # 遍历所有压缩算法
        for method in self.compression_algorithms:
            total_time = self._calculate_compression_time(client_profile, image_profile, method)
            method_times[method] = total_time
        
        # 按时间升序排序
        sorted_methods = sorted(method_times.items(), key=lambda x: x[1])
        
        return sorted_methods
    
    def _calculate_compression_time(self, client_profile, image_profile, method):
        """
        计算特定压缩算法的总时间
        总时间 = 压缩时间 + 传输时间 + 解压时间
        """
        # 解析算法和等级
        parts = method.split('-')
        algo = parts[0]
        level_str = '-'.join(parts[1:])
        level = int(level_str) if level_str.isdigit() else 1
        
        # 从配置文件中获取参数
        bandwidth_mbps = client_profile.get('bandwidth_mbps', 10.0)
        cpu_score = client_profile.get('cpu_score', 2000)
        decompression_speed = client_profile.get('decompression_speed', 100)
        
        image_size_mb = image_profile.get('total_size_mb', 100.0)
        entropy = image_profile.get('avg_layer_entropy', 0.5)
        
        # 计算压缩时间（根据算法和等级）
        compression_time = self._get_compression_time(algo, level, image_size_mb, cpu_score, entropy)
        
        # 计算压缩比，得到压缩后大小
        compression_ratio = self._get_compression_ratio(algo, level, entropy)
        compressed_size_mb = image_size_mb * compression_ratio
        
        # 计算传输时间
        transmission_time = (compressed_size_mb * 8.0) / bandwidth_mbps
        
        # 计算解压时间
        decomp_time = (compressed_size_mb * 8.0) / decompression_speed
        
        # 总时间
        total_time = compression_time + transmission_time + decomp_time
        
        return total_time
    
    def _get_compression_time(self, algo, level, size_mb, cpu_score, entropy):
        """
        根据算法、等级、文件大小、CPU性能和熵值估算压缩时间
        """
        # 不同算法的基础压缩时间因子
        algo_factors = {
            'gzip': 0.05,   # gzip相对较慢
            'zstd': 0.02,   # zstd速度快
            'lz4': 0.01,    # lz4最快
            'brotli': 0.08  # brotli较慢
        }
        
        factor = algo_factors.get(algo, 0.05)
        
        # 等级对时间的影响（通常等级越高，时间越长）
        level_multiplier = 1.0
        if algo == 'gzip':
            level_multiplier = 0.5 + (level * 0.05)  # gzip 1-9
        elif algo == 'zstd':
            level_multiplier = 0.3 + (level * 0.03)  # zstd 1-19
        elif algo == 'brotli':
            level_multiplier = 0.2 + (level * 0.06)  # brotli 1-11
        elif algo == 'lz4':
            level_multiplier = 0.8 + (level * 0.05)  # lz4不同等级
        
        # CPU性能调整
        cpu_factor = 1000.0 / cpu_score
        
        # 熵值影响（熵值越高，压缩越困难）
        entropy_factor = 1.0 + (1.0 - entropy) * 0.5
        
        return size_mb * factor * level_multiplier * cpu_factor * entropy_factor
    
    def _get_compression_ratio(self, algo, level, entropy):
        """
        根据算法、等级和熵值估算压缩比
        """
        # 基础压缩比（熵值为0.5时）
        base_ratios = {
            'gzip': 0.7,
            'zstd': 0.65,
            'lz4': 0.85,
            'brotli': 0.6
        }
        
        base_ratio = base_ratios.get(algo, 0.7)
        
        # 等级对压缩比的影响
        level_effect = 0.0
        if algo == 'gzip':
            level_effect = min(0.2, level * 0.025)  # gzip 1-9
        elif algo == 'zstd':
            level_effect = min(0.25, level * 0.015)  # zstd 1-19
        elif algo == 'brotli':
            level_effect = min(0.3, level * 0.025)  # brotli 1-11
        elif algo == 'lz4':
            level_effect = min(0.1, level * 0.01)  # lz4不同等级
        
        # 熵值对压缩比的影响（熵值越高，越难压缩）
        entropy_effect = max(0.1, entropy)
        
        # 实际压缩比
        ratio = base_ratio - level_effect
        ratio = max(0.05, min(0.95, ratio))  # 限制在合理范围内
        
        # 根据熵值调整
        final_ratio = ratio + (1 - entropy_effect) * 0.3
        
        return max(0.05, min(0.95, final_ratio))


# ==============================================================================
# 战术层：保持不变 (略)
# ==============================================================================
class CAGSTacticalLayer:
    def __init__(self):
        self.buffer = {}
        self.next_needed_id = 0

    def on_download_complete(self, chunk_id, data_size_kb):
        self.buffer[chunk_id] = data_size_kb
        # print(f"    📦 [Buffer] 收到块 #{chunk_id}, 当前缓冲: {list(self.buffer.keys())}")
        
        while self.next_needed_id in self.buffer:
            # print(f"    ✅ [Stream] 提交块 #{self.next_needed_id} 至解压引擎")
            del self.buffer[self.next_needed_id]
            self.next_needed_id += 1

# ==============================================================================
# 修正层：修复过于激进的 AIMD
# ==============================================================================
class CAGSCorrectionLayer:
    def __init__(self, initial_chunk_size, min_size=256*1024, max_size=4*1024*1024):
        self.current_size = initial_chunk_size
        self.min_size = min_size
        self.max_size = max_size
        self.success_streak = 0
        self.fail_streak = 0
        
        # [修复点 3] 引入容忍阈值，防止因为单次网络抖动导致性能腰斩
        self.tolerance_threshold = 2 

    def feedback(self, status, rtt_ms=None):
        if status == 'TIMEOUT':
            self.fail_streak += 1
            self.success_streak = 0
            
            # [修复点 3] 只有连续失败超过阈值，才触发乘性减
            if self.fail_streak >= self.tolerance_threshold:
                old = self.current_size
                self.current_size = max(self.min_size, self.current_size // 2)
                # print(f"🚨 [AIMD] 确认拥塞! 乘性减: {old//1024}KB -> {self.current_size//1024}KB")
                self.fail_streak = 0 # 重置计数
        
        elif status == 'SUCCESS':
            self.success_streak += 1
            self.fail_streak = 0 # 成功一次就重置失败计数，因为TCP只要通了就说明没拥塞
            
            # 加性增 (Additive Increase)
            if self.success_streak > 5:
                if self.current_size < self.max_size:
                    self.current_size = min(self.max_size, self.current_size + 256*1024)
                    # print(f"📈 [AIMD] 探测带宽，加性增: -> {self.current_size//1024}KB")
                self.success_streak = 0
                
        return self.current_size