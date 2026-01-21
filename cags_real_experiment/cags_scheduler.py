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