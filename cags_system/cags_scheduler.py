import numpy as np
import math

# ==============================================================================
# 第一层：战略层 (Strategy Layer) - 基于势垒函数的资源感知效用模型
# ==============================================================================

class CAGSStrategyLayer:
    def __init__(self, alpha=1.0, beta=0.5, gamma=2.0, uncertainty_weight=5.0):
        self.alpha = alpha  # 时间权重
        self.beta = beta    # 算力权重
        self.gamma = gamma  # 风险权重
        self.uncertainty_weight = uncertainty_weight # [新增] 不确定性惩罚系数
        
        # 决策空间
        self.chunk_sizes = [256*1024, 512*1024, 1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024]
        self.concurrencies = [1, 2, 4, 8, 16]

    def optimize(self, predicted_bw_mbps, predicted_loss_rate, client_cpu_load, model_uncertainty=0.0):
        """
        执行非凸优化决策。
        
        参数:
        - model_uncertainty (float): AI 模型的预测不确定性 (0.0 ~ 1.0)。
                                     0.0 表示完全确信，1.0 表示完全瞎猜。
        """
        best_cost = float('inf')
        best_config = (1024*1024, 1)

        # 基础物理参数
        bw_bps = predicted_bw_mbps * 1024 * 1024 / 8.0
        MTU = 1460 

        # [核心逻辑升级] 计算风险放大因子
        # 如果模型不确定性很高 (e.g. 0.8)，这个因子会变大 (1 + 5*0.8 = 5.0)
        # 这意味着所有的风险成本将被放大 5 倍，迫使系统选择风险极小(小切片)的方案。
        risk_amplifier = 1.0 + (self.uncertainty_weight * model_uncertainty)

        for s in self.chunk_sizes:
            for n in self.concurrencies:
                # 1. === 传输时间成本 ===
                # 并发增益 (边际递减)
                concurrency_gain = n ** 0.9 
                effective_bw = bw_bps * concurrency_gain
                t_trans = s / effective_bw

                # 2. === 计算势垒成本 ===
                # 预估任务负载: 线程开销 + 系统调用开销
                thread_overhead = 0.02 * n
                syscall_overhead = 0.005 * (1024*1024 / s) # 单位数据量的小包开销更高，但系数更温和
                task_load = thread_overhead + syscall_overhead
                
                # 限制最大负载
                current_total_load = min(0.99, client_cpu_load + task_load)
                # 指数势垒
                c_cpu = math.exp(4 * current_total_load) 

                # 3. === 风险概率成本 (融入不确定性) ===
                num_packets = s / MTU
                # 伯努利试验：切片传输失败的概率
                prob_fail = 1 - (1 - predicted_loss_rate) ** num_packets
                
                # [关键修改]：风险成本 = 基础风险 * 风险放大因子
                # 原理：当 AI 甚至不知道当前是不是弱网时，为了安全，我们将潜在的重传代价人为放大。
                # 这会让大包的 Cost 变得极高，从而在数学上“自然地”滑落到小包配置。
                r_risk = (prob_fail * t_trans * 10) * risk_amplifier

                # === 总广义成本 ===
                cost = self.alpha * t_trans + self.beta * c_cpu + self.gamma * r_risk

                if cost < best_cost:
                    best_cost = cost
                    best_config = (s, n)

        return best_config, best_cost

# ==============================================================================
# 第二层：战术层 (Tactical Layer) - [流水线与乱序重排]
# ==============================================================================
class CAGSTacticalLayer:
    def __init__(self):
        self.reorder_buffer = {} 
        self.expected_id = 0      

    def on_download_complete(self, chunk_id, data_size_kb):
        """模拟下载完成回调：解决乱序到达问题 (HOL Blocking)"""
        # print(f"⬇️ [Net] Chunk {chunk_id} ({data_size_kb:.1f}KB) 下载完成")
        self.reorder_buffer[chunk_id] = data_size_kb
        self._flush_buffer()

    def _flush_buffer(self):
        # 只有当 expected_id 到达时，才推送给解压引擎
        while self.expected_id in self.reorder_buffer:
            size = self.reorder_buffer.pop(self.expected_id)
            # print(f"✅ [Buffer] Chunk {self.expected_id} 顺序正确 -> 推送解压流水线")
            self.expected_id += 1
        
        # 调试信息：如果缓冲区有残留，说明发生了乱序
        # if self.reorder_buffer:
        #    keys = sorted(list(self.reorder_buffer.keys()))
        #    print(f"⏳ [Buffer] 暂存乱序块: {keys} (阻塞中: 等待 Chunk {self.expected_id})")

# ==============================================================================
# 第三层：修正层 (Correction Layer) - [AIMD 动态流控]
# ==============================================================================

class CAGSCorrectionLayer:
    def __init__(self, initial_chunk_size, min_size=256*1024, max_size=16*1024*1024):
        self.current_size = initial_chunk_size
        self.min_size = min_size
        self.max_size = max_size
        self.success_streak = 0
        self.fail_streak = 0
        # 引入容忍机制，防止单次抖动就导致窗口腰斩
        self.tolerance_threshold = 2 

    def feedback(self, status, rtt_ms=None):
        """
        基于应用层反馈的 AIMD 控制算法
        status: 'SUCCESS' 或 'TIMEOUT' (模拟 RTO 触发)
        rtt_ms: 当前 RTT (虽然此处仅用于记录，但在真实 TCP 中用于计算 RTO)
        """
        if status == 'TIMEOUT':
            self.fail_streak += 1
            self.success_streak = 0
            
            # 只有连续失败超过容忍阈值，才认为是真正的拥塞
            if self.fail_streak >= self.tolerance_threshold:
                old = self.current_size
                # 乘性减 (Multiplicative Decrease): 窗口减半
                self.current_size = max(self.min_size, self.current_size // 2)
                # print(f"🚨 [AIMD] 确认拥塞! 乘性减: {old//1024}KB -> {self.current_size//1024}KB")
                self.fail_streak = 0 # 重置计数器
        elif status == 'SUCCESS':
            self.success_streak += 1
            self.fail_streak = 0 # 成功一次就重置失败计数
            
            # 加性增 (Additive Increase)
            if self.success_streak > 5:
                if self.current_size < self.max_size:
                    self.current_size = min(self.max_size, self.current_size + 256*1024)
                    # print(f"📈 [AIMD] 探测带宽，加性增: -> {self.current_size//1024}KB")
                self.success_streak = 0
                
        return self.current_size