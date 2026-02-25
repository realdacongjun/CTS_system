import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import namedtuple

# 定义传输配置数据结构
TransmissionConfig = namedtuple('TransmissionConfig', 
                                 ['threads', 'cpu_quota', 'chunk_size', 'algo_name'])

class CAGSDecisionEngine:
    def __init__(self, model, scaler_c, scaler_i, enc, cols_c, cols_i, device='cpu'):
        """
        初始化CAGS决策引擎
        :param model: 加载好的CompactCFTNetV2模型
        :param scaler_c: 客户端特征标准化器
        :param scaler_i: 镜像特征标准化器
        :param enc: 算法名称编码器
        :param cols_c: 客户端特征列名列表
        :param cols_i: 镜像特征列名列表
        :param device: 运行设备
        """
        self.model = model
        self.scaler_c = scaler_c
        self.scaler_i = scaler_i
        self.enc = enc
        self.cols_c = cols_c
        self.cols_i = cols_i
        self.device = device
        self.model.eval()
        
        # 定义完整决策空间 (240组候选配置)
        self._init_decision_space()
        
    def _init_decision_space(self):
        """初始化论文定义的完整决策空间"""
        self.threads_list = [1, 2, 4, 8, 16]
        self.cpu_quota_list = [0.2, 0.5, 1.0, 2.0]
        self.chunk_size_list = [64, 256, 1024] # KB
        self.algo_list = self.enc.classes_.tolist() # 从训练数据中获取的算法列表
        
        self.configs = []
        for t in self.threads_list:
            for c in self.cpu_quota_list:
                for s in self.chunk_size_list:
                    for a in self.algo_list:
                        self.configs.append(TransmissionConfig(t, c, s, a))
        print(f"✅ 决策空间初始化完成: {len(self.configs)} 组候选配置")

    def _predict_batch(self, env_state, image_features):
        """
        批量预测所有候选配置的性能与不确定性
        :param env_state: dict, 当前环境状态 (bandwidth_mbps, cpu_limit, network_rtt, mem_limit_mb)
        :param image_features: dict, 当前镜像特征
        :return: DataFrame, 包含所有配置的预测结果
        """
        import torch
        
        # 1. 构造客户端特征 (包含物理交叉特征)
        df_env = pd.DataFrame([env_state])
        df_img = pd.DataFrame([image_features])
        
        # 计算物理交叉特征 (必须与训练时一致!)
        df_env['theoretical_time'] = df_img['total_size_mb'].values[0] / (df_env['bandwidth_mbps'] / 8 + 1e-8)
        df_env['cpu_to_size_ratio'] = df_env['cpu_limit'] / (df_img['total_size_mb'].values[0] + 1e-8)
        df_env['mem_to_size_ratio'] = df_env['mem_limit_mb'] / (df_img['total_size_mb'].values[0] + 1e-8)
        df_env['network_score'] = df_env['bandwidth_mbps'] / (df_env['network_rtt'] + 1e-8)
        
        # 按训练时的顺序提取特征
        Xc_raw = df_env[self.cols_c].values
        Xi_raw = df_img[self.cols_i].values
        
        # 标准化
        Xc = self.scaler_c.transform(Xc_raw)
        Xi = self.scaler_i.transform(Xi_raw)
        
        # 2. 批量构造所有配置的输入
        results = []
        batch_size = 64 # 防止显存溢出
        
        for i in range(0, len(self.configs), batch_size):
            batch_configs = self.configs[i:i+batch_size]
            
            # 构造Batch Tensor
            cx_batch = torch.FloatTensor(Xc).repeat(len(batch_configs), 1)
            ix_batch = torch.FloatTensor(Xi).repeat(len(batch_configs), 1)
            
            # 编码算法
            algo_names = [c.algo_name for c in batch_configs]
            # 安全编码
            known = set(self.enc.classes_)
            ax_batch = []
            for a in algo_names:
                if a in known:
                    ax_batch.append(self.enc.transform([a])[0])
                else:
                    ax_batch.append(self.enc.transform([self.algo_list[0]])[0]) # 默认回退
            ax_batch = torch.LongTensor(ax_batch)
            
            # 移至设备
            cx_batch = cx_batch.to(self.device)
            ix_batch = ix_batch.to(self.device)
            ax_batch = ax_batch.to(self.device)
            
            # 预测
            with torch.no_grad():
                preds = self.model(cx_batch, ix_batch, ax_batch)
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
                # 转换回原始空间
                pred_time_ori = torch.expm1(gamma).cpu().numpy()
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_log = torch.sqrt(var_log + 1e-6)
                std_ori = torch.exp(gamma) * std_log
                epistemic_unc = (beta / (v * (alpha - 1) + 1e-6)).cpu().numpy() # 认知不确定性
            
            # 保存结果
            for j, cfg in enumerate(batch_configs):
                # 估算吞吐量 (MB/s) 和 CPU成本 (简化模型)
                throughput = df_img['total_size_mb'].values[0] / (pred_time_ori[j] + 1e-8)
                # CPU成本简化为: 线程数 * CPU配额 * 时间
                cpu_cost = cfg.threads * cfg.cpu_quota * pred_time_ori[j]
                
                results.append({
                    'threads': cfg.threads,
                    'cpu_quota': cfg.cpu_quota,
                    'chunk_size': cfg.chunk_size,
                    'algo_name': cfg.algo_name,
                    'pred_time': pred_time_ori[j],
                    'uncertainty': std_ori[j].cpu().numpy(),
                    'epistemic_unc': epistemic_unc[j],
                    'throughput': throughput,
                    'cpu_cost': cpu_cost
                })
        
        return pd.DataFrame(results)

    def _detect_dpc(self, pred_df, env_state):
        """
        动态帕累托坍缩 (DPC) 检测
        :param pred_df: 预测结果DataFrame
        :param env_state: 当前环境状态
        :return: str, 拓扑类型 ('VERTICAL', 'CONVEX', 'TRANSITION')
        """
        bw = env_state['bandwidth_mbps']
        rtt = env_state['network_rtt']
        
        # 论文中的简化判定规则 (可基于实测数据优化)
        if bw < 5 and rtt > 50:
            return 'VERTICAL' # 弱网：垂直坍缩
        elif bw > 500 and rtt < 10:
            return 'CONVEX'   # 强网：凸拓扑
        else:
            return 'TRANSITION' # 过渡

    def _cid_sort(self, pred_df, safety_threshold=0.5):
        """
        基于置信区间支配 (CID) 的鲁棒排序
        :param pred_df: 预测结果
        :param safety_threshold: 认知不确定性安全阈值
        :return: DataFrame, 鲁棒帕累托前沿
        """
        # 1. 预剔除高风险配置 (OOD检测)
        safe_df = pred_df[pred_df['epistemic_unc'] < safety_threshold].copy()
        if len(safe_df) == 0:
            print("⚠️ 警告: 所有配置不确定性过高，回退至保守配置")
            return pred_df.head(1) # 回退
        
        # 2. 帕累托非支配排序 (最小化时间，最小化CPU成本 -> 等价于最大化吞吐量，最小化成本)
        # 这里简化为双目标: pred_time (越小越好), cpu_cost (越小越好)
        points = safe_df[['pred_time', 'cpu_cost']].values
        
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                # 检查是否有其他点支配当前点
                is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1) | np.all(points[is_efficient] == c, axis=1)
                is_efficient[i] = True # 保留自己
        
        pareto_front = safe_df[is_efficient].copy()
        return pareto_front

    def _akt_knee_tracking(self, pareto_front, collapse_type):
        """
        自适应拐点追踪 (AKT)
        :param pareto_front: 鲁棒帕累托前沿
        :param collapse_type: DPC拓扑类型
        :return: Series, 最优配置
        """
        if len(pareto_front) == 1:
            return pareto_front.iloc[0]
            
        # 归一化目标
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pareto_front[['pred_time', 'cpu_cost']])
        
        if collapse_type == 'VERTICAL':
            # 弱网：激进抢占吞吐量 (最小化pred_time)
            idx = pareto_front['pred_time'].idxmin()
        elif collapse_type == 'CONVEX':
            # 强网：优先节省CPU (最小化cpu_cost)
            idx = pareto_front['cpu_cost'].idxmin()
        else:
            # 过渡：找帕累托前沿的膝点 (到乌托邦点的切比雪夫距离最小)
            utopia = np.array([0, 0]) # 理想点
            distances = np.max(np.abs(scaled - utopia), axis=1)
            idx = pareto_front.index[np.argmin(distances)]
            
        return pareto_front.loc[idx]


    def make_decision(self, env_state, image_features, safety_threshold=0.5, enable_uncertainty=True, enable_dpc=True):
    # 1. 批量预测
        pred_df = self._predict_batch(env_state, image_features)
    
    # 2. DPC检测（受enable_dpc开关控制）
        if enable_dpc:
            collapse_type = self._detect_dpc(pred_df, env_state)
        else:
            collapse_type = 'TRANSITION' # 关闭DPC时固定为过渡类型
        print(f"🔍 DPC拓扑检测: {collapse_type} (enable_dpc={enable_dpc})")
    
    # 3. CID鲁棒排序（受enable_uncertainty开关控制）
        if enable_uncertainty:
            pareto_front = self._cid_sort(pred_df, safety_threshold)
        else:
        # 关闭不确定性时，直接用原始预测值做帕累托排序
            pareto_front = self._cid_sort(pred_df, safety_threshold=1000) # 关闭安全过滤
    
    # 4. AKT拐点追踪
        best_config = self._akt_knee_tracking(pareto_front, collapse_type)
        
        # 格式化输出
        config_dict = {
            'threads': int(best_config['threads']),
            'cpu_quota': best_config['cpu_quota'],
            'chunk_size_kb': int(best_config['chunk_size']),
            'algo_name': best_config['algo_name']
        }
        
        metrics_dict = {
            'pred_time_s': best_config['pred_time'],
            'uncertainty_s': best_config['uncertainty'],
            'throughput_mbps': best_config['throughput'] * 8, # 转Mbps
            'collapse_type': collapse_type,
            'pareto_size': len(pareto_front)
        }
        
        return config_dict, metrics_dict, pareto_front