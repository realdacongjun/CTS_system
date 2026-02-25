import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from collections import namedtuple

# 定义传输配置数据结构
TransmissionConfig = namedtuple('TransmissionConfig', 
                                 ['threads', 'cpu_quota', 'chunk_size_kb', 'algo_name'])

class CAGSDecisionEngine:
    def __init__(self, model, scaler_c, scaler_i, enc, cols_c, cols_i, device='cpu'):
        """
        初始化CAGS决策引擎
        :param model: 加载好的CompactCFTNetV2模型
        :param scaler_c: 客户端特征标准化器
        :param scaler_i: 镜像特征标准化器
        :param enc: 算法名称编码器
        :param cols_c: 客户端特征列名列表 (必须包含训练时的8个特征)
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
        
        # 定义完整决策空间 (240组候选配置，完全对齐论文)
        self._init_decision_space()
        
    def _init_decision_space(self):
        """初始化论文定义的完整决策空间"""
        self.threads_list = [1, 2, 4, 8, 16]
        self.cpu_quota_list = [0.2, 0.5, 1.0, 2.0]
        self.chunk_size_list = [64, 256, 1024] # KB
        
        # 从训练时的编码器获取算法列表
        self.algo_list = self.enc.classes_.tolist() 
        
        self.configs = []
        for t in self.threads_list:
            for c in self.cpu_quota_list:
                for s in self.chunk_size_list:
                    for a in self.algo_list:
                        self.configs.append(TransmissionConfig(t, c, s, a))
        print(f"✅ CAGS决策引擎初始化完成: {len(self.configs)} 组候选配置")

    def _predict_batch(self, env_state: dict, image_features: dict):
        """
        批量预测所有候选配置的性能与不确定性
        :param env_state: dict, 当前环境状态 
            (必须包含: bandwidth_mbps, cpu_limit, network_rtt, mem_limit_mb)
        :param image_features: dict, 当前镜像特征
        :return: DataFrame, 包含所有配置的预测结果
        """
        # 1. 构造特征 (必须和训练时完全一致，包含4个物理交叉特征)
        df_env = pd.DataFrame([env_state])
        df_img = pd.DataFrame([image_features])
        
        # ==========================================
        # 【关键适配】复现训练时的物理交叉特征
        # ==========================================
        df_env['theoretical_time'] = df_img['total_size_mb'].values[0] / (df_env['bandwidth_mbps'] / 8 + 1e-8)
        df_env['cpu_to_size_ratio'] = df_env['cpu_limit'] / (df_img['total_size_mb'].values[0] + 1e-8)
        df_env['mem_to_size_ratio'] = df_env['mem_limit_mb'] / (df_img['total_size_mb'].values[0] + 1e-8)
        df_env['network_score'] = df_env['bandwidth_mbps'] / (df_env['network_rtt'] + 1e-8)
        
        # 按训练时的顺序提取特征
        try:
            Xc_raw = df_env[self.cols_c].values
            Xi_raw = df_img[self.cols_i].values
        except KeyError as e:
            print(f"❌ 特征列名不匹配，请检查: {e}")
            print(f"   训练时的客户端特征: {self.cols_c}")
            print(f"   训练时的镜像特征: {self.cols_i}")
            raise
        
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
                # 【关键适配】你的模型输出是 [gamma, v, alpha, beta]
                preds = self.model(cx_batch, ix_batch, ax_batch)
                gamma, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
                
                # 转换回原始时间空间
                pred_time_ori = torch.expm1(gamma).cpu().numpy()
                
                # 计算不确定性
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_log = torch.sqrt(var_log + 1e-6)
                std_ori = torch.exp(gamma) * std_log # 原始空间的标准差
                
                # 认知不确定性 (用于OOD检测)
                epistemic_unc = (1.0 / (v + 1e-6)).cpu().numpy()
            
            # 保存结果
            for j, cfg in enumerate(batch_configs):
                img_size_mb = image_features['total_size_mb']
                throughput = img_size_mb / (pred_time_ori[j] + 1e-8) # MB/s
                # 简化CPU成本估算
                cpu_cost = cfg.threads * cfg.cpu_quota * pred_time_ori[j]
                
                results.append({
                    'threads': cfg.threads,
                    'cpu_quota': cfg.cpu_quota,
                    'chunk_size_kb': cfg.chunk_size_kb,
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
        基于论文规则的简化实现
        """
        bw = env_state['bandwidth_mbps']
        rtt = env_state['network_rtt']
        
        if bw < 5 and rtt > 50:
            return 'VERTICAL' # 弱网：垂直坍缩，激进抢占吞吐量
        elif bw > 500 and rtt < 10:
            return 'CONVEX'   # 强网：凸拓扑，优先节省CPU
        else:
            return 'TRANSITION' # 过渡：平衡

    def _cid_sort(self, pred_df, safety_threshold=0.5):
        """
        基于置信区间支配 (CID) 的鲁棒排序
        """
        # 1. 预剔除高风险配置 (OOD检测)
        safe_df = pred_df[pred_df['epistemic_unc'] < safety_threshold].copy()
        if len(safe_df) == 0:
            print("⚠️  警告: 所有配置不确定性过高，回退至保守配置")
            return pred_df.head(1)
        
        # 2. 帕累托非支配排序 (双目标: 最小化pred_time, 最小化cpu_cost)
        points = safe_df[['pred_time', 'cpu_cost']].values
        
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                # 检查是否有其他点支配当前点
                is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1) | np.all(points[is_efficient] == c, axis=1)
                is_efficient[i] = True
        
        pareto_front = safe_df[is_efficient].copy()
        return pareto_front

    def _akt_knee_tracking(self, pareto_front, collapse_type):
        """
        自适应拐点追踪 (AKT)
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
            # 过渡：找帕累托前沿的膝点 (切比雪夫距离)
            utopia = np.array([0, 0])
            distances = np.max(np.abs(scaled - utopia), axis=1)
            idx = pareto_front.index[np.argmin(distances)]
            
        return pareto_front.loc[idx]

    def make_decision(self, env_state: dict, image_features: dict, 
                      safety_threshold=0.5, enable_uncertainty=True, enable_dpc=True):
        """
        端到端决策入口
        :param env_state: 环境状态
        :param image_features: 镜像特征
        :param enable_uncertainty: 消融实验开关：是否启用不确定性
        :param enable_dpc: 消融实验开关：是否启用DPC
        :return: (config_dict, metrics_dict, pareto_front_df)
        """
        # 1. 批量预测
        pred_df = self._predict_batch(env_state, image_features)
        
        # 2. DPC检测
        if enable_dpc:
            collapse_type = self._detect_dpc(pred_df, env_state)
        else:
            collapse_type = 'TRANSITION'
        
        # 3. CID鲁棒排序
        if enable_uncertainty:
            pareto_front = self._cid_sort(pred_df, safety_threshold)
        else:
            # 关闭不确定性时，直接用原始预测值做帕累托排序
            pareto_front = self._cid_sort(pred_df, safety_threshold=1000.0)
        
        # 4. AKT拐点追踪
        best_config = self._akt_knee_tracking(pareto_front, collapse_type)
        
        # 格式化输出 (适配实验脚本的skopeo参数)
        config_dict = {
            'threads': int(best_config['threads']),
            'cpu_quota': best_config['cpu_quota'],
            'chunk_size_kb': int(best_config['chunk_size_kb']),
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