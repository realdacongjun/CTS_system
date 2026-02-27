import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from collections import namedtuple
from typing import Dict, Tuple
import warnings

# 定义传输配置数据结构
TransmissionConfig = namedtuple('TransmissionConfig', 
                                 ['threads', 'cpu_quota', 'chunk_size_kb', 'algo_name'])

class CAGSDecisionEngine:
    def __init__(self, model, scaler_c, scaler_i, enc, cols_c, cols_i, device='cpu'):
        """
        初始化CAGS决策引擎
        完全兼容已训练的CompactCFTNetV2模型，不做任何架构修改
        """
        self.model = model
        self.scaler_c = scaler_c
        self.scaler_i = scaler_i
        self.enc = enc
        self.cols_c = cols_c
        self.cols_i = cols_i
        self.device = device
        self.model.eval()
        
        # 验证特征维度匹配
        print(f"🔧 CAGS初始化:")
        print(f"   客户端特征: {cols_c} (维度: {len(cols_c)})")
        print(f"   镜像特征: {cols_i} (维度: {len(cols_i)})")
        
        self._init_decision_space()
        
    def _init_decision_space(self):
        """初始化决策空间"""
        self.threads_list = [1, 2, 4, 8, 16]
        self.cpu_quota_list = [0.2, 0.5, 1.0, 2.0]
        self.chunk_size_list = [64, 256, 1024]  # KB
        
        self.algo_list = self.enc.classes_.tolist() 
        
        self.configs = []
        for t in self.threads_list:
            for c in self.cpu_quota_list:
                for s in self.chunk_size_list:
                    for a in self.algo_list:
                        self.configs.append(TransmissionConfig(t, c, s, a))
        print(f"✅ 决策空间: {len(self.configs)} 组配置")

    def _predict_batch(self, env_state: dict, image_features: dict):
        """
        批量预测 - 完全兼容原模型输出格式 [gamma, v, alpha, beta]
        """
        # 构造特征DataFrame
        df_env = pd.DataFrame([env_state])
        df_img = pd.DataFrame([image_features])
        
        # 计算物理交叉特征（必须与训练时一致）
        df_env['theoretical_time'] = df_img['total_size_mb'].values[0] / (df_env['bandwidth_mbps'] / 8 + 1e-8)
        df_env['cpu_to_size_ratio'] = df_env['cpu_limit'] / (df_img['total_size_mb'].values[0] + 1e-8)
        df_env['mem_to_size_ratio'] = df_env['mem_limit_mb'] / (df_img['total_size_mb'].values[0] + 1e-8)
        df_env['network_score'] = df_env['bandwidth_mbps'] / (df_env['network_rtt'] + 1e-8)
        
        # 提取特征（严格按训练时顺序）
        try:
            Xc_raw = df_env[self.cols_c].values
            Xi_raw = df_img[self.cols_i].values
        except KeyError as e:
            print(f"❌ 特征缺失: {e}")
            print(f"   可用列: {list(df_env.columns)}")
            raise
        
        # 标准化
        Xc = self.scaler_c.transform(Xc_raw)
        Xi = self.scaler_i.transform(Xi_raw)
        
        # 批量预测
        results = []
        batch_size = 64
        
        for i in range(0, len(self.configs), batch_size):
            batch_configs = self.configs[i:i+batch_size]
            
            # 构造输入张量
            cx_batch = torch.FloatTensor(Xc).repeat(len(batch_configs), 1)
            ix_batch = torch.FloatTensor(Xi).repeat(len(batch_configs), 1)
            
            # 算法编码
            algo_names = [c.algo_name for c in batch_configs]
            known = set(self.enc.classes_)
            ax_batch = []
            for a in algo_names:
                if a in known:
                    ax_batch.append(self.enc.transform([a])[0])
                else:
                    ax_batch.append(0)  # 回退到第一个算法
            ax_batch = torch.LongTensor(ax_batch)
            
            cx_batch = cx_batch.to(self.device)
            ix_batch = ix_batch.to(self.device)
            ax_batch = ax_batch.to(self.device)
            
            # 预测 - 不修改模型，直接解析输出
            with torch.no_grad():
                preds = self.model(cx_batch, ix_batch, ax_batch)  # [N, 4]
                
                # 解析输出: [gamma, v, alpha, beta]
                gamma = preds[:, 0]
                v = preds[:, 1]
                alpha = preds[:, 2]
                beta = preds[:, 3]
                
                # 转原始空间
                pred_time_ori = torch.expm1(gamma).cpu().numpy()
                
                # 计算不确定性
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_log = torch.sqrt(var_log + 1e-6)
                std_ori = torch.exp(gamma) * std_log
                
                # 认知不确定性
                epistemic_unc = (1.0 / (v + 1e-6)).cpu().numpy()
            
            # 保存结果
            for j, cfg in enumerate(batch_configs):
                img_size = image_features.get('total_size_mb', 50)
                throughput = img_size / (pred_time_ori[j] + 1e-8)
                cpu_cost = cfg.threads * cfg.cpu_quota * pred_time_ori[j]
                
                results.append({
                    'threads': cfg.threads,
                    'cpu_quota': cfg.cpu_quota,
                    'chunk_size_kb': cfg.chunk_size_kb,
                    'algo_name': cfg.algo_name,
                    'pred_time': float(pred_time_ori[j]),
                    'uncertainty': float(std_ori[j]),
                    'epistemic_unc': float(epistemic_unc[j]),
                    'throughput': float(throughput),
                    'cpu_cost': float(cpu_cost)
                })
        
        return pd.DataFrame(results)

    def _detect_dpc(self, pred_df: pd.DataFrame, env_state: dict) -> str:
        """动态帕累托坍缩检测"""
        bw = env_state.get('bandwidth_mbps', 100)
        rtt = env_state.get('network_rtt', 10)
        
        if bw < 5 or rtt > 100:
            return 'VERTICAL'
        elif bw > 500 and rtt < 5:
            return 'CONVEX'
        else:
            return 'TRANSITION'

    def _cid_sort(self, pred_df: pd.DataFrame, safety_threshold: float = 0.5) -> pd.DataFrame:
        """置信区间支配排序 - 修复帕累托前沿计算"""
        # OOD过滤
        safe_df = pred_df[pred_df['epistemic_unc'] < safety_threshold].copy()
        if len(safe_df) == 0:
            print("⚠️  警告: 所有配置不确定性过高，选择最保守配置")
            return pred_df.nsmallest(1, 'epistemic_unc')
        
        # 帕累托非支配排序 [修复版]
        points = safe_df[['pred_time', 'cpu_cost']].values
        n_points = len(points)
        is_efficient = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_efficient[i]:
                # 点i被支配的条件：存在j，使得j在所有目标上≤i，且至少一个<
                dominated_by_i = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
                is_efficient[dominated_by_i] = False
        
        pareto_front = safe_df.iloc[is_efficient].copy()
        
        if len(pareto_front) == 0:
            return safe_df
            
        return pareto_front

    def _akt_knee_tracking(self, pareto_front: pd.DataFrame, collapse_type: str) -> pd.Series:
        """自适应拐点追踪"""
        if len(pareto_front) == 1:
            return pareto_front.iloc[0]
        
        # 归一化
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pareto_front[['pred_time', 'cpu_cost']])
        
        if collapse_type == 'VERTICAL':
            # 弱网：最小化时间
            idx = pareto_front['pred_time'].idxmin()
        elif collapse_type == 'CONVEX':
            # 强网：最小化CPU成本
            idx = pareto_front['cpu_cost'].idxmin()
        else:
            # 过渡：膝点（切比雪夫距离）
            utopia = np.array([0, 0])
            distances = np.max(np.abs(scaled - utopia), axis=1)
            knee_pos = np.argmin(distances)
            idx = pareto_front.index[knee_pos]
            
        return pareto_front.loc[idx]

    def make_decision(self, env_state: dict, image_features: dict, 
                      safety_threshold: float = 0.5, 
                      enable_uncertainty: bool = True, 
                      enable_dpc: bool = True) -> Tuple[dict, dict, pd.DataFrame]:
        """
        端到端决策入口
        """
        # 1. 批量预测
        pred_df = self._predict_batch(env_state, image_features)
        
        # 2. DPC检测
        collapse_type = self._detect_dpc(pred_df, env_state) if enable_dpc else 'TRANSITION'
        
        # 3. CID排序
        threshold = safety_threshold if enable_uncertainty else 1e6
        pareto_front = self._cid_sort(pred_df, threshold)
        
        # 4. AKT选择
        best = self._akt_knee_tracking(pareto_front, collapse_type)
        
        # 格式化输出（确保Python原生类型）
        config_dict = {
            'threads': int(best['threads']),
            'cpu_quota': float(best['cpu_quota']),
            'chunk_size_kb': int(best['chunk_size_kb']),
            'algo_name': str(best['algo_name'])
        }
        
        metrics_dict = {
            'pred_time_s': float(best['pred_time']),
            'uncertainty_s': float(best['uncertainty']),
            'throughput_mbps': float(best['throughput'] * 8),
            'epistemic_unc': float(best['epistemic_unc']),
            'collapse_type': str(collapse_type),
            'pareto_size': int(len(pareto_front))
        }
        
        return config_dict, metrics_dict, pareto_front