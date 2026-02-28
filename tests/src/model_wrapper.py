# src/model_wrapper.py
import torch
import pickle
import numpy as np
import pandas as pd
import logging
import time
import yaml
from typing import Dict, Any, Tuple, List, Optional
import sys
import os

class CFTNetWrapper:
    def __init__(self, model_config_path: str):
        with open(model_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 🔧 修复1：直接读取顶层key，而非 config['model']['xxx']
        self.model_path = self.config['model_path']
        self.preprocess_path = self.config['preprocessing_path']
        
        # 处理相对路径：相对于 model_config.yaml 所在目录解析
        config_dir = os.path.dirname(model_config_path)
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(config_dir, self.model_path)
        if not os.path.isabs(self.preprocess_path):
            self.preprocess_path = os.path.join(config_dir, self.preprocess_path)

        # 设备选择
        requested_device = self.config.get('device', 'cpu')
        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logging.warning(f"⚠️  CUDA不可用，自动回退到CPU")
        else:
            self.device = torch.device(requested_device)
        
        self.logger = logging.getLogger(__name__)
        self._start_time = time.time()
        self._load_everything()
        self._load_time = time.time() - self._start_time
        self.logger.info(f"✅ 系统加载完成，耗时: {self._load_time:.2f}s")

    def _load_everything(self):
        try:
            self.logger.info(f"   正在加载预处理器: {self.preprocess_path}")
            with open(self.preprocess_path, 'rb') as f:
                preprocess = pickle.load(f)
            
            self.scaler_c = preprocess['scaler_c']
            self.scaler_i = preprocess['scaler_i']
            self.enc = preprocess['enc']
            self.cols_c = preprocess['cols_c']
            self.cols_i = preprocess['cols_i']
            self.num_algos = len(self.enc.classes_)
            self.algo_list = self.enc.classes_.tolist()
            
            self.logger.info(f"   正在导入模型类...")
            try:
                from cts_model import CompactCFTNetV2
                self.model_class = CompactCFTNetV2
            except ImportError:
                raise ImportError("❌ 无法导入 CompactCFTNetV2")
            
            embed_dim = self.config.get('embed_dim', 64)
            
            self.logger.info(f"   正在初始化模型...")
            self.model = self.model_class(
                client_feats=len(self.cols_c),
                image_feats=len(self.cols_i),
                num_algos=self.num_algos,
                embed_dim=embed_dim
            ).to(self.device)
            
            self.logger.info(f"   正在加载模型权重: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.logger.info(f"✅ CompactCFTNetV2 模型加载成功")
            
        except Exception as e:
            self.logger.error(f"❌ 系统加载失败: {e}", exc_info=True)
            raise

    # ... (其余函数: get_model_info, _construct_features, predict_single_config 等保持不变，只需确保上面的初始化部分正确) ...
    # 为了节省篇幅，这里假设后面的函数保持原样，只需复制你原来代码中 _construct_features 及之后的内容即可

    # ==========================================
    # 【新增】模型信息查询接口 (用于实验四)
    # ==========================================
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 估算模型大小
        params_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        # 估算预处理器大小 (约10KB)
        preproc_bytes = 10 * 1024
        total_bytes = params_bytes + preproc_bytes
        
        return {
            'model_class': self.model_class.__name__,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(total_params - trainable_params),
            'model_params_size_mb': round(params_bytes / (1024 * 1024), 3),
            'total_system_size_mb': round(total_bytes / (1024 * 1024), 2),
            'client_feature_dim': len(self.cols_c),
            'image_feature_dim': len(self.cols_i),
            'num_algorithms': self.num_algos,
            'algorithm_list': self.algo_list,
            'device': str(self.device),
            'load_time_seconds': round(self._load_time, 2)
        }

    # ==========================================
    # 核心预测接口
    # ==========================================
    def _construct_features(
        self, 
        env_state: Dict[str, Any], 
        image_features: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【优化点4】独立的特征构造函数
        确保与训练时100%一致
        """
        df_env = pd.DataFrame([env_state])
        df_img = pd.DataFrame([image_features])
        
        # 安全检查：镜像大小必须存在
        if 'total_size_mb' not in df_img.columns:
            raise ValueError("image_features 必须包含 'total_size_mb' 字段")
        
        img_size = df_img['total_size_mb'].values[0]
        
        # 计算物理交叉特征 (必须与训练代码完全一致)
        df_env['theoretical_time'] = img_size / (df_env['bandwidth_mbps'] / 8 + 1e-8)
        df_env['cpu_to_size_ratio'] = df_env['cpu_limit'] / (img_size + 1e-8)
        df_env['mem_to_size_ratio'] = df_env['mem_limit_mb'] / (img_size + 1e-8)
        df_env['network_score'] = df_env['bandwidth_mbps'] / (df_env['network_rtt'] + 1e-8)
        
        return df_env, df_img

    def predict_single_config(
        self, 
        env_state: Dict[str, Any], 
        image_features: Dict[str, Any], 
        algo_name: str
    ) -> Tuple[float, float]:
        """
        预测单个特定配置的性能
        
        Args:
            env_state: 客户端环境特征
            image_features: 镜像特征
            algo_name: 压缩算法名称
            
        Returns:
            (预测耗时秒数, 不确定性分数秒数)
        """
        try:
            # 1. 构造特征
            df_env, df_img = self._construct_features(env_state, image_features)
            
            # 2. 提取特征 (严格按 cols_c/cols_i 顺序)
            Xc_raw = df_env[self.cols_c].values
            Xi_raw = df_img[self.cols_i].values
            
            # 3. 标准化
            Xc = self.scaler_c.transform(Xc_raw)
            Xi = self.scaler_i.transform(Xi_raw)
            
            # 4. 算法编码
            if algo_name in self.algo_list:
                ax = self.enc.transform([algo_name])[0]
            else:
                self.logger.warning(f"⚠️  算法 {algo_name} 未见过，使用默认算法 {self.algo_list[0]}")
                ax = 0
            
            # 5. 模型推理
            cx = torch.FloatTensor(Xc).to(self.device)
            ix = torch.FloatTensor(Xi).to(self.device)
            ax_tensor = torch.LongTensor([ax]).to(self.device)
            
            with torch.no_grad():
                preds = self.model(cx, ix, ax_tensor)
                gamma, v, alpha, beta = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
                
                # 转换回原始时间空间
                pred_time_log = gamma
                pred_time_ori = torch.expm1(pred_time_log).cpu().numpy()[0]
                
                # 计算不确定性 (std)
                var_log = beta / (v * (alpha - 1) + 1e-6)
                std_log = torch.sqrt(var_log + 1e-6)
                std_ori = (torch.exp(pred_time_log) * std_log).cpu().numpy()[0]
            
            return float(pred_time_ori), float(std_ori)
            
        except Exception as e:
            self.logger.error(f"❌ 预测失败: {e}", exc_info=True)
            return 100.0, 10.0  # 返回保守默认值

    def predict(
        self, 
        features: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        通用预测接口 (兼容现有代码)
        默认预测第一个算法
        """
        # 解析 features
        env_state = {k: features[k] for k in features if k in self.cols_c or k in [
            'bandwidth_mbps', 'network_rtt', 'cpu_limit', 'mem_limit_mb',
            'cpu_available', 'mem_available', 'packet_loss'
        ]}
        image_features = {k: features[k] for k in features if k in self.cols_i or k in ['total_size_mb']}
        
        # 默认使用第一个算法
        default_algo = self.algo_list[0]
        
        return self.predict_single_config(env_state, image_features, default_algo)

    # ==========================================
    # 【新增】批量预测接口 (性能优化)
    # ==========================================
    def batch_predict(
        self,
        env_state: Dict[str, Any],
        image_features: Dict[str, Any],
        algo_name_list: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量预测多个算法的性能
        
        Args:
            env_state: 客户端环境特征
            image_features: 镜像特征
            algo_name_list: 要预测的算法列表，None则预测所有算法
            
        Returns:
            预测结果列表
        """
        if algo_name_list is None:
            algo_name_list = self.algo_list
        
        results = []
        for algo_name in algo_name_list:
            pred_time, unc = self.predict_single_config(env_state, image_features, algo_name)
            results.append({
                'algo_name': algo_name,
                'pred_time_s': pred_time,
                'uncertainty_s': unc
            })
        
        return results