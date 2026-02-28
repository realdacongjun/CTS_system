#!/usr/bin/env python3
import asyncio
import logging
import sys
import os
import pandas as pd
from pathlib import Path
from aiohttp import web

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

class CTSServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()
        self.engine = None
        self.wrapper = None
        self.image_features_df = None
        self._load_cts_engine()
        self._load_image_features()

    def _complete_env_features(self, env_state: dict, image_features: dict):
        """🔧 新增：补全物理交叉特征（必须与训练代码完全一致）"""
        img_size = image_features.get('total_size_mb', 100.0)
        env_state = env_state.copy()
        
        # 确保基础字段存在
        env_state.setdefault('bandwidth_mbps', 100.0)
        env_state.setdefault('network_rtt', 10.0)
        env_state.setdefault('cpu_limit', 4.0)
        env_state.setdefault('mem_limit_mb', 8192.0)
        
        # 计算物理特征
        env_state['theoretical_time'] = img_size / (env_state['bandwidth_mbps'] / 8 + 1e-8)
        env_state['cpu_to_size_ratio'] = env_state['cpu_limit'] / (img_size + 1e-8)
        env_state['mem_to_size_ratio'] = env_state['mem_limit_mb'] / (img_size + 1e-8)
        env_state['network_score'] = env_state['bandwidth_mbps'] / (env_state['network_rtt'] + 1e-8)
        
        return env_state

    def _load_cts_engine(self):
        try:
            from model_wrapper import CFTNetWrapper
            from cags_decision import CAGSDecisionEngine
            
            # 🔧 修改1：简化配置加载，直接指向 model_config.yaml
            # 不再依赖旧的 config.yaml 嵌套结构
            model_cfg_path = PROJECT_ROOT / "configs" / "model_config.yaml"
            
            if not model_cfg_path.exists():
                logger.error(f"❌ 模型配置不存在: {model_cfg_path}")
                return

            self.wrapper = CFTNetWrapper(str(model_cfg_path))
            self.engine = CAGSDecisionEngine(
                model=self.wrapper.model, 
                scaler_c=self.wrapper.scaler_c,
                scaler_i=self.wrapper.scaler_i, 
                enc=self.wrapper.enc,
                cols_c=self.wrapper.cols_c, 
                cols_i=self.wrapper.cols_i, 
                device=self.wrapper.device
            )
            logger.info("✅ CTS 决策引擎加载完成")
        except Exception as e:
            logger.error(f"❌ 引擎加载失败: {e}", exc_info=True)

    def _load_image_features(self):
        try:
            path = PROJECT_ROOT / "data" / "image_features_database.csv"
            if path.exists():
                self.image_features_df = pd.read_csv(path)
                logger.info(f"✅ 镜像特征库加载完成: {len(self.image_features_df)} 条")
        except Exception as e:
            logger.warning(f"⚠️  镜像特征库加载失败: {e}")

    def _get_image_features(self, image_name):
        if self.image_features_df is None:
            return {'total_size_mb': 200.0} # 更合理的默认值
        base = image_name.split(':')[0]
        # 容错：大小写不敏感
        match = self.image_features_df[self.image_features_df['image_name'].str.contains(base, na=False, case=False)]
        return match.iloc[0].to_dict() if not match.empty else {'total_size_mb': 200.0}

    def _setup_routes(self):
        self.app.router.add_get('/health', self.health)
        self.app.router.add_post('/strategy', self.get_strategy)

    async def health(self, req):
        return web.json_response({"status": "healthy", "engine_loaded": self.engine is not None})

    async def get_strategy(self, request):
        try:
            data = await request.json()
            image = data['image']
            env = data.get('environment', {})
            
            # 1. 获取镜像特征
            img_feat = self._get_image_features(image)
            
            # 2. 🔧 关键：补全物理特征
            full_env = self._complete_env_features(env, img_feat)
            
            if self.engine:
                cfg, metrics, _ = self.engine.make_decision(
                    env_state=full_env, 
                    image_features=img_feat
                )
                strategy = {
                    "algorithm": cfg['algo_name'],
                    "threads": cfg['threads'],
                    "chunk_size_kb": cfg.get('chunk_size_kb', 256),
                    "cpu_quota": cfg.get('cpu_quota', 1.0),
                    "predicted_time_s": metrics.get('pred_time_s', 0.0), # 🔧 新增：返回预测时间供调用方参考
                    "uncertainty": metrics.get('uncertainty_s', 0.0)
                }
                logger.info(f"🤖 决策: {image} -> {strategy['algorithm']} @ {strategy['threads']} threads")
            else:
                strategy = {"algorithm": "zstd_l3", "threads": 4, "chunk_size_kb": 256, "cpu_quota": 1.0}
                logger.warning(f"⚠️  引擎未加载，返回默认策略")
            
            return web.json_response({"strategy": strategy})
        except Exception as e:
            logger.error(f"❌ 决策请求失败: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"🚀 CTS Server 运行在 http://{self.host}:{self.port}")
        while True: 
            await asyncio.sleep(3600)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(CTSServer().start())