"""
压缩决策器（Compression Decision Engine）
目的：根据客户端和镜像的特征，选择最优的压缩算法和级别

决策公式：
总成本 = 压缩端耗时 + 传输耗时 + 客户端解压时间
"""


class CompressionDecisionEngine:
    """压缩决策引擎"""
    
    def __init__(self, strategy="cost_model"):
        """
        初始化压缩决策引擎
        
        Args:
            strategy: 决策策略，可选 "cost_model"（基于成本模型）或 "ml_model"（基于机器学习模型）
        """
        # 预定义的压缩算法选项
        self.compression_options = [
            "gzip-1", "gzip-6", "gzip-9",
            "zstd-1", "zstd-3", "zstd-5", "zstd-10", "zstd-19",
            "lz4-fast", "lz4-10", "lz4-12"
        ]
        
        self.strategy = strategy
        self.ml_model = None
        
        # 如果使用机器学习策略，加载模型
        if self.strategy == "ml_model":
            self._load_ml_model()
    
    def _load_ml_model(self):
        """
        加载机器学习模型
        这里预留接口，可以根据需要加载XGBoost、随机森林等模型
        """
        try:
            # 示例：加载XGBoost模型
            # import xgboost as xgb
            # self.ml_model = xgb.Booster(model_file='models/compression_decision_model.json')
            print("机器学习模型加载成功")
        except Exception as e:
            print(f"机器学习模型加载失败: {e}")
            # 回退到成本模型
            self.strategy = "cost_model"
    
    def calculate_cost(self, client_profile, image_profile, compression_method):
        """
        计算特定压缩方法的总成本
        
        Args:
            client_profile: 客户端配置
            image_profile: 镜像配置
            compression_method: 压缩方法
            
        Returns:
            cost: 总成本
            details: 成本明细
        """
        # 获取基础参数
        bandwidth_mbps = client_profile.get("bandwidth_mbps", 50)
        decompression_speed = client_profile.get("decompression_speed", 100)
        cpu_score = client_profile.get("cpu_score", 1000)
        
        total_size_mb = image_profile.get("total_size_mb", 100)
        entropy = image_profile.get("avg_layer_entropy", 0.5)
        text_ratio = image_profile.get("text_ratio", 0.2)
        binary_ratio = image_profile.get("binary_ratio", 0.7)
        
        # 根据压缩方法估算压缩比和压缩时间
        compression_ratio, compression_time_multiplier = self._get_compression_stats(compression_method, entropy, text_ratio)
        
        # 计算压缩后大小
        compressed_size_mb = total_size_mb * compression_ratio
        
        # 计算各项成本
        # 传输时间（秒）= 大小（Mb）* 8 / 带宽（Mbps）
        transmission_time = (compressed_size_mb * 8) / bandwidth_mbps
        
        # 压缩时间（与CPU性能和压缩算法相关）
        compression_time = total_size_mb * compression_time_multiplier * (1000 / cpu_score)
        
        # 解压时间（与客户端解压性能相关）
        decompression_time = (compressed_size_mb * 10) / decompression_speed  # 简化的计算模型
        
        # 总成本
        total_cost = compression_time + transmission_time + decompression_time
        
        details = {
            "compressed_size_mb": compressed_size_mb,
            "transmission_time": transmission_time,
            "compression_time": compression_time,
            "decompression_time": decompression_time
        }
        
        return total_cost, details
    
    def _get_compression_stats(self, compression_method, entropy, text_ratio):
        """
        根据压缩方法和数据特征估算压缩比和时间乘数
        
        Returns:
            compression_ratio: 压缩比（0-1，越小越好）
            time_multiplier: 时间乘数（越大表示越耗时）
        """
        # 基于经验和启发式规则的简单模型
        if compression_method.startswith("gzip"):
            level = int(compression_method.split("-")[1])
            # gzip压缩比模型：熵值越低，压缩效果越好
            ratio_base = 0.3 + 0.5 * (1 - entropy) + 0.2 * text_ratio
            compression_ratio = max(0.1, ratio_base * (1 - level * 0.05))
            time_multiplier = 1.0 + level * 0.5
            
        elif compression_method.startswith("zstd"):
            level = int(compression_method.split("-")[1]) if "-" in compression_method else 3
            # zstd压缩比模型
            ratio_base = 0.25 + 0.5 * (1 - entropy) + 0.25 * text_ratio
            compression_ratio = max(0.05, ratio_base * (1 - level * 0.03))
            time_multiplier = 1.0 + level * 0.3
            
        elif compression_method.startswith("lz4"):
            if "fast" in compression_method:
                compression_ratio = 0.6 + 0.2 * (1 - entropy)
                time_multiplier = 0.5
            else:
                level = int(compression_method.split("-")[1])
                ratio_base = 0.5 + 0.3 * (1 - entropy) + 0.2 * text_ratio
                compression_ratio = max(0.2, ratio_base * (1 - level * 0.02))
                time_multiplier = 0.8 + level * 0.1
        else:
            # 默认情况
            compression_ratio = 0.5
            time_multiplier = 1.0
            
        return compression_ratio, time_multiplier
    
    def _prepare_features(self, client_profile, image_profile, compression_method):
        """
        准备用于机器学习模型的特征向量
        
        Args:
            client_profile: 客户端配置
            image_profile: 镜像配置
            compression_method: 压缩方法
            
        Returns:
            features: 特征向量
        """
        # 提取客户端特征
        bandwidth_mbps = client_profile.get("bandwidth_mbps", 50)
        decompression_speed = client_profile.get("decompression_speed", 100)
        cpu_score = client_profile.get("cpu_score", 1000)
        
        # 提取镜像特征
        total_size_mb = image_profile.get("total_size_mb", 100)
        entropy = image_profile.get("avg_layer_entropy", 0.5)
        text_ratio = image_profile.get("text_ratio", 0.2)
        binary_ratio = image_profile.get("binary_ratio", 0.7)
        compressed_file_ratio = image_profile.get("compressed_file_ratio", 0.1)
        layer_count = image_profile.get("layer_count", 1)
        
        # 压缩方法特征编码
        method_features = [0] * len(self.compression_options)
        if compression_method in self.compression_options:
            method_features[self.compression_options.index(compression_method)] = 1
        
        # 构建特征向量
        features = [
            bandwidth_mbps / 1000.0,  # 归一化带宽
            decompression_speed / 1000.0,  # 归一化解压速度
            cpu_score / 10000.0,  # 归一化CPU得分
            total_size_mb / 1000.0,  # 归一化镜像大小
            entropy,
            text_ratio,
            binary_ratio,
            compressed_file_ratio,
            layer_count / 50.0,  # 归一化层数
        ] + method_features  # 添加压缩方法独热编码
        
        return features
    
    def _predict_with_ml_model(self, client_profile, image_profile):
        """
        使用机器学习模型进行预测
        
        Args:
            client_profile: 客户端配置
            image_profile: 镜像配置
            
        Returns:
            best_method: 最佳压缩方法
        """
        if self.ml_model is None:
            # 如果没有模型，回退到成本模型
            return self._make_decision_with_cost_model(client_profile, image_profile)
        
        best_score = float('-inf')
        best_method = None
        
        # 遍历所有压缩方法，预测得分
        for method in self.compression_options:
            features = self._prepare_features(client_profile, image_profile, method)
            # 示例：使用XGBoost模型预测
            # score = self.ml_model.predict(xgb.DMatrix([features]))[0]
            # 临时模拟预测得分
            import random
            score = random.random()
            
            if score > best_score:
                best_score = score
                best_method = method
                
        return best_method if best_method else "lz4-fast"
    
    def _make_decision_with_cost_model(self, client_profile, image_profile):
        """
        使用传统成本模型进行决策
        
        Args:
            client_profile: 客户端配置
            image_profile: 镜像配置
            
        Returns:
            decision: 决策结果
        """
        best_cost = float('inf')
        best_method = None
        best_details = None
        
        # 遍历所有压缩选项，找到成本最低的
        for method in self.compression_options:
            cost, details = self.calculate_cost(client_profile, image_profile, method)
            if cost < best_cost:
                best_cost = cost
                best_method = method
                best_details = details
        
        # 生成决策原因
        reason = self._generate_reason(client_profile, image_profile, best_method)
        
        return {
            "selected_compression": best_method,
            "estimated_total_cost": round(best_cost, 2),
            "expected_compressed_size": round(best_details["compressed_size_mb"], 2),
            "transmission_time": round(best_details["transmission_time"], 2),
            "compression_time": round(best_details["compression_time"], 2),
            "decompression_time": round(best_details["decompression_time"], 2),
            "reason": reason
        }
    
    def make_decision(self, client_profile, image_profile):
        """
        做出压缩决策
        
        Args:
            client_profile: 客户端配置
            image_profile: 镜像配置
            
        Returns:
            decision: 决策结果
        """
        if self.strategy == "ml_model":
            # 使用机器学习模型进行决策
            best_method = self._predict_with_ml_model(client_profile, image_profile)
            
            # 为了保持接口一致性，仍然计算成本明细
            cost, details = self.calculate_cost(client_profile, image_profile, best_method)
            
            return {
                "selected_compression": best_method,
                "estimated_total_cost": round(cost, 2),
                "expected_compressed_size": round(details["compressed_size_mb"], 2),
                "transmission_time": round(details["transmission_time"], 2),
                "compression_time": round(details["compression_time"], 2),
                "decompression_time": round(details["decompression_time"], 2),
                "reason": "基于机器学习模型的决策"
            }
        else:
            # 使用传统的成本模型进行决策
            return self._make_decision_with_cost_model(client_profile, image_profile)
    
    def _generate_reason(self, client_profile, image_profile, selected_method):
        """生成选择该压缩方法的原因"""
        reasons = []
        
        bandwidth = client_profile.get("bandwidth_mbps", 0)
        decompression_speed = client_profile.get("decompression_speed", 0)
        entropy = image_profile.get("avg_layer_entropy", 0)
        text_ratio = image_profile.get("text_ratio", 0)
        binary_ratio = image_profile.get("binary_ratio", 0)
        
        if "gzip" in selected_method and text_ratio > 0.3:
            reasons.append("文本文件比例高，gzip压缩效果好")
        elif "zstd" in selected_method and binary_ratio > 0.5:
            reasons.append("二进制文件比例高，zstd更适合")
        elif "lz4" in selected_method and bandwidth < 30:
            reasons.append("网络带宽较低，选择快速压缩算法")
        elif "lz4" in selected_method and entropy > 0.7:
            reasons.append("数据熵值高难压缩，选择快速算法")
        
        if not reasons:
            reasons.append("基于综合成本考量选择了此算法")
            
        return "; ".join(reasons)


def main():
    """测试压缩决策引擎"""
    # 测试传统成本模型
    print("测试传统成本模型:")
    engine_cost = CompressionDecisionEngine(strategy="cost_model")
    
    # 模拟客户端和镜像配置
    client_profile = {
        "cpu_score": 3100,
        "bandwidth_mbps": 68,
        "decompression_speed": 280,
        "latency_requirement": 400
    }
    
    image_profile = {
        "layer_count": 6,
        "total_size_mb": 742,
        "avg_layer_entropy": 0.78,
        "text_ratio": 0.18,
        "binary_ratio": 0.82,
        "compressed_file_ratio": 0.12,
        "avg_file_size_kb": 42.3,
        "median_file_size_kb": 8.1,
        "small_file_ratio": 0.67,
        "big_file_ratio": 0.15,
        "predict_popularity": 0.54
    }
    
    decision = engine_cost.make_decision(client_profile, image_profile)
    print("压缩决策结果:")
    for key, value in decision.items():
        print(f"  {key}: {value}")
    
    print("\n测试机器学习模型（模拟）:")
    # 测试机器学习模型（模拟）
    engine_ml = CompressionDecisionEngine(strategy="ml_model")
    decision_ml = engine_ml.make_decision(client_profile, image_profile)
    print("压缩决策结果:")
    for key, value in decision_ml.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()