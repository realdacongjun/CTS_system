import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# CTS 通用模型定义 (解决 DRY 原则问题)
# ==============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_features, embed_dim))
        # 使用 Xavier 初始化保证数值稳定性
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights + self.biases

class TransformerTower(nn.Module):
    def __init__(self, num_features, embed_dim, nhead=4, num_layers=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        tokens = self.tokenizer(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        out = self.transformer(tokens)
        return out[:, 0, :]

class CTSDualTowerModel(nn.Module):
    def __init__(self, client_feats, image_feats, num_algos, embed_dim=32):
        super().__init__()
        self.client_tower = TransformerTower(client_feats, embed_dim)
        self.image_tower = TransformerTower(image_feats, embed_dim)
        self.algo_embed = nn.Embedding(num_algos, embed_dim)
        
        fusion_input_dim = embed_dim * 3 
        self.hidden = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 输出层：Gamma, v, Alpha, Beta
        self.head = nn.Linear(64, 4) 

    def forward(self, cx, ix, ax):
        c_vec = self.client_tower(cx)
        i_vec = self.image_tower(ix)
        a_vec = self.algo_embed(ax)
        combined = torch.cat([c_vec, i_vec, a_vec], dim=1)
        hidden = self.hidden(combined)
        out = self.head(hidden)
        
        # 施加数学约束 (Softplus) 保证参数 > 0
        gamma = out[:, 0]
        v     = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        
        return torch.stack([gamma, v, alpha, beta], dim=1)