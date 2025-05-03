import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 假设每个UE的特征维度： [x坐标, y坐标, 任务计算量, 延迟要求, 信道质量]
FEATURE_DIM = 5  
EMBED_DIM = 128  # Transformer嵌入维度
NHEAD = 8        # 注意力头数
NUM_LAYERS = 3   # Transformer编码器层数

class UETransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征嵌入层（将原始特征映射到高维空间）
        self.feature_embed = nn.Linear(FEATURE_DIM, EMBED_DIM)
        
        # 可学习的位置编码（与Transformer原版不同，此处无需预设最大长度）
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
        
        # Transformer编码器层
        encoder_layers = TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=NHEAD,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, NUM_LAYERS)
        
        # 聚合层（将多个UE的嵌入聚合成全局状态）
        self.global_pool = nn.Linear(EMBED_DIM, EMBED_DIM)
        
    def forward(self, ue_list):
        """
        输入: ue_list - 形状为 [batch_size, num_ue, FEATURE_DIM] 的Tensor
        输出: 聚合后的全局状态 [batch_size, EMBED_DIM]
        """
        # 特征嵌入
        embedded = self.feature_embed(ue_list)  # [B, N, D]
        
        # 添加位置编码（可学习方式，与输入无关）
        B, N, D = embedded.shape
        embedded += self.pos_encoder.repeat(B, N, 1)
        
        # 调整维度顺序: [N, B, D] （Transformer需要的格式）
        embedded = embedded.permute(1, 0, 2)
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(embedded)  # [N, B, D]
        
        # 全局平均池化 + 非线性变换
        pooled = encoded.mean(dim=0)                  # [B, D]
        global_state = torch.tanh(self.global_pool(pooled))
        
        return global_state

# ----------------- 使用示例 ------------------
if __name__ == "__main__":
    # 假设一个批次的UE列表（batch_size=2，第一个样本有3个UE，第二个有2个UE）
    ue_list = [
        # 样本1: 3个UE
        [
            [0.1, 0.2, 100, 2.0, 0.8],  # UE1特征
            [0.3, 0.5, 200, 1.5, 0.6],  # UE2特征
            [0.7, 0.9, 150, 3.0, 0.7]   # UE3特征
        ],
        # 样本2: 2个UE
        [
            [0.4, 0.3, 80, 2.5, 0.9],
            [0.6, 0.1, 300, 2.0, 0.5]
        ]
    ]
    
    # 转换为不等长Tensor（实际使用时需用DataLoader处理）
    ue_tensors = [torch.tensor(u, dtype=torch.float32) for u in ue_list]
    padded_ue = nn.utils.rnn.pad_sequence(ue_tensors, batch_first=True)  # [B, max_N, D]
    
    # 初始化模型
    model = UETransformerEncoder()
    
    # 前向传播
    global_state = model(padded_ue)  # 输出形状: [2, 128]
    print("聚合后的全局状态:", global_state.shape)  
    print(global_state)