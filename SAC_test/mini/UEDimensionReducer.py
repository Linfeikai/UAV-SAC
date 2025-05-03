# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # 随机种子（保证可重复性）
# np.random.seed(42)

# # 生成20个UE的数据
# num_ue = 20
# ue_data = []

# for i in range(num_ue):
#     ue = {
#         "UE_ID": i+1,
#         "X": np.random.uniform(0, 1),          # 归一化坐标（0~1）
#         "Y": np.random.uniform(0, 1),
#         "Task_Compute": int(np.random.uniform(50, 500)),  # 任务计算量（MI）
#         "Delay_Req": np.random.uniform(1.0, 5.0),        # 延迟要求（秒）
#         "Channel_Quality": np.random.uniform(0.4, 0.95)  # 信道质量（0~1）
#     }
#     ue_data.append(ue)

# # 转换为DataFrame
# df_ue = pd.DataFrame(ue_data)
# print(df_ue.round(2))

# plt.figure(figsize=(8, 6))
# plt.scatter(df_ue["X"], df_ue["Y"], 
#             c=df_ue["Channel_Quality"], 
#             s=df_ue["Task_Compute"]/10,  # 点大小表示任务量
#             cmap='viridis')

# plt.colorbar(label='Channel Quality')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('UE Distribution: Task Size (Marker) vs Channel Quality (Color)')
# # plt.show()
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class UEDimensionReducer(nn.Module):
    def __init__(self, input_dim=5, embed_dim=64, output_dim=32):
        super().__init__()
        # Step 1: 嵌入层（每个UE的5维→64维）
        self.embed = nn.Linear(input_dim, embed_dim)
        
        # Step 2: Transformer Encoder（无位置编码，因UE顺序无关）
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,  # 注意力头数
                dim_feedforward=128,
                dropout=0.1
            ),
            num_layers=2  # 堆叠2层Encoder
        )
        
        # Step 3: 全局平均池化 + 投影到目标维度
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, ue_states):
        """
        输入: ue_states - [batch_size, num_ue=20, input_dim=5]
        输出: 压缩后的全局状态 [batch_size, output_dim=32]
        """
        # 嵌入
        embedded = self.embed(ue_states)  # [B,20,64]
        
        # Transformer处理需要调整维度为 [序列长度, batch_size, 嵌入维度]
        encoded = self.encoder(embedded.permute(1, 0, 2))  # [20,B,64]
        
        # 全局平均池化（按序列维度）
        global_state = encoded.mean(dim=0)  # [B,64]
        
        # 投影到最终维度
        return self.projection(global_state)  # [B,32]

# 测试示例
batch_size = 2
ue_states = torch.randn(batch_size, 20, 5)  # 随机输入
model = UEDimensionReducer()
output = model(ue_states)
print("输入维度:", ue_states.shape)  # [8,20,5]
print("输出维度:", output.shape)     # [8,32]
print("输出数据:", output)          # [8,32]