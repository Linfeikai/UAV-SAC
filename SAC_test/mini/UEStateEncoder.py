import torch.nn as nn

class UEStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
            )
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=2)

    def forward(self, x):
        # x: [batch, 20, 9]
        batch_size = x.shape[0]
        ue_emb = self.encoder(x)  # [batch,20,16]
        
        # 注意力聚合
        ue_emb = ue_emb.transpose(0,1)  # [20,batch,16]
        attn_out, _ = self.attention(
            query=ue_emb.mean(0, keepdim=True),
            key=ue_emb,
            value=ue_emb
        )
        return attn_out.squeeze(0)  # [batch,16]