import torch
from torch import nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)

class xLSTMBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,               # 输入和输出的维度 (D)
        num_layers: int = 2,        # xLSTM 块的层数
        num_heads: int = 4,         # 注意力头数 (用于 sLSTM 和 mLSTM)
        max_seq_len: int = 1024,    # 最大上下文长度 (xLSTM 需要预设 Context Length)
        dropout: float = 0.1,
        use_mlstm: bool = True,     # 是否混合使用 mLSTM (矩阵记忆块)
        bidirectional: bool = True # 是否双向 (Backbone 常用配置)
    ):
        """
        xLSTM Backbone: 
        输入 (B, S, D) -> xLSTM Blocks -> 输出 (B, S, D)
        """
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        # --- 1. 配置 sLSTM (Scalar LSTM) ---
        # sLSTM 擅长序列控制和局部特征提取
        slstm_cfg = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=num_heads,
                conv1d_kernel_size=4,  # 局部卷积核大小，通常设为 4
            )
        )

        # --- 2. 配置 mLSTM (Matrix LSTM) ---
        # mLSTM 类似于 Transformer 的 Attention，擅长全局记忆
        # 如果 use_mlstm=False，则纯用 sLSTM
        mlstm_cfg = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                num_heads=num_heads,
                embedding_dim=d_model, 
            )
        ) if use_mlstm else None

        # --- 3. 配置 Block Stack ---
        # slstm_at="all" 表示每一层都包含 sLSTM，mLSTM 会根据配置穿插
        cfg = xLSTMBlockStackConfig(
            slstm_block=slstm_cfg,
            mlstm_block=mlstm_cfg,
            num_blocks=num_layers,
            embedding_dim=d_model,
            dropout=dropout,
            slstm_at="all", 
            context_length=max_seq_len, # 必须指定，用于内部优化
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)

        # --- 4. 双向处理的融合层 (可选) ---
        # 如果是双向，拼接后的维度是 2*D，需要映射回 D
        if self.bidirectional:
            self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, SeqLen, D_Model)
        Returns:
            out: Output tensor of shape (Batch, SeqLen, D_Model)
        """
        # 1. 正向传播 (Causal Forward)
        # xLSTM 本身是因果的 (只能看过去)
        out_fwd = self.xlstm_stack(x)

        if not self.bidirectional:
            return out_fwd, None

        # 2. 双向逻辑 (Bidirectional Logic - 如果需要)
        # 翻转序列 -> 通过 xLSTM -> 翻转回来
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.xlstm_stack(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])

        # 3. 拼接与融合
        # (B, S, D) + (B, S, D) -> (B, S, 2D)
        combined = torch.cat([out_fwd, out_rev], dim=-1)
        
        # (B, S, 2D) -> (B, S, D)
        out = self.output_proj(combined)
        

        return out, None