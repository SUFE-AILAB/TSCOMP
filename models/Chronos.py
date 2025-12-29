import torch
import torch.nn as nn
from chronos import Chronos2Pipeline

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 加载 Chronos-Bolt 模型
        # 常见 ID: 'amazon/chronos-bolt-tiny', 'amazon/chronos-bolt-small', 'amazon/chronos-bolt-base'
        # 使用 Pipeline 加载，它会自动处理 Tokenization 和各个 Bolt 变体的差异
        # torch_dtype=torch.bfloat16 是 Bolt 推荐的精度
        self.pipeline = Chronos2Pipeline.from_pretrained(
            'amazon/chronos-t5-base',
            device_map=self.device,
            torch_dtype=torch.bfloat16, 
        )

        # 2. 冻结模型参数
        # Chronos 是拿来即用的 Zero-shot 模型，通常不需要微调内部参数
        if configs.frozen and configs.pretrain:
            for param in self.pipeline.model.parameters():
                param.requires_grad = False

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x: [Batch, Input_Len, N_Vars]
        """
        B, L, M = x.shape

        # --- 维度变换: Multivariate -> Univariate Batch ---
        # TSlib 是 [Batch, Seq_Len, N_Vars]
        # Chronos 视每个变量为独立序列，需要变成 [Batch * N_Vars, Seq_Len]
        # [B, L, M] -> [B, M, L] -> [B*M, L]
        context_tensor = x.permute(0, 2, 1).reshape(B * M, L)
        
        # 确保数据类型匹配 (bfloat16)
        # context_tensor = context_tensor.to(dtype=torch.bfloat16) 

        # --- 3. 推理 (Inference) ---
        # ChronosPipeline 接受 tensor 输入
        # limit_prediction_length=False 允许预测任意长度
        # num_samples=1 表示我们取确定性结果（通常是中位数路径）
        forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=self.pred_len,
            num_samples=1, 
            limit_prediction_length=False
        )
        # forecast shape: [B*M, num_samples, pred_len] -> [B*M, 1, pred_len]
        
        # --- 4. 维度还原 ---
        # [B*M, 1, pred_len] -> [B*M, pred_len]
        forecast = forecast.squeeze(1)
        
        # [B*M, pred_len] -> [B, M, pred_len] -> [B, pred_len, M]
        forecast = forecast.reshape(B, M, self.pred_len).permute(0, 2, 1)

        return forecast # [Batch, Pred_Len, N_Vars]