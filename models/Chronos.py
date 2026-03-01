import torch
import torch.nn as nn
from chronos import Chronos2Model
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 加载 Chronos-2 模型
        self.model = Chronos2Model.from_pretrained(
            './models/llm/Chronos',
            device_map="cpu",
        )
        # 获取补丁大小以计算所需的补丁数量
        self.patch_size = self.model.chronos_config.output_patch_size

        # 2. 冻结模型参数
        if configs.frozen and configs.pretrain:
            for param in self.model.parameters():
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
        
        # 计算需要的输出 patch 数量
        num_output_patches = math.ceil(self.pred_len / self.patch_size)

        # --- 3. 推理 (Inference) ---
        # 直接调用 Chronos2Model 的 forward
        # num_samples (samples/quantiles) 取决于模型配置，通常包含中位数
        model_output = self.model(
            context=context_tensor,
            num_output_patches=num_output_patches,
        )
        
        # quantile_preds shape: [B*M, num_quantiles, horizon]
        # output horizon 可能比 pred_len 长 (因为是 patch 的倍数)
        forecast = model_output.quantile_preds
        
        #为了确定性预测，我们通常取中位数。
        # 假设配置中的分位数包含 0.5。如果没有，取中间的。
        # Chronos2 默认 quantiles 可能是 [0.1, 0.2, ..., 0.9]
        # 我们取中间的作为点预测
        median_idx = forecast.shape[1] // 2
        forecast = forecast[:, median_idx, :self.pred_len] 
        
        # --- 4. 维度还原 ---
        # [B*M, pred_len] -> [B, M, pred_len] -> [B, pred_len, M]
        forecast = forecast.reshape(B, M, self.pred_len).permute(0, 2, 1)

        return forecast # [Batch, Pred_Len, N_Vars]