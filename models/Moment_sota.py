# https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb

import torch
from torch import nn
from momentfm import MOMENTPipeline

# todo: https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb
# 注意原始Moment代码里面有mixed precision training, scheduler, gradient clipping
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        self.model = MOMENTPipeline.from_pretrained(
            f"./models/llm/MOMENT-base", 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': self.pred_len,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': configs.frozen,
                'freeze_embedder': configs.frozen,
                'freeze_head': False,
            },
            # local_files_only=True
        )
        self.model.init()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        # MOMENT expects: [Batch, Channels, Seq_Len]
        
        x = x_enc.permute(0, 2, 1) # [B, C, S]
        B, C, S = x.shape
        
        # MOMENT context length is usually 512.
        if S < 512:
            pad_len = 512 - S
            padding = torch.zeros(B, C, pad_len).to(x.device)
            x = torch.cat((x, padding), dim=-1)
            
            input_mask = torch.ones(B, 512).to(x.device)
            input_mask[:, S:] = 0
        else:
            if S > 512:
                x = x[:, :, -512:]
            input_mask = torch.ones(B, 512).to(x.device)
            
        output = self.model(x_enc=x, input_mask=input_mask)
        
        # Output forecast: [Batch, Channels, Horizon]
        dec_out = output.forecast
        
        # Return [Batch, Horizon, Channels]
        dec_out = dec_out.permute(0, 2, 1)
        
        return dec_out