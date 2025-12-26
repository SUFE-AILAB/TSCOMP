import numpy as np
import torch
import torch.nn as nn
from torch import optim

from einops import rearrange
from transformers import GPT2Config, GPT2Model
class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_len = configs.patch_len
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_len) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        
        if configs.is_gpt:
            self.gpt2_config = GPT2Config.from_pretrained('./models/llm/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.gpt2 = GPT2Model.from_pretrained(
                './models/llm/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )
            self.gpt2.h = self.gpt2.h[:configs.llm_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_len, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.frozen and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # for layer in (self.gpt2, self.in_layer, self.out_layer):
        #     layer.to(device=configs.device)
        #     layer.train()
        
        self.cnt = 0


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs