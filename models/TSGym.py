import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Retrieval import TSGymRetrievalFusionBlock, TSGymRetrievalTool
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Encoder_ori, LinearEncoder
from layers.SelfAttention_Family import FullAttention, ProbAttention, DSAttention, FourierCrossAttention, AutoCorrelation
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding, DataEmbedding_inverted, PatchEmbedding_wo_pos, PatchEmbedding, PatchEmbedding_wo_pos_CD, PatchEmbedding_CD
from layers.StandardNorm import Normalize, DishTS
from layers.SeriesDecom import series_decomp, series_decomp_multi, DFT_series_decomp
from layers.xlstm_backbone import xLSTMBackbone
import numpy as np
from copy import deepcopy
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoModelForCausalLM
from models import TimeLLM, Moment
import warnings
warnings.filterwarnings("ignore")

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PatchingHead_CD(nn.Module):
    def __init__(self, d_model, n_vars, pred_len, patch_num, patch_len, head_dropout=0):
        super().__init__()
        # B, patch_num, d_model -> B, patch_num, C*patch_len -> B, NP, C, PL -> B, C, NP, PL -> B, C, NP*PL -> B,C,predlen
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.patch_num = patch_num
        self.pred_len = pred_len
        self.linear1 = nn.Linear(d_model, n_vars * patch_len)
        self.linear2 = nn.Linear(patch_num * patch_len, pred_len)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):  # x: [bs x patch_num x d_model]
        assert self.patch_num == x.shape[1]
        x = self.linear1(x)  # B, patch_num, C*patch_len
        x = self.dropout(x)
        x = x.view(x.shape[0], self.patch_num, self.n_vars, self.patch_len)  # B, patch_num, C, patch_len
        x = x.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)  # B, C, patch_num, patch_len # B, C, NP*PL 
        x = self.linear2(x)  # B,C,predlen
        return x

def calculate_patch_num(seq_len, patch_len=8):
    # 原本stride=1, 例如seq_len=60会产生60-patch_len+1个patches, 高度overlapped
    # 现在stride=patch_len, 产生的是non-overlapped patches
    stride = patch_len
    padding = patch_len - seq_len % patch_len if seq_len % patch_len != 0 else 0
    patch_num = int((seq_len + padding - patch_len) / stride + 1)
    return stride, padding, patch_num, patch_len

# todo: encoder-decoder architecture
# todo: non-stationary
class DNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(configs.d_model, configs.d_model),
                                nn.GELU(),
                                nn.Linear(configs.d_model, configs.d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        x = self.fc(x)
        return x, None
    
class GRU(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder = nn.GRU(input_size=configs.d_model,
                              hidden_size=configs.d_model,
                              num_layers=configs.e_layers,
                              batch_first=True)

    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        x, _ = self.encoder(x)
        return x, None
    
class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y

# GPT4TS, TimeLLM
class LLM(nn.Module):
    def __init__(self, configs,
                 network_architecture,
                 frozen=True, gpt_layers=6):
        super().__init__()
        self.network_architecture = network_architecture

        if network_architecture == 'LLM-GPT4TS':
            self.encoder = GPT2Model.from_pretrained('./models/llm/gpt2', output_attentions=False, output_hidden_states=True)
            self.encoder.h = self.encoder.h[:gpt_layers]
            if frozen:
                for i, (name, param) in enumerate(self.encoder.named_parameters()):
                    if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                        param.requires_grad = True
                    elif 'mlp' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                pass
            self.proj = nn.Linear(768, configs.d_model) # todo: 截断还是linear projection?
        elif network_architecture == 'LLM-TimeLLM':
            self.encoder = TimeLLM.Model(configs, frozen=frozen)
        else:
            raise NotImplementedError
    
    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        if self.network_architecture == 'LLM-GPT4TS':
            # padding zero on the last dimension, BxTx768
            x = torch.nn.functional.pad(x, (0, 768 - x.shape[-1]))
            x = self.encoder(inputs_embeds=x).last_hidden_state
            x = self.proj(x)
        elif self.network_architecture == 'LLM-TimeLLM':
            x = self.encoder(x)
        else:
            raise NotImplementedError
        return x, None

# Timer, Moment
class TSFM(nn.Module):
    def __init__(self, configs,
                 network_architecture,
                 frozen=True):
        super().__init__()
        self.network_architecture = network_architecture

        if network_architecture == 'TSFM-Timer':
            # Timer里面的decoder实际上是encoder
            self.encoder = Encoder(attn_layers=[EncoderLayer(AttentionLayer(FullAttention(configs=configs,
                                                                                        mask_flag=False,
                                                                                        factor=configs.factor,
                                                                                        attention_dropout=configs.dropout, 
                                                                                        output_attention=False),
                                                                            configs.d_model, configs.n_heads),
                                                            configs.d_model,
                                                            configs.d_ff,
                                                            dropout=configs.dropout,
                                                            activation=configs.activation) for _ in range(configs.e_layers)],
                                norm_layer=torch.nn.LayerNorm(configs.d_model))
            sd = torch.load('./models/llm/timer/Timer_forecast_1.0.ckpt', map_location="cpu", weights_only=False)["state_dict"]
            for k in sd.keys():
                print(k)
            sd = {k[6:]: v for k, v in sd.items() if 'decoder' in k}
            self.encoder.load_state_dict(sd, strict=False)
            if frozen:
                # frozen attention weights
                for name, param in self.encoder.named_parameters():
                    # 只finetune attention layer中的layer norm仿射变换的参数
                    if 'attn_layers' in name and 'norm' not in name:
                        param.requires_grad = False
            else:
                pass
        elif network_architecture == 'TSFM-TimerXL':
            raise NotImplementedError
        elif network_architecture == 'TSFM-Moment':
            self.encoder = Moment.Model(frozen=frozen)
            self.proj = nn.Linear(768, configs.d_model)
        elif network_architecture == 'TSFM-TimeMoE':
            self.encoder = AutoModelForCausalLM.from_pretrained(
                './models/llm/Time-MoE',
                device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
                use_cache=False
            )
            self.proj = nn.Linear(768, configs.d_model)
        elif network_architecture == 'TSFM-Chronos':
            from chronos import Chronos2Model
            self.encoder = Chronos2Model.from_pretrained(
                "./models/llm/Chronos",
                device_map="cpu"
            )
            self.proj = nn.Linear(768, configs.d_model)
        else:
            raise NotImplementedError

    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        if self.network_architecture == 'TSFM-Moment':
            x = torch.nn.functional.pad(x, (0, 768 - x.shape[-1])) # padding to 768
            x, _ = self.encoder(x)
            x = self.proj(x) # 768 -> d_model
        elif self.network_architecture == 'TSFM-Timer':
            x, _ = self.encoder(x)
        elif self.network_architecture == 'TSFM-TimeMoE':
            x = torch.nn.functional.pad(x, (0, 768 - x.shape[-1])) # padding to 768
            x = self.encoder(inputs_embeds=x, output_hidden_states=True).hidden_states[-1]
            x = self.proj(x) # 768 -> d_model
        elif self.network_architecture == 'TSFM-Chronos':
            x = torch.nn.functional.pad(x, (0, 768 - x.shape[-1])) # padding to 768
            x = self.encoder.encoder(inputs_embeds=x, group_ids=torch.arange(x.shape[0]).to(x.device), output_attentions=False).last_hidden_state
            x = self.proj(x) # 768 -> d_model
        else:
            raise NotImplementedError
        return x, None

# Update 20251109 cc: 重构模型框架，将各个模块解耦方便后续扩展
# Update 20251226 ls:增加rag组件
# Update 20251218 cc: 加入回归任务
def get_q_mat_path(seqlen, predlen, dataset_name):
    ratio = 0.6 if dataset_name in ['ETTh1','ETTh2','ETTm1','ETTm2'] else 0.7 
    q_mat_path = f"{dataset_name}_{seqlen}_ratio{ratio}.npy"
    q_out_mat_path = f"{dataset_name}_{predlen}_ratio{ratio}.npy"
    return q_mat_path, q_out_mat_path

class Model(nn.Module):
    def __init__(self, configs,
                 gym_x_mark=True,
                 gym_series_sampling=False,
                 gym_series_norm=None,
                 gym_series_decomp=None,
                 gym_channel_independent=False,
                 gym_input_embed='series-encoding',
                 gym_network_architecture='Transformer',
                 gym_attn='self-attention',
                 gym_feature_attn='self-attention',
                 gym_encoder_only=True,
                 gym_frozen=True,
                 gym_rag=False
                 ):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.configs = configs

        self.gym_x_mark = eval(gym_x_mark) if isinstance(gym_x_mark, str) else gym_x_mark
        self.gym_series_sampling = eval(gym_series_sampling) if isinstance(gym_series_sampling, str) else gym_series_sampling
        self.gym_series_norm = gym_series_norm
        self.gym_series_decomp = gym_series_decomp
        self.gym_channel_independent = eval(gym_channel_independent) if isinstance(gym_channel_independent, str) else gym_channel_independent
        self.gym_input_embed = gym_input_embed
        self.gym_network_architecture = gym_network_architecture
        self.gym_attn = gym_attn
        self.gym_feature_attn = gym_feature_attn
        self.gym_encoder_only = eval(gym_encoder_only) if isinstance(gym_encoder_only, str) else gym_encoder_only
        self.gym_frozen = eval(gym_frozen) if isinstance(gym_frozen, str) else gym_frozen
        self.gym_rag = eval(gym_rag) if isinstance(gym_rag, str) else gym_rag
        # Olinear Q-Mat
        if not self.gym_series_sampling:
            self.q_mat_dir, self.q_out_mat_dir = get_q_mat_path(self.seq_len, self.pred_len, configs.data_path.split('.')[0])
            self.q_mat_dir, self.q_out_mat_dir = os.path.join(configs.root_path, self.q_mat_dir), os.path.join(configs.root_path, self.q_out_mat_dir)
        else:
            _, self.q_out_mat_dir = get_q_mat_path(self.seq_len, self.pred_len, configs.data_path.split('.')[0])
            self.q_out_mat_dir = os.path.join(configs.root_path, self.q_out_mat_dir)
            self.q_mat_dir = []
            self.sampling_seqlen = []
            for _seqlen_ratio in [1,2,4,8]:
                _seqlen = self.seq_len // _seqlen_ratio
                self.sampling_seqlen.append(_seqlen)
                _qmat_path, _ = get_q_mat_path(_seqlen, self.pred_len, configs.data_path.split('.')[0])
                self.q_mat_dir.append(os.path.join(configs.root_path, _qmat_path))


        # build model
        self._build_series_preprocessing()
        self._build_series_tokenization()
        self._build_multi_granularity_interation()
        self._build_feature_attention()
        self._build_backbone()
        self._build_outputhead()

    def _build_series_preprocessing(self):
        # Series Preprocessing Modules: Series Sampling, Normalization, Decomposition
        # Series Sampling (multi-granularity)
        if self.gym_series_sampling:
            self.series_sampling = self.multi_scale_process_inputs
        else:
            self.series_sampling = None

        # Series Normalization
        if self.gym_series_norm == 'None':
            self.series_norm = Normalize(self.configs.enc_in, affine=False, non_norm=True)
        elif self.gym_series_norm == 'Stat':
            self.series_norm = Normalize(self.configs.enc_in, affine=False, non_norm=False)
        elif self.gym_series_norm == 'RevIN':
            self.series_norm = Normalize(self.configs.enc_in, affine=True, non_norm=False)
        elif self.gym_series_norm == 'DishTS':
            self.series_norm = DishTS(self.configs)
        else:
            raise NotImplementedError
        
        if self.series_sampling: # series normalization有可学习参数
            if self.gym_series_norm == 'DishTS':
                self.series_norm = nn.ModuleList([DishTS(self.configs, seq_len=self.seq_len // (self.configs.down_sampling_window ** i)) 
                                                  for i in range(self.configs.down_sampling_layers + 1)])
            else:
                self.series_norm = nn.ModuleList(deepcopy(self.series_norm) 
                                                 for i in range(self.configs.down_sampling_layers + 1))

        # Series Decomposition
        if self.gym_series_decomp == 'None':
            self.series_decompsition = None
        elif self.gym_series_decomp == 'MA':
            self.series_decompsition = series_decomp(self.configs.moving_avg)
        elif self.gym_series_decomp == 'MoEMA':
            self.series_decompsition = series_decomp_multi([15, 25, 35])
        elif self.gym_series_decomp == 'DFT':
            self.series_decompsition = DFT_series_decomp()
        else:
            raise NotImplementedError
        
    def _build_series_tokenization(self):
        # Series Tokenization
        # channel-independent: no-patching: BxSxD -> (BxD)xSx1; patching: BxSxD -> (BxD)xS -> (BxD) x patch_num x patch_len
        if self.gym_channel_independent: # channel-independent
            if self.gym_input_embed == 'series-encoding': # CI + series-encoding
                if self.gym_network_architecture in ['GRU', 'MLP']:
                    self.enc_embedding = DataEmbedding_wo_pos(1, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else DataEmbedding_wo_pos(
                        1, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout) # encoder only时为None节省显存
                else:
                    self.enc_embedding = DataEmbedding(1, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else DataEmbedding(
                        1, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout) # encoder only时为None节省显存
                self.enc_embedding_S = self.seq_len # enc_in tensor 的三元组的中间维度
                
                if self.series_sampling: # CI + series-encoding + series-sampling
                    self.enc_embedding = nn.ModuleList(deepcopy(self.enc_embedding) for i in range(self.configs.down_sampling_layers + 1))
                    self.dec_embedding = None if self.gym_encoder_only else nn.ModuleList(deepcopy(self.dec_embedding) 
                                                                                          for i in range(self.configs.down_sampling_layers + 1))
                    self.enc_embedding_S = self.sampling_seqlen # enc_in tensor 的三元组的中间维度
            elif self.gym_input_embed == 'series-patching': # CI + series-patching
                stride, padding, self.patch_num, patch_len = calculate_patch_num(self.configs.seq_len)
                if self.gym_network_architecture in ['GRU','MLP']:
                    self.enc_embedding = PatchEmbedding_wo_pos(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                               padding=padding, dropout=self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else PatchEmbedding_wo_pos(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                               padding=padding, dropout=self.configs.dropout)
                else:
                    self.enc_embedding = PatchEmbedding(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                        padding=padding, dropout=self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else PatchEmbedding(d_model=self.configs.d_model, patch_len=patch_len, 
                                                                                           stride=stride, padding=padding, dropout=self.configs.dropout)
                self.enc_embedding_S = self.patch_num # enc_in tensor 的三元组的中间维度
                
                if self.series_sampling: # CI + series-patching + series-sampling
                    self.enc_embedding = torch.nn.ModuleList()
                    self.dec_embedding = None if self.gym_encoder_only else torch.nn.ModuleList()
                    self.enc_embedding_S = []
                    for i in range(self.configs.down_sampling_layers + 1):
                        seq_len_ = self.configs.seq_len // (self.configs.down_sampling_window ** i)
                        stride, padding, patch_num, patch_len = calculate_patch_num(seq_len_)
                        if self.gym_network_architecture in ['GRU','MLP']:
                            enc_embedding_ = PatchEmbedding_wo_pos(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                                   padding=padding, dropout=self.configs.dropout)
                            dec_embedding_ = None if self.gym_encoder_only else PatchEmbedding_wo_pos(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                                   padding=padding, dropout=self.configs.dropout)
                        else:
                            enc_embedding_ = PatchEmbedding(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                            padding=padding, dropout=self.configs.dropout)
                            dec_embedding_ = None if self.gym_encoder_only else PatchEmbedding(d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                            padding=padding, dropout=self.configs.dropout)
                            
                        self.enc_embedding.append(enc_embedding_)
                        if not self.gym_encoder_only: self.dec_embedding.append(dec_embedding_)
                        self.enc_embedding_S.append(patch_num)
                    
                    self.enc_embedding = nn.ModuleList(self.enc_embedding)
                    self.dec_embedding = None if self.gym_encoder_only else nn.ModuleList(self.dec_embedding)
        else: # channel-dependent
            if self.gym_input_embed == 'series-patching': # CD + series-patching
                stride, padding, self.patch_num, patch_len = calculate_patch_num(self.configs.seq_len)
                if self.gym_network_architecture in ['GRU','MLP']:
                    self.enc_embedding = PatchEmbedding_wo_pos_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                               padding=padding, dropout=self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else PatchEmbedding_wo_pos_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len,
                                                                                                  stride=stride, padding=padding, dropout=self.configs.dropout)
                else:
                    self.enc_embedding = PatchEmbedding_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                        padding=padding, dropout=self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else PatchEmbedding_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len, 
                                                                                           stride=stride, padding=padding, dropout=self.configs.dropout)
                self.enc_embedding_S = self.patch_num
                if self.series_sampling: # CI + series-patching + series-sampling
                    self.enc_embedding = torch.nn.ModuleList()
                    self.dec_embedding = None if self.gym_encoder_only else torch.nn.ModuleList()
                    self.enc_embedding_S = []
                    for i in range(self.configs.down_sampling_layers + 1):
                        seq_len_ = self.configs.seq_len // (self.configs.down_sampling_window ** i)
                        stride, padding, patch_num, patch_len = calculate_patch_num(seq_len_)
                        if self.gym_network_architecture in ['GRU','MLP']:
                            enc_embedding_ = PatchEmbedding_wo_pos_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                                   padding=padding, dropout=self.configs.dropout)
                            dec_embedding_ = None if self.gym_encoder_only else PatchEmbedding_wo_pos_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len,
                                                                                                      stride=stride, padding=padding, dropout=self.configs.dropout)
                        else:
                            enc_embedding_ = PatchEmbedding_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                            padding=padding, dropout=self.configs.dropout)
                            dec_embedding_ = None if self.gym_encoder_only else PatchEmbedding_CD(enc_in=self.configs.enc_in, d_model=self.configs.d_model, patch_len=patch_len, stride=stride,
                                                            padding=padding, dropout=self.configs.dropout)
                            
                        self.enc_embedding.append(enc_embedding_)
                        if not self.gym_encoder_only: self.dec_embedding.append(dec_embedding_)
                        self.enc_embedding_S.append(patch_num)
                    
                    self.enc_embedding = nn.ModuleList(self.enc_embedding)
                    self.dec_embedding = None if self.gym_encoder_only else nn.ModuleList(self.dec_embedding)
            elif self.gym_input_embed == 'inverted-encoding': # CD + inverted-encoding
                self.enc_embedding = DataEmbedding_inverted(self.configs.seq_len, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                self.dec_embedding = None if self.gym_encoder_only else DataEmbedding_inverted(
                    self.configs.seq_len, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                if self.series_sampling: raise NotImplementedError # CD + inverted-encoding + series-sampling not supported
                self.enc_embedding_S = self.configs.enc_in+4 if self.gym_x_mark else self.configs.enc_in
            elif self.gym_input_embed == 'ortho-encoding': # CD + ortho-encoding, from OLinear, update1228 cc
                self.enc_embedding = nn.Linear(1, self.configs.d_model, bias=False)
                Q_out_mat = torch.from_numpy(np.load(self.q_out_mat_dir)).to(torch.float32)
                if not self.series_sampling:
                    Q_mat = torch.from_numpy(np.load(self.q_mat_dir)).to(torch.float32).transpose(-1, -2)
                    self.register_buffer('Q_mat', Q_mat)
                    self.delta1 = nn.Parameter(torch.zeros(1, self.configs.enc_in, 1, self.seq_len))
                    self.ortho_input_encoder = nn.Linear(self.seq_len*self.configs.d_model, self.configs.d_model)
                self.register_buffer('Q_out_mat', Q_out_mat)
                self.dec_embedding = None
                self.delta2 = nn.Parameter(torch.zeros(1, self.configs.enc_in, 1, self.pred_len))
                self.enc_embedding_S = self.configs.enc_in
                
                if self.series_sampling: # CD + ortho-encoding + series_sampling
                    self.enc_embedding = nn.ModuleList(deepcopy(self.enc_embedding) for i in range(self.configs.down_sampling_layers + 1))
                    self.delta1 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.configs.enc_in, 1, _seqlen)) for _seqlen in self.sampling_seqlen])
                    self.ortho_input_encoder = nn.ModuleList([nn.Linear(_seqlen*self.configs.d_model, self.configs.d_model) for _seqlen in self.sampling_seqlen])
                    self.enc_embedding_S = []
                    for i,_seqlen in enumerate(self.sampling_seqlen):
                        Q_mat = torch.from_numpy(np.load(self.q_mat_dir[i])).to(torch.float32).transpose(-1, -2)
                        assert Q_mat.shape[0] == _seqlen
                        self.register_buffer(f'Q_mat_{i}', Q_mat)
                        self.enc_embedding_S.append(self.configs.enc_in)
            elif self.gym_input_embed == 'series-encoding': # CD + series-encoding
                if self.gym_network_architecture in ['GRU','MLP']:
                    self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else DataEmbedding_wo_pos(
                        self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                else:
                    self.enc_embedding = DataEmbedding(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                    self.dec_embedding = None if self.gym_encoder_only else DataEmbedding(
                        self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                self.enc_embedding_S = self.seq_len
                if self.series_sampling: # CD + series-encoding + series-sampling
                    self.enc_embedding = nn.ModuleList(deepcopy(self.enc_embedding) for i in range(self.configs.down_sampling_layers + 1))
                    self.dec_embedding = None if self.gym_encoder_only else nn.ModuleList(deepcopy(self.dec_embedding) 
                                                                                          for i in range(self.configs.down_sampling_layers + 1))
                    self.enc_embedding_S = self.sampling_seqlen
            else:
                raise NotImplementedError

    def _build_multi_granularity_interation(self):
        # multi-granularity interation (based-on self-attention)
        # seasonal: high -> low; trend: low -> high
        if self.series_sampling and self.series_decompsition:
            self.seasonal_mixing = AttentionLayer(
                                    FullAttention(mask_flag=False,
                                                  factor=self.configs.factor,
                                                  attention_dropout=self.configs.dropout,
                                                  output_attention=False), # masked multi-head attention
                                    self.configs.d_model, self.configs.n_heads)
            self.trend_mixing = AttentionLayer(
                                    FullAttention(mask_flag=False,
                                                  factor=self.configs.factor,
                                                  attention_dropout=self.configs.dropout,
                                                  output_attention=False), # masked multi-head attention
                                    self.configs.d_model, self.configs.n_heads)
            
            self.seasonal_mixing = nn.ModuleList([deepcopy(self.seasonal_mixing) 
                                                  for i in range(self.configs.down_sampling_layers)])
            self.trend_mixing = nn.ModuleList([deepcopy(self.trend_mixing) 
                                               for i in range(self.configs.down_sampling_layers)])
        else:
            pass
    
    def _build_feature_attention(self):
        # feature attention
        if self.gym_channel_independent: # CI no feature attention
            self.feature_encoder = None
        else:
            if self.gym_feature_attn == 'null':
                self.feature_encoder = None
            else:
                if self.gym_feature_attn == 'self-attention':
                    FeatureAttention = FullAttention
                elif self.gym_feature_attn == 'sparse-attention':
                    FeatureAttention = ProbAttention
                else:
                    raise NotImplementedError

                # Feature Encoder
                self.feature_encoder = Encoder(
                        [
                            EncoderLayer(
                                AttentionLayer(FeatureAttention(configs=self.configs,
                                                                mask_flag=False,
                                                                factor=self.configs.factor,
                                                                attention_dropout=self.configs.dropout, 
                                                                output_attention=False),
                                            self.configs.d_model, self.configs.n_heads),
                                self.configs.d_model,
                                self.configs.d_ff,
                                dropout=self.configs.dropout,
                                activation=self.configs.activation
                            ) for l in range(self.configs.e_layers)
                        ],
                        norm_layer=torch.nn.LayerNorm(self.configs.d_model)
                    )
                if self.series_sampling:
                    self.feature_embedding = nn.ModuleList([DataEmbedding_inverted(self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                                                                self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                                                            for i in range(self.configs.down_sampling_layers + 1)])
                    self.feature_encoder = nn.ModuleList(deepcopy(self.feature_encoder) for i in range(self.configs.down_sampling_layers + 1))
                    if self.gym_input_embed == 'ortho-encoding':
                        self.seq_projector = nn.ModuleList([nn.Linear(self.configs.d_model, self.configs.pred_len, bias=True)
                                                            for i in range(self.configs.down_sampling_layers + 1)])
                    else:
                        self.seq_projector = nn.ModuleList([nn.Linear(self.configs.d_model, self.configs.seq_len // (self.configs.down_sampling_window ** i), bias=True)
                                                            for i in range(self.configs.down_sampling_layers + 1)])
                else:
                    self.feature_embedding = DataEmbedding_inverted(self.configs.seq_len, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
                    if self.gym_input_embed == 'ortho-encoding':
                        self.seq_projector = nn.Linear(self.configs.d_model, self.configs.pred_len, bias=True)
                    else:
                        self.seq_projector = nn.Linear(self.configs.d_model, self.configs.seq_len, bias=True)
    
    def _build_backbone(self):
        # encoder(-decoder)                        
        if self.gym_network_architecture == 'Transformer':
            # Attention
            if self.gym_attn == 'self-attention':
                Attention = FullAttention
            elif self.gym_attn == 'auto-correlation':
                Attention = AutoCorrelation
            elif self.gym_attn == 'sparse-attention':
                Attention = ProbAttention
            elif self.gym_attn == 'frequency-enhanced-attention' or self.gym_feature_attn == 'frequency-attention':
                Attention = FourierCrossAttention
            elif self.gym_attn == 'destationary-attention':
                # https://github.com/thuml/Time-Series-Library/blob/3aed70eb3d7b8e5b51e8aafe1f9e69ed06d11de8/models/Nonstationary_Transformer.py#L113
                if self.gym_series_norm != 'Stat': raise NotImplementedError
                if self.gym_input_embed != 'series-encoding': raise NotImplementedError
                Attention = DSAttention

                if self.gym_series_sampling:
                    self.tau_learner = nn.ModuleList([Projector(enc_in=1 if self.gym_channel_independent else self.configs.enc_in, seq_len=self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                                                hidden_dims=self.configs.p_hidden_dims,
                                                                hidden_layers=self.configs.p_hidden_layers,
                                                                output_dim=1)
                                                                for i in range(self.configs.down_sampling_layers + 1)])
                    self.delta_learner = nn.ModuleList([Projector(enc_in=1 if self.gym_channel_independent else self.configs.enc_in, seq_len=self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                                                  hidden_dims=self.configs.p_hidden_dims,
                                                                  hidden_layers=self.configs.p_hidden_layers,
                                                                  output_dim=self.configs.seq_len // (self.configs.down_sampling_window ** i)) 
                                                                  for i in range(self.configs.down_sampling_layers + 1)])             
                
                else:
                    self.tau_learner = Projector(enc_in=1 if self.gym_channel_independent else self.configs.enc_in, seq_len=self.configs.seq_len, hidden_dims=self.configs.p_hidden_dims,
                                                hidden_layers=self.configs.p_hidden_layers, output_dim=1)
                    self.delta_learner = Projector(enc_in=1 if self.gym_channel_independent else self.configs.enc_in, seq_len=self.configs.seq_len,
                                                hidden_dims=self.configs.p_hidden_dims, hidden_layers=self.configs.p_hidden_layers,
                                                output_dim=self.configs.seq_len)
            else:
                raise NotImplementedError
            
            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attention(configs=self.configs,
                                                 mask_flag=False,
                                                 factor=self.configs.factor,
                                                 attention_dropout=self.configs.dropout, 
                                                 output_attention=False),
                                    self.configs.d_model, self.configs.n_heads),
                        self.configs.d_model,
                        self.configs.d_ff,
                        dropout=self.configs.dropout,
                        activation=self.configs.activation
                    ) for l in range(self.configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.configs.d_model)
            )

            # Decoder
            if not self.gym_encoder_only:
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.decoder = Decoder(
                        [
                            DecoderLayer(
                                AttentionLayer(
                                    Attention(mask_flag=True, factor=self.configs.factor, attention_dropout=self.configs.dropout,
                                                output_attention=False), # masked multi-head attention
                                    self.configs.d_model, self.configs.n_heads),
                                AttentionLayer(
                                    Attention(mask_flag=False, factor=self.configs.factor, attention_dropout=self.configs.dropout,
                                            output_attention=False),
                                    self.configs.d_model, self.configs.n_heads),
                                self.configs.d_model,
                                self.configs.d_ff,
                                dropout=self.configs.dropout,
                                activation=self.configs.activation,
                            )
                            for l in range(self.configs.d_layers)
                        ],
                        norm_layer=torch.nn.LayerNorm(self.configs.d_model)
                    )
                else:
                    raise NotImplementedError
            else:
                self.decoder = None
        elif 'LLM' in self.gym_network_architecture:
            # loads a pretrained GPT-2 base model
            self.encoder = LLM(self.configs, network_architecture=self.gym_network_architecture, frozen=self.gym_frozen)
            self.decoder = None
        elif 'TSFM' in self.gym_network_architecture:
            self.encoder = TSFM(self.configs, network_architecture=self.gym_network_architecture, frozen=self.gym_frozen)
            self.decoder = None
        else:
            if not self.gym_encoder_only:
                raise NotImplementedError
            elif self.gym_network_architecture == 'MLP':
                if self.gym_attn == 'DNN': # gym_attn is not real attention, just the network-architecture
                    self.encoder = DNN(self.configs)
                elif self.gym_attn == 'NormLin': # From Olinear
                    if not self.gym_series_sampling: # NormLin + nosampling
                        self.encoder = Encoder_ori(
                            [
                                LinearEncoder(
                                    d_model=self.configs.d_model, d_ff=self.configs.d_ff, CovMat=None,
                                    dropout=self.configs.dropout, activation=self.configs.activation, token_num=self.enc_embedding_S,
                                ) for _ in range(self.configs.e_layers)
                            ],
                            norm_layer=nn.LayerNorm(self.configs.d_model),
                            one_output=False,
                            CKA_flag=False
                        )
                    else:
                        self.encoder = nn.ModuleList() # NormLin + sampling
                        for _enc_embedding_S in self.enc_embedding_S:
                            # print(self.enc_embedding_S)
                            _encoder = Encoder_ori(
                                [
                                    LinearEncoder(
                                        d_model=self.configs.d_model, d_ff=self.configs.d_ff, CovMat=None,
                                        dropout=self.configs.dropout, activation=self.configs.activation, token_num=_enc_embedding_S,
                                    ) for _ in range(self.configs.e_layers)
                                ],
                                norm_layer=nn.LayerNorm(self.configs.d_model),
                                one_output=False,
                                CKA_flag=False
                            )
                            self.encoder.append(deepcopy(_encoder))
                else: # default is DNN
                    self.encoder = DNN(self.configs)
                self.decoder = None
            elif self.gym_network_architecture == 'GRU':
                if self.gym_attn == 'xLSTM':
                    self.encoder = xLSTMBackbone(
                        d_model=self.configs.d_model, 
                        num_layers=self.configs.e_layers, 
                        num_heads=self.configs.n_heads, 
                        dropout=self.configs.dropout
                        )
                else:
                    self.encoder = GRU(self.configs)
                self.decoder = None
            else:
                raise NotImplementedError
            
        # If series decomposition, deepcopy encoder to cosntruct seasonal & trend branches
        if self.series_decompsition:
            if not self.gym_encoder_only:
                raise NotImplementedError
            self.encoder_seasonal = deepcopy(self.encoder)
            self.encoder_trend = deepcopy(self.encoder)
            if self.series_sampling:
                if self.gym_attn == 'NormLin':
                    pass
                else:
                    self.encoder_seasonal = nn.ModuleList(deepcopy(self.encoder_seasonal) for i in range(self.configs.down_sampling_layers + 1))
                    self.encoder_trend = nn.ModuleList(deepcopy(self.encoder_trend) for i in range(self.configs.down_sampling_layers + 1))
            del self.encoder
        else:
            if self.series_sampling:
                if self.gym_attn == 'NormLin':
                    pass
                else:
                    self.encoder = nn.ModuleList(deepcopy(self.encoder) for i in range(self.configs.down_sampling_layers + 1))
                if self.decoder is not None:
                    self.decoder = nn.ModuleList(deepcopy(self.decoder) for i in range(self.configs.down_sampling_layers + 1))
            else:
                pass

    def _build_outputhead(self):
        # CI
        if self.gym_channel_independent:
            if self.gym_input_embed == 'series-encoding': # CI + series-encoding
                self.decoder_projection = nn.Linear(self.configs.d_model, 1, bias=True) # (B*C, S, d_model) -> (B*C, S, 1)
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast': # 预测任务
                    self.head = nn.Linear(self.configs.seq_len, self.configs.pred_len) # (B, C, S) -> (B, C, pred_len)
                elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation': # 异常检测/插补 做重构任务
                    # TODO：classification-based anomaly detection
                    self.head = nn.Linear(self.configs.seq_len, self.configs.seq_len) # (B, C, S) -> (B, C, S)
                elif self.task_name == 'classification':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.enc_in * self.configs.seq_len, self.configs.num_classes)
                    ) # (B, C, S) -> (B, num_classes)
                elif self.task_name == 'finance_regressing':
                    self.head = nn.Linear(self.configs.enc_in, 1) # (B, C) -> (B, 1)
                else:
                    raise NotImplementedError
                # CI + series-encoding + series-sampling
                if self.gym_series_sampling:
                    self.decoder_projection = nn.ModuleList(nn.Linear(self.configs.d_model, 1, bias=True)
                                              for i in range(self.configs.down_sampling_layers + 1)) # decoder_projection只有在series-encoding下才有意义
                    if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast': # 预测任务
                        self.head = torch.nn.ModuleList(
                        [
                            torch.nn.Linear(
                                self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                self.configs.pred_len,
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation': # 异常检测/插补 做重构任务
                        # TODO：classification-based anomaly detection
                        self.head = torch.nn.ModuleList(
                        [
                            torch.nn.Linear(
                                self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                self.configs.seq_len,
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'classification':
                        self.head = torch.nn.ModuleList([
                            nn.Sequential(
                                nn.Flatten(start_dim=1),
                                nn.Dropout(self.configs.dropout),
                                nn.Linear(self.configs.enc_in * self.configs.seq_len // (self.configs.down_sampling_window ** i), self.configs.num_classes)
                            ) for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'finance_regressing':
                        self.head = torch.nn.ModuleList([
                            nn.Linear(self.configs.enc_in,1) for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    else:
                        raise NotImplementedError
            elif self.gym_input_embed == 'series-patching': # CI + series-patching
                stride, padding, patch_num, patch_len = calculate_patch_num(self.configs.seq_len)
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.head = FlattenHead(self.configs.enc_in, self.configs.d_model * patch_num, self.configs.pred_len,
                                            head_dropout=self.configs.dropout) # (B, C, patch_num, d_model) -> (B, C, pred_len)
                elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
                    self.head = FlattenHead(self.configs.enc_in, self.configs.d_model * patch_num, self.configs.seq_len,
                                            head_dropout=self.configs.dropout) # (B, C, patch_num, d_model) -> (B, C, seq_len)
                elif self.task_name == 'classification':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.enc_in * self.configs.d_model * patch_num, self.configs.num_classes)
                    ) # (B, C, patch_num, d_model) -> (B, num_classes)
                elif self.task_name == 'finance_regressing':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.enc_in * self.configs.d_model, 1)
                    ) # (B, C, patch_num, d_model) -> (B, 1)
                else:
                    raise NotImplementedError
                # CI + series-patching + series-sampling
                if self.gym_series_sampling:
                    self.head = torch.nn.ModuleList()
                    for i in range(self.configs.down_sampling_layers + 1):
                        seq_len_ = self.configs.seq_len // (self.configs.down_sampling_window ** i)
                        stride, padding, patch_num, patch_len = calculate_patch_num(seq_len_)
                        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                            _head = FlattenHead(self.configs.enc_in, self.configs.d_model * patch_num, self.configs.pred_len,
                                                    head_dropout=self.configs.dropout) # (B, C, patch_num, d_model) -> (B, C, pred_len)
                        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
                            _head = FlattenHead(self.configs.enc_in, self.configs.d_model * patch_num, self.configs.seq_len,
                                                    head_dropout=self.configs.dropout) # (B, C, patch_num, d_model) -> (B, C, seq_len)
                        elif self.task_name == 'classification':
                            _head = nn.Sequential(
                                nn.Flatten(start_dim=1),
                                nn.Dropout(self.configs.dropout),
                                nn.Linear(self.configs.enc_in * self.configs.d_model * patch_num, self.configs.num_classes)
                            ) # (B, C, patch_num, d_model) -> (B, num_classes)
                        elif self.task_name == 'finance_regressing':
                            _head = nn.Sequential(
                                nn.Flatten(start_dim=1),
                                nn.Dropout(self.configs.dropout),
                                nn.Linear(self.configs.enc_in * self.configs.d_model, 1)
                            ) # (B, C, d_model) -> (B, 1)
                        self.head.append(_head)
                    self.head = nn.ModuleList(self.head)
            else:
                raise NotImplementedError
        else: # channel-dependent
            if self.gym_input_embed == 'series-encoding':
                # decoder projection
                self.decoder_projection = nn.Linear(self.configs.d_model, self.configs.c_out, bias=True) # (B, S, d_model) -> (B, S, C)
                # output head
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.head = nn.Linear(self.configs.seq_len, self.configs.pred_len) # (B, C, S) -> (B, C, pred_len)
                elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation': # 异常检测/插补 做重构任务
                    self.head = nn.Linear(self.configs.seq_len, self.configs.seq_len) # (B, C, S) -> (B, C, S)
                elif self.task_name == 'classification':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.enc_in * self.configs.seq_len, self.configs.num_classes)
                    )
                elif self.task_name == 'finance_regressing':
                    self.head = nn.Linear(self.configs.enc_in, 1)
                else:
                    raise NotImplementedError
                # CD + series-encoding + series-sampling
                if self.gym_series_sampling:
                    self.decoder_projection = nn.ModuleList(deepcopy(self.decoder_projection) for i in range(self.configs.down_sampling_layers + 1))
                    if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                        self.head = torch.nn.ModuleList(
                        [
                            torch.nn.Linear(
                                self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                self.configs.pred_len,
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
                        self.head = torch.nn.ModuleList(
                        [
                            torch.nn.Linear(
                                self.configs.seq_len // (self.configs.down_sampling_window ** i),
                                self.configs.seq_len,
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'classification':
                        self.head = torch.nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Flatten(start_dim=1),
                                nn.Dropout(self.configs.dropout),
                                nn.Linear(self.configs.enc_in * self.configs.seq_len // (self.configs.down_sampling_window ** i), self.configs.num_classes)
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'finance_regressing':
                        self.head = torch.nn.ModuleList([
                            nn.Linear(self.configs.enc_in,1) for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    else:
                        raise NotImplementedError
            elif self.gym_input_embed == 'inverted-encoding':# CD + inverted-encoding
                # predicting head
                self.decoder_projection = None
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.head = nn.Linear(self.configs.d_model, self.configs.pred_len, bias=True) # (B, C, d_model) -> (B, C, pred_len)
                elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation': # 异常检测/插补 做重构任务
                    self.head = nn.Linear(self.configs.d_model, self.configs.seq_len, bias=True) # (B, C, d_model) -> (B, C, S)
                elif self.task_name == 'classification':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.d_model * self.configs.enc_in, self.configs.num_classes)
                    ) # (B, C, d_model) -> (B, num_classes)
                elif self.task_name == 'finance_regressing':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.d_model * self.configs.enc_in, 1)
                    ) # (B, C, d_model) -> (B, num_classes)
                else:
                    raise NotImplementedError
            elif self.gym_input_embed == 'ortho-encoding':
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.decoder_projection = nn.Linear(self.configs.d_model, self.pred_len * self.configs.d_model)
                    self.head = nn.Sequential(
                        nn.Linear(self.pred_len * self.configs.d_model, self.configs.d_ff),
                        nn.GELU(),
                        nn.Linear(self.configs.d_ff, self.pred_len)
                    )
                elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation': # 异常检测/插补 做重构任务
                    self.head = nn.Sequential(
                        nn.Linear(self.configs.d_model, self.seq_len * self.configs.d_model),
                        nn.Linear(self.seq_len * self.configs.d_model, self.configs.d_ff),
                        nn.GELU(),
                        nn.Linear(self.configs.d_ff, self.seq_len)
                    )
                elif self.task_name == 'finance_regressing':
                    self.head = nn.Sequential(
                        nn.Linear(self.configs.enc_in*self.configs.d_model, self.configs.d_ff),
                        nn.GELU(),
                        nn.Linear(self.configs.d_ff, 1)
                    )
                else:
                    raise NotImplementedError
                if self.gym_series_sampling:
                    if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast': # 预测任务
                        self.decoder_projection = nn.Linear(self.configs.d_model, self.pred_len * self.configs.d_model)
                        self.head = torch.nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.pred_len * self.configs.d_model, self.configs.d_ff),
                                nn.GELU(),
                                nn.Linear(self.configs.d_ff, self.pred_len)
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation': # 异常检测/插补 做重构任务
                        # TODO：classification-based anomaly detection
                        self.head = torch.nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.configs.d_model, self.seq_len * self.configs.d_model),
                                nn.Linear(self.seq_len * self.configs.d_model, self.configs.d_ff),
                                nn.GELU(),
                                nn.Linear(self.configs.d_ff, self.seq_len)
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    elif self.task_name == 'finance_regressing':
                        self.head = torch.nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.configs.enc_in*self.configs.d_model, self.configs.d_ff),
                                nn.GELU(),
                                nn.Linear(self.configs.d_ff, 1)
                            )
                            for i in range(self.configs.down_sampling_layers + 1)
                        ])
                    else:
                        raise NotImplementedError
            elif self.gym_input_embed == 'series-patching': # CD + series-patching
                stride, padding, patch_num, patch_len = calculate_patch_num(self.configs.seq_len)
                # B, patch_num, d_model -> B, patch_num, C*patch_len -> B, NP, C, PL -> B, C, NP, PL -> B, C, S -> B,C,predlen
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.head = PatchingHead_CD(d_model=self.configs.d_model, n_vars=self.configs.enc_in, pred_len=self.configs.pred_len, 
                                                patch_num=patch_num, patch_len=patch_len, head_dropout=self.configs.dropout) # (B, patchnum, dmodel) -> (B, C, pred_len)
                elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
                    self.head = PatchingHead_CD(d_model=self.configs.d_model, n_vars=self.configs.enc_in, pred_len=self.configs.seq_len, 
                                                patch_num=patch_num, patch_len=patch_len, head_dropout=self.configs.dropout) # (B, patchnum, dmodel) -> (B, C, pred_len)
                elif self.task_name == 'classification':
                    # B, np, dmodel -> B,num_classes
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.d_model * patch_num, self.configs.num_classes)
                    ) # (B, C, patch_num, d_model) -> (B, num_classes)
                elif self.task_name == 'finance_regressing':
                    self.head = nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Dropout(self.configs.dropout),
                        nn.Linear(self.configs.d_model * patch_num, 1)
                    ) # (B, C, patch_num, d_model) -> (B, 1)
                else:
                    raise NotImplementedError
                # CD + series-patching + series-sampling
                if self.gym_series_sampling:
                    self.head = torch.nn.ModuleList()
                    for i in range(self.configs.down_sampling_layers + 1):
                        seq_len_ = self.configs.seq_len // (self.configs.down_sampling_window ** i)
                        stride, padding, patch_num, patch_len = calculate_patch_num(seq_len_)
                        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                            _head = PatchingHead_CD(d_model=self.configs.d_model, n_vars=self.configs.enc_in, pred_len=self.configs.pred_len, 
                                                        patch_num=patch_num, patch_len=patch_len, head_dropout=self.configs.dropout) # (B, patchnum, dmodel) -> (B, C, pred_len)
                            # _head = FlattenHead(self.configs.enc_in, self.configs.d_model * patch_num, self.configs.pred_len,
                            #                         head_dropout=self.configs.dropout) # (B, C, patch_num, d_model) -> (B, C, pred_len)
                        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
                            _head = PatchingHead_CD(d_model=self.configs.d_model, n_vars=self.configs.enc_in, pred_len=self.configs.seq_len, 
                                                        patch_num=patch_num, patch_len=patch_len, head_dropout=self.configs.dropout) # (B, patchnum, dmodel) -> (B, C, seq_len)
                            # _head = FlattenHead(self.configs.enc_in, self.configs.d_model * patch_num, self.configs.seq_len,
                            #                         head_dropout=self.configs.dropout) # (B, C, patch_num, d_model) -> (B, C, seq_len)
                        elif self.task_name == 'classification':
                            _head = nn.Sequential(
                                nn.Flatten(start_dim=1),
                                nn.Dropout(self.configs.dropout),
                                nn.Linear(self.configs.d_model * patch_num, self.configs.num_classes)
                            ) # (B, C, patch_num, d_model) -> (B, num_classes)
                        elif self.task_name == 'finance_regressing':
                            _head = nn.Sequential(
                                nn.Flatten(start_dim=1),
                                nn.Dropout(self.configs.dropout),
                                nn.Linear(self.configs.d_model, 1)
                            ) # (B, C, d_model) -> (B, 1)
                        self.head.append(_head)
                    self.head = nn.ModuleList(self.head)
            else:
                raise NotImplementedError
    
        if self.gym_rag:
            self.rt = TSGymRetrievalTool(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                channels=self.configs.enc_in,
                n_period=self.configs.n_period, # 与多尺度解耦
                topm=self.configs.topm,
                channel_independence=self.gym_channel_independent
            )
            self.rfb = TSGymRetrievalFusionBlock(
                period_num = self.rt.period_num, 
                pred_len = self.pred_len, 
                channels=self.configs.enc_in)
            
    def multi_scale_process_inputs(self, x_enc, x_mark_enc=None):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def prepare_dataset(self, train_data, valid_data, test_data):
        """RAG: prepare retrieval features for train/valid/test sets"""
        self.rt.prepare_dataset(train_data)
        
        self.retrieval_dict = {}
        
        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False)

        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def fetch_batch(self, index, mode):
        """
        根据 index 获取当前 batch 的检索特征
        """
        # 注意：为了节省显存，大字典通常存在 CPU 上，取 batch 时再移到 GPU
        # 假设缓存的维度是 [G, Total_Samples, P, C]
        pred_from_retrieval = self.retrieval_dict[mode][:, index.cpu()] # G, B, P, C
        return pred_from_retrieval.to(index.device)
    
    def f_series_preprocessing_sampling(self, x_enc, x_mark_enc=None):
        '''
        input:
        - x_enc, (B, S, C)
        - x_mark_enc, (B, S, C)
        series sampling
        series normalization
        feature attention
        channel-independent and series-encoding
        learn tau and delta for non-stationary attention (if necessary)
        return:
        - x_enc: list of sampled x_enc, each with shape (B or B*C, S_i, C or 1)
        - x_mark_enc: list of sampled x_mark_enc, each with shape (B or B*C, S_i, C or 1)
        - tau: list of tau for non-stationary attention
        - delta: list of delta for non-stationary attention
        - enc_out_fa: list of feature attention output

        '''
        self.B, self.S, self.C = x_enc.shape
        # series sampling
        # x_enc -> list with different seq_len
        assert self.series_sampling
        x_enc, x_mark_enc = self.multi_scale_process_inputs(x_enc, x_mark_enc)
        # series normalization
        x_enc = [self.series_norm[i](_, 'norm') for i, _ in enumerate(x_enc)]
        # feature attention
        if self.gym_feature_attn != 'null':
            x_enc_fa = [self.feature_embedding[i](_, None) for i, _ in enumerate(x_enc)]
            enc_out_fa = [self.feature_encoder[i](_)[0] for i, _ in enumerate(x_enc_fa)]
            enc_out_fa = [self.seq_projector[i](_).permute(0, 2, 1) for i, _ in enumerate(enc_out_fa)]
        else:
            enc_out_fa = [None] * len(x_enc)
        # channel-independent and series-encoding
        if self.gym_channel_independent and self.gym_input_embed  == 'series-encoding':
            x_enc = [_.permute(0, 2, 1).contiguous().reshape(self.B * self.C, _.shape[1], 1) for _ in x_enc]
            if x_mark_enc is not None: x_mark_enc = [_.repeat(self.C, 1, 1) for _ in x_mark_enc]
        if self.gym_input_embed == 'ortho-encoding':
            x_enc = [_.permute(0, 2, 1).unsqueeze(-1) for _ in x_enc] # B,C,S,1
        # learn tau and delta for non-stationary attention (if necessary)
        if self.gym_attn == 'destationary-attention':
            tau, delta = [], []
            for i, x_enc_ in enumerate(x_enc):
                mean_enc = x_enc_.mean(1, keepdim=True).detach()
                std_enc = torch.sqrt(torch.var(x_enc_, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                tau.append(self.tau_learner[i](x_enc_.clone().detach(), std_enc).exp())
                delta.append(self.delta_learner[i](x_enc_.clone().detach(), mean_enc))
        else:
            tau, delta = [None] * len(x_enc), [None] * len(x_enc)
        return x_enc, x_mark_enc, tau, delta, enc_out_fa

    def f_series_preprocessing_wosampling(self, x_enc, x_mark_enc=None):
        '''
        input:
        - x_enc, (B, S, C)
        - x_mark_enc, (B, S, C)
        series normalization
        feature attention
        channel-independent and series-encoding
        learn tau and delta for non-stationary attention (if necessary)
        return:
        - x_enc: (B or B*C, S, C or 1)
        - x_mark_enc: (B or B*C, S, C or 1)
        - tau: for non-stationary attention
        - delta: for non-stationary attention
        - enc_out_fa: feature attention output
        '''
        self.B, self.S, self.C = x_enc.shape
        assert not self.series_sampling
        # series normalization
        x_enc = self.series_norm(x_enc, 'norm')
        # feature attention
        if self.gym_feature_attn != 'null':
            x_enc_fa = self.feature_embedding(x_enc, None) # BxSxD -> BxDxS -> BxDxd_model
            enc_out_fa, _ = self.feature_encoder(x_enc_fa) # BxDxd_model
            enc_out_fa = self.seq_projector(enc_out_fa).permute(0, 2, 1) # BxDxd_model -> BxDxS -> BxSxD
        else:
            enc_out_fa = None
        # channel-independent and series-encoding
        if self.gym_channel_independent and self.gym_input_embed  == 'series-encoding':
            x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(self.B * self.C, self.S, 1)
            if x_mark_enc is not None: x_mark_enc = x_mark_enc.repeat(self.C, 1, 1)
        if self.gym_input_embed == 'ortho-encoding':
            x_enc = x_enc.permute(0,2,1).unsqueeze(-1) # B,C,S,1
        # learn tau and delta for non-stationary attention (if necessary)
        if self.gym_attn == 'destationary-attention':
            mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
            # B x S x E, B x 1 x E -> B x 1, positive scalar
            tau = self.tau_learner(x_enc.clone().detach(), std_enc).exp() # x_raw
            # B x S x E, B x 1 x E -> B x S
            delta = self.delta_learner(x_enc.clone().detach(), mean_enc)
        else:
            tau, delta = None, None
        return x_enc, x_mark_enc, tau, delta, enc_out_fa

    def f_series_encoding_decomposition_sampling(self, x_enc, x_mark_enc, tau, delta, enc_out_fa):
        '''
        series encoding with series decomposition and series sampling
        input:
        - x_enc: list of sampled x_enc, each with shape (B or B*C, S_i, C or 1)
        - x_mark_enc: list of sampled x_mark_enc, each with shape (B or B*C, S_i, C or 1)
        - tau: list of tau for non-stationary attention
        - delta: list of delta for non-stationary attention
        - enc_out_fa: list of feature attention output
        return:
        - enc_out: list of encoder output
        -- if series-encoding, each with shape (B or B*C, S_i, d_model)
        -- if series-patching, each with shape (B*C, patch_num, d_model)
        '''
        assert self.gym_series_decomp and self.series_sampling
        if not self.gym_encoder_only: # now only support encoder-only architecture with series decomposition
            raise NotImplementedError
        if self.gym_input_embed == 'inverted-encoding': # now not support inverted-encoding with series sampling
            raise NotImplementedError
        # series decomposition
        seasonal_init, trend_init = [], []
        for _ in x_enc:
            seasonal_init_, trend_init_ = self.series_decompsition(_)
            seasonal_init.append(seasonal_init_); trend_init.append(trend_init_)
        # input encoding
        enc_out_seasonal, enc_out_trend = [], []
        for i, (seasonal_init_, trend_init_) in enumerate(zip(seasonal_init, trend_init)):
            if self.gym_input_embed == 'series-encoding':
                enc_out_seasonal_ = self.enc_embedding[i](seasonal_init_, x_mark_enc[i] if x_mark_enc is not None else None) # todo: x_mark
                enc_out_trend_ = self.enc_embedding[i](trend_init_, x_mark_enc[i] if x_mark_enc is not None else None)
            elif self.gym_input_embed == 'series-patching':
                enc_out_seasonal_, self.n_vars = self.enc_embedding[i](seasonal_init_.permute(0, 2, 1)) # BxSxD -> BxDxS
                enc_out_trend_, self.n_vars = self.enc_embedding[i](trend_init_.permute(0, 2, 1))
            else:
                raise NotImplementedError
            enc_out_seasonal.append(enc_out_seasonal_); enc_out_trend.append(enc_out_trend_)

        # multi-granularity mixing
        # seasonal: high -> low
        enc_out_seasonal_mixed = []
        for i, enc_out_seasonal_ in enumerate(enc_out_seasonal):
            if i == 0:
                enc_out_seasonal_mixed.append(enc_out_seasonal_)
            else:
                enc_out_seasonal_ = self.seasonal_mixing[i-1](queries=enc_out_seasonal_,
                                                                keys=enc_out_seasonal_previous_,
                                                                values=enc_out_seasonal_previous_,
                                                                attn_mask=None)[0]
                enc_out_seasonal_mixed.append(enc_out_seasonal_)
            enc_out_seasonal_previous_ = enc_out_seasonal_
        enc_out_seasonal = enc_out_seasonal_mixed; del enc_out_seasonal_mixed
        
        # tred: low -> high
        enc_out_trend_mixed = []
        for i, enc_out_trend_ in enumerate(enc_out_trend[::-1]):
            if i == 0:
                enc_out_trend_mixed.append(enc_out_trend_)
            else:
                enc_out_trend_ = self.trend_mixing[i-1](queries=enc_out_trend_,
                                                        keys=enc_out_trend_previous_,
                                                        values=enc_out_trend_previous_,
                                                        attn_mask=None)[0]
                enc_out_trend_mixed.append(enc_out_trend_)
            enc_out_trend_previous_ = enc_out_trend_
        enc_out_trend = enc_out_trend_mixed[::-1]; del enc_out_trend_mixed

        # seasonal + trend
        enc_out_seasonal = [self.encoder_seasonal[i](_, attn_mask=None, tau=tau[i], delta=delta[i])[0] for i, _ in enumerate(enc_out_seasonal)]
        enc_out_trend = [self.encoder_trend[i](_, attn_mask=None, tau=tau[i], delta=delta[i])[0] for i, _ in enumerate(enc_out_trend)]
        enc_out = [enc_out_seasonal_ + enc_out_trend_ for enc_out_seasonal_, enc_out_trend_
                    in zip(enc_out_seasonal, enc_out_trend)]
        
        del enc_out_seasonal, enc_out_trend
        return enc_out

    def f_series_encoding_decomposition_wosampling(self, x_enc, x_mark_enc, tau, delta, enc_out_fa):
        '''
        series encoding with series decomposition and series sampling
        input:
        - x_enc: (B or B*C, S, C or 1)
        - x_mark_enc: (B or B*C, S, C or 1)
        - tau: for non-stationary attention
        - delta: for non-stationary attention
        - enc_out_fa: feature attention output
        return:
        - enc_out: encoder output
        -- if series-encoding, (B or B*C, S_i, d_model)
        -- if series-patching, (B*C, patch_num, d_model)
        -- if inverted-encoding, (B, S, d_model)
        '''
        assert self.gym_series_decomp and not self.series_sampling
        if not self.gym_encoder_only: # now only support encoder-only architecture with series decomposition
            raise NotImplementedError
    
        # series decomposition
        seasonal_init, trend_init = self.series_decompsition(x_enc)
        # series tokenization
        if self.gym_input_embed == 'inverted-encoding' or self.gym_input_embed == 'inverted':
            enc_out_seasonal = self.enc_embedding(seasonal_init, x_mark_enc)
            enc_out_trend = self.enc_embedding(trend_init, x_mark_enc)
        elif self.gym_input_embed == 'series-encoding':
            enc_out_seasonal = self.enc_embedding(seasonal_init, x_mark_enc)
            enc_out_trend = self.enc_embedding(trend_init, x_mark_enc)
        elif self.gym_input_embed == 'series-patching':
            enc_out_seasonal, self.n_vars = self.enc_embedding(seasonal_init.permute(0, 2, 1)) # BxSxD -> BxDxS
            enc_out_trend, self.n_vars = self.enc_embedding(trend_init.permute(0, 2, 1))
        else:
            raise NotImplementedError

        # seasonal + trend
        enc_out_seasonal, _ = self.encoder_seasonal(enc_out_seasonal, attn_mask=None, tau=tau, delta=delta)
        enc_out_trend, _ = self.encoder_trend(enc_out_trend, attn_mask=None, tau=tau, delta=delta)
        enc_out = enc_out_seasonal + enc_out_trend
        del enc_out_seasonal, enc_out_trend
        return enc_out
    
    def f_series_encoding_wodecomposition_sampling(self, x_enc, x_mark_enc, tau, delta, enc_out_fa):
        '''
        series encoding without series decomposition but series sampling
        input:
        - x_enc: list of sampled x_enc, each with shape (B or B*C, S_i, C or 1)
        - x_mark_enc: list of sampled x_mark_enc, each with shape (B or B*C, S_i, C or 1)
        - tau: list of tau for non-stationary attention
        - delta: list of delta for non-stationary attention
        - enc_out_fa: list of feature attention output
        return:
        - enc_out: list of encoder output
        -- if series-encoding, each with shape (B or B*C, S_i, d_model)
        -- if series-patching, each with shape (B*C, patch_num, d_model)
        '''
        if self.gym_input_embed == 'inverted-encoding': # now not support inverted-encoding with series sampling
            raise NotImplementedError
        if self.gym_input_embed == 'series-encoding':
            if x_mark_enc is not None: assert len(x_enc) == len(x_mark_enc)
            enc_out = [self.enc_embedding[i](x_enc[i], x_mark_enc[i] if x_mark_enc is not None else None) for i in range(len(x_enc))]
        elif self.gym_input_embed == 'series-patching':
            x_enc = [_.permute(0, 2, 1) for _ in x_enc]
            enc_out = []
            for i, x_enc_ in enumerate(x_enc):
                enc_out_, self.n_vars = self.enc_embedding[i](x_enc_)
                enc_out.append(enc_out_)
        elif self.gym_input_embed == 'ortho-encoding':
            # x_enc: [B,C,S,1] -> [B,C,S,D]
            enc_out = [self.enc_embedding[i](x_enc[i]) for i in range(len(x_enc))]
            # [B,C,D,S]
            enc_out = [_.permute(0,1,3,2) for _ in enc_out]
            # [B,C,D,S]
            enc_out = [torch.einsum('bndt,tv->bndv', _, getattr(self, f"Q_mat_{i}")) + self.delta1[i] for i,_ in enumerate(enc_out)]
            # [B,C,D]
            enc_out = [self.ortho_input_encoder[i](_.flatten(start_dim=-2)) for i,_ in enumerate(enc_out)]
        else:
            raise NotImplementedError
        # attention in encoder, the shape of enc_out
        # no patching: [bs x seq_len x d_model]
        # patching: [(bs * nvars) x patch_num x d_model]
        enc_out = [self.encoder[i](_, attn_mask=None, tau=tau[i], delta=delta[i])[0] for i, _ in enumerate(enc_out)]
        return enc_out

    def f_series_encoding_wodecomposition_wosampling(self, x_enc, x_mark_enc, tau, delta, enc_out_fa):
        '''
        series encoding without series decomposition and series sampling
        input:
        - x_enc: (B or B*C, S, C or 1)
        - x_mark_enc: (B or B*C, S, C or 1)
        - tau: for non-stationary attention
        - delta: for non-stationary attention
        - enc_out_fa: feature attention output
        return:
        - enc_out: encoder output
        -- if series-encoding, (B or B*C, S_i, d_model)
        -- if series-patching, (B*C, patch_num, d_model)
        -- if inverted-encoding, (B, S, d_model)
        '''
        # encoder input embedding
        if self.gym_input_embed == 'inverted-encoding' or self.gym_input_embed == 'inverted':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        elif self.gym_input_embed == 'series-encoding':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        elif self.gym_input_embed == 'series-patching':
            # do patching and embedding
            x_enc = x_enc.permute(0, 2, 1) # BxSxD -> BxDxS
            # u: [bs * nvars x patch_num x d_model]
            enc_out, self.n_vars = self.enc_embedding(x_enc)
        elif self.gym_input_embed == 'ortho-encoding':
            # B,C,S,1 -> B,C,S,D -> B,C,D,S
            enc_out = self.enc_embedding(x_enc).permute(0,1,3,2)
            # B,C,D,S
            enc_out = torch.einsum('bndt,tv->bndv', enc_out, self.Q_mat)
            # B,C,D,S -> B,C,D
            enc_out = self.ortho_input_encoder(enc_out.flatten(start_dim=-2))
        else:
            raise NotImplementedError
        # attention in encoder, the shape of enc_out
        # no patching: [bs x seq_len x d_model]
        # patching: [(bs * nvars) x patch_num x d_model]
        # print(enc_out.shape)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        return enc_out

    def f_series_decoing_sampling_forecasting(self, enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data=None):
        if self.gym_encoder_only: # encoder-only
            if self.gym_input_embed == 'series-encoding':
                dec_out = [self.decoder_projection[i](_) for i, _ in enumerate(enc_out)] # enc_out: B/B*C, S, dmodel -> B/B*C, S, C/1
                if self.gym_feature_attn != 'null': dec_out = [dec_out_ + enc_out_fa_ for dec_out_, enc_out_fa_ in zip(dec_out, enc_out_fa)]
                dec_out = [self.head[i](_.permute(0, 2, 1)).permute(0, 2, 1) for i, _ in enumerate(dec_out)] # dec_out: B/B*C, 1, S -> B/B*C, pred_len, 1
                dec_out = [_[:, -self.pred_len:, :] for _ in dec_out]
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
                if self.gym_channel_independent: dec_out = dec_out.reshape(self.B, self.C, self.pred_len).permute(0, 2, 1).contiguous()
            elif self.gym_input_embed == 'series-patching' and self.gym_channel_independent:
                enc_out = [torch.reshape(_, (-1, self.n_vars, _.shape[-2], _.shape[-1])) for _ in enc_out]
                enc_out = [_.permute(0, 1, 3, 2) for _ in enc_out]
                dec_out = [self.head[i](_) for i, _ in enumerate(enc_out)]
                dec_out = [_.permute(0, 2, 1) for _ in dec_out]
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
            elif self.gym_input_embed == 'series-patching' and not self.gym_channel_independent:
                # [B, patchnum, dmodel]
                dec_out = [self.head[i](_) for i,_ in enumerate(enc_out)]
                dec_out = [_.permute(0, 2, 1) for _ in dec_out]
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
            elif self.gym_input_embed == 'ortho-encoding':
                dec_out = [self.decoder_projection(_) for i,_ in enumerate(enc_out)]
                # reshape dot product
                dec_out = [_.reshape(self.B, self.C, self.configs.d_model, self.pred_len) for _ in dec_out]
                dec_out = [torch.einsum('bndt,tv->bndv', _, self.Q_out_mat) + self.delta2 for i,_ in enumerate(dec_out)]
                dec_out = [self.head[i](_.flatten(start_dim=-2)) for i,_ in enumerate(dec_out)]
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
                # B,pred,C
                dec_out = dec_out.permute(0, 2, 1).contiguous()
            else:
                raise NotImplementedError
        else: # encoder-decoder
            raise NotImplementedError # series sampling with encoder-decoder not implemented yet
        # de-normalization layer (if necessary)
        # 如果混合粒度情况, 用第一(0)层参照TimeMixer
        dec_out = self.series_norm[0](dec_out, 'denorm') if self.series_sampling else self.series_norm(dec_out, 'denorm')
        
        if self.gym_rag:
            retrieval_pred = self.rfb(rag_raw_data)
            dec_out = torch.cat([dec_out, retrieval_pred], dim=1)
            dec_out = self.rfb.linear_pred(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        return dec_out
    
    def f_series_decoing_sampling_regressing(self, enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data=None):
        if self.gym_encoder_only: # encoder-only
            if self.gym_input_embed == 'series-encoding':
                dec_out = [self.decoder_projection[i](_) for i, _ in enumerate(enc_out)] # enc_out: B/B*C, S, dmodel -> B/B*C, S, C/1
                if self.gym_feature_attn != 'null': dec_out = [dec_out_ + enc_out_fa_ for dec_out_, enc_out_fa_ in zip(dec_out, enc_out_fa)]
                # channel independent: B*C, S, 1 -> B,C,S,1 -> B,C,S -> B,S,C
                # channel dependent: B,S,C
                if self.gym_channel_independent: dec_out = [_.reshape(self.B, self.C, self.seq_len, 1).squeeze(-1).permute(0,2,1) for _ in dec_out]
                dec_out = [_[:, -1, :] for _ in dec_out] #B,C
                dec_out = [self.head[i](_) for i, _ in enumerate(dec_out)] # dec_out:B,
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
            elif self.gym_input_embed == 'series-patching' and self.gym_channel_independent:
                enc_out = [_[:, -1, :] for _ in enc_out] # B*C, patchnum, dmodel -> B*C, dmodel
                enc_out = [torch.reshape(_, (-1, self.n_vars, _.shape[-1])) for _ in enc_out] # B, C, dmodel
                dec_out = [self.head[i](_) for i, _ in enumerate(enc_out)] # B, 1
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
            elif self.gym_input_embed == 'series-patching' and not self.gym_channel_independent:
                enc_out = [_[:, -1, :] for _ in enc_out] # B, patchnum, dmodel -> B, dmodel
                dec_out = [self.head[i](_) for i,_ in enumerate(enc_out)] # B, 1
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
            elif self.gym_input_embed == 'ortho-encoding':
                # B,C,D -> B,C*D -> B
                dec_out = [self.head[i](_.flatten(start_dim=-2)) for i,_ in enumerate(enc_out)]
                dec_out = torch.stack(dec_out, dim=-1).sum(-1)
            else:
                raise NotImplementedError
        else: # encoder-decoder
            raise NotImplementedError # series sampling with encoder-decoder not implemented yet
        return dec_out

    def f_series_decoding_wosampling_forecasting(self, enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data=None):
        if self.gym_encoder_only: # encoder-only
            if self.gym_input_embed == 'inverted-encoding' or self.gym_input_embed == 'inverted':
                dec_out = self.head(enc_out) # BxDxd_model -> BxDxpred_len
                dec_out = dec_out.permute(0, 2, 1) # BxDxpred_len -> Bxpred_lenxD
                dec_out = dec_out[:, :, :self.C]
            elif self.gym_input_embed == 'series-encoding':
                dec_out = self.decoder_projection(enc_out) # BxSxd_model -> BxSxD
                if self.gym_feature_attn != 'null': dec_out = dec_out + enc_out_fa
                # projection to predicting length
                dec_out = self.head(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # BxSxD -> BxDxS -> BxDxpred_len -> Bxpred_lenxD
                dec_out = dec_out[:, -self.pred_len:, :]
                if self.gym_channel_independent: dec_out = dec_out.reshape(self.B, self.C, self.pred_len).permute(0, 2, 1).contiguous()
            elif self.gym_input_embed == 'series-patching' and self.gym_channel_independent:
                # z: [bs x nvars x patch_num x d_model]
                enc_out = torch.reshape(enc_out, (-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1]))
                # z: [bs x nvars x d_model x patch_num]
                enc_out = enc_out.permute(0, 1, 3, 2)
                dec_out = self.head(enc_out)  # B x D x pred_len
                dec_out = dec_out.permute(0, 2, 1) # B x pred_len x D
            elif self.gym_input_embed == 'series-patching' and not self.gym_channel_independent:
                # z:[bs, patch_num, d_model]
                dec_out = self.head(enc_out)  # B x D x pred_len
                dec_out = dec_out.permute(0, 2, 1) # B x pred_len x D
            elif self.gym_input_embed == 'ortho-encoding':
                # print(f"decoding shape:{enc_out.shape}")
                # B,C,D -> B,C,D*predlen
                dec_out = self.decoder_projection(enc_out)
                # print(f"decoding shape:{dec_out.shape}")
                dec_out = dec_out.reshape(self.B, self.C, self.configs.d_model, self.pred_len)
                # print(f"decoding shape:{dec_out.shape}") : B,C,dmodel,predlen
                dec_out = torch.einsum('bndt,tv->bndv', dec_out, self.Q_out_mat) + self.delta2
                if self.gym_feature_attn != 'null': dec_out = dec_out + enc_out_fa.permute(0,2,1).unsqueeze(-2)
                # print(f"decoding shape:{dec_out.shape}")
                dec_out = self.head(dec_out.flatten(start_dim=-2))
                # B,pred,C
                dec_out = dec_out.permute(0, 2, 1).contiguous()
            else:
                raise NotImplementedError
        else: # encoder-decoder
            # decoder input embedding
            if self.gym_input_embed == 'series-encoding':
                dec_out = self.dec_embedding(x_dec, x_mark_dec)
            elif self.gym_input_embed == 'series-patching' and self.gym_channel_independent:
                # do patching and embedding
                x_dec = x_dec.permute(0, 2, 1) # BxSxD -> BxDxS
                # u: [(bs * nvars) x patch_num x d_model]
                dec_out, n_vars = self.dec_embedding(x_dec)
            else:
                raise NotImplementedError
            # attention in decoder
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
            # output layer
            if self.gym_input_embed == 'series-encoding':
                dec_out = self.decoder_projection(dec_out) # B x L x d_model -> B x L x D
                dec_out = dec_out[:, -self.pred_len:, :] # B x pred_len x D
            elif self.gym_input_embed == 'series-patching':
                # z: [bs x nvars x patch_num x d_model]
                dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
                # z: [bs x nvars x d_model x patch_num]
                dec_out = dec_out.permute(0, 1, 3, 2)
                # Decoder
                dec_out = self.head(dec_out)  # B x D x pred_len
                dec_out = dec_out.permute(0, 2, 1) # B x pred_len x D
            else:
                raise NotImplementedError

        # de-normalization layer (if necessary)
        # 如果混合粒度情况, 用第一(0)层参照TimeMixer
        dec_out = self.series_norm[0](dec_out, 'denorm') if self.series_sampling else self.series_norm(dec_out, 'denorm')

        if self.gym_rag:
            retrieval_pred = self.rfb(rag_raw_data)
            dec_out = torch.cat([dec_out, retrieval_pred], dim=1)
            dec_out = self.rfb.linear_pred(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        return dec_out
    
    def f_series_decoding_wosampling_regressing(self, enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data=None):
        if self.gym_encoder_only: # encoder-only
            if self.gym_input_embed == 'inverted-encoding' or self.gym_input_embed == 'inverted':
                dec_out = self.head(enc_out) # BxDxd_model -> Bx1
            elif self.gym_input_embed == 'series-encoding':
                dec_out = self.decoder_projection(enc_out) # BxSxd_model -> BxSxD or B*C,S,dmodel -> B*C,S,1
                if self.gym_feature_attn != 'null': dec_out = dec_out + enc_out_fa
                # B*C,S,1 -> B,C,S,1 -> B,C,S -> B,S,C
                if self.gym_channel_independent: dec_out = dec_out.reshape(self.B, self.C, self.seq_len, 1).squeeze(-1).permute(0,2,1)
                dec_out = dec_out[:, -1, :] # B,C
                # projection to predicting length
                dec_out = self.head(dec_out) # B,
            elif self.gym_input_embed == 'series-patching' and self.gym_channel_independent:
                enc_out = enc_out[:, -1, :] # bs*nvars, dmodel
                # z: [bs x nvars x d_model]
                enc_out = torch.reshape(enc_out, (-1, self.n_vars, enc_out.shape[-1]))
                # z: [bs x nvars x d_model]
                dec_out = self.head(enc_out)  # B x 1
            elif self.gym_input_embed == 'series-patching' and not self.gym_channel_independent:
                # z: b,np,dm
                enc_out = enc_out[:, -1, :] # bs, dmodel
                dec_out = self.head(enc_out)  # B x 1
            elif self.gym_input_embed == 'ortho-encoding':
                # enc_out: B,C,dmodel
                dec_out = self.head(enc_out.flatten(start_dim=-2))
            else:
                raise NotImplementedError
        else: # encoder-decoder
            raise NotImplementedError

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, rag_raw_data=None):
        # 【1】series preprocessing
        # f_series_preprocessing_sampling:
        # return:
        # - x_enc: list of sampled x_enc, each with shape (B or B*C, S_i, C or 1)
        # - x_mark_enc: list of sampled x_mark_enc, each with shape (B or B*C, S_i, C or 1)
        # - tau: list of tau for non-stationary attention
        # - delta: list of delta for non-stationary attention
        # - enc_out_fa: list of feature attention output
        if self.series_sampling:
            x_enc, x_mark_enc, tau, delta, enc_out_fa = self.f_series_preprocessing_sampling(x_enc, x_mark_enc)
        # f_series_preprocessing_wosampling:
        # return:
        # - x_enc: (B or B*C, S, C or 1)
        # - x_mark_enc: (B or B*C, S, C or 1)
        # - tau: for non-stationary attention
        # - delta: for non-stationary attention
        # - enc_out_fa: feature attention output
        if not self.series_sampling:
            x_enc, x_mark_enc, tau, delta, enc_out_fa = self.f_series_preprocessing_wosampling(x_enc, x_mark_enc)

        # 【2】series encoding
        # f_series_encoding_decomposition_sampling:
        # return:
        # - enc_out: list of encoder output
        # -- if series-encoding, each with shape (B or B*C, S_i, d_model)
        # -- if series-patching, each with shape (B*C, patch_num, d_model)
        if self.series_sampling and self.gym_series_decomp != 'None':
            enc_out = self.f_series_encoding_decomposition_sampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        # f_series_encoding_decomposition_wosampling:
        # return:
        # - enc_out: encoder output
        # -- if series-encoding, (B or B*C, S_i, d_model)
        # -- if series-patching, (B*C, patch_num, d_model)
        # -- if inverted-encoding, (B, S, d_model)
        if not self.series_sampling and self.gym_series_decomp != 'None':
            enc_out = self.f_series_encoding_decomposition_wosampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        # f_series_encoding_wodecomposition_sampling:
        # return:
        # - enc_out: list of encoder output
        # -- if series-encoding, each with shape (B or B*C, S_i, d_model)
        # -- if series-patching, each with shape (B*C, patch_num, d_model)
        if self.series_sampling and self.gym_series_decomp == 'None':
            enc_out = self.f_series_encoding_wodecomposition_sampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        
        # f_series_encoding_wodecomposition_wosampling:
        # return:
        # - enc_out: encoder output
        # -- if series-encoding, (B or B*C, S_i, d_model)
        # -- if series-patching, (B*C, patch_num, d_model)
        # -- if inverted-encoding, (B, C, d_model)
        if not self.series_sampling and self.gym_series_decomp == 'None':
            enc_out = self.f_series_encoding_wodecomposition_wosampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        # 【3】series decoding
        # f_series_decoing_sampling or f_series_decoding_wosampling:
        if self.series_sampling:
            dec_out = self.f_series_decoing_sampling_forecasting(enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data=rag_raw_data)
        else:
            dec_out = self.f_series_decoding_wosampling_forecasting(enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data=rag_raw_data)
        # return: dec_out
        return dec_out
    
    def regressing(self, x_enc, x_mark_enc, x_dec, x_mark_dec, rag_raw_data=None):
        # 【1】series preprocessing
        # f_series_preprocessing_sampling:
        # return:
        # - x_enc: list of sampled x_enc, each with shape (B or B*C, S_i, C or 1)
        # - x_mark_enc: list of sampled x_mark_enc, each with shape (B or B*C, S_i, C or 1)
        # - tau: list of tau for non-stationary attention
        # - delta: list of delta for non-stationary attention
        # - enc_out_fa: list of feature attention output
        if self.series_sampling:
            x_enc, x_mark_enc, tau, delta, enc_out_fa = self.f_series_preprocessing_sampling(x_enc, x_mark_enc)
        # f_series_preprocessing_wosampling:
        # return:
        # - x_enc: (B or B*C, S, C or 1)
        # - x_mark_enc: (B or B*C, S, C or 1)
        # - tau: for non-stationary attention
        # - delta: for non-stationary attention
        # - enc_out_fa: feature attention output
        if not self.series_sampling:
            x_enc, x_mark_enc, tau, delta, enc_out_fa = self.f_series_preprocessing_wosampling(x_enc, x_mark_enc)
        

        # 【2】series encoding
        # f_series_encoding_decomposition_sampling:
        # return:
        # - enc_out: list of encoder output
        # -- if series-encoding, each with shape (B or B*C, S_i, d_model)
        # -- if series-patching, each with shape (B*C, patch_num, d_model)
        if self.series_sampling and self.gym_series_decomp != 'None':
            enc_out = self.f_series_encoding_decomposition_sampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        # f_series_encoding_decomposition_wosampling:
        # return:
        # - enc_out: encoder output
        # -- if series-encoding, (B or B*C, S_i, d_model)
        # -- if series-patching, (B*C, patch_num, d_model)
        # -- if inverted-encoding, (B, S, d_model)
        if not self.series_sampling and self.gym_series_decomp != 'None':
            enc_out = self.f_series_encoding_decomposition_wosampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        # f_series_encoding_wodecomposition_sampling:
        # return:
        # - enc_out: list of encoder output
        # -- if series-encoding, each with shape (B or B*C, S_i, d_model)
        # -- if series-patching, each with shape (B*C, patch_num, d_model)
        if self.series_sampling and self.gym_series_decomp == 'None':
            enc_out = self.f_series_encoding_wodecomposition_sampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        # f_series_encoding_wodecomposition_wosampling:
        # return:
        # - enc_out: encoder output
        # -- if series-encoding, (B or B*C, S_i, d_model)
        # -- if series-patching, (B*C, patch_num, d_model)
        # -- if inverted-encoding, (B, S, d_model)
        if not self.series_sampling and self.gym_series_decomp == 'None':
            enc_out = self.f_series_encoding_wodecomposition_wosampling(x_enc, x_mark_enc, tau, delta, enc_out_fa)
        
            
        # 【3】series decoding
        # f_series_decoing_sampling or f_series_decoding_wosampling:
        if self.series_sampling:
            dec_out = self.f_series_decoing_sampling_regressing(enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data)
        else:
            dec_out = self.f_series_decoding_wosampling_regressing(enc_out, enc_out_fa, x_dec, x_mark_dec, rag_raw_data)
        
        # return: dec_out
        dec_out = dec_out.flatten()
        assert len(dec_out) == self.B
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, rag_raw_data=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_mark_enc = x_mark_enc if self.gym_x_mark else None
            x_mark_dec = x_mark_dec if self.gym_x_mark else None
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, rag_raw_data)
        elif self.task_name == 'finance_regressing':
            x_mark_enc = x_mark_enc if self.gym_x_mark else None
            x_mark_dec = x_mark_dec if self.gym_x_mark else None
            dec_out = self.regressing(x_enc, x_mark_enc, x_dec, x_mark_dec, rag_raw_data)
        else:
            raise NotImplementedError
        return dec_out
