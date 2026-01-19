import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class TSGymRetrievalFusionBlock(nn.Module):
    def __init__(self, period_num, pred_len, channels):
        """
        参数:
            period_num (list): 周期列表，例如 [1, 12, 24]
            pred_len (int): 预测序列长度
            channels (int): 输入特征通道数 (C)
        """
        super().__init__()
        self.period_num = period_num
        self.pred_len = pred_len
        self.channels = channels
        
        # --- 核心组件：多尺度投影层 ---
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        
    def forward(self, pred_from_retrieval):
        """
        参数:
            pred_from_retrieval: 从 RAFTStore 取出的原始数据
                          Shape: [G, B, Pred_Len, Channels]
        返回:
            fused_features: 融合后的特征
                            Shape: [B, Pred_Len, D_Model]
        """
        G, B, P, C = pred_from_retrieval.shape
        assert P == self.pred_len, f"预测长度不匹配: 输入 {P} vs 定义 {self.pred_len}"
        assert C == self.channels, f"通道数不匹配: 输入 {C} vs 定义 {self.channels}"

        # 目的：消除局部窗口的偏移，让 Linear 只学形状
        seq_mean = torch.mean(pred_from_retrieval, dim=2, keepdim=True)
        seq_std = torch.std(pred_from_retrieval, dim=2, keepdim=True) + 1e-5
        x_norm = (pred_from_retrieval - seq_mean) / seq_std

        retrieval_pred_list = []
        # 遍历每一个时间尺度 (Grid)
        for i, pr in enumerate(x_norm):
            g = self.period_num[i]
            pr = pr.reshape(B, P // g, g, C)
            pr = pr[:, :, 0, :]

            # 投影/特征提取 (Projection)
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(B, self.pred_len, self.channels)
            retrieval_pred_list.append(pr)

        # 多尺度融合 (Stack & Sum)
        # [B, P, C]
        fused_out_norm = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)
        
        final_mean = torch.mean(seq_mean, dim=0) # [B, 1, C]
        final_std = torch.mean(seq_std, dim=0)
        
        fused_out_real = fused_out_norm * final_std + final_mean

        return fused_out_real

class TSGymRetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        channel_independence=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key

        self.channel_independence = channel_independence
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            
            # Ensure scalar or array is converted to tensor
            if isinstance(td[0], np.ndarray):
                train_data_all.append(torch.from_numpy(td[0]))
            else:
                train_data_all.append(td[0])
            
            if self.with_dec:
                y_slice = td[1][-(train_data.pred_len + train_data.label_len):]
            else:
                y_slice = td[1][-train_data.pred_len:]
                
            if isinstance(y_slice, np.ndarray):
                y_data_all.append(torch.from_numpy(y_slice))
            else:
                y_data_all.append(y_slice)
            
        self.train_data_all = torch.stack(train_data_all, dim=0).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.stack(y_data_all, dim=0).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)
        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C
            
        return mg, None
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        # assert(seq_len == self.seq_len, channels == self.channels)
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        if self.channel_independence:
            # === 通道独立模式 (CI) ===
            # 逻辑: 把 C 提到前面，和 G 融合。这意味着我们有 G*C 个独立的检索任务在并行运行
            
            # Query: (G, B, S, C) -> (G, C, B, S) -> (G*C, B, S)
            query_feat = x_mg.permute(0, 3, 1, 2).reshape(-1, bsz, seq_len)
            
            # Database: (G, T, S, C) -> (G, C, T, S) -> (G*C, T, S)
            db_feat = self.train_data_all_mg.permute(0, 3, 1, 2).reshape(-1, self.n_train, seq_len)
            
            # Target (Label): (G, T, P, C) -> (G, C, T, P) -> (G*C, T, P)
            y_data_flat = self.y_data_all_mg.permute(0, 3, 1, 2).reshape(-1, self.n_train, self.pred_len)

            # 计算相似度 (periodic_batch_corr 是通用的，它会自动处理第一维)
            # Sim Shape: (G*C, B, T)
            sim = self.periodic_batch_corr(db_feat, query_feat)
            
            num_parallel = self.n_period * self.channels
        else:
            # === 通道依赖模式 (Joint, 原有逻辑) ===
            # 逻辑: 把 C 融合进特征维 S 里
            y_data_flat = self.y_data_all_mg.flatten(start_dim=2) # (G, T, P*C)
            sim = self.periodic_batch_corr(
                self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
                x_mg.flatten(start_dim=2), # G, B, S * C
            ) # G, B, T
            num_parallel = self.n_period
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(num_parallel, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T
            

        # Flatten for TopK (G*C*B or G*B)
        sim = sim.reshape(num_parallel * bsz, self.n_train)
        # sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(num_parallel, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(num_parallel, bsz, self.n_train) # G, B, T

        # Avoid NaN when all values are -inf (happens in small datasets like Illness with large seq_len)
        if torch.isinf(ranking_sim).all():
             ranking_sim.fill_(0.0)
        else:
             # Handle batch items individually if necessary, but global check is safer for now
             # Replace rows that are all -inf with 0
             row_max = torch.max(ranking_sim, dim=-1, keepdim=True)[0]
             mask_all_inf = torch.isinf(row_max)
             if mask_all_inf.any():
                 ranking_sim = torch.where(mask_all_inf, torch.zeros_like(ranking_sim), ranking_sim)

        data_len, seq_len, channels = self.train_data_all.shape
            
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)

        # ranking_prob: (Parallel, B, T)
        # y_data_flat:  (Parallel, T, Feature_Dim)
        # CI 模式: (G*C, B, T) * (G*C, T, P) -> (G*C, B, P)
        # Joint 模式: (G, B, T) * (G, T, P*C) -> (G, B, P*C)
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_flat.to(x.device))

        if self.channel_independence:
            # CI Output: (G*C, B, P) -> 还原为 (G, C, B, P) -> 转置为 (G, B, P, C)
            pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, self.channels, bsz, -1)
            pred_from_retrieval = pred_from_retrieval.permute(0, 2, 3, 1) # G, B, P, C
        else:
            # Joint Output: (G, B, P*C) -> 还原为 (G, B, P, C)
            pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, self.channels)

        pred_from_retrieval = pred_from_retrieval.to(x.device)
        
        return pred_from_retrieval
    
    def retrieve_cpu(self, x, index, train=True):
        # index = index.to(x.device) # index device handling moved below
        
        bsz, seq_len, channels = x.shape
        # assert(seq_len == self.seq_len, channels == self.channels)
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        if self.channel_independence:
            # === 通道独立模式 (CI) ===
            # 逻辑: 把 C 提到前面，和 G 融合。这意味着我们有 G*C 个独立的检索任务在并行运行
            
            # Query: (G, B, S, C) -> (G, C, B, S) -> (G*C, B, S)
            query_feat = x_mg.permute(0, 3, 1, 2).reshape(-1, bsz, seq_len)
            
            # Database: (G, T, S, C) -> (G, C, T, S) -> (G*C, T, S)
            db_feat = self.train_data_all_mg.permute(0, 3, 1, 2).reshape(-1, self.n_train, seq_len)
            
            # Target (Label): (G, T, P, C) -> (G, C, T, P) -> (G*C, T, P)
            y_data_flat = self.y_data_all_mg.permute(0, 3, 1, 2).reshape(-1, self.n_train, self.pred_len)
            num_parallel = self.n_period * self.channels
        else:
            # === 通道依赖模式 (Joint) ===
            # 逻辑: 把 C 融合进特征维 S 里
            query_feat = x_mg.flatten(start_dim=2) # G, B, S * C
            db_feat = self.train_data_all_mg.flatten(start_dim=2) # G, T, S * C
            y_data_flat = self.y_data_all_mg.flatten(start_dim=2) # (G, T, P*C)
            num_parallel = self.n_period

        # Normalize Query (GPU)
        query_feat = query_feat - torch.mean(query_feat, dim=2, keepdim=True)
        query_feat = F.normalize(query_feat, dim=2)

        # Prepare Masking Indices (GPU)
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1, device=x.device)
            sliding_index = sliding_index.unsqueeze(0).repeat(bsz, 1)
            index_dev = index.to(x.device)
            sliding_index = sliding_index + (index_dev - self.seq_len - self.pred_len + 1).unsqueeze(1)

        # Block-wise TopK
        chunk_size = 512 
        train_len = self.n_train
        iters = math.ceil(train_len / chunk_size)
        
        topk_values = None
        topk_indices = None
        
        for i in range(iters):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, train_len)
            current_chunk_len = end_idx - start_idx
            
            # Load DB Chunk (CPU -> GPU)
            db_chunk = db_feat[:, start_idx:end_idx].to(x.device)
            
            # Normalize DB Chunk
            db_chunk = db_chunk - torch.mean(db_chunk, dim=2, keepdim=True)
            db_chunk = F.normalize(db_chunk, dim=2)
            
            # Compute Sim: [Num_Parallel, Batch, Chunk_Len]
            sim_chunk = torch.bmm(query_feat, db_chunk.transpose(1, 2))
            
            # Apply Mask
            if train:
                curr_sliding = sliding_index - start_idx
                valid = (curr_sliding >= 0) & (curr_sliding < current_chunk_len)
                if valid.any():
                    b_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand_as(curr_sliding)
                    valid_b = b_idx[valid]
                    valid_c = curr_sliding[valid]
                    
                    mask_chunk = torch.zeros(bsz, current_chunk_len, device=x.device, dtype=torch.bool)
                    mask_chunk[valid_b, valid_c] = True
                    mask_chunk = mask_chunk.unsqueeze(0).expand(num_parallel, -1, -1)
                    
                    sim_chunk = sim_chunk.masked_fill(mask_chunk, float('-inf'))

            # TopK on Chunk
            k = min(self.topm, current_chunk_len)
            chunk_val, chunk_idx = torch.topk(sim_chunk, k, dim=2)
            chunk_idx += start_idx
            
            # Merge
            if topk_values is None:
                topk_values = chunk_val
                topk_indices = chunk_idx
            else:
                combined_val = torch.cat([topk_values, chunk_val], dim=2)
                combined_idx = torch.cat([topk_indices, chunk_idx], dim=2)
                
                topk_values, meta_idx = torch.topk(combined_val, self.topm, dim=2)
                topk_indices = torch.gather(combined_idx, 2, meta_idx)

        # Softmax
        ranking_prob = F.softmax(topk_values / self.temperature, dim=2)
        
        # Gather Y (CPU -> GPU)
        gathered_y = []
        topk_indices_cpu = topk_indices.cpu()
        
        for i in range(num_parallel):
            idxs = topk_indices_cpu[i]
            val = y_data_flat[i][idxs]
            gathered_y.append(val)
            
        gathered_y = torch.stack(gathered_y, dim=0).to(x.device)
        
        pred_from_retrieval = torch.matmul(ranking_prob.unsqueeze(2), gathered_y).squeeze(2)

        if self.channel_independence:
            # CI Output: (G*C, B, P) -> 还原为 (G, C, B, P) -> 转置为 (G, B, P, C)
            pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, self.channels, bsz, -1)
            pred_from_retrieval = pred_from_retrieval.permute(0, 2, 3, 1) # G, B, P, C
        else:
            # Joint Output: (G, B, P*C) -> 还原为 (G, B, P, C)
            pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, self.channels)

        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False):
        assert(self.train_data_all_mg != None)
        
        # data_with_index = IndexWrapper(data)

        rt_loader = DataLoader(
            data, # data_with_index
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in tqdm(rt_loader, desc="Retrieval Progress"):
                if 'traffic' in data.root_path or 'electricity' in data.root_path:
                    pred_from_retrieval = self.retrieve_cpu(batch_x.float().to(data.device), index, train=train)
                else:
                    pred_from_retrieval = self.retrieve(batch_x.float().to(data.device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        
        return retrievals
    
class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            
            # Ensure scalar or array is converted to tensor
            if isinstance(td[0], np.ndarray):
                train_data_all.append(torch.from_numpy(td[0]))
            else:
                train_data_all.append(td[0])
            
            if self.with_dec:
                y_slice = td[1][-(train_data.pred_len + train_data.label_len):]
            else:
                y_slice = td[1][-train_data.pred_len:]

            if isinstance(y_slice, np.ndarray):
                y_data_all.append(torch.from_numpy(y_slice))
            else:
                y_data_all.append(y_slice)
            
        self.train_data_all = torch.stack(train_data_all, dim=0).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.stack(y_data_all, dim=0).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        # assert(seq_len == self.seq_len, channels == self.channels)
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        # Avoid NaN when all values are -inf (happens in small datasets like Illness with large seq_len)
        if torch.isinf(ranking_sim).all():
             ranking_sim.fill_(0.0)
        else:
             # Handle batch items individually if necessary, but global check is safer for now
             # Replace rows that are all -inf with 0
             row_max = torch.max(ranking_sim, dim=-1, keepdim=True)[0]
             mask_all_inf = torch.isinf(row_max)
             if mask_all_inf.any():
                 ranking_sim = torch.where(mask_all_inf, torch.zeros_like(ranking_sim), ranking_sim)

        data_len, seq_len, channels = self.train_data_all.shape
            
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        # ranking_prob = ranking_prob.detach().cpu() # G, B, T
        
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all.to(x.device)).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)
        
        return pred_from_retrieval
    
    def retrieve_cpu(self, x, index, train=True):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len, f"seq_len mismatch: {seq_len} vs {self.seq_len}"
        assert channels == self.channels, f"channels mismatch: {channels} vs {self.channels}"
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        # === 通道依赖模式 (Joint) ===
        query_feat = x_mg.flatten(start_dim=2)
        db_feat = self.train_data_all_mg.flatten(start_dim=2)
        y_data_flat = self.y_data_all_mg.flatten(start_dim=2)
        num_parallel = self.n_period

        # Normalize Query (GPU)
        query_feat = query_feat - torch.mean(query_feat, dim=2, keepdim=True)
        query_feat = F.normalize(query_feat, dim=2)

        # Prepare Masking Indices (GPU)
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1, device=x.device)
            sliding_index = sliding_index.unsqueeze(0).repeat(bsz, 1)
            index_dev = index.to(x.device)
            sliding_index = sliding_index + (index_dev - self.seq_len - self.pred_len + 1).unsqueeze(1)

        # Block-wise TopK
        chunk_size = 512 
        train_len = self.n_train
        iters = math.ceil(train_len / chunk_size)
        
        topk_values = None
        topk_indices = None
        
        for i in range(iters):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, train_len)
            current_chunk_len = end_idx - start_idx
            
            # Load DB Chunk (CPU -> GPU)
            db_chunk = db_feat[:, start_idx:end_idx].to(x.device)
            
            # Normalize DB Chunk
            db_chunk = db_chunk - torch.mean(db_chunk, dim=2, keepdim=True)
            db_chunk = F.normalize(db_chunk, dim=2)
            
            # Compute Sim: [Num_Parallel, Batch, Chunk_Len]
            sim_chunk = torch.bmm(query_feat, db_chunk.transpose(1, 2))
            
            # Apply Mask
            if train:
                curr_sliding = sliding_index - start_idx
                valid = (curr_sliding >= 0) & (curr_sliding < current_chunk_len)
                if valid.any():
                    b_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand_as(curr_sliding)
                    valid_b = b_idx[valid]
                    valid_c = curr_sliding[valid]
                    
                    mask_chunk = torch.zeros(bsz, current_chunk_len, device=x.device, dtype=torch.bool)
                    mask_chunk[valid_b, valid_c] = True
                    mask_chunk = mask_chunk.unsqueeze(0).expand(num_parallel, -1, -1)
                    
                    sim_chunk = sim_chunk.masked_fill(mask_chunk, float('-inf'))

            # TopK on Chunk
            k = min(self.topm, current_chunk_len)
            chunk_val, chunk_idx = torch.topk(sim_chunk, k, dim=2)
            chunk_idx += start_idx
            
            # Merge
            if topk_values is None:
                topk_values = chunk_val
                topk_indices = chunk_idx
            else:
                combined_val = torch.cat([topk_values, chunk_val], dim=2)
                combined_idx = torch.cat([topk_indices, chunk_idx], dim=2)
                
                topk_values, meta_idx = torch.topk(combined_val, self.topm, dim=2)
                topk_indices = torch.gather(combined_idx, 2, meta_idx)

        # Softmax
        ranking_prob = F.softmax(topk_values / self.temperature, dim=2)
        
        # Gather Y (CPU -> GPU)
        gathered_y = []
        topk_indices_cpu = topk_indices.cpu()
        
        for i in range(num_parallel):
            idxs = topk_indices_cpu[i]
            val = y_data_flat[i][idxs]
            gathered_y.append(val)
            
        gathered_y = torch.stack(gathered_y, dim=0).to(x.device)
        
        pred_from_retrieval = torch.matmul(ranking_prob.unsqueeze(2), gathered_y).squeeze(2)
        pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, self.channels)

        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False):
        assert(self.train_data_all_mg != None)
        
        # data_with_index = IndexWrapper(data)

        rt_loader = DataLoader(
            data, # data_with_index
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in tqdm(rt_loader):
                if 'traffic' in data.root_path or 'electricity' in data.root_path:
                    pred_from_retrieval = self.retrieve_cpu(batch_x.float().to(data.device), index, train=train)
                else:
                    pred_from_retrieval = self.retrieve(batch_x.float().to(data.device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        
        return retrievals
    
class IndexWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        # 调用原始 dataset 的 getitem
        data = self.dataset[index]
        # 强行把 index 加进去返回
        return index, *data 
    
    def __len__(self):
        return len(self.dataset)
