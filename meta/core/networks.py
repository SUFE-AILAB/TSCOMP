"""
Neural Network Model Definition Module.

This module contains implementations of various neural network models for meta-learning, mainly including:

Transformer Components:
    - TransformerEncoderLayerWithAttn: Custom Transformer Encoder Layer that supports returning attention weights
    - SimpleAttentionLayer: Simplified Attention Layer without Q/K/V linear projection

ICL (In-Context Learning) Models:
    - BaseMetaICL: Base class for all ICL models, contains attention mask generation logic
    - MetaICL: Standard multi-head attention ICL model
    - MetaSimpleICL: Simplified ICL model
    - MetaICLFrozenComp: ICL model with frozen component embeddings
    - MetaICLAddComp: ICL model with additive component embeddings
    - MetaICLLabelEncoder: ICL model using label encoding
    - MetaICLDeepInput: ICL model with deep input projection

Other Models:
    - meta_predictor: Basic MLP meta-predictor
    - MetaICLTabPFN: TabPFN-based ICL model

model_type Parameter Description:
    The model_type parameter controls the attention mask strategy, format is "{architecture}-{mask_strategy}".
    Mask strategy is parsed through the BaseMetaICL._get_mask() method:

    - nomask: No mask, fully connected attention
    - simplemask: Simple mask, test points cannot see each other
    - mask-similar-meta: Mask training samples with same meta-features (formerly icl-mq)
    - mask-train-self: Training samples only see themselves/diagonal (formerly icl-hls)
    - mask-test-train: Test samples cannot see training samples (formerly icl-ls)
    - mask-train-peers: Training samples cannot see each other (formerly icl-yx)

Author: TSGym
"""
import torch
from torch import nn
from torch.nn import functional as F
import inspect

# ============================================================================
# Custom Transformer Components with Attention Weight Support
# ============================================================================

class TransformerEncoderLayerWithAttn(nn.Module):
    """
    Custom Transformer Encoder Layer that supports returning Softmax-processed Attention Scores.
    Functions the same as nn.TransformerEncoderLayer, but can return attention scores.

    Note: The returned attention scores have already been processed by Softmax, i.e.:
        attn_scores = softmax(Q @ K^T / sqrt(d_k) / temporal)
    Each row sums to 1, representing the attention weight distribution of that position to other positions.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True, k=0.0, temporal=1.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.k = k
        self.temporal = temporal
        self.batch_first = batch_first

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def _multi_head_attention(self, q, k, v, attn_mask=None, need_weights=False):
        batch_size, seq_len, d_model = q.shape
        d_k = d_model // self.nhead

        q = q.view(batch_size, seq_len, self.nhead, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)

        if self.k > 0.0:
            inf_mask = attn_scores == float('-inf')
            attn_scores_masked = attn_scores.clone()
            attn_scores_masked[inf_mask] = float('inf')

            k_percent = self.k / 100.0
            valid_count = (~inf_mask).sum(dim=-1, keepdim=True)
            k_count = (valid_count * k_percent).long().clamp(min=1)
            k_count = torch.minimum(k_count, valid_count - 1)

            sorted_scores, _ = torch.sort(attn_scores_masked, dim=-1)
            kth_idx = (k_count - 1).squeeze(-1)
            kth_idx_expanded = kth_idx.unsqueeze(-1)
            kth_values = torch.gather(sorted_scores, dim=-1, index=kth_idx_expanded)

            truncate_mask = (attn_scores <= kth_values) & (~inf_mask)
            attn_scores[truncate_mask] = float('-inf')

        attn_weights = F.softmax(attn_scores / self.temporal, dim=-1)
        attn_weights = self.dropout1(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)

        if need_weights:
            attn_scores_avg = attn_weights.mean(dim=1)
            return output, attn_scores_avg
        return output, None

    def forward(self, src, src_mask=None, need_weights=False):
        if not self.batch_first:
            src = src.transpose(0, 1)

        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)

        attn_output, attn_scores = self._multi_head_attention(q, k, v, attn_mask=src_mask, need_weights=need_weights)

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        if not self.batch_first:
            src = src.transpose(0, 1)

        if need_weights:
            return src, attn_scores
        return src, None


class SimpleAttentionLayer(nn.Module):
    """Simplified Attention Layer without Q/K/V linear projection."""
    def __init__(self, d_model, dim_feedforward=256, dropout=0.1, k=0.0, temporal=1.0):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** -0.5
        self.k = k
        self.temporal = temporal

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src, src_mask=None, need_weights=False):
        normed = self.norm1(src)
        attn_scores = torch.bmm(normed, normed.transpose(1, 2)) * self.scale

        if src_mask is not None:
            attn_scores = attn_scores + src_mask.unsqueeze(0)

        if self.k > 0.0:
            batch_size, seq_len, _ = attn_scores.shape
            inf_mask = attn_scores == float('-inf')
            attn_scores_masked = attn_scores.clone()
            attn_scores_masked[inf_mask] = float('inf')

            k_percent = self.k / 100.0
            valid_count = (~inf_mask).sum(dim=-1, keepdim=True)
            k_count = (valid_count * k_percent).long().clamp(min=1)
            k_count = torch.minimum(k_count, valid_count - 1)

            sorted_scores, _ = torch.sort(attn_scores_masked, dim=-1)
            kth_idx = (k_count - 1).squeeze(-1)
            kth_idx_expanded = kth_idx.unsqueeze(-1)
            kth_values = torch.gather(sorted_scores, dim=-1, index=kth_idx_expanded)

            truncate_mask = (attn_scores <= kth_values) & (~inf_mask)
            attn_scores[truncate_mask] = float('-inf')

        attn_weights = F.softmax(attn_scores / self.temporal, dim=-1)
        attn_weights = self.dropout1(attn_weights)

        attn_output = torch.bmm(attn_weights, src)
        src = src + attn_output

        normed2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(normed2))))
        src = src + self.dropout2(ff_output)

        if need_weights:
            return src, attn_weights
        return src, None


# ============================================================================
# Base ICL Model
# ============================================================================

class BaseMetaICL(nn.Module):
    """Base class for all ICL models."""
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1,
                 num_layers=2, model_type='icl-nomasktrain', k=0.0, temporal=1.0):
        super().__init__()
        self.model_type = model_type
        self.d_model = d_model
        self.num_layers = num_layers
        self.k = k
        self.temporal = temporal
        self.n_col = n_col

        embedding_dim = embed_dim_meta_feature // len(n_col)
        self.embeddings = nn.ModuleList([nn.Embedding(int(_), embedding_dim) for _ in n_col])
        embed_dim_component = len(n_col) * embedding_dim

        input_size = embed_dim_component + embed_dim_meta_feature
        self.input_proj = self._build_input_proj(input_size, d_model)
        self.transformer_layers = self._build_transformer_layers(d_model, dropout, num_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, 1)
        )

    def _build_input_proj(self, input_size, d_model):
        """Override in subclasses for custom input projection."""
        return nn.Linear(input_size, d_model)

    def _build_transformer_layers(self, d_model, dropout, num_layers):
        """Override in subclasses for custom transformer layers."""
        raise NotImplementedError

    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), model=None):
        if model is not None:
            param_dict = {pn: p for pn, p in model.named_parameters()}
        else:
            param_dict = {pn: p for pn, p in self.named_parameters()}

        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extras_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extras_args)
        return optimizer

    def _get_embedding(self, components, meta_feature):
        """Override in subclasses for custom embedding."""
        component_embedding = torch.hstack([e(components[:, i]) for i, e in enumerate(self.embeddings)])
        embedding = torch.cat((component_embedding, meta_feature), dim=1)
        return self.input_proj(embedding)

    def _get_mask(self, N_train, N_test, device, train_meta=None):
        """Generate attention mask based on model_type.

        Mask strategies:
            - nomask: No mask, full attention
            - simplemask: Test samples cannot see each other
            - mask-similar-meta: Mask train samples with same meta-features (formerly icl-mq)
            - mask-train-self: Train samples only see themselves (diagonal) (formerly icl-hls)
            - mask-test-train: Test samples cannot see train samples (formerly icl-ls)
            - mask-train-peers: Train samples cannot see each other (formerly icl-yx)
        """
        L = N_train + N_test
        mask = torch.zeros((L, L), device=device)

        if 'nomask' in self.model_type:
            return mask

        mask[:N_train, N_train:] = float('-inf')

        if self.model_type in ['mask-similar-meta'] and train_meta is not None:
            # Mask train samples with same meta-features (formerly icl-mq)
            sim_mask = (train_meta.unsqueeze(1) == train_meta.unsqueeze(0)).all(dim=-1)
            sim_mask.fill_diagonal_(False)
            tl_mask = torch.zeros((N_train, N_train), device=device)
            tl_mask.masked_fill_(sim_mask, float('-inf'))
            mask[:N_train, :N_train] = tl_mask
        elif self.model_type in ['mask-train-self', 'mask-test-train']:
            # Train samples only see themselves (diagonal) (formerly icl-hls, icl-ls)
            tl_mask = torch.full((N_train, N_train), float('-inf'), device=device)
            tl_mask.fill_diagonal_(0.0)
            mask[:N_train, :N_train] = tl_mask
        elif self.model_type == 'mask-train-peers':
            # Train samples cannot see each other (formerly icl-yx)
            tl_mask = torch.zeros((N_train, N_train), device=device)
            tl_mask.fill_diagonal_(float('-inf'))
            mask[:N_train, :N_train] = tl_mask

        if self.model_type == 'mask-test-train':
            # Test samples cannot see train samples (formerly icl-ls)
            mask[N_train:, :N_train] = float('-inf')

        if N_test > 0:
            if self.model_type == 'mask-train-peers':
                # Test samples cannot see each other (formerly icl-yx)
                mask[N_train:, N_train:] = float('-inf')
            elif 'simplemask' in self.model_type:
                mask[N_train:, N_train:] = float('-inf')
            else:
                br_mask = torch.full((N_test, N_test), float('-inf'), device=device)
                br_mask.fill_diagonal_(0.0)
                mask[N_train:, N_train:] = br_mask

        return mask

    def _forward_transformer(self, src, mask, need_weights=False):
        output = src
        all_attn_scores = []

        for layer in self.transformer_layers:
            output, attn_scores = layer(output, src_mask=mask, need_weights=need_weights)
            if need_weights and attn_scores is not None:
                all_attn_scores.append(attn_scores)

        if need_weights:
            return output, all_attn_scores
        return output, None

    def forward(self, train_data, test_data=None, return_attention=False):
        train_comp, train_meta = train_data
        train_tokens = self._get_embedding(train_comp, train_meta)

        if test_data is None:
            N_train = train_tokens.size(0)
            mask = self._get_mask(N_train, 0, train_tokens.device, train_meta)

            src = train_tokens.unsqueeze(0)
            output, all_attn_scores = self._forward_transformer(src, mask, need_weights=return_attention)
            output = output.squeeze(0)
            pred = self.out(output)

            if return_attention:
                raise NotImplementedError("Not implemented for training mode")
            return None, pred

        else:
            test_comp, test_meta = test_data
            test_tokens = self._get_embedding(test_comp, test_meta)

            N_train = train_tokens.size(0)
            N_test = test_tokens.size(0)

            tokens = torch.cat([train_tokens, test_tokens], dim=0)
            src = tokens.unsqueeze(0)

            mask = self._get_mask(N_train, N_test, tokens.device, train_meta)

            output, all_attn_scores = self._forward_transformer(src, mask, need_weights=return_attention)
            output = output.squeeze(0)

            test_output = output[N_train:]
            pred = self.out(test_output)

            if return_attention:
                train_to_train_list = []
                train_to_test_list = []
                test_to_train_list = []
                test_to_test_list = []

                for layer_scores in all_attn_scores:
                    scores = layer_scores[0]
                    train_to_train_list.append(scores[:N_train, :N_train])
                    train_to_test_list.append(scores[:N_train, N_train:])
                    test_to_train_list.append(scores[N_train:, :N_train])
                    test_to_test_list.append(scores[N_train:, N_train:])

                train_to_train = torch.stack(train_to_train_list, dim=0)
                train_to_test = torch.stack(train_to_test_list, dim=0)
                test_to_train = torch.stack(test_to_train_list, dim=0)
                test_to_test = torch.stack(test_to_test_list, dim=0)

                attention_info = {
                    'attention_scores': all_attn_scores,
                    'N_train': N_train,
                    'N_test': N_test,
                    'train_to_train': train_to_train,
                    'train_to_test': train_to_test,
                    'test_to_train': test_to_train,
                    'test_to_test': test_to_test,
                }
                return None, pred, attention_info

            return None, pred


# ============================================================================
# ICL Model Variants
# ============================================================================

class MetaICL(BaseMetaICL):
    """Standard ICL with multi-head attention."""
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1, nhead=4,
                 num_layers=2, model_type='icl-nomasktrain', k=0.0, temporal=1.0):
        self.nhead = nhead
        super().__init__(n_col, embed_dim_meta_feature, d_model, dropout, num_layers, model_type, k, temporal)

    def _build_transformer_layers(self, d_model, dropout, num_layers):
        return nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=d_model, nhead=self.nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, k=self.k, temporal=self.temporal
            ) for _ in range(num_layers)
        ])


class MetaSimpleICL(BaseMetaICL):
    """Simplified ICL without Q/K/V projections."""
    def _build_transformer_layers(self, d_model, dropout, num_layers):
        return nn.ModuleList([
            SimpleAttentionLayer(
                d_model=d_model, dim_feedforward=d_model * 4,
                dropout=dropout, k=self.k, temporal=self.temporal
            ) for _ in range(num_layers)
        ])


class MetaICLFrozenComp(MetaICL):
    """ICL with frozen component embeddings."""
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1, nhead=4,
                 num_layers=2, model_type='icl-frozencomp', k=0.0, temporal=1.0):
        super().__init__(n_col, embed_dim_meta_feature, d_model, dropout, nhead, num_layers, model_type, k, temporal)
        for emb in self.embeddings:
            emb.weight.requires_grad = False
        print(f"MetaICLFrozenComp: Component embeddings FROZEN")


class MetaICLAddComp(MetaICL):
    """ICL with additive component embeddings."""
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1, nhead=4,
                 num_layers=2, model_type='icl-addcomp', k=0.0, temporal=1.0, add_embed_dim=256):
        super().__init__(n_col, embed_dim_meta_feature, d_model, dropout, nhead, num_layers, model_type, k, temporal)
        self.add_embed_dim = add_embed_dim

        # Recreate embeddings with additive embedding dimension
        self.embeddings = nn.ModuleList([nn.Embedding(int(_), add_embed_dim) for _ in n_col])
        input_size = add_embed_dim + embed_dim_meta_feature
        self.input_proj = nn.Linear(input_size, d_model)

        print(f"MetaICLAddComp: add_embed_dim={add_embed_dim}")

    def _get_embedding(self, components, meta_feature):
        component_embeddings = [e(components[:, i]) for i, e in enumerate(self.embeddings)]
        component_embedding = torch.stack(component_embeddings, dim=0).sum(dim=0)
        embedding = torch.cat((component_embedding, meta_feature), dim=1)
        return self.input_proj(embedding)


class MetaICLLabelEncoder(MetaICL):
    """ICL without nn.Embedding, using standardized label values."""
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1, nhead=4,
                 num_layers=2, model_type='icl-labelencoder', k=0.0, temporal=1.0):
        self.n_components = len(n_col)
        nn.Module.__init__(self)
        self.model_type = model_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.k = k
        self.temporal = temporal

        self.register_buffer('n_col', torch.tensor(n_col, dtype=torch.float32))
        input_size = self.n_components + embed_dim_meta_feature
        self.input_proj = nn.Linear(input_size, d_model)
        self.transformer_layers = self._build_transformer_layers(d_model, dropout, num_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(),
            nn.Dropout(p=dropout), nn.Linear(d_model, 1)
        )
        print(f"MetaICLLabelEncoder: No embedding")

    def _get_embedding(self, components, meta_feature):
        components_float = components.float()
        n_col_safe = torch.clamp(self.n_col - 1, min=1.0)
        components_normalized = (components_float / n_col_safe) - 0.5
        embedding = torch.cat((components_normalized, meta_feature), dim=1)
        return self.input_proj(embedding)

    @property
    def embeddings(self):
        return nn.ModuleList()


class MetaICLDeepInput(MetaICL):
    """ICL with deeper input projection."""
    def _build_input_proj(self, input_size, d_model):
        return nn.Sequential(
            nn.Linear(input_size, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )


# ============================================================================
# MLP Model
# ============================================================================

class meta_predictor(nn.Module):
    """Basic MLP meta-predictor."""
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1, model_pred_len=False, n_layers=2):
        super().__init__()
        embedding_dim = embed_dim_meta_feature // len(n_col)
        self.embeddings = nn.ModuleList([nn.Embedding(int(_), embedding_dim) for _ in n_col])
        embed_dim_component = len(n_col) * embedding_dim

        self.model_pred_len = model_pred_len
        if model_pred_len:
            embed_dim_pred_len = embed_dim_component
            self.embeddings_pred_len = nn.Embedding(8, embed_dim_pred_len)
            input_size = embed_dim_component + embed_dim_meta_feature + embed_dim_pred_len
        else:
            input_size = embed_dim_component + embed_dim_meta_feature

        layers = [nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(), nn.Dropout(p=dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(), nn.Dropout(p=dropout)])
        layers.append(nn.Linear(d_model, 1))
        self.out = nn.Sequential(*layers)

    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), model=None):
        if model is not None:
            param_dict = {pn: p for pn, p in model.named_parameters()}
        else:
            param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extras_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extras_args)
        return optimizer

    def forward(self, components, meta_feature, pred_len=None):
        component_embedding = torch.hstack([e(components[:, i]) for i, e in enumerate(self.embeddings)])
        if pred_len is not None and self.model_pred_len:
            pred_len_embedding = self.embeddings_pred_len(pred_len)
            embedding = torch.cat((component_embedding, meta_feature, pred_len_embedding.squeeze()), dim=1)
        else:
            embedding = torch.cat((component_embedding, meta_feature), dim=1)
        pred = self.out(embedding)
        return component_embedding, pred


# ============================================================================
# TabPFN Model
# ============================================================================

from tabpfn import TabPFNRegressor

class MetaICLTabPFN(nn.Module):
    """ICL using pre-trained TabPFN."""
    def __init__(self, n_col, model_path='/data2/coding/tsgym/tabpfn-v2-regressor/tabpfn-v2-regressor-v2_default.ckpt', device='cuda'):
        super().__init__()
        self.n_categorical = len(n_col)
        self.categorical_indices = list(range(self.n_categorical))
        self.model = TabPFNRegressor(model_path=model_path, device=device, categorical_features_indices=self.categorical_indices)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        return torch.optim.SGD([torch.tensor([0.0], requires_grad=True)], lr=learning_rate)

    def forward(self, train_data, test_data=None, train_targets=None):
        if test_data is None:
            return None, torch.zeros(train_data[0].shape[0], device=train_data[0].device)

        train_comp, train_meta = train_data
        test_comp, test_meta = test_data

        if train_targets is None:
            raise ValueError("train_targets must be provided for MetaICLTabPFN")

        X_train = torch.cat([train_comp.float(), train_meta], dim=1)
        X_test = torch.cat([test_comp.float(), test_meta], dim=1)
        y_train = train_targets

        original_device = X_train.device

        X_train_np = X_train.detach().cpu().numpy()
        y_train_np = y_train.detach().cpu().numpy()
        X_test_np = X_test.detach().cpu().numpy()

        self.model.fit(X_train_np, y_train_np)
        y_pred = self.model.predict(X_test_np)

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)

        return None, y_pred.to(original_device)
