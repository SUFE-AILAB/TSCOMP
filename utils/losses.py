# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np
import pdb
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, preds, trues):
        epsilon = 1e-6
        loss = torch.mean(torch.abs((trues - preds) / (trues + epsilon)))
        # torch.mean(torch.abs(divide_no_nan(preds, trues) - 1))
        return loss
    
    
class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)

class PSLoss(nn.Module):
    # https://github.com/Dilfiraa/PS_Loss/blob/main/DLinear/exp/exp_main.py
    def __init__(self, patch_len_threshold):
        super(PSLoss, self).__init__()
        self.patch_len_threshold = patch_len_threshold
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def create_patches(self, x, patch_len, stride):
        
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(self, true, pred):

        # Get patch length an stride
        true_fft = torch.fft.rfft(true.float(), dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list).item()
        top_index = max(1, top_index)
        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        patch_len = max(2, int(patch_len))
        stride = patch_len // 2
        
        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch
    
    def patch_wise_structural_loss(self, true_patch, pred_patch):
        # Cast to float32 for stability
        true_patch = true_patch.float()
        pred_patch = pred_patch.float()
        
        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        
        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var + 1e-6)
        pred_patch_std = torch.sqrt(pred_patch_var + 1e-6)
        
        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        
        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        
        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        
        if torch.isnan(linear_corr_loss) or torch.isnan(var_loss) or torch.isnan(mean_loss):
             print(f"Structural Loss NaNs: Corr={linear_corr_loss.item()}, Var={var_loss.item()}, Mean={mean_loss.item()}")

        return linear_corr_loss, var_loss, mean_loss
    
    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss, parameters):
        
        # Cast to float32 for stability
        true = true.float()
        pred = pred.float()
        
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var + 1e-6)
        pred_std = torch.sqrt(pred_var + 1e-6)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2*true_std*pred_std + 1e-5) / (true_var + pred_var + 1e-5)
   
        # Gradiant based dynamic weighting
        # Ensure loss terms are on the same device/dtype as parameters if needed, but here we want gradients.
        # Since we cast true/pred to float for stats, the loss values are float.
        # parameters are bfloat16. autograd handles this.
        
        corr_gradient = torch.autograd.grad(corr_loss, parameters, create_graph=True)[0] # 这里使用什么参数是关键，每个sota模型都不一样
        var_gradient = torch.autograd.grad(var_loss, parameters, create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, parameters, create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        eps = 1e-8
        aplha = gradiant_avg.norm().detach() / (corr_gradient.norm().detach() + eps)
        beta =  gradiant_avg.norm().detach() /  (var_gradient.norm().detach() + eps)
        gamma = gradiant_avg.norm().detach() / (mean_gradient.norm().detach() + eps)
        gamma = gamma * torch.mean(linear_sim*var_sim).detach()
        
        gamma = gamma * torch.mean(linear_sim*var_sim).detach()
        
        if torch.isnan(aplha) or torch.isnan(beta) or torch.isnan(gamma):
             print(f"Weights NaNs: Alpha={aplha}, Beta={beta}, Gamma={gamma}")
             print(f"Grad Norms: Corr={corr_gradient.norm().item()}, Var={var_gradient.norm().item()}, Mean={mean_gradient.norm().item()}")
             print(f"Grad Avg Norm: {gradiant_avg.norm().item()}")
        
        return aplha, beta, gamma

    def ps_loss(self, true, pred, parameters):

        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)
        
        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)
        
        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss, parameters)

        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss
        
        return ps_loss

    def get_last_parameters(self, model):
        # 只针对TSGym
        # get the last layer parameters for finetuning
        params = model.head.parameters()
        if not model.gym_series_sampling and not model.gym_encoder_only and model.gym_input_embed == 'series-encoding':
            params = model.decoder_projection.parameters()
            
        return list(params)

    def forward(self, trues, preds, model):
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print(f"NaN/Inf detected in model predictions! Max: {preds.max().item()}, Min: {preds.min().item()}")
            
        parameters = self.get_last_parameters(model)
        loss = self.ps_loss(trues, preds, parameters)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf detected in PSLoss output!")
            
        return loss

class FreDFLoss(nn.Module):
    # https://github.com/Master-PLC/FreDF/blob/main/exp/exp_long_term_forecasting.py
    def __init__(self, args, device, mask=None):
        super(FreDFLoss, self).__init__()
        self.args = args
        self.device = device
        self.mask = None # 暂时不涉及add_noise noise_amp

    def forward(self, outputs, batch_y):
        loss_auxi = 0
        if self.args.auxi_lambda:
            # fft shape: [B, P, D]
            if self.args.auxi_mode == "fft":
                loss_auxi =  - torch.fft.fft(batch_y.float(), dim=1)

            elif self.args.auxi_mode == "rfft":
                if self.args.auxi_type == 'complex':
                    loss_auxi = torch.fft.rfft(outputs.float(), dim=1) - torch.fft.rfft(batch_y.float(), dim=1)
                elif self.args.auxi_type == 'complex-phase':
                    loss_auxi = (torch.fft.rfft(outputs.float(), dim=1) - torch.fft.rfft(batch_y.float(), dim=1)).angle()
                elif self.args.auxi_type == 'complex-mag-phase':
                    loss_auxi_mag = (torch.fft.rfft(outputs.float(), dim=1) - torch.fft.rfft(batch_y.float(), dim=1)).abs()
                    loss_auxi_phase = (torch.fft.rfft(outputs.float(), dim=1) - torch.fft.rfft(batch_y.float(), dim=1)).angle()
                    loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                elif self.args.auxi_type == 'phase':
                    loss_auxi = torch.fft.rfft(outputs.float(), dim=1).angle() - torch.fft.rfft(batch_y.float(), dim=1).angle()
                elif self.args.auxi_type == 'mag':
                    loss_auxi = torch.fft.rfft(outputs.float(), dim=1).abs() - torch.fft.rfft(batch_y.float(), dim=1).abs()
                elif self.args.auxi_type == 'mag-phase':
                    loss_auxi_mag = torch.fft.rfft(outputs.float(), dim=1).abs() - torch.fft.rfft(batch_y.float(), dim=1).abs()
                    loss_auxi_phase = torch.fft.rfft(outputs.float(), dim=1).angle() - torch.fft.rfft(batch_y.float(), dim=1).angle()
                    loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                else:
                    raise NotImplementedError

            elif self.args.auxi_mode == "rfft-D":
                loss_auxi = torch.fft.rfft(outputs.float(), dim=-1) - torch.fft.rfft(batch_y.float(), dim=-1)

            elif self.args.auxi_mode == "rfft-2D":
                loss_auxi = torch.fft.rfft2(outputs.float()) - torch.fft.rfft2(batch_y.float())
            
            elif self.args.auxi_mode == "legendre":
                loss_auxi = leg_torch(outputs, self.args.leg_degree, device=self.device) - leg_torch(batch_y, self.args.leg_degree, device=self.device)
            
            elif self.args.auxi_mode == "chebyshev":
                loss_auxi = chebyshev_torch(outputs, self.args.leg_degree, device=self.device) - chebyshev_torch(batch_y, self.args.leg_degree, device=self.device)
            
            elif self.args.auxi_mode == "hermite":
                loss_auxi = hermite_torch(outputs, self.args.leg_degree, device=self.device) - hermite_torch(batch_y, self.args.leg_degree, device=self.device)
            
            elif self.args.auxi_mode == "laguerre":
                loss_auxi = laguerre_torch(outputs, self.args.leg_degree, device=self.device) - laguerre_torch(batch_y, self.args.leg_degree, device=self.device)
            else:
                raise NotImplementedError

            if self.mask is not None:
                loss_auxi *= self.mask

            if self.args.offload:
                loss_auxi = loss_auxi.cpu()

            if self.args.auxi_loss == "MAE":
                # MAE, 最小化element-wise error的模长
                loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
            elif self.args.auxi_loss == "MSE":
                # MSE, 最小化element-wise error的模长
                loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
            else:
                raise NotImplementedError

            if self.args.offload:
                loss_auxi = loss_auxi.to(self.device)
        return loss_auxi

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)

class DECOMP(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, alpha):
        super(DECOMP, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average

class DBLoss(nn.Module):
    """自定义分解损失函数（趋势+季节双损失）"""

    def __init__(self, alpha, beta):
        super().__init__()
        self.decomp = DECOMP(alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, pred, target):
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss
    
class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        self.alpha = alpha
        self.loss_mode = loss_mode
        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        # [b,l,n]
        if pred.ndim == 1:
            # imputation
            mask = torch.isnan(gt)
            if torch.any(mask):
                # pred, gt = pred.masked_fill(mask, 0), gt.masked_fill(mask, 0)
                pred, gt = pred[~mask], gt[~mask]

            loss_fun = nn.L1Loss(reduction='mean')
            weightedLoss = loss_fun(pred, gt)
        else:
            L = pred.shape[1]
            weights = (torch.tensor([(i + 1) ** (-self.alpha) for i in range(L)]).unsqueeze(dim=0).unsqueeze(dim=-1)
                       .to(pred.device))
            if self.loss_mode in ['L1', 'L2']:
                loss_vec = self.loss_fun(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights)
            elif self.loss_mode == 'L1L2':
                loss_vec = self.loss_fun1(pred, gt)
                loss_vec2 = self.loss_fun2(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights + loss_vec2 * weights)
            else:
                raise NotImplementedError
        return weightedLoss