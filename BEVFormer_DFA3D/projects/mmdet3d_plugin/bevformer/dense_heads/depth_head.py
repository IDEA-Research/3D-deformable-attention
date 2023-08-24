# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from BEVDepth (https://github.com/Megvii-BaseDetection/BEVDepth)
# Copyright 2022 Megvii-BaseDetection. All rights reserved
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
#  Modified by Hongyang Li
# ---------------------------------------------
from operator import index
from turtle import down, forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from copy import deepcopy
import numpy as np
import torch.utils.checkpoint as checkpoint
from torchvision.ops.misc import FrozenBatchNorm2d

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class ASPP_NoDp(ASPP):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super().__init__(inplanes, mid_channels, BatchNorm)
        self.dropout = None
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x
@HEADS.register_module()
class DepthHead(nn.Module):
    def __init__(self, in_channels, cam_channel, mid_channels, out_channels, downsample_factor, dbound, loss_weight, num_cams=6, max_tol=0, sfm_or_sig=True, indice_layer=2):
        super().__init__()
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.max_tol = max_tol
        self.downsample_factor = downsample_factor
        self.dbound = dbound
        self.loss_weight = loss_weight
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.sfm_or_sig = sfm_or_sig # True: sfm, False:sig
        self.indice_layer = indice_layer
        self.out_channels = out_channels
        self.cam_channel  = cam_channel
        if cam_channel>0:
            self.cam_ref = torch.load("data/cam_param_ref.pth") + 1e-5
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, cam_channel))
            from torch.nn.init import normal_
            normal_(self.cams_embeds)
            self.cam_encoder = nn.Sequential(
                nn.Conv2d(16,
                        cam_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0),
                nn.BatchNorm2d(cam_channel),
                nn.ReLU(inplace=False),
            )
        else:
            self.cam_encoder = None
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels + cam_channel if cam_channel>0 else in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP_NoDp(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      self.depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
    @auto_fp16(apply_to=('x', 'x_cam'))
    def forward(self, x, x_cam=None, return_dpt=False):
        # x_cam: lidar2image, bs num_cam 16
        # x: bs num_cam C H W
        batch_size, num_cam, C, H, W = x[self.indice_layer].shape
        if self.cam_channel > 0:
            x_cam_normed = x_cam / self.cam_ref.unsqueeze(0).repeat(x_cam.shape[0], 1, 1).to(x_cam.device)
            feat_cam = self.cam_encoder(x_cam_normed.transpose(1,2).unsqueeze(-1))
            feat_cam = feat_cam + self.cams_embeds.unsqueeze(0).repeat(x_cam.shape[0], 1, 1).transpose(1,2).unsqueeze(-1)  # bs cam_channel num_cam 1 --> bs*num_cam cam_channel 1 --> bs*num_cam cam_channel H W
            feat_cam = feat_cam.permute(0,2,1,3).unsqueeze(-1).flatten(0,1).repeat(1,1,H,W)
            feat = torch.cat([x[self.indice_layer].flatten(0,1), feat_cam], dim=1)
        else:
            feat = x[self.indice_layer].flatten(0,1)
        feat = self.reduce_conv(feat)  # batch_size*num_cam, C, H, W

        feat_dpt = self.depth_conv(feat)  # batch_size*num_cam, C_out+C_dpt, H, W 
        if self.sfm_or_sig:
            dpt_dist = feat_dpt.softmax(1)  # [:, self.out_channels:].softmax(1)  # batch_size*num_cam, C_dpt(sum=1), H, W
        else:
            dpt_dist = feat_dpt.sigmoid()  # [:, self.out_channels:].sigmoid()
        # batch_size*num_cam, C_out, C_dpt(sum=1), H, W
        # get multi scale depth distribution
        dpt_dists = []
        multi_scales = [[*x_scale.shape[3:]] for x_scale in x]
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(dpt_dist, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        if not return_dpt:
            return dpt_dists
        return dpt_dist, dpt_dists
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]
        gt_depths_errtol = self.error_tol(gt_depths)
        return gt_depths_errtol.float()
    def error_tol(self, gt_depths_onehot_):
        if self.max_tol < 1:
            return gt_depths_onehot_
        error_tol = [-self.max_tol, self.max_tol+1]
        padding = gt_depths_onehot_.new_zeros(gt_depths_onehot_.shape[0], 1)
        gt_depths_onehot = gt_depths_onehot_.clone()
        for error in range(error_tol[0], error_tol[1]):
            if error < 0 :  # move left
                gt_depths_onehot = gt_depths_onehot + torch.cat([gt_depths_onehot[..., 1:], padding], dim=-1)
            elif error > 0:  # move right
                gt_depths_onehot = gt_depths_onehot + torch.cat([padding, gt_depths_onehot[..., :-1]], dim=-1)
        gt_depths_onehot = (gt_depths_onehot / (gt_depths_onehot + 1e-5))
        return gt_depths_onehot
    @force_fp32(apply_to=('depth_preds'))
    def loss(self, depth_labels, depth_preds):
        if depth_labels.dim() == 3:
            depth_labels = depth_labels.unsqueeze(0)
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)  # batch_size*num_cam, C H W --> batch_size*num_cam H W C 
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return {"loss_dpt": self.loss_weight * depth_loss}
    def get_dpt_multi_scale_gt(self, gt_depths_, multi_scales, dtype):
        # gt_depths: [B, N, H, W]
        BN, _, H, W = gt_depths_.shape
        gt_depths = gt_depths_.view(
            BN,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(BN, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1)
        gt_depths = gt_depths[..., 1:]
        # B*N H/ds W/ds C
        gt_depths = gt_depths.permute(0,3,1,2).to(dtype)  # B*N C H/ds W/ds
        dpt_dists = []
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(gt_depths, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        return dpt_dists
@HEADS.register_module()
class DepthHead_GTDpt(DepthHead):
    @auto_fp16(apply_to=('x', 'x_cam'))
    def forward(self, x, x_cam=None, gt_dpt=None, return_dpt=False):
        # x_cam: lidar2image, bs num_cam 16
        # x: bs num_cam C H W
        batch_size, num_cam, C, H, W = x[self.indice_layer].shape
        dpt_dist = self.get_gt_dpt_dist(gt_dpt.view(batch_size, num_cam, *gt_dpt.shape[1:]).squeeze(2)).permute(0,3,1,2)
        # batch_size*num_cam, C_out, C_dpt(sum=1), H, W
        # get multi scale depth distribution
        dpt_dists = []
        multi_scales = [[*x_scale.shape[3:]] for x_scale in x]
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(dpt_dist, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        if not return_dpt:
            return dpt_dists
        return dpt_dist, dpt_dists
    def get_gt_dpt_dist(self, gt_depths_):
        B, N, H, W = gt_depths_.shape
        gt_depths = gt_depths_.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1)[..., 1:]  # B*N H/ds W/ds d
        # .view(-1, self.depth_channels + 1)[:, 1:]
        error_tol = [-self.max_tol, self.max_tol+1]
        padding = gt_depths.new_zeros(*gt_depths.shape[:-1], self.max_tol)
        for error in range(error_tol[0], error_tol[1]):
            if error < 0 :  # move left
                gt_depths = gt_depths + torch.cat([gt_depths[..., 1:], padding[..., :1]], dim=-1)
            elif error > 0:  # move right
                gt_depths = gt_depths + torch.cat([padding[..., :1], gt_depths[..., :-1]], dim=-1)
        gt_depths = (gt_depths / (gt_depths + 1e-5))

        return gt_depths.float()

@HEADS.register_module()
class DepthHead_MLVGDpt(DepthHead):
    def __init__(self, in_channels, cam_channel, mid_channels, out_channels, downsample_factor, dbound, loss_weight, levels=[2, 4, 8], num_cams=6, max_tol=0, sfm_or_sig=True, indice_layer=2):
        super().__init__(in_channels, cam_channel, mid_channels, out_channels, downsample_factor, dbound, loss_weight, num_cams, max_tol, sfm_or_sig, indice_layer)
        self.levels = torch.tensor(levels)
        self.num_per_lvl = int((self.depth_channels / self.levels.sum()).item())
        self.dim_depth_code = self.num_per_lvl * len(self.levels)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP_NoDp(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      self.dim_depth_code,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        
    @auto_fp16(apply_to=('x', 'x_cam'))
    def forward(self, x, x_cam=None, return_dpt=False):
        # x_cam: lidar2image, bs num_cam 16
        # x: bs num_cam C H W
        batch_size, num_cam, C, H, W = x[self.indice_layer].shape
        if self.cam_channel > 0:
            x_cam_normed = x_cam / self.cam_ref.unsqueeze(0).repeat(x_cam.shape[0], 1, 1).to(x_cam.device)
            feat_cam = self.cam_encoder(x_cam_normed.transpose(1,2).unsqueeze(-1))
            feat_cam = feat_cam + self.cams_embeds.unsqueeze(0).repeat(x_cam.shape[0], 1, 1).transpose(1,2).unsqueeze(-1)  # bs cam_channel num_cam 1 --> bs*num_cam cam_channel 1 --> bs*num_cam cam_channel H W
            feat_cam = feat_cam.permute(0,2,1,3).unsqueeze(-1).flatten(0,1).repeat(1,1,H,W)
            feat = torch.cat([x[self.indice_layer].flatten(0,1), feat_cam], dim=1)
        else:
            feat = x[self.indice_layer].flatten(0,1)
        feat = self.reduce_conv(feat)  # batch_size*num_cam, C, H, W

        dpt_dist = self.depth_conv(feat)  # batch_size*num_cam, C_out+C_dpt, H, W 
        if self.sfm_or_sig:
            dpt_dist = dpt_dist.softmax(1)
        else:
            dpt_dist = dpt_dist.sigmoid()
        dpt_dist_decoded = self.decode_prediction(dpt_dist)
        # batch_size*num_cam, C_out, C_dpt(sum=1), H, W
        # get multi scale depth distribution
        dpt_dists = []
        multi_scales = [[*x_scale.shape[3:]] for x_scale in x]
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(dpt_dist_decoded, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        if not return_dpt:
            return dpt_dists
        return dpt_dist, dpt_dists
    def decode_prediction(self, prediction):
        decoded_prediction = []
        for l_id, level in enumerate(self.levels):
            pred_level = prediction[:, l_id*self.num_per_lvl:(l_id+1)*self.num_per_lvl]
            pred_level = pred_level.unsqueeze(2).repeat(1, 1, level, 1, 1).flatten(1,2)
            decoded_prediction.append(pred_level)
        decoded_prediction = torch.cat(decoded_prediction, dim=1)
        return decoded_prediction
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]
        gt_depths = self.encode_dpt_dist(gt_depths)
        gt_depths_errtol = self.error_tol(gt_depths)
        return gt_depths_errtol.float()
    def encode_dpt_dist(self, dpt_dist_):
        # dpt_dist: N C
        dpt_dist_grouped = dpt_dist_.new_zeros(dpt_dist_.shape[0], self.dim_depth_code)
        N, C = dpt_dist_.shape
        count_channel = 0
        for l_id, lvl in enumerate(self.levels):
            dpt_dist_level = dpt_dist_.view(N, C//lvl, lvl)  # N C//lvl lvl
            dpt_dist_level = dpt_dist_level[:, count_channel//lvl: count_channel//lvl + self.num_per_lvl]  # N num_per_lvl lvl
            dpt_dist_level = dpt_dist_level.sum(dim=2)  # N num_per_lvl
            # dpt_dist_level = dpt_dist_level.repeat(1,1,lvl)  # N num_per_lvl lvl H W
            # dpt_dist_level = dpt_dist_level.flatten(1,2)  # N num_per_lvl*lvl H W
            dpt_dist_grouped[:, l_id*self.num_per_lvl:(l_id+1)*self.num_per_lvl] = dpt_dist_level
            count_channel += self.num_per_lvl*lvl
        return dpt_dist_grouped
    @force_fp32(apply_to=('depth_preds'))
    def loss(self, depth_labels, depth_preds):
        if depth_labels.dim() == 3:
            depth_labels = depth_labels.unsqueeze(0)
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.dim_depth_code)  # batch_size*num_cam, C H W --> batch_size*num_cam H W C 
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return {"loss_dpt": self.loss_weight * depth_loss}
    def get_dpt_multi_scale_gt(self, gt_depths_, multi_scales, dtype):
        # gt_depths: [B, N, H, W]
        BN, _, H, W = gt_depths_.shape
        gt_depths = gt_depths_.view(
            BN,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(BN, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1)
        gt_depths = gt_depths[..., 1:]
        # B*N H/ds W/ds C
        dpt_dist_grouped = gt_depths.new_zeros(gt_depths.shape)
        _, _, _, C = dpt_dist_grouped.shape
        count_channel = 0
        for l_id, lvl in enumerate(self.levels):
            dpt_dist_level = gt_depths.view(*gt_depths.shape[:-1], C//lvl, lvl)  # N H W C//lvl lvl
            dpt_dist_level = dpt_dist_level[:, :, :, count_channel//lvl: count_channel//lvl + self.num_per_lvl]  # N H W num_per_lvl lvl
            dpt_dist_level = dpt_dist_level.sum(dim=-1, keepdim=True)  # N H W num_per_lvl 1
            dpt_dist_level = dpt_dist_level.repeat(1, 1, 1, 1, lvl)  # N H W num_per_lvl lvl
            dpt_dist_level = dpt_dist_level.flatten(3)  # N H W num_per_lvl*lvl
            # dpt_dist_level = dpt_dist_level.repeat(1,1,lvl)  # N num_per_lvl lvl H W
            # dpt_dist_level = dpt_dist_level.flatten(1,2)  # N num_per_lvl*lvl H W
            dpt_dist_grouped[:, :, :, count_channel:count_channel+self.num_per_lvl*lvl] = dpt_dist_level
            count_channel += self.num_per_lvl*lvl
        gt_depths = dpt_dist_grouped.permute(0,3,1,2).to(dtype)
        # gt_depths = gt_depths.view(*gt_depths.shape[:-1], C_dpt//self.group_size, self.group_size)\
        #         .sum(dim=-1, keepdim=True).repeat(1,1,1,1,self.group_size).flatten(3).permute(0,3,1,2).to(dtype)
        dpt_dists = []
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(gt_depths, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        return dpt_dists
@HEADS.register_module()
class DepthHead_GDpt(DepthHead):
    def __init__(self, in_channels, cam_channel, mid_channels, out_channels, downsample_factor, dbound, loss_weight, group_size=4, num_cams=6, max_tol=0, sfm_or_sig=True, indice_layer=2):
        super().__init__(in_channels, cam_channel, mid_channels, out_channels, downsample_factor, dbound, loss_weight, num_cams, max_tol, sfm_or_sig, indice_layer)
        self.group_size = group_size
        self.num_group = self.depth_channels // self.group_size
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP_NoDp(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      self.num_group,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
    @auto_fp16(apply_to=('x', 'x_cam'))
    def forward(self, x, x_cam=None, return_dpt=False):
        # x_cam: lidar2image, bs num_cam 16
        # x: bs num_cam C H W
        batch_size, num_cam, C, H, W = x[self.indice_layer].shape
        if self.cam_channel > 0:
            x_cam_normed = x_cam / self.cam_ref.unsqueeze(0).repeat(x_cam.shape[0], 1, 1).to(x_cam.device)
            feat_cam = self.cam_encoder(x_cam_normed.transpose(1,2).unsqueeze(-1))
            feat_cam = feat_cam + self.cams_embeds.unsqueeze(0).repeat(x_cam.shape[0], 1, 1).transpose(1,2).unsqueeze(-1)  # bs cam_channel num_cam 1 --> bs*num_cam cam_channel 1 --> bs*num_cam cam_channel H W
            feat_cam = feat_cam.permute(0,2,1,3).unsqueeze(-1).flatten(0,1).repeat(1,1,H,W)
            feat = torch.cat([x[self.indice_layer].flatten(0,1), feat_cam], dim=1)
        else:
            feat = x[self.indice_layer].flatten(0,1)
        feat = self.reduce_conv(feat)  # batch_size*num_cam, C, H, W

        dpt_dist = self.depth_conv(feat)  # batch_size*num_cam, C_out+C_dpt, H, W 
        if self.sfm_or_sig:
            dpt_dist = dpt_dist.softmax(1)  # batch_size*num_cam, C_dpt(sum=1), H, W
        else:
            dpt_dist = dpt_dist.sigmoid()
        # batch_size*num_cam, C_out, C_dpt(sum=1), H, W
        # get multi scale depth distribution
        dpt_dist_decoded = self.decode_prediction(dpt_dist)
        dpt_dists = []
        multi_scales = [[*x_scale.shape[3:]] for x_scale in x]
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(dpt_dist_decoded, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        if not return_dpt:
            return dpt_dists
        return dpt_dist, dpt_dists
    def decode_prediction(self, prediction_):
        # N C H W
        prediction = prediction_.unsqueeze(2).repeat(1, 1, self.group_size, 1, 1).flatten(1,2)
        return prediction
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]
        gt_depths = self.encode_dpt_dist(gt_depths)
        gt_depths_errtol = self.error_tol(gt_depths)
        return gt_depths_errtol.float()
    def encode_dpt_dist(self, dpt_dist_):
        # N C
        N, C = dpt_dist_.shape
        dpt_dist = dpt_dist_.view(N, C//self.group_size, self.group_size).sum(dim=-1)
        return dpt_dist
    @force_fp32(apply_to=('depth_preds'))
    def loss(self, depth_labels, depth_preds):
        if depth_labels.dim() == 3:
            depth_labels = depth_labels.unsqueeze(0)
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.num_group)  # batch_size*num_cam, C H W --> batch_size*num_cam H W C 
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return {"loss_dpt": self.loss_weight * depth_loss}
    def get_dpt_multi_scale_gt(self, gt_depths_, multi_scales, dtype):
        # gt_depths: [B, N, H, W]
        BN, _, H, W = gt_depths_.shape
        gt_depths = gt_depths_.view(
            BN,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )  # B*N H/ds ds W/ds ds 1
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
        gt_depths = gt_depths.view(BN, H // self.downsample_factor,
                                   W // self.downsample_factor)  # B*N H/ds W/ds

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1)
        gt_depths = gt_depths[..., 1:]
        _, _, _, C_dpt = gt_depths.shape
        gt_depths = gt_depths.view(*gt_depths.shape[:-1], C_dpt//self.group_size, self.group_size)\
                .sum(dim=-1, keepdim=True).repeat(1,1,1,1,self.group_size).flatten(3).permute(0,3,1,2).to(dtype)
        dpt_dists = []
        for scale in multi_scales:
            dpt_dist_scale = F.interpolate(gt_depths, size=scale, mode="nearest")
            dpt_dists.append(dpt_dist_scale)
        #
        return dpt_dists
