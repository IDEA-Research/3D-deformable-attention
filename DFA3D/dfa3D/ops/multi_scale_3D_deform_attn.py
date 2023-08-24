# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmcv (https://github.com/open-mmlab/mmcv)
# Copyright (c) OpenMMLab. All rights reserved
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------ 

import torch
from torch.autograd.function import Function, once_differentiable

from .. import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['wms_deform_attn_backward', 'wms_deform_attn_forward', 'ms_depth_score_sample_forward', 'ms_depth_score_sample_backward'])

class MultiScaleDepthScoreSampleFunction(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step
        output = ext_module.ms_depth_score_sample_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations)
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)

        ext_module.ms_depth_score_sample_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, None
class WeightedMultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                depth_score: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        """GPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.wms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            depth_score,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights, depth_score)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights, depth_score = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        grad_depth_score = torch.zeros_like(depth_score)

        ext_module.wms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            depth_score,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            grad_depth_score,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, grad_depth_score, None
class MultiScale3DDeformableAttnFunction(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, value_dpt_dist: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step

        depth_score = ext_module.ms_depth_score_sample_forward(
            value_dpt_dist,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            im2col_step=ctx.im2col_step)
        
        output = ext_module.wms_deform_attn_forward(
            value,
            value_spatial_shapes[..., :2].contiguous(),
            value_level_start_index,
            sampling_locations[..., :2].contiguous(),
            attention_weights,
            depth_score,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_dpt_dist, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights, depth_score)
        return output, depth_score
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_dpt_dist, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights, depth_score = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        grad_depth_score = torch.zeros_like(depth_score)

        ext_module.wms_deform_attn_backward(
            value,
            value_spatial_shapes[..., :2].contiguous(),
            value_level_start_index,
            sampling_locations[..., :2].contiguous(),
            attention_weights,
            depth_score,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc[..., :2].contiguous(),
            grad_attn_weight,
            grad_depth_score,
            im2col_step=ctx.im2col_step)
        
        grad_value_dpt_dist = torch.zeros_like(value_dpt_dist)
        grad_sampling_loc_ = torch.zeros_like(sampling_locations)
        ext_module.ms_depth_score_sample_backward(
            value_dpt_dist,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            grad_depth_score.contiguous(),
            grad_value_dpt_dist,
            grad_sampling_loc,
            im2col_step=ctx.im2col_step)
        grad_sampling_loc = grad_sampling_loc + grad_sampling_loc_


        return grad_value, grad_value_dpt_dist, None, None, \
            grad_sampling_loc, grad_attn_weight, None, None
    