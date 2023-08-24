// ------------------------------------------------------------------------
// DFA3D
// Copyright (c) 2023 IDEA. All Rights Reserved.
// Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
// ------------------------------------------------------------------------
// Modified from mmcv (https://github.com/open-mmlab/mmcv)
// Copyright (c) OpenMMLab. All rights reserved
// Licensed under the Apache License, Version 2.0 [see LICENSE for details]
// ------------------------------------------------------------------------
// Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
// Copyright 2018-2019 Open-MMLab. All rights reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details]
// ------------------------------------------------------------------------ 
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor wms_deform_attn_cuda_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const Tensor& depth_score,
                                   const int im2col_step);  // an announcement, which will be linked to the implementation in new_ops.cpp when linking.
void wms_deform_attn_cuda_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& depth_score, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, Tensor& grad_depth_score, const int im2col_step);
Tensor wms_deform_attn_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const Tensor& depth_score,
                                   const int im2col_step);
void wms_deform_attn_impl_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& depth_score, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, Tensor& grad_depth_score, const int im2col_step);
REGISTER_DEVICE_IMPL(wms_deform_attn_impl_forward, CUDA,
                     wms_deform_attn_cuda_forward);
REGISTER_DEVICE_IMPL(wms_deform_attn_impl_backward, CUDA,
                     wms_deform_attn_cuda_backward);
Tensor ms_depth_score_sample_cuda_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const int im2col_step); 
Tensor ms_depth_score_sample_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const int im2col_step); 
void ms_depth_score_sample_cuda_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             const int im2col_step);
void ms_depth_score_sample_impl_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc,
    const int im2col_step);
REGISTER_DEVICE_IMPL(ms_depth_score_sample_impl_forward, CUDA,
                     ms_depth_score_sample_cuda_forward);
REGISTER_DEVICE_IMPL(ms_depth_score_sample_impl_backward, CUDA,
                     ms_depth_score_sample_cuda_backward);

