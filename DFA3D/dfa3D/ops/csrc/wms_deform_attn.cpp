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

Tensor wms_deform_attn_impl_forward(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const Tensor &depth_score,
                                   const int im2col_step) {
  return DISPATCH_DEVICE_IMPL(wms_deform_attn_impl_forward, value,
                              spatial_shapes, level_start_index, sampling_loc,
                              attn_weight, depth_score, im2col_step);
}

void wms_deform_attn_impl_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &depth_score, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight, Tensor &grad_depth_score,
    const int im2col_step) {
  DISPATCH_DEVICE_IMPL(wms_deform_attn_impl_backward, value, spatial_shapes,
                       level_start_index, sampling_loc, attn_weight, depth_score,
                       grad_output, grad_value, grad_sampling_loc,
                       grad_attn_weight, grad_depth_score, im2col_step);
}

Tensor wms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight,
                              const Tensor &depth_score,
                              const int im2col_step) {
  at::DeviceGuard guard(value.device());
  return wms_deform_attn_impl_forward(value, spatial_shapes, level_start_index,
                                     sampling_loc, attn_weight, depth_score, im2col_step);
}

void wms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight, const Tensor &depth_score,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, Tensor &grad_depth_score, const int im2col_step) {
  at::DeviceGuard guard(value.device());
  wms_deform_attn_impl_backward(value, spatial_shapes, level_start_index,
                               sampling_loc, attn_weight, depth_score, grad_output,
                               grad_value, grad_sampling_loc, grad_attn_weight, grad_depth_score,
                               im2col_step);
}
