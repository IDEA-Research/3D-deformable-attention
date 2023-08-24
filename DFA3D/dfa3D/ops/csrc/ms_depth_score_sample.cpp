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

Tensor ms_depth_score_sample_impl_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index, const Tensor &sampling_loc, const int im2col_step){
    return DISPATCH_DEVICE_IMPL(ms_depth_score_sample_impl_forward, value,
                              spatial_shapes, level_start_index, sampling_loc,
                              im2col_step);
}

void ms_depth_score_sample_impl_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc,
    const int im2col_step) {
  DISPATCH_DEVICE_IMPL(ms_depth_score_sample_impl_backward, value, spatial_shapes,
                       level_start_index, sampling_loc,
                       grad_output, grad_value, grad_sampling_loc,
                       im2col_step);
}
Tensor ms_depth_score_sample_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const int im2col_step) {
  at::DeviceGuard guard(value.device());
  return ms_depth_score_sample_impl_forward(value, spatial_shapes, level_start_index,
                                     sampling_loc, im2col_step);
}
void ms_depth_score_sample_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             const int im2col_step) {
  at::DeviceGuard guard(value.device());
  ms_depth_score_sample_impl_backward(value, spatial_shapes, level_start_index,
                               sampling_loc, grad_output,
                               grad_value, grad_sampling_loc,
                               im2col_step);
}