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

std::string get_compiler_version();
std::string get_compiling_cuda_version();

Tensor wms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight, const Tensor &depth_score, const int im2col_step);

void wms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight, const Tensor &depth_score,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, Tensor &grad_depth_score, const int im2col_step);

Tensor ms_depth_score_sample_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index, const Tensor &sampling_loc, const int im2col_step);
void ms_depth_score_sample_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             const int im2col_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wms_deform_attn_forward", &wms_deform_attn_forward,
        "forward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("depth_scores"), py::arg("im2col_step"));
  m.def("wms_deform_attn_backward", &wms_deform_attn_backward,
        "backward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("depth_scores"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("grad_attn_weight"), py::arg("grad_depth_score"), py::arg("im2col_step"));
  m.def("ms_depth_score_sample_forward", &ms_depth_score_sample_forward,
        "forward function of multi-scale sampling 3D attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("im2col_step"));
  m.def("ms_depth_score_sample_backward", &ms_depth_score_sample_backward,
        "forward function of multi-scale sampling 3D attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("im2col_step"));
}
