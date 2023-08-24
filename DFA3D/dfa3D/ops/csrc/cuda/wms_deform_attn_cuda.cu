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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>
#include <vector>

#include "wms_deform_attn_cuda_kernel.cuh"

template <typename scalar_t>
void wms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const scalar_t *data_depth_score,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  wms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, data_depth_score, batch_size, spatial_size,
          num_heads, channels, num_levels, num_query, num_point, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in wms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void wms_deformable_col2im_cuda(
    cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  const int num_threads =
      (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > CUDA_NUM_THREADS) {
    if ((channels & CUDA_NUM_THREADS - 1) == 0) {
      wms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
             num_threads * 7 * sizeof(scalar_t), stream>>>(
              num_kernels, grad_col, data_value, data_spatial_shapes,
              data_level_start_index, data_sampling_loc, data_attn_weight, data_depth_score,
              batch_size, spatial_size, num_heads, channels, num_levels,
              num_query, num_point, grad_value, grad_sampling_loc,
              grad_attn_weight, grad_depth_score);
    } else {
      wms_deformable_col2im_gpu_kernel_gm<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight, grad_depth_score);
    }
  } else {
    switch (channels) {
      case 1:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      1>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 2:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      2>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 4:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      4>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 8:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      8>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 16:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      16>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 32:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      32>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 64:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      64>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 128:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      128>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 256:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      256>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      case 512:
        wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      512>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, data_depth_score, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight, grad_depth_score);
        break;
      default:
        if (channels < 64) {
          wms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                 num_threads * 7 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight, data_depth_score,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight, grad_depth_score);
        } else {
          wms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                 num_threads * 7 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight, data_depth_score,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight, grad_depth_score);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in wms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

at::Tensor wms_deform_attn_cuda_forward(const at::Tensor &value,
                                       const at::Tensor &spatial_shapes,
                                       const at::Tensor &level_start_index,
                                       const at::Tensor &sampling_loc,
                                       const at::Tensor &attn_weight,
                                       const at::Tensor &depth_score,
                                       const int im2col_step) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(depth_score.is_contiguous(),
             "depth_score tensor has to be contiguous");

  AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.is_cuda(),
             "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
  AT_ASSERTM(depth_score.is_cuda(), "depth_score must be a CUDA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);

  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  const int im2col_step_ = std::min(batch, im2col_step);

  AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)",
             batch, im2col_step_);

  auto output =
      at::zeros({batch, num_query, num_heads, channels}, value.options());

  const int batch_n = im2col_step_;
  auto output_n = output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto per_depth_score_size = num_query * num_heads * num_levels * num_point * 4;
  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "wms_deform_attn_forward_cuda", ([&] {
          wms_deformable_im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              depth_score.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_depth_score_size,
              batch_n, spatial_size, num_heads, channels, num_levels, num_query,
              num_point, columns.data_ptr<scalar_t>());
        }));
  }

  output = output.view({batch, num_query, num_heads * channels});

  return output;
}


void wms_deform_attn_cuda_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &depth_score, const at::Tensor &grad_output,
    at::Tensor &grad_value, at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, at::Tensor &grad_depth_score, const int im2col_step) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(depth_score.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(grad_output.is_contiguous(),
             "grad_output tensor has to be contiguous");

  AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.is_cuda(),
             "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
  AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);

  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  const int im2col_step_ = std::min(batch, im2col_step);

  AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)",
             batch, im2col_step_);

  const int batch_n = im2col_step_;
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto per_depth_score_size = num_query * num_heads * num_levels * num_point * 4;
  auto grad_output_n = grad_output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto grad_output_g = grad_output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "wms_deform_attn_backward_cuda", ([&] {
          wms_deformable_col2im_cuda(
              at::cuda::getCurrentCUDAStream(),
              grad_output_g.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              depth_score.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_depth_score_size,
              batch_n, spatial_size, num_heads, channels, num_levels, num_query,
              num_point,
              grad_value.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_value_size,
              grad_sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              grad_attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              grad_depth_score.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_depth_score_size);
        }));
  }
}