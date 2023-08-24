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
#ifndef SAMPLE_3D_ATTN_CUDA_KERNEL
#define SAMPLE_3D_ATTN_CUDA_KERNEL

#include "common_cuda_helper.hpp"
#include "pytorch_cuda_helper.hpp"

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}
template <typename scalar_t>
__device__ void im2col_trilinear(
    const scalar_t *&bottom_data, const int &height, const int &width, const int &depth,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const scalar_t &d, const int &m, scalar_t *val) {
  // depth_score: in the order of left_top, right_top, left_bottom, right_bottom
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int d_low = floorf(d);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int d_high = d_low + 1;

  //   const scalar_t lh = h - h_low;
  //   const scalar_t lw = w - w_low;
  const scalar_t ld = d - d_low;
  //   const scalar_t hh = 1 - lh, hw = 1 - lw; 
  const scalar_t hd = 1 - ld;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  //   const int base_ptr = m * channels + c;
  const int base_ptr = m * channels;

  scalar_t v1_1 = 0, v1_2 = 0;
  if (h_low >= 0 && w_low >= 0 && d_low >= 0) {
    const int ptr1_1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr + d_low;
    v1_1 = bottom_data[ptr1_1];
  }
  if (h_low >= 0 && w_low >= 0 && d_high <= depth-1) {
    const int ptr1_2 = h_low_ptr_offset + w_low_ptr_offset + base_ptr + d_high;
    v1_2 = bottom_data[ptr1_2];
  }
  scalar_t v2_1 = 0, v2_2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && d_low >= 0) {
    const int ptr2_1 = h_low_ptr_offset + w_high_ptr_offset + base_ptr + d_low;
    v2_1 = bottom_data[ptr2_1];
  }
  if (h_low >= 0 && w_high <= width - 1 && d_high <= depth-1) {
    const int ptr2_2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr + d_high;
    v2_2 = bottom_data[ptr2_2];
  }
  scalar_t v3_1 = 0, v3_2 = 0;
  if (h_high <= height - 1 && w_low >= 0 && d_low >= 0) {
    const int ptr3_1 = h_high_ptr_offset + w_low_ptr_offset + base_ptr + d_low;
    v3_1 = bottom_data[ptr3_1];
  }
  if (h_high <= height - 1 && w_low >= 0 && d_high <= depth-1) {
    const int ptr3_2 = h_high_ptr_offset + w_low_ptr_offset + base_ptr + d_high;
    v3_2 = bottom_data[ptr3_2];
  }
  scalar_t v4_1 = 0, v4_2 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && d_low >= 0) {
    const int ptr4_1 = h_high_ptr_offset + w_high_ptr_offset + base_ptr + d_low;
    v4_1 = bottom_data[ptr4_1];
  }
  if (h_high <= height - 1 && w_high <= width - 1 && d_high <= depth-1) {
    const int ptr4_2 = h_high_ptr_offset + w_high_ptr_offset + base_ptr + d_high;
    v4_2 = bottom_data[ptr4_2];
  }

  val[0] += v1_1 * hd + v1_2 * ld;
  val[1] += v2_1 * hd + v2_2 * ld;
  val[3] += v3_1 * hd + v3_2 * ld;
  val[2] += v4_1 * hd + v4_2 * ld;
}
template <typename scalar_t>
__global__ void ms_depth_score_sample_im2col_gpu_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const int batch_size,
    const int spatial_size, const int num_heads, const int channels, const int channels_out,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    // index of bs
    const int b_col = _temp;
    // ptr of curren col.
    scalar_t *data_col_ptr = data_col + index*num_levels*num_point*channels_out;
    scalar_t *data_col_ptr_cur = data_col_ptr;
    // ptr offset of the weight for current sampling point.
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_d = data_spatial_shapes[spatial_h_ptr + 2];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t d_im = loc_d * spatial_d - 0.5;

        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d) {
           im2col_trilinear(data_value_ptr, spatial_h,
                                                spatial_w, spatial_d, num_heads, channels,
                                                h_im, w_im, d_im, m_col, data_col_ptr_cur);
        }
        
        data_loc_w_ptr += 3;
        data_col_ptr_cur += 4;
      }
    }
  }
}

template <typename scalar_t>
__device__ void col2im_trilinear(
    const scalar_t *&bottom_data, const int &height, const int &width, const int &depth,
    const int &nheads, const int &channels, const int &channels_out, const scalar_t &h,
    const scalar_t &w, const scalar_t &d, const int &m, const int &c_out, const scalar_t &top_grad,
    scalar_t *&grad_value,
    scalar_t *grad_sampling_loc) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int d_low = floorf(d);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const int d_high = d_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t ld = d - d_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw, hd = 1 - ld;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const int c_1 = d_low, c_2 = d_high;
  const scalar_t top_grad_value = top_grad;
  scalar_t grad_d_weight = 0;
  scalar_t v1 = 0, v2 = 0;
  switch (c_out)
  {
  case 0:
    if (h_low >= 0 && w_low >= 0 && d_low >= 0) {
      const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + m * channels + d_low;
      v1 = bottom_data[ptr1];
      atomicAdd(grad_value + ptr1, hd * top_grad_value);
    }
    if (h_low >= 0 && w_low >= 0 && d_high <= (depth-1)) {
      const int ptr2 = h_low_ptr_offset + w_low_ptr_offset + m * channels + d_high;
      v2 = bottom_data[ptr2];
      atomicAdd(grad_value + ptr2, ld * top_grad_value);
    }
    grad_d_weight += top_grad_value * (v2 - v1);
    break;
  case 1:
    if (h_low >= 0 && w_high <= width-1 && d_low >= 0) {
      const int ptr1 = h_low_ptr_offset + w_high_ptr_offset + m * channels + d_low;
      v1 = bottom_data[ptr1];
      atomicAdd(grad_value + ptr1, hd * top_grad_value);
    }
    if (h_low >= 0 && w_high <= width-1 && d_high <= (depth-1)) {
      const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + m * channels + d_high;
      v2 = bottom_data[ptr2];
      atomicAdd(grad_value + ptr2, ld * top_grad_value);
    }
    grad_d_weight += top_grad_value * (v2 - v1);
    break;
  case 2:
    if (h_high <= height-1 && w_high <= width-1 && d_low >= 0) {
      const int ptr1 = h_high_ptr_offset + w_high_ptr_offset + m * channels + d_low;
      v1 = bottom_data[ptr1];
      atomicAdd(grad_value + ptr1, hd * top_grad_value);
    }
    if (h_high <= height-1 && w_high <= width-1 && d_high <= (depth-1)) {
      const int ptr2 = h_high_ptr_offset + w_high_ptr_offset + m * channels + d_high;
      v2 = bottom_data[ptr2];
      atomicAdd(grad_value + ptr2, ld * top_grad_value);
    }
    grad_d_weight += top_grad_value * (v2 - v1);
    break;
  case 3:
    if (h_high <= height-1 && w_low >=0 && d_low >= 0) {
      const int ptr1 = h_high_ptr_offset + w_low_ptr_offset + m * channels + d_low;
      v1 = bottom_data[ptr1];
      atomicAdd(grad_value + ptr1, hd * top_grad_value);
    }
    if (h_high <= height-1 && w_low >=0 && d_high <= (depth-1)) {
      const int ptr2 = h_high_ptr_offset + w_low_ptr_offset + m * channels + d_high;
      v2 = bottom_data[ptr2];
      atomicAdd(grad_value + ptr2, ld * top_grad_value);
    }
    grad_d_weight += top_grad_value * (v2 - v1);
    break;
  default:
    break;
  }

  *grad_sampling_loc = 0;
  *(grad_sampling_loc + 1) = 0;
  *(grad_sampling_loc + 2) = depth * grad_d_weight;
}
template <typename scalar_t, unsigned int blockSize>
__global__ void ms_depth_score_sample_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, 
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int channels_out, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 3];
  unsigned int tid = threadIdx.x;
  const int qid_stride = num_heads * channels;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels_out;
    _temp /= channels_out;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;


    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;
    const int grad_sampling_ptr = data_weight_ptr;
    const int grad_output_ptr = data_weight_ptr<<2;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr * 3);
    const int grad_loc_stride = 3;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_d = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        const scalar_t d_im = loc_d * spatial_d - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x * 3)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x * 3) + 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x * 3) + 2)) = 0;
        const scalar_t top_grad = grad_col[grad_output_ptr + l_col*num_point*4 + p_col*4 + c_col];
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d) {
          col2im_trilinear(
              data_value_ptr, spatial_h, spatial_w, spatial_d, num_heads, channels, channels_out, h_im,
              w_im, d_im, m_col, c_col, top_grad, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x * 3));
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_d = cache_grad_sampling_loc[2];
          int sid = 3;
          for (unsigned int _tid = 1; _tid < blockSize; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_d += cache_grad_sampling_loc[sid + 2];
            sid += 3;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *(grad_sampling_loc_out + 2) = _grad_d;
        }
        __syncthreads();

        data_loc_w_ptr += 3;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

#endif  // SAMPLE_3D_ATTN_CUDA_KERNEL