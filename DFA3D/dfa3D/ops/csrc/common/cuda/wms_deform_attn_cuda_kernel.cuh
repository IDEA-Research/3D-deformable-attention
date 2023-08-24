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
#ifndef DEFORM_ATTN_CUDA_KERNEL
#define DEFORM_ATTN_CUDA_KERNEL

#include "common_cuda_helper.hpp"
#include "pytorch_cuda_helper.hpp"

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}
template <typename scalar_t>
__device__ scalar_t wms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t *&depth_score) {
  // depth_score: in the order of left_top, right_top, left_bottom, right_bottom
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  scalar_t ds_w1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    ds_w1 = depth_score[0];
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  scalar_t ds_w2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    ds_w2 = depth_score[1];
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  scalar_t ds_w3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    ds_w3 = depth_score[3];
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  scalar_t ds_w4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    ds_w4 = depth_score[2];
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw * ds_w1, w2 = hh * lw * ds_w2, w3 = lh * hw * ds_w3, w4 = lh * lw * ds_w4;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void wms_deform_attn_col2im_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, const scalar_t *&depth_score, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw * depth_score[0], w2 = hh * lw * depth_score[1], w3 = lh * hw * depth_score[3], w4 = lh * lw * depth_score[2];
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  scalar_t ds_w1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    ds_w1 = depth_score[0];
    grad_h_weight -= ds_w1 * hw * v1;
    grad_w_weight -= ds_w1 * hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    atomicAdd(grad_depth_score + 0, v1 * hh * hw * top_grad_value);
  }
  scalar_t v2 = 0;
  scalar_t ds_w2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    ds_w2 = depth_score[1];
    grad_h_weight -= ds_w2 * lw * v2;
    grad_w_weight += ds_w2 * hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    atomicAdd(grad_depth_score + 1, v2 * hh * lw * top_grad_value);
  }
  scalar_t v3 = 0;
  scalar_t ds_w3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    ds_w3 = depth_score[3];
    grad_h_weight += ds_w3 * hw * v3;
    grad_w_weight -= ds_w3 * lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    atomicAdd(grad_depth_score + 3, v3 * lh * hw * top_grad_value);
  }
  scalar_t v4 = 0;
  scalar_t ds_w4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    ds_w4 = depth_score[2];
    grad_h_weight += ds_w4 * lw * v4;
    grad_w_weight += ds_w4 * lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    atomicAdd(grad_depth_score + 2, v4 * lh * lw * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void wms_deform_attn_col2im_bilinear_gm(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, const scalar_t *&depth_score, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw * depth_score[0], w2 = hh * lw * depth_score[1], w3 = lh * hw * depth_score[3], w4 = lh * lw * depth_score[2];
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  scalar_t ds_w1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    ds_w1 = depth_score[0];
    grad_h_weight -= ds_w1 * hw * v1;
    grad_w_weight -= ds_w1 * hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    atomicAdd(grad_depth_score + 0, v1 * hh * hw * top_grad_value);
  }
  scalar_t v2 = 0;
  scalar_t ds_w2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    ds_w2 = depth_score[1];
    grad_h_weight -= ds_w2 * lw * v2;
    grad_w_weight += ds_w2 * hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    atomicAdd(grad_depth_score + 1, v2 * hh * lw * top_grad_value);
  }
  scalar_t v3 = 0;
  scalar_t ds_w3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    ds_w3 = depth_score[3];
    grad_h_weight += ds_w3 * hw * v3;
    grad_w_weight -= ds_w3 * lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    atomicAdd(grad_depth_score + 3, v3 * lh * hw * top_grad_value);
  }
  scalar_t v4 = 0;
  scalar_t ds_w4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    ds_w4 = depth_score[2];
    grad_h_weight += ds_w4 * lw * v4;
    grad_w_weight += ds_w4 * lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    atomicAdd(grad_depth_score + 2, v4 * lh * lw * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

template <typename scalar_t>
__global__ void wms_deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const scalar_t *data_depth_score, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    // index of channel for current thread.
    const int c_col = _temp % channels;
    _temp /= channels;
    // index of sampling_point for ..
    const int sampling_index = _temp;
    // index of head for current sampling point
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    // index of bs
    const int b_col = _temp;
    // ptr of curren col.
    scalar_t *data_col_ptr = data_col + index;
    // ptr offset of the weight for current sampling point.
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_depth_score_ptr = data_weight_ptr << 2;
    // ptr offset of the sampling location for current sampling point.
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t *depth_score = data_depth_score + data_depth_score_ptr;

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += wms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                spatial_w, num_heads, channels,
                                                h_im, w_im, m_col, c_col, depth_score) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        data_depth_score_ptr += 4;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  __shared__ scalar_t cache_grad_depth_score[blockSize*4];
  unsigned int tid = threadIdx.x;
  const int qid_stride = num_heads * channels;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    int data_depth_score_ptr = data_weight_ptr << 2;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    scalar_t *grad_depth_score_out = grad_depth_score + (grad_sampling_ptr << 2);
    const int grad_weight_stride = 1;
    const int grad_depth_score_stride = 4;
    const int grad_loc_stride = 2;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t *depth_score = data_depth_score + data_depth_score_ptr;

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        *(cache_grad_depth_score + (threadIdx.x << 2)    )= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 1)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 2)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 3)= 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          wms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, depth_score, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x, cache_grad_depth_score + (threadIdx.x << 2));
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0],
                   _grad_b1 = cache_grad_depth_score[0],
                   _grad_b2 = cache_grad_depth_score[1],
                   _grad_b3 = cache_grad_depth_score[2],
                   _grad_b4 = cache_grad_depth_score[3];
          int sid = 2;
          int bid = 4;
          for (unsigned int _tid = 1; _tid < blockSize; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[_tid];
            _grad_b1 += cache_grad_depth_score[bid];
            _grad_b2 += cache_grad_depth_score[bid+1];
            _grad_b3 += cache_grad_depth_score[bid+2];
            _grad_b4 += cache_grad_depth_score[bid+3];
            sid += 2;
            bid += 4;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *grad_attn_weight_out = _grad_a;
          *grad_depth_score_out     = _grad_b1;
          *(grad_depth_score_out+1) = _grad_b2;
          *(grad_depth_score_out+2) = _grad_b3;
          *(grad_depth_score_out+3) = _grad_b4;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        data_depth_score_ptr += 4;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
        grad_depth_score_out += grad_depth_score_stride;
      }
    }
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void wms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  __shared__ scalar_t cache_grad_depth_score[blockSize*4];
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    int data_depth_score_ptr = data_weight_ptr << 2;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    scalar_t *grad_depth_score_out = grad_depth_score + (grad_sampling_ptr << 2);
    const int grad_weight_stride = 1;
    const int grad_depth_score_stride = 4;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t *depth_score = data_depth_score + data_depth_score_ptr;

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        *(cache_grad_depth_score + (threadIdx.x << 2)    )= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 1)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 2)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 3)= 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          wms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, depth_score, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x, cache_grad_depth_score + (threadIdx.x << 2));
        }

        __syncthreads();
        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            const unsigned int xid3 = tid << 2;
            const unsigned int xid4 = (tid + s) << 2;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            cache_grad_depth_score[xid3 + 0] += cache_grad_depth_score[xid4 + 0];
            cache_grad_depth_score[xid3 + 1] += cache_grad_depth_score[xid4 + 1];
            cache_grad_depth_score[xid3 + 2] += cache_grad_depth_score[xid4 + 2];
            cache_grad_depth_score[xid3 + 3] += cache_grad_depth_score[xid4 + 3];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
          *(grad_depth_score_out + 0) = cache_grad_depth_score[0];
          *(grad_depth_score_out + 1) = cache_grad_depth_score[1];
          *(grad_depth_score_out + 2) = cache_grad_depth_score[2];
          *(grad_depth_score_out + 3) = cache_grad_depth_score[3];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        data_depth_score_ptr += 4;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
        grad_depth_score_out += grad_depth_score_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void wms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {

  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  scalar_t *cache_grad_depth_score  = cache_grad_sampling_loc + 3 * blockDim.x;

  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    int data_depth_score_ptr = data_weight_ptr << 2;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    scalar_t *grad_depth_score_out = grad_depth_score + (grad_sampling_ptr << 2);
    const int grad_weight_stride = 1;
    const int grad_depth_score_stride = 4;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t *depth_score = data_depth_score + data_depth_score_ptr;

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        *(cache_grad_depth_score + (threadIdx.x << 2)    )= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 1)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 2)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 3)= 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          wms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, depth_score, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x, cache_grad_depth_score + (threadIdx.x << 2));
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0],
                   _grad_b1 = cache_grad_depth_score[0],
                   _grad_b2 = cache_grad_depth_score[1],
                   _grad_b3 = cache_grad_depth_score[2],
                   _grad_b4 = cache_grad_depth_score[3];
          int sid = 2;
          int bid = 4;
          for (unsigned int _tid = 1; _tid < blockDim.x; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[_tid];
            _grad_b1 += cache_grad_depth_score[bid];
            _grad_b2 += cache_grad_depth_score[bid+1];
            _grad_b3 += cache_grad_depth_score[bid+2];
            _grad_b4 += cache_grad_depth_score[bid+3];
            sid += 2;
            bid += 4;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *grad_attn_weight_out = _grad_a;
          *grad_depth_score_out     = _grad_b1;
          *(grad_depth_score_out+1) = _grad_b2;
          *(grad_depth_score_out+2) = _grad_b3;
          *(grad_depth_score_out+3) = _grad_b4;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        data_depth_score_ptr += 4;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
        grad_depth_score_out += grad_depth_score_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void wms_deformable_col2im_gpu_kernel_shm_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {

  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  scalar_t *cache_grad_depth_score  = cache_grad_sampling_loc + 3 * blockDim.x;

  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    int data_depth_score_ptr = data_weight_ptr << 2;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    scalar_t *grad_depth_score_out = grad_depth_score + (grad_sampling_ptr << 2);
    const int grad_weight_stride = 1;
    const int grad_depth_score_stride = 4;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t *depth_score = data_depth_score + data_depth_score_ptr;

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        *(cache_grad_depth_score + (threadIdx.x << 2)    )= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 1)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 2)= 0;
        *(cache_grad_depth_score + (threadIdx.x << 2) + 3)= 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          wms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, depth_score, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x, cache_grad_depth_score + (threadIdx.x << 2));
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            const unsigned int xid3 = tid << 2;
            const unsigned int xid4 = (tid + s) << 2;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            cache_grad_depth_score[xid3] += cache_grad_depth_score[xid4];
            cache_grad_depth_score[xid3 + 1] += cache_grad_depth_score[xid4 + 1];
            cache_grad_depth_score[xid3 + 2] += cache_grad_depth_score[xid4 + 2];
            cache_grad_depth_score[xid3 + 3] += cache_grad_depth_score[xid4 + 3];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
              cache_grad_depth_score[xid3] += 
                  cache_grad_depth_score[xid4 + (s << 2)];
              cache_grad_depth_score[xid3+1] += 
                  cache_grad_depth_score[xid4 + (s << 2) + 1];
              cache_grad_depth_score[xid3+2] += 
                  cache_grad_depth_score[xid4 + (s << 2) + 2];
              cache_grad_depth_score[xid3+3] += 
                  cache_grad_depth_score[xid4 + (s << 2) + 3];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
          *grad_depth_score_out     = cache_grad_depth_score[0];
          *(grad_depth_score_out+1) = cache_grad_depth_score[1];
          *(grad_depth_score_out+2) = cache_grad_depth_score[2];
          *(grad_depth_score_out+3) = cache_grad_depth_score[3];
        }
      __syncthreads();

      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
      data_depth_score_ptr += 4;
      grad_attn_weight_out += grad_weight_stride;
      grad_sampling_loc_out += grad_loc_stride;
      grad_depth_score_out += grad_depth_score_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void wms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  scalar_t *cache_grad_depth_score = cache_grad_sampling_loc + 3 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    int data_depth_score_ptr = data_weight_ptr << 2;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    scalar_t *grad_depth_score_out = grad_depth_score + (grad_sampling_ptr << 2);
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int grad_depth_score_stride = 4;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const scalar_t *depth_score = data_depth_score + data_depth_score_ptr;

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        *(cache_grad_depth_score + (threadIdx.x << 2)) = 0;
        *(cache_grad_depth_score + ((threadIdx.x << 2)+1)) = 0;
        *(cache_grad_depth_score + ((threadIdx.x << 2)+2)) = 0;
        *(cache_grad_depth_score + ((threadIdx.x << 2)+3)) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          wms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, depth_score, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x, cache_grad_depth_score + (threadIdx.x << 2));
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            const unsigned int xid3 = tid << 2;
            const unsigned int xid4 = (tid + s) << 2;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            cache_grad_depth_score[xid3] += cache_grad_depth_score[xid4];
            cache_grad_depth_score[xid3 + 1] += cache_grad_depth_score[xid4 + 1];
            cache_grad_depth_score[xid3 + 2] += cache_grad_depth_score[xid4 + 2];
            cache_grad_depth_score[xid3 + 3] += cache_grad_depth_score[xid4 + 3];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
              cache_grad_depth_score[xid3] += 
                  cache_grad_depth_score[xid4 + (s << 2)];
              cache_grad_depth_score[xid3+1] += 
                  cache_grad_depth_score[xid4 + (s << 2) + 1];
              cache_grad_depth_score[xid3+2] += 
                  cache_grad_depth_score[xid4 + (s << 2) + 2];
              cache_grad_depth_score[xid3+3] += 
                  cache_grad_depth_score[xid4 + (s << 2) + 3];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc_out, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc_out + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight_out, cache_grad_attn_weight[0]);
          atomicAdd(grad_depth_score_out , cache_grad_depth_score[0]);
          atomicAdd(grad_depth_score_out + 1, cache_grad_depth_score[1]);
          atomicAdd(grad_depth_score_out + 2, cache_grad_depth_score[2]);
          atomicAdd(grad_depth_score_out + 3, cache_grad_depth_score[3]);
        }
        __syncthreads();

      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
      data_depth_score_ptr += 4;
      grad_attn_weight_out += grad_weight_stride;
      grad_sampling_loc_out += grad_loc_stride;
      grad_depth_score_out += grad_depth_score_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void wms_deformable_col2im_gpu_kernel_gm(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const scalar_t *data_depth_score,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight, scalar_t *grad_depth_score) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          wms_deform_attn_col2im_bilinear_gm(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, data_depth_score, grad_value_ptr,
              grad_sampling_loc_out, grad_attn_weight_out, grad_depth_score);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}
#endif  // DEFORM_ATTN_CUDA_KERNEL
