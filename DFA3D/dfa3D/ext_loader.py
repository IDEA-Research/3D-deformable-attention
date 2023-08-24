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

import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch

if torch.__version__ != 'parrots':

    def load_ext(name, funcs):
        ext = importlib.import_module('dfa3D.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
else:
    from parrots import extension
    from parrots.base import ParrotsException

    has_return_value_ops = [
        'nms',
        'softnms',
        'nms_match',
        'nms_rotated',
        'top_pool_forward',
        'top_pool_backward',
        'bottom_pool_forward',
        'bottom_pool_backward',
        'left_pool_forward',
        'left_pool_backward',
        'right_pool_forward',
        'right_pool_backward',
        'fused_bias_leakyrelu',
        'upfirdn2d',
        'ms_deform_attn_forward',
        'pixel_group',
        'contour_expand',
    ]

    def get_fake_func(name, e):

        def fake_func(*args, **kwargs):
            warnings.warn(f'{name} is not supported in parrots now')
            raise e

        return fake_func

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            try:
                ext_fun = extension.load(fun, name, lib_dir=lib_root)
            except ParrotsException as e:
                if 'No element registered' not in e.message:
                    warnings.warn(e.message)
                ext_fun = get_fake_func(fun, e)
                ext_list.append(ext_fun)
            else:
                if fun in has_return_value_ops:
                    ext_list.append(ext_fun.op)
                else:
                    ext_list.append(ext_fun.op_)
        return ExtModule(*ext_list)


def check_ops_exist():
    ext_loader = pkgutil.find_loader('dfa3D._ext')
    return ext_loader is not None
