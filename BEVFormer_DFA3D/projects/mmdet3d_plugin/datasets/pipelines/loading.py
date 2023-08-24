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
# Modified from bevdepth (https://github.com/Megvii-BaseDetection/BEVDepth)
# Copyright (c) 2022 Megvii-BaseDetection
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------ 
#  Modified by Hongyang Li
# ---------------------------------------------

import os
import pdb
import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES
@PIPELINES.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size     = map_size
    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2]
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []
        for img_path in img_paths:
            dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt", img_path.split("/")[-1]+".bin")
            point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
            dpt_paths.append(dpt_path)
            if self.is_to_depth_map:
                map_depth = self.to_depth_map(point_depth)
                map_depths.append(map_depth)
        # img is of shape (h, w, c, num_views)
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """


        depth_coords = point_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(self.map_size)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]

        return depth_map
