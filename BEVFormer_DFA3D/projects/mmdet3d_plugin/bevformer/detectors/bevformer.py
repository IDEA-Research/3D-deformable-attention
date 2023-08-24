# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from BEVFormer (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) fundamentalvision. All rights reserved
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#  Modified by Hongyang Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask, GridMaskHybrid
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmdet.models import builder
import math


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

@DETECTORS.register_module()
class BEVFormer_DFA3D(BEVFormer):
    def __init__(self, use_grid_mask=False, 
                    pts_voxel_layer=None, 
                    pts_voxel_encoder=None, 
                    pts_middle_encoder=None, 
                    pts_fusion_layer=None, 
                    img_backbone=None, 
                    pts_backbone=None, 
                    img_neck=None, 
                    pts_neck=None, 
                    pts_bbox_head=None, 
                    img_roi_head=None, 
                    img_rpn_head=None, 
                    train_cfg=None, 
                    test_cfg=None, 
                    pretrained=None, 
                    video_test_mode=False,
                    d_bound=None,
                    pts_dpt_head=None):
        super().__init__(use_grid_mask, pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer, img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head, train_cfg, test_cfg, pretrained, video_test_mode)
        self.pts_dpt_head = builder.build_head(pts_dpt_head)
        self.grid_mask = GridMaskHybrid(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.img_backbone.train()

    def extract_img_feat(self, img, img_metas, dpt=None, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
                if not (dpt is None):
                    dpt.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
                if not (dpt is None):
                    _, _, C_dpt, _, _ = dpt.size()
                    dpt = dpt.reshape(B * N, C_dpt, H, W)
            if self.use_grid_mask:
                if not (dpt is None):
                    img, dpt = self.grid_mask(img, dpt)
                else:
                    img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        if len_queue is not None:
            if not (dpt is None):
                dpt = dpt.view(int(B/len_queue), len_queue, int(BN / B), *dpt.shape[1:])
        return img_feats_reshaped, dpt
    @auto_fp16(apply_to=('img', "dpt"))
    def extract_feat(self, img, dpt=None, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats, dpt_masked = self.extract_img_feat(img, img_metas, dpt, len_queue=len_queue)
        
        return img_feats, dpt_masked
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list, _ = self.extract_feat(img=imgs_queue, len_queue=len_queue)  # len([])=N_scale; bs, len_q, N_cam, C, H, W
            lidar2img_list = []
            for bs_id in range(bs):
                lidar2img_sample = []
                for q_id in range(len_queue):
                    lidar2img_sample.append(torch.FloatTensor(np.stack(img_metas_list[bs_id][q_id]["lidar2img"])))
                lidar2img_sample = torch.stack(lidar2img_sample, dim=0).to(img_feats_list[0].device)
                lidar2img_list.append(lidar2img_sample)
            lidar2img_list = torch.stack(lidar2img_list, dim = 0)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                lidar2img = lidar2img_list[:, i]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                dpt_dist, out_dpt_multiscale = self.pts_dpt_head(img_feats, lidar2img.flatten(2), return_dpt=True)
                prev_bev = self.pts_bbox_head(
                    img_feats, [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale], img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'dpt', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      dpt=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        dpt = dpt[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats, dpt_masked = self.extract_feat(img=img, dpt=dpt, img_metas=img_metas)

        lidar2img = torch.stack([torch.FloatTensor(np.stack(each["lidar2img"])) for each in img_metas], dim=0).\
            to(img_feats[0].device)
        dpt_dist, out_dpt_multiscale = self.pts_dpt_head(img_feats, lidar2img.flatten(2), return_dpt=True)
        losses = dict()

        loss_dpt_input = [dpt_masked, dpt_dist]
        loss_dpt = self.pts_dpt_head.loss(*loss_dpt_input)
        losses_pts = self.forward_pts_train(img_feats, [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale], 
                                            gt_bboxes_3d, gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        losses.update(loss_dpt)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          pts_dpt_dist,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, pts_dpt_dist, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses
    
    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False, **kwargs):
        """Test function"""
        lidar2img = torch.stack([torch.FloatTensor(np.stack(each["lidar2img"])) for each in img_metas], dim=0).\
            to(x[0].device)
        out_dpt_multiscale = self.pts_dpt_head(x, lidar2img.flatten(2), return_dpt=False)
        
        outs = self.pts_bbox_head(x, [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale], img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats, dpt_masked = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale, img=img, **kwargs)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

@DETECTORS.register_module()
class BEVFormer_DFA3D_GTDpt(BEVFormer_DFA3D):
    def __init__(self, use_grid_mask=False, pts_voxel_layer=None, pts_voxel_encoder=None, pts_middle_encoder=None, pts_fusion_layer=None, img_backbone=None, pts_backbone=None, img_neck=None, pts_neck=None, pts_bbox_head=None, img_roi_head=None, img_rpn_head=None, train_cfg=None, test_cfg=None, pretrained=None, video_test_mode=False, d_bound=None, pts_dpt_head=None):
        super().__init__(use_grid_mask, pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer, img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head, train_cfg, test_cfg, pretrained, video_test_mode, d_bound, pts_dpt_head)
        for key, value in self.pts_dpt_head.named_parameters():
            value.requires_grad = False
    @auto_fp16(apply_to=('img', 'dpt', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      dpt=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_dpt = dpt[:, :-1, ...]
        img = img[:, -1, ...]
        dpt = dpt[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_dpt, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats, dpt_masked = self.extract_feat(img=img, dpt=dpt, img_metas=img_metas)

        lidar2img = torch.stack([torch.FloatTensor(np.stack(each["lidar2img"])) for each in img_metas], dim=0).\
            to(img_feats[0].device)
        dpt_dist, out_dpt_multiscale = self.pts_dpt_head(img_feats, lidar2img.flatten(2), dpt_masked, return_dpt=True)
        losses = dict()

        loss_dpt_input = [dpt_masked, dpt_dist]
        loss_dpt = self.pts_dpt_head.loss(*loss_dpt_input)
        losses_pts = self.forward_pts_train(img_feats, [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale], 
                                            gt_bboxes_3d, gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        losses.update(loss_dpt)
        return losses
    def obtain_history_bev(self, imgs_queue, dpt_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            dpt_queue  = dpt_queue.reshape(bs*len_queue, num_cams, 1, H, W)
            img_feats_list, dpt_masked_list = self.extract_feat(img=imgs_queue, dpt=dpt_queue, len_queue=len_queue)  # len([])=N_scale; bs, len_q, N_cam, C, H, W
            lidar2img_list = []
            for bs_id in range(bs):
                lidar2img_sample = []
                for q_id in range(len_queue):
                    lidar2img_sample.append(torch.FloatTensor(np.stack(img_metas_list[bs_id][q_id]["lidar2img"])))
                lidar2img_sample = torch.stack(lidar2img_sample, dim=0).to(img_feats_list[0].device)
                lidar2img_list.append(lidar2img_sample)
            lidar2img_list = torch.stack(lidar2img_list, dim = 0)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                lidar2img = lidar2img_list[:, i]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                dpt_masked = dpt_masked_list[:, i].flatten(0,1)
                dpt_dist, out_dpt_multiscale = self.pts_dpt_head(img_feats, lidar2img.flatten(2), dpt_masked, return_dpt=True)
                prev_bev = self.pts_bbox_head(
                    img_feats, [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale], img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def simple_test(self, img_metas, img=None, dpt=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats, dpt_masked = self.extract_feat(img=img, dpt=dpt[0], img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, dpt_masked, img_metas, prev_bev, rescale=rescale, img=img, **kwargs)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
    def simple_test_pts(self, x, dpt_masked, img_metas, prev_bev=None, rescale=False, **kwargs):
        """Test function"""
        lidar2img = torch.stack([torch.FloatTensor(np.stack(each["lidar2img"])) for each in img_metas], dim=0).\
            to(x[0].device)
        out_dpt_multiscale = self.pts_dpt_head(x, lidar2img.flatten(2), dpt_masked, return_dpt=False)
        
        outs = self.pts_bbox_head(x, [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale], img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results
