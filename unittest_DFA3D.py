import os
import sys; sys.path.insert(0, '../')
from dfa3D.ops.multi_scale_3D_deform_attn import (
    WeightedMultiScaleDeformableAttnFunction,
    MultiScaleDepthScoreSampleFunction, MultiScale3DDeformableAttnFunction
)
import torch

def conduct_3D_deformable_attention(data):
    # (two-stage DFA3D) 1. get depth score
    depth_score = MultiScaleDepthScoreSampleFunction.apply(
        data["value_dpt_dist"],  # bs, spatial_size(h*w), num_head, D
        data["spatial_shapes_3D"],  # num_level 3
        data["level_start_index"],  # num_level
        data["sampling_locations"],  # bs, num_query, num_heads, num_point, num_points, 3
        data["im2col_step"]
    )
    # (two-stage DFA3D) 2. weighted 2D deformable attention
    output = WeightedMultiScaleDeformableAttnFunction.apply(
        data["value"], data["spatial_shapes"], data["level_start_index"], data["sampling_locations_2D"],
        data["attention_weights"], depth_score, data["im2col_step"]
    )

    # one stage DFA3D
    output_2 = MultiScale3DDeformableAttnFunction.apply(
        data["value"], data["value_dpt_dist"], data["spatial_shapes_3D"], data["level_start_index"], data["sampling_locations"],
        data["attention_weights"], data["im2col_step"]
    )
    return output, output_2
def get_multi_scale_features(data):
    spatial_shapes_3D = data["spatial_shapes_3D"]
    count = 0
    feat_multi_scale = []
    dist_multi_scale = []
    for spatial_shape_3D in spatial_shapes_3D:
        spatial_shape_2D = spatial_shape_3D[:2]
        feat_scale = data["value"][:, count:count+spatial_shape_2D[0]*spatial_shape_2D[1]]
        dist_scale = data["value_dpt_dist"][:, count:count+spatial_shape_2D[0]*spatial_shape_2D[1]]
        feat_multi_scale.append(feat_scale.view(feat_scale.shape[0], *spatial_shape_2D, *feat_scale.shape[2:]))
        dist_multi_scale.append(dist_scale.view(dist_scale.shape[0], *spatial_shape_2D, *dist_scale.shape[2:]))
        count += spatial_shape_2D[0]*spatial_shape_2D[1]
    return feat_multi_scale, dist_multi_scale
def get_random_data():
    data = {
        "value_dpt_dist": torch.randn([6, 30825, 8, 112]).softmax(dim=-1).cuda(),
        "spatial_shapes_3D": torch.tensor([[116, 200, 112],
                    [ 58, 100, 112],
                    [ 29,  50, 112],
                    [ 15,  25, 112]]).cuda(),
        "level_start_index": torch.tensor([    0, 23200, 29000, 30450]).cuda(),
        "sampling_locations": torch.rand([6, 9502, 8, 4, 8, 3]).cuda(),
        "value": torch.randn([6, 30825, 8, 32]).cuda(),
        "attention_weights": torch.rand([6, 9502, 8, 4, 8]).cuda(),
        "im2col_step": 32
    }
    return data
if __name__ == "__main__":
    # ===================== get&prepare data =====================
    data = get_random_data()
    data["spatial_shapes"] = data["spatial_shapes_3D"][:, :2]
    data["sampling_locations_2D"] = data["sampling_locations"][..., :2].contiguous()
    for n, v in data.items():
        try:
            data[n] = v.contiguous().detach()
        except:
            continue
    feat_multi_scale, dist_multi_scale = get_multi_scale_features(data)
    data["feat_multi_scale"] = feat_multi_scale
    data["dist_multi_scale"] = dist_multi_scale
    # ===================== 3D Deformable Attention =====================
    output, output_2 = conduct_3D_deformable_attention(data)
    if torch.isnan(output.sum()):
        print("Error")