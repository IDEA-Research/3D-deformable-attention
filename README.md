3D Deformable Attention (DFA3D)
========
By [Hongyang Li*](https://scholar.google.com.hk/citations?view_op=list_works&hl=zh-CN&user=zdgHNmkAAAAJ&gmla=AMpAcmTJNHoetv6zgfzZkIRcYsFr0UkGGDyl5tAp5etuBqhz3lzYZCQrVDot02xVQ1XTbnMS1fPdAfe0-2--aTXOtewokjyShNLOQQyyhtkolwaz0hvENZpi-pJ-Wg), [Hao Zhang*](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=zh-CN), [Zhaoyang Zeng](https://scholar.google.com.hk/citations?user=U_cvvUwAAAAJ&hl=zh-CN&oi=sra), [Shilong Liu](https://scholar.google.com/citations?hl=zh-CN&user=nkSVY3MAAAAJ), [Feng Li](https://scholar.google.com.hk/citations?user=ybRe9GcAAAAJ&hl=zh-CN&oi=sra), [Tianhe Ren](https://scholar.google.com.hk/citations?user=cW4ILs0AAAAJ&hl=zh-CN&oi=sra), and [Lei Zhang](https://scholar.google.com/citations?hl=zh-CN&user=fIlGZToAAAAJ).

This repository is the official implementation of the paper "DFA3D: 3D Deformable Attention For 2D-to-3D Feature Lifting".

# News
[2023/7/15] Our paper is accepted by ICCV2023.


# TODO List
- [ ] Release 3D Deformable Attention.
- [ ] Release BEVFormer-DFA3D-PredDepth (-base & -small) and BEVFormer-DFA3D-GTDepth (including code and checkpoints).
- [ ] Release 3D attention visualization tool.


# Abstract
In this paper, we propose a new operator, called 3D DeFormable Attention (DFA3D), for 2D-to-3D feature lifting, which transforms multi-view 2D image features into a unified 3D space for 3D object detection. 
Existing feature lifting approaches, such as Lift-Splat-based and 2D attention-based, either use estimated depth to get pseudo LiDAR features and then splat them to a 3D space, which is a one-pass operation without feature refinement, or ignore depth and lift features by 2D attention mechanisms, which achieve finer semantics while suffering from a depth ambiguity problem. 
In contrast, our DFA3D-based method first leverages the estimated depth to expand each view's 2D feature map to 3D and then utilizes DFA3D to aggregate features from the expanded 3D feature maps. With the help of DFA3D, the depth ambiguity problem can be effectively alleviated from the root, and the lifted features can be progressively refined layer by layer, thanks to the Transformer-like architecture. In addition, we propose a mathematically equivalent implementation of DFA3D which can significantly improve its memory efficiency and computational speed. We integrate DFA3D into several methods that use 2D attention-based feature lifting with only a few modifications in code and evaluate on the nuScenes dataset. The experiment results show a consistent improvement of +1.41 mAP on average, and up to +15.1 mAP improvement when high-quality depth information is available, demonstrating the superiority, applicability, and huge potential of DFA3D.

# Method
## Comparisons of feature lifting methods.
<img src="figures/Comparisons.png">


## Improvements.
Our DFA3D brings consistent improvement on several methods, including two concurrent works ([DA-BEV](https://arxiv.org/abs/2302.13002)  and [Sparse4D](https://arxiv.org/abs/2211.10581)).

<img src="figures/Main_results.png" width="400px">

Improving the quality of depth will bring further gains (up to 15.1% mAP).

<img src="figures/Depth.png" width="400px">

## How to transform your 2D Attention-based feature lifting into our 3D Deformable Attention-based one.
Here, we take 2D Deformable Attention as an example, only a few modifications in code are required. For more details, please refer to our examples provided in Model Zoo (TODO).
<img src="figures/Modifications.png">

# Citation
```
@misc{li2023dfa3d,
      title={DFA3D: 3D Deformable Attention For 2D-to-3D Feature Lifting}, 
      author={Hongyang Li and Hao Zhang and Zhaoyang Zeng and Shilong Liu and Feng Li and Tianhe Ren and Lei Zhang},
      year={2023},
      eprint={2307.12972},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
