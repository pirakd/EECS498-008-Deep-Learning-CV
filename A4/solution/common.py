"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)
        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        self.fpn_params['c3_latheral'] = nn.Conv2d(dummy_out['c3'].shape[1], self.out_channels, kernel_size=1, stride=1)
        self.fpn_params['c4_latheral'] = nn.Conv2d(dummy_out['c4'].shape[1], self.out_channels, kernel_size=1, stride=1)
        self.fpn_params['c5_latheral'] = nn.Conv2d(dummy_out['c5'].shape[1], self.out_channels, kernel_size=1, stride=1)
        self.fpn_params['p3'] = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.fpn_params['p4'] = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.fpn_params['p5'] = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        fpn_feats['p5'] = self.fpn_params['p5'](self.fpn_params['c5_latheral'](backbone_feats['c5']))

        fpn_feats['p4'] = self.fpn_params['p4'](self.fpn_params['c4_latheral'](backbone_feats['c4'])) +\
                          F.interpolate(fpn_feats['p5'], scale_factor=(2, 2))

        fpn_feats['p3'] = self.fpn_params['p3'](self.fpn_params['c3_latheral'](backbone_feats['c3'])) +\
                          F.interpolate(fpn_feats['p4'], scale_factor=2 )
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code

        locations = torch.empty(feat_shape[2]*feat_shape[3], 2)
        locations[:, 0] = (torch.arange(feat_shape[2]) + 0.5).repeat_interleave(feat_shape[3])
        locations[:, 1] = (torch.arange(feat_shape[3]) + 0.5).repeat(feat_shape[2])
        locations = locations * level_stride
        location_coords[level_name] = locations.to(device)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code

    def iou(box, boxes):
        """
        :param box1: tensor (x1, y1, x2, y2)
        :param box_2: tensor[n,4] (x1, y1, x2, y2)]
        :return:
        """
        x_11, y_11, x_12, y_12 = box
        x_21, y_21, x_22, y_22 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w_1 = x_12 - x_11
        h_1 = y_12 - y_11
        w_2 = x_22 - x_21
        h_2 = y_22 - y_21
        left_edge = torch.maximum(x_11, x_21)
        upper_edge = torch.maximum(y_11, y_21)
        right_edge = torch.minimum(x_12, x_22)
        lower_edge = torch.minimum(y_12, y_22)

        intersect_height = torch.clip(lower_edge - upper_edge, min=0)
        intersect_width = torch.clip(right_edge - left_edge, min=0)
        intersection_area = intersect_height * intersect_width
        union_area = (w_1 * h_1 + w_2 * h_2) - intersection_area
        return intersection_area / union_area
    keep = []
    alive_indices = set(list(range(boxes.shape[0])))
    sorted_scores_idx = torch.argsort(-scores)

    for highest_score_idx in sorted_scores_idx:
        highest_score_idx = highest_score_idx.item()
        if highest_score_idx not in alive_indices:
            continue
        keep.append(highest_score_idx)
        alive_indices.remove(highest_score_idx)
        alive_list = list(alive_indices)
        pairs_iou = iou(boxes[highest_score_idx, :], boxes[alive_list, :])
        alive_indices = alive_indices.difference(set([alive_list[x] for x in torch.nonzero(pairs_iou >= iou_threshold)]))
        if len(alive_indices) == 0:
            break
    return torch.tensor(keep)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################



def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


if __name__ == '__main__':
    import time
    import torchvision
    DEVICE = 'cpu'
    boxes = (100.0 * torch.rand(5000, 4)).round()
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0] + 1.0
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1] + 1.0
    scores = torch.randn(5000)

    names = ["your_cpu", "torchvision_cpu", "torchvision_cuda"]
    iou_thresholds = [0.3, 0.5, 0.7]
    elapsed = dict(zip(names, [0.0] * len(names)))
    intersects = dict(zip(names[1:], [0.0] * (len(names) - 1)))

    for iou_threshold in iou_thresholds:
        tic = time.time()
        my_keep = nms(boxes, scores, iou_threshold)
        elapsed["your_cpu"] += time.time() - tic

        tic = time.time()
        tv_keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        elapsed["torchvision_cpu"] += time.time() - tic
        intersect = len(set(tv_keep.tolist()).intersection(my_keep.tolist())) / len(tv_keep)
        intersects["torchvision_cpu"] += intersect

        tic = time.time()
        tv_cuda_keep = torchvision.ops.nms(boxes.to(device=DEVICE), scores.to(device=DEVICE), iou_threshold).to(
            my_keep.device
        )
        # torch.cuda.synchronize()
        elapsed["torchvision_cuda"] += time.time() - tic
        intersect = len(set(tv_cuda_keep.tolist()).intersection(my_keep.tolist())) / len(
            tv_cuda_keep
        )
        intersects["torchvision_cuda"] += intersect

    for key in intersects:
        intersects[key] /= len(iou_thresholds)

    # You should see < 1% difference
    print("Testing NMS:")
    print("Your        CPU  implementation: %fs" % elapsed["your_cpu"])
    print("torchvision CPU  implementation: %fs" % elapsed["torchvision_cpu"])
    print("torchvision CUDA implementation: %fs" % elapsed["torchvision_cuda"])
    print("Speedup CPU : %fx" % (elapsed["your_cpu"] / elapsed["torchvision_cpu"]))
    print("Speedup CUDA: %fx" % (elapsed["your_cpu"] / elapsed["torchvision_cuda"]))
    print(
        "Difference CPU : ", 1.0 - intersects["torchvision_cpu"]
    )  # in the order of 1e-3 or less
    print(
        "Difference CUDA: ", 1.0 - intersects["torchvision_cuda"]
    )  # in the order of 1e-3 or less