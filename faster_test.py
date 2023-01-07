import torch
import torchvision
import numpy as np
from torchvision.ops import MultiScaleRoIAlign
from typing import List, Tuple

from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.transform import RCNNTransform
from faster_rcnn.rpn import RegionProposalNetwork, RPNHead
from faster_rcnn.anchor import AnchorGener
from faster_rcnn.rcnn_head import RCNNHead
from faster_rcnn.rcnn import RCNNHeadMLP, RCNNPredictor


def test_transform(images, targets, show=False):
    t = RCNNTransform()
    images, targets = t(images, targets)
    if show:
        print(images.tensors[0])
        print("*" * 30)
        print(targets)
    return images, targets


def test_backbone(images, show=False):
    backbone = torchvision.models.vgg16(pretrained=True).features
    # backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    backbone.out_channels = 512
    features = backbone(images.tensors)
    if show:
        print(features)
    return features


def test_rpn(images, targets, features, show=False):
    out_channels = 512

    rpn_anchor_gener = AnchorGener()
    rpn_head = RPNHead(out_channels, rpn_anchor_gener.num_anchors_per_location)

    rpn = RegionProposalNetwork(rpn_anchor_gener, rpn_head)

    boxes, losses = rpn(images, targets, features)
    if show:
        print(boxes)
        print(losses)
    return boxes, losses


def test_rcnn_head(features, proposals, image_sizes, targets):

    num_classes = 6
    fc_nerual_num = 1024
    out_channels = 512

    rcnn_roi_pool_outsize = 7
    rcnn_fc = RCNNHeadMLP(out_channels * rcnn_roi_pool_outsize ** 2, fc_nerual_num)
    rcnn_predictor = RCNNPredictor(fc_nerual_num, num_classes)

    rcnn_fg_iou_thresh = 0.5
    rcnn_bg_iou_thresh = 0.5
    rcnn_batch_size_per_image = 512
    rcnn_positive_proportion = 0.25
    rcnn_score_thresh = 0.05
    rcnn_nms_thresh = 0.5
    rcnn_detections_per_img = 100

    rcnn_head = RCNNHead(
        rcnn_fc, rcnn_predictor, rcnn_roi_pool_outsize,
        rcnn_fg_iou_thresh, rcnn_bg_iou_thresh,
        rcnn_batch_size_per_image, rcnn_positive_proportion,
        rcnn_score_thresh, rcnn_nms_thresh, rcnn_detections_per_img
    )

    result, losses = rcnn_head(features, proposals, image_sizes, targets)
    print(result)
    print(losses)


if __name__ == "__main__":

    # images = torch.rand(2, 3, 400, 600)

    # images = torch.rand(2, 3, 600, 1200)
    images = [torch.rand(3, 500, 375), torch.rand(3, 500, 333)]

    targets = [
        {
            "boxes": torch.tensor([
                [100, 100, 300, 300],
                [200, 200, 500, 500],
                [300, 200, 400, 500]]),
            "labels": torch.tensor([3, 4, 1])
        },
        {
            "boxes": torch.tensor([
                [100, 200, 300, 500],
                [100, 100, 200, 400]]),
            "labels": torch.tensor([5, 2])
        }
    ]
    # images, targets = test_transform(images, targets)
    # features = test_backbone(images)
    # proposals, losses = test_rpn(images, targets, features)
    # test_rcnn_head(features, proposals, images.image_sizes, targets)
    backbone = torchvision.models.vgg16(pretrained=True).features
    # backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    backbone.out_channels = 512
    fr = FasterRCNN(backbone, num_classes=6)

    result, losses = fr(images, targets)
    print(result)
    print(losses)
