import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign


from .rcnn import RCNN, RCNNHeadMLP, RCNNPredictor
from .rcnn_head import RCNNHead
from .transform import RCNNTransform
from .rpn import RegionProposalNetwork, RPNHead
from .anchor import AnchorGener


class FasterRCNN(RCNN):
    """
    Implements Faster R-CNN.
    """

    def __init__(self, backbone, num_classes=None,
                 # transform:
                 min_size=512, max_size=1280,
                 image_mean=None, image_std=None,
                 # rpn:
                 rpn_anchor_gener=None, rpn_head=None,
                 rpn_anchor_sizes=None, rpn_anchor_aspect_ratios=None,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_nms_thresh=0.7, rpn_score_thresh=.0,
                 rpn_pn_proportion=0.5, rpn_batch_size_per_image=256,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 # rcnn head:
                 rcnn_fc=None, rcnn_predictor=None, rcnn_roi_pool_outsize=7,
                 rcnn_fg_iou_thresh=0.5, rcnn_bg_iou_thresh=0.5,
                 rcnn_batch_size_per_image=512, rcnn_positive_proportion=0.25,
                 rcnn_score_thresh=0.05, rcnn_nms_thresh=0.5, rcnn_detections_per_img=100
                 ):

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = RCNNTransform(min_size, max_size, image_mean, image_std)

        out_channels = backbone.out_channels
        if rpn_anchor_gener is None:
            rpn_anchor_gener = AnchorGener(rpn_anchor_sizes, rpn_anchor_aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_gener.num_anchors_per_location)

        rpn = RegionProposalNetwork(
            rpn_anchor_gener, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_nms_thresh, rpn_score_thresh,
            rpn_pn_proportion, rpn_batch_size_per_image,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
        )

        fc_nerual_num = 1024
        if rcnn_fc is None:
            rcnn_fc = RCNNHeadMLP(out_channels * rcnn_roi_pool_outsize ** 2, fc_nerual_num)

        if rcnn_predictor is None:
            rcnn_predictor = RCNNPredictor(fc_nerual_num, num_classes)

        rcnn_head = RCNNHead(
            rcnn_fc, rcnn_predictor, rcnn_roi_pool_outsize,
            rcnn_fg_iou_thresh, rcnn_bg_iou_thresh,
            rcnn_batch_size_per_image, rcnn_positive_proportion,
            rcnn_score_thresh, rcnn_nms_thresh, rcnn_detections_per_img
        )

        super(FasterRCNN, self).__init__(transform, backbone, rpn, rcnn_head)
