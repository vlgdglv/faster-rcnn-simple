import math
import torch
from torch.nn import functional as F
from torch import nn, Tensor

import torchvision
from torchvision.ops import nms

from .utils import *

class RPNHead(nn.Module):

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.object_pred = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1), stride=(1, 1))
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1))

        for layer in self.children():
            nn.init.normal_(layer.weight, std=1e-2)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feature: Tensor):
        o = F.relu(self.conv3x3(feature))
        object_res = self.object_pred(o)
        bbox_res = self.bbox_pred(o)
        return object_res, bbox_res


class RegionProposalNetwork(nn.Module):

    def __init__(self,
                 # structures
                 anchor_generater, rpn_head,
                 # thresholds
                 fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                 nms_thresh=0.7, score_thresh=.0,
                 # sampler
                 positive_proportion=0.5, batch_size_per_image=256,
                 #
                 pre_nms_top_n_train=2000, pre_nms_top_n_test=1000,
                 post_nms_top_n_train=2000, post_nms_top_n_test=1000,
                 ):
        super(RegionProposalNetwork, self).__init__()

        self.rpn_head = rpn_head
        self.anchor_generater = anchor_generater

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.po_proportion = positive_proportion
        self.batch_size_per_image = batch_size_per_image
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test
        self.min_box_size = 1e-3

    def forward(self, images, targets, feature):
        num_images = len(images.tensors)
        objectness, bbox_deltas = self.rpn_head(feature)
        anchors = self.anchor_generater(images, feature)
        num_anchors = anchors[0].shape[0]
        objectness, bbox_deltas = concat_rpn_head_outputs(objectness, bbox_deltas)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        proposals = decode_bbox(bbox_deltas, anchors, weights)
        proposals = proposals.view(num_images, -1, 4)

        boxes, prob = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors)

        losses = {}
        if self.training:
            # labels: -1: discard, 0: background, 1: foreground
            labels, matched_gt_boxes = self.match_targets_to_anchors(anchors, targets)
            sampled_idx = self.select_training_samples(labels)

            sampled_objectness = objectness[sampled_idx]
            sampled_bbox_deltas = bbox_deltas[sampled_idx]

            labels = torch.cat(labels, dim=0)
            sampled_label = labels[sampled_idx]
            sampled_objectness = sampled_objectness.flatten()
            loss_objectness = F.binary_cross_entropy_with_logits(sampled_objectness, sampled_label)

            regression_targets = get_regression_targets(anchors, matched_gt_boxes, sampled_idx)
            loss_bbox = smooth_l1_loss(sampled_bbox_deltas, regression_targets, beta=1/9) / sampled_idx.numel()

            losses = {
                "rpn_loss_objectness": loss_objectness,
                "rpn_loss_bbox": loss_bbox,
            }
        return boxes, losses

    def filter_proposals(self, proposals, objectness, image_sizes, num_anchors):
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        topk_idx = self.get_top_n_idx(objectness, num_anchors)
        num_images = torch.arange(num_images, device=device)
        batch_idx = num_images[:, None]

        objectness = objectness[batch_idx, topk_idx]
        proposals = proposals[batch_idx, topk_idx]

        objectness_prob = torch.sigmoid(objectness)

        ret_boxes = []
        ret_prob = []
        for boxes, prob, image_size in zip(proposals, objectness_prob, image_sizes):
            boxes = clip_box_within_image(boxes, image_size)
            kept = remove_small_boxes(boxes, self.min_box_size)
            boxes, prob = boxes[kept], prob[kept]

            kept = torch.where(prob >= self.score_thresh)[0]
            boxes, prob = boxes[kept], prob[kept]

            kept = nms(boxes, prob, self.nms_thresh)
            kept = kept[:self.get_post_nms_top_n()]
            boxes, prob = boxes[kept], prob[kept]
            
            ret_boxes.append(boxes)
            ret_prob.append(prob)

        return ret_boxes, ret_prob

    def match_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_matrix = box_iou(gt_boxes, anchors_per_image)
                # match_matrix is M (gt) x N (predicted)
                # Max over gt elements (dim 0) to find best gt candidate for each prediction
                matched_vals, matches = match_matrix.max(dim=0)
                matches_copy = matches.clone()
                
                bg_idx = matched_vals < self.bg_iou_thresh
                matches[bg_idx] = -1
                abandons_idx = (matched_vals >= self.bg_iou_thresh) & (matched_vals < self.fg_iou_thresh)
                matches[abandons_idx] = -2
                
                # low quality match
                highest_quality_match, _ = match_matrix.max(dim=1)
                # find all matches that has highest quality score
                highest_quality_match_all = torch.where(match_matrix == highest_quality_match[:, None])
                # update low quality match
                highest_match_idx = highest_quality_match_all[1]
                matches[highest_match_idx] = matches_copy[highest_match_idx]

                matched_gt_boxes_per_image = gt_boxes[matches.clamp(min=0)]
                labels_per_image = (matches >= 0)
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                # backgrounds:
                labels_per_image[matches == -1] = .0
                # discard betweens
                labels_per_image[matches == -2] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def select_training_samples(self, labels):
        # sample background and foreground
        sampled_fg_idx = []
        sampled_bg_idx = []
        for label_per_image in labels:
            positives = torch.where(label_per_image >= 1)[0]
            negatives = torch.where(label_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.po_proportion)
            num_pos = min(num_pos, positives.numel())
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(num_neg, negatives.numel())

            # randomly select positive and negative examples
            perm_pos = torch.randperm(positives.numel(), device=positives.device)[:num_pos]
            perm_neg = torch.randperm(negatives.numel(), device=negatives.device)[:num_neg]
            pos_idx = positives[perm_pos]
            neg_idx = negatives[perm_neg]
            # create binary mask from indices
            pos_idx_mask = torch.zeros_like(label_per_image, dtype=torch.uint8)
            neg_idx_mask = torch.zeros_like(label_per_image, dtype=torch.uint8)
            pos_idx_mask[pos_idx] = 1
            neg_idx_mask[neg_idx] = 1

            sampled_fg_idx.append(pos_idx_mask)
            sampled_bg_idx.append(neg_idx_mask)
        # concat bg and fg
        sampled_fg_idx = torch.where(torch.cat(sampled_fg_idx, dim=0))[0]
        sampled_bg_idx = torch.where(torch.cat(sampled_bg_idx, dim=0))[0]
        return torch.cat([sampled_fg_idx, sampled_bg_idx], dim=0)

    def get_top_n_idx(self, objectness, num_anchors):
        which_top_n = self.get_pre_nms_top_n()
        pre_nms_top_n = min(num_anchors, which_top_n)
        _, top_n_idx = objectness.topk(pre_nms_top_n, dim=1)
        return top_n_idx

    def get_pre_nms_top_n(self):
        if self.training:
            return self.pre_nms_top_n_train
        else:
            return self.pre_nms_top_n_test

    def get_post_nms_top_n(self):
        if self.training:
            return self.post_nms_top_n_train
        else:
            return self.post_nms_top_n_test


def concat_rpn_head_outputs(cls, deltas):
    N, AxC, H, W = cls.shape
    Ax4 = deltas.shape[1]
    A = Ax4 // 4
    C = AxC // A
    cls = permute_and_flatten(cls, N, C, H, W)
    deltas = permute_and_flatten(deltas, N, 4, H, W)
    cls = cls.flatten(0, -2)
    deltas = deltas.reshape(-1, 4)
    return cls, deltas


def permute_and_flatten(inuput, N, C, H, W):
    inuput = inuput.view(N, -1, C, H, W)
    inuput = inuput.permute(0, 3, 4, 1, 2)
    inuput = inuput.reshape(N, -1, C)
    return inuput


def get_regression_targets(anchors, matched_gt_boxes, sampled_idx):
    anchors = torch.cat(anchors, dim=0)
    matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
    anchors = anchors[sampled_idx]
    matched_gt_boxes = matched_gt_boxes[sampled_idx]
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    return encode_box(matched_gt_boxes, anchors, weights)
