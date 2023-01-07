import math
import torch
from .utils import *
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.roi_align import roi_align
from torchvision.ops import batched_nms


class RCNNHead(nn.Module):
    def __init__(self, rcnn_fc, rcnn_predictor, roi_pool_outsize,
                 # train settings:
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_proportion,
                 # inference settings:
                 score_thresh, nms_thresh, detections_per_img):
        super(RCNNHead, self).__init__()
        self.rcnn_fc = rcnn_fc
        self.rcnn_predictor = rcnn_predictor
        self.roi_pool_outsize = roi_pool_outsize
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.po_proportion = positive_proportion
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(self, features, proposals, image_sizes, targets=None):
        """

        @param features: Tensor, shape: ([N, 512, H, W])
        @param proposals: List[Tensor], Tensor shape: ([S, 4])
        @param image_sizes: List[Size]
        @param targets: List[Dict[Str: Tensor]]
        @return:
        """
        if self.training:
            # matched idx: the gt box matches the corresponding proposal box
            proposals, labels, matched_idxs, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        _features = roi_align(features, proposals, output_size=self.roi_pool_outsize, sampling_ratio=2)
        _features = self.rcnn_fc(_features)
        cls_pred, bbox_delta_pred = self.rcnn_predictor(_features)

        result = []
        losses = {}
        if self.training:
            labels = torch.cat(labels, dim=0)
            loss_classifier = F.cross_entropy(cls_pred, labels)

            pos_idx = torch.where(labels > 0)[0]
            pos_labels = labels[pos_idx]

            N, num_classes = cls_pred.shape
            bbox_delta_pred = bbox_delta_pred.reshape(N, -1, 4)
            bbox_delta_pred = bbox_delta_pred[pos_idx, pos_labels]
            regression_targets = regression_targets[pos_idx]
            loss_bbox_reg = smooth_l1_loss(bbox_delta_pred, regression_targets) / labels.numel()
            losses = {
                "rcnn_loss_classifier": loss_classifier,
                "rnn_loss_bbox_reg": loss_bbox_reg
            }
        else:
            result_bboxes, result_labels, result_scores = \
                self.process_result(cls_pred, bbox_delta_pred, proposals, image_sizes)
            num_images = len(result_bboxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": result_bboxes[i],
                        "labels": result_labels[i],
                        "scores": result_scores[i],
                    }
                )

        return result, losses

    def select_training_samples(self, proposals, targets):
        """
        @param proposals: List[Tensor]
        @param targets: List[Dict[Str: Tensor]]
        @return: List[Tensor], List[Tensor], List[Tensor], Tensor
        """
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        labels, matched_idxs = self.match_targets_to_proposals(proposals, gt_boxes, gt_labels)
        sampled_idx = self.subsample(labels)
        matched_gt_boxes = []
        for image_idx in range(len(proposals)):
            image_sampled_idx = sampled_idx[image_idx]
            proposals[image_idx] = proposals[image_idx][image_sampled_idx]
            labels[image_idx] = labels[image_idx][image_sampled_idx]
            matched_idxs[image_idx] = matched_idxs[image_idx][image_sampled_idx]
            image_gt_boxes = gt_boxes[image_idx]
            if image_gt_boxes.numel() == 0:
                image_gt_boxes = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(image_gt_boxes[matched_idxs[image_idx]])

        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
        concat_proposals = torch.cat(proposals, dim=0)
        weights = torch.tensor([10.0, 10.0, 5.0, 5.0,])
        regression_targets = encode_box(matched_gt_boxes, concat_proposals, weights)
        return proposals, labels, matched_idxs, regression_targets

    def match_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        labels = []
        matched_idxs = []
        for proposal, gt_box, gt_label in zip(proposals, gt_boxes, gt_labels):
            if gt_box.numel() == 0:
                # Background image
                device = proposal.device
                matches = torch.zeros((proposal.shape[0],), dtype=torch.int64, device=device)
                matched_gt_labels = torch.zeros((proposal.shape[0],), dtype=torch.int64, device=device)
            else:
                match_matrix = box_iou(gt_box, proposal)
                # match_matrix is M (gt) x N (predicted)
                # Max over gt elements (dim 0) to find best gt candidate for each prediction
                matched_vals, matches = match_matrix.max(dim=0)
                bg_idx = matched_vals < self.bg_iou_thresh
                abandons_idx = (matched_vals >= self.bg_iou_thresh) & (matched_vals <= self.fg_iou_thresh)

                matches = matches.clamp(min=0)
                matched_gt_labels = gt_label[matches]
                matched_gt_labels = matched_gt_labels.to(dtype=torch.int64)

                # backgrounds:
                matched_gt_labels[bg_idx] = .0
                # discard betweens
                matched_gt_labels[abandons_idx] = -1.0
            labels.append(matched_gt_labels)
            matched_idxs.append(matches)
        return labels, matched_idxs

    def subsample(self, labels):
        sampled_idx = []
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

            sampled_idx.append(torch.where(pos_idx_mask | neg_idx_mask)[0])
        return sampled_idx

    def process_result(self, cls_pred, bbox_delta_pred, proposals, image_shapes):
        device = cls_pred.device
        num_classes = cls_pred.shape[-1]
        num_boxes_per_image = [proposal.shape[0] for proposal in proposals]

        cls_scores = F.softmax(cls_pred, -1)
        cls_scores = cls_scores.split(num_boxes_per_image, 0)

        weights = torch.tensor([10.0, 10.0, 5.0, 5.0,])
        bbox_pred = decode_bbox(bbox_delta_pred, proposals, weights)
        bbox_pred = bbox_pred.split(num_boxes_per_image, 0)

        result_bboxes = []
        result_scores = []
        result_labels = []
        for scores, boxes, image_shape in zip(cls_scores, bbox_pred, image_shapes):
            boxes = clip_box_within_image(boxes, image_shape)
            labels = torch.arange(0, num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            scores = scores[:, 1:]
            boxes = boxes[:, 1:]
            labels = labels[:, 1:]

            scores = scores.reshape(-1)
            boxes = boxes.reshape(-1, 4)
            labels = labels.reshape(-1)

            kept = torch.where(scores > self.score_thresh)[0]
            scores, boxes, labels = scores[kept], boxes[kept], labels[kept]

            kept = remove_small_boxes(boxes, min_size=1e-2)
            scores, boxes, labels = scores[kept], boxes[kept], labels[kept]

            kept = batched_nms(boxes, scores, labels, self.nms_thresh)
            kept = kept[:self.detections_per_img]
            scores, boxes, labels = scores[kept], boxes[kept], labels[kept]

            result_scores.append(scores)
            result_bboxes.append(boxes)
            result_labels.append(labels)
        return result_bboxes, result_labels, result_scores


