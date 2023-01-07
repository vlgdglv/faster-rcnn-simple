import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import check_images, check_targets


class RCNN(nn.Module):
    """
    Abstract RCNN
    """
    def __init__(self, transform, backbone, rpn, rcnn_heads):
        super(RCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.rcnn_heads = rcnn_heads

    def forward(self, images, targets=None):
        """
        :param images: list[Tensor], raw images
        :param targets: list[dict[Tensor]], ground truth boxes and corresponding labels
                dict[Tensor]: targets['boxes']=tensor((N,4)), targets['labels']=tensor((N,))
        :return: detection results and losses
        """

        check_images(images)
        if self.training and not check_targets(targets):
            raise ValueError("Invalid targets, but I'm too lazy to tell the details.")

        # step 1: image and targets transform
        images, targets = self.transform(images, targets)

        # step 2: feed into convolutional backbone, generating feature map
        features = self.backbone(images.tensors)

        # step 3: feed features into rpn to generate proposals
        proposals, rpn_losses = self.rpn(images, targets, features)

        # step 4: feed proposals into rcnn head to get final prediction results
        detections, rcnn_losses = self.rcnn_heads(features, proposals, images.image_sizes, targets)

        # step 5: post process, e.g. transform box and image scales
        detections = self.transform.postprocess(detections, images.image_sizes, images.original_image_sizes)

        # losses in training
        losses = {}
        losses.update(rpn_losses)
        losses.update(rcnn_losses)

        return detections, losses


class RCNNHeadMLP(nn.Module):
    """
    RCNN fully connected layers
    """
    def __init__(self, in_channels, out_channels):
        super(RCNNHeadMLP, self).__init__()
        self.fc6 = nn.Linear(in_channels, out_channels)
        self.fc7 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class RCNNPredictor(nn.Module):
    """
    RCNN predict head
    """
    def __init__(self, in_channels, num_classes):
        super(RCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_reg = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        cls = self.cls_score(x)
        reg = self.bbox_reg(x)

        return cls, reg
