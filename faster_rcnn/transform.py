import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from .utils import *
from .image import ImageList


class RCNNTransform(nn.Module):

    def __init__(self, min_size=600, max_size=1333, image_mean=None, image_std=None):
        super(RCNNTransform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        # make copies
        images = [img for img in images]
        if targets is not None:
            # targets_copy = [{k: v for k, v in target.item()} for target in targets]
            targets_copy = []
            for t in targets:
                data = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        # store origianl sizes
        original_image_sizes = [img.shape[-2:] for img in images]

        # transform each image
        for idx, image in enumerate(images):
            target = targets[idx] if targets is not None else None
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[idx] = image
            if targets is not None and target is not None:
                targets[idx] = target

        # get transformed sizes
        image_sizes = [img.shape[-2:] for img in images]

        # batch images
        images = batch_up(images)
        images_list = ImageList(images, image_sizes, original_image_sizes)
        return images_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        img_shape = torch.tensor(image.shape[-2:])
        img_min_size = float(torch.min(img_shape))
        img_max_size = float(torch.max(img_shape))
        min_size = float(self.min_size)
        max_size = float(self.max_size)
        scale_ratio = min_size / img_min_size
        if img_max_size * scale_ratio > max_size:
            scale_ratio = max_size / img_max_size
        image = nn.functional.interpolate(image[None],
                                          scale_factor=scale_ratio, mode='bilinear',
                                          recompute_scale_factor=True, align_corners=False)[0]

        if target is not None:
            bbox = target["boxes"]
            new_img_shape = image.shape[-2:]
            height_ratio = torch.tensor(new_img_shape[0], dtype=torch.float32, device=bbox.device) / \
                           img_shape[0].clone().detach()
            width_ratio = torch.tensor(new_img_shape[1], dtype=torch.float32, device=bbox.device) / \
                          img_shape[1].clone().detach()
            xmin, ymin, xmax, ymax = bbox.unbind(1)
            xmin = xmin * width_ratio
            xmax = xmax * width_ratio
            ymin = ymin * height_ratio
            ymax = ymax * height_ratio
            target["boxes"] = torch.stack((xmin, ymin, xmax, ymax), dim=1)

        return image, target

    def postprocess(self, detections, image_sizes, original_image_sizes):
        if self.training:
            return detections
        else:
            for i, (det, image_size, original_size) in enumerate(zip(detections, image_sizes, original_image_sizes)):
                boxes = det["boxes"]
                boxes = recover_boxes(boxes, image_size, original_size)
                boxes = boxes.to(torch.int32)
                detections[i]["boxes"] = boxes
            return detections
